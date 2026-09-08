from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from multi_agent_sim.controller_catalog import (
    CONTROLLER_ARCHITECTURE,
    CONTROLLER_MANIFEST_VERSION,
    LEGACY_OBJECTIVE_SCHEMA_VERSION,
    discover_saved_controllers,
    publish_saved_controller,
)
from multi_agent_sim.session import SimulationSession, create_initial_scenario
from multi_agent_sim.episode import OBJECTIVE_SCHEMA_VERSION


TORCH_INSTALLED = importlib.util.find_spec("torch") is not None
if TORCH_INSTALLED:
    import torch
    from multi_agent_sim.learning import SharedMLP, load_checkpoint, save_checkpoint
    from multi_agent_sim.saved_mlp_controllers import (
        create_registry_with_saved_controllers,
    )


def _publication(checkpoint: Path, directory: Path, **overrides: object):
    values: dict[str, object] = {
        "directory": directory,
        "run_id": "12345678abcdef",
        "controller_name": "Warehouse Policy",
        "qualified_stage": 1,
        "evaluated_stage": 2,
        "success_rate": 0.87,
        "mean_cost": 1.25,
        "episodes_trained": 43750,
        "optimizer_updates": 1367,
        "promotion_success_rate": 0.8,
        "use_action_masks": True,
        "torch_version": "2.6.0",
        "curriculum": (
            {
                "number": 1,
                "width": 4,
                "height": 4,
                "num_robots": 1,
                "num_items": 1,
                "max_steps": 32,
            },
            {
                "number": 2,
                "width": 4,
                "height": 4,
                "num_robots": 1,
                "num_items": 2,
                "max_steps": 64,
            },
        ),
        "cost_weights": {"failure": 10.0, "item_progress": 0.1},
        "training_config": {"promotion_success_rate": 0.8},
        "created_at": datetime(2026, 9, 7, 12, 0, tzinfo=timezone.utc),
    }
    values.update(overrides)
    return publish_saved_controller(checkpoint, **values)


class ControllerCatalogTests(unittest.TestCase):
    def test_publish_uses_informative_name_manifest_commit_and_checksum_dedup(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "best.pt"
            checkpoint.write_bytes(b"fixed model bytes")
            catalog_directory = root / "controllers"

            first = _publication(checkpoint, catalog_directory)
            second = _publication(checkpoint, catalog_directory)
            catalog = discover_saved_controllers(catalog_directory)

            self.assertTrue(first.created)
            self.assertFalse(second.created)
            self.assertEqual(first.entry, second.entry)
            self.assertEqual(len(catalog.entries), 1)
            name = first.entry.model_path.name
            for fragment in (
                "warehouse-policy-s02-4x4-r1-i2-sr087",
                "ep043750",
                "20260907T120000Z",
                "12345678",
            ):
                self.assertIn(fragment, name)
            manifest = first.entry.manifest
            self.assertEqual(manifest.architecture, CONTROLLER_ARCHITECTURE)
            self.assertEqual(manifest.manifest_version, CONTROLLER_MANIFEST_VERSION)
            self.assertEqual(manifest.objective_schema_version, OBJECTIVE_SCHEMA_VERSION)
            self.assertTrue(manifest.qualified)
            self.assertEqual(manifest.qualified_stage, 1)
            self.assertEqual(manifest.evaluated_stage, 2)
            self.assertEqual(manifest.display_name, "Warehouse Policy · Stage 1 · 87% · ep 43,750")
            self.assertTrue(first.entry.manifest_path.exists())
            self.assertFalse(any(catalog_directory.glob(".*.tmp")))

    def test_unqualified_controller_is_clearly_labeled(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "best.pt"
            checkpoint.write_bytes(b"candidate")
            published = _publication(
                checkpoint,
                root / "controllers",
                controller_name=None,
                qualified_stage=0,
                success_rate=0.42,
            )

            self.assertFalse(published.entry.manifest.qualified)
            self.assertIn("Unqualified", published.entry.manifest.display_name)

    def test_discovery_skips_missing_modified_traversing_and_duplicate_entries(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "best.pt"
            checkpoint.write_bytes(b"model")
            catalog_directory = root / "controllers"
            published = _publication(checkpoint, catalog_directory)
            valid_payload = published.entry.manifest.to_dict()

            missing = dict(valid_payload)
            missing["controller_id"] = "missing"
            missing["model_file"] = "missing.pt"
            (catalog_directory / "missing.controller.json").write_text(
                json.dumps(missing), encoding="utf-8"
            )
            traversal = dict(valid_payload)
            traversal["controller_id"] = "traversal"
            traversal["model_file"] = "../outside.pt"
            (catalog_directory / "traversal.controller.json").write_text(
                json.dumps(traversal), encoding="utf-8"
            )
            duplicate = dict(valid_payload)
            (catalog_directory / "zz-duplicate.controller.json").write_text(
                json.dumps(duplicate), encoding="utf-8"
            )
            published.entry.model_path.write_bytes(b"modified")

            catalog = discover_saved_controllers(catalog_directory)

            self.assertEqual(catalog.entries, ())
            self.assertEqual(len(catalog.warnings), 4)
            self.assertTrue(any("checksum" in warning for warning in catalog.warnings))
            self.assertTrue(any("missing" in warning for warning in catalog.warnings))
            self.assertTrue(any("filename within" in warning for warning in catalog.warnings))

    def test_discovery_is_deterministic_and_rejects_duplicate_ids_and_versions(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            catalog_directory = root / "controllers"
            first_checkpoint = root / "first.pt"
            second_checkpoint = root / "second.pt"
            first_checkpoint.write_bytes(b"first")
            second_checkpoint.write_bytes(b"second")
            zebra = _publication(
                first_checkpoint, catalog_directory, controller_name="Zebra"
            )
            alpha = _publication(
                second_checkpoint, catalog_directory, controller_name="Alpha"
            )
            duplicate_payload = zebra.entry.manifest.to_dict()
            (catalog_directory / "zz-duplicate.controller.json").write_text(
                json.dumps(duplicate_payload), encoding="utf-8"
            )
            incompatible_payload = alpha.entry.manifest.to_dict()
            incompatible_payload["controller_id"] = "future"
            incompatible_payload["manifest_version"] = 999
            (catalog_directory / "future.controller.json").write_text(
                json.dumps(incompatible_payload), encoding="utf-8"
            )

            catalog = discover_saved_controllers(catalog_directory)

            self.assertEqual(
                tuple(entry.manifest.display_name.split(" · ")[0] for entry in catalog.entries),
                ("Alpha", "Zebra"),
            )
            self.assertTrue(any("duplicate controller ID" in warning for warning in catalog.warnings))
            self.assertTrue(any("manifest version" in warning for warning in catalog.warnings))

    def test_legacy_manifest_without_objective_version_remains_discoverable(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "old-best.pt"
            checkpoint.write_bytes(b"old inference weights")
            published = _publication(checkpoint, root / "controllers")
            payload = published.entry.manifest.to_dict()
            payload["manifest_version"] = 1
            payload.pop("objective_schema_version")
            published.entry.manifest_path.write_text(
                json.dumps(payload), encoding="utf-8"
            )

            catalog = discover_saved_controllers(root / "controllers")

            self.assertEqual(len(catalog.entries), 1)
            self.assertEqual(catalog.warnings, ())
            self.assertEqual(
                catalog.entries[0].manifest.objective_schema_version,
                LEGACY_OBJECTIVE_SCHEMA_VERSION,
            )


@unittest.skipUnless(TORCH_INSTALLED, "PyTorch training extra is not installed")
class SavedMLPSessionControllerTests(unittest.TestCase):
    def _catalog(self, root: Path):
        checkpoint = root / "best.pt"
        model = SharedMLP()
        with torch.no_grad():
            final = model.layers[-1]
            final.weight.zero_()
            final.bias.zero_()
            final.bias[5] = 10.0  # Wait
        save_checkpoint(model, checkpoint, {})
        return _publication(checkpoint, root / "controllers")

    def test_selected_model_loads_once_and_controls_multiple_robots(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            published = self._catalog(root)
            result = create_registry_with_saved_controllers(
                directory=root / "controllers"
            )
            key = published.entry.manifest.registry_key
            scenario = create_initial_scenario(
                4, 2, 2, 1, seed=9, controller_keys=(key, key)
            )

            with patch(
                "multi_agent_sim.saved_mlp_controllers.load_checkpoint",
                wraps=load_checkpoint,
            ) as loader:
                session = SimulationSession(
                    scenario, max_steps=3, controller_registry=result.registry
                )
                action_results = session.step_forward()

            self.assertEqual(loader.call_count, 1)
            self.assertTrue(all(value.action.value == "wait" for value in action_results.values()))
            self.assertEqual(
                session.controller_display_name_for("robot_1"),
                published.entry.manifest.display_name,
            )

    def test_saved_controller_limits_are_rejected_before_session_start(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            published = self._catalog(root)
            registry = create_registry_with_saved_controllers(
                directory=root / "controllers"
            ).registry
            key = published.entry.manifest.registry_key
            scenario = create_initial_scenario(
                5, 2, 5, 0, seed=2, controller_keys=(key,) * 5
            )

            with self.assertRaisesRegex(ValueError, "does not support"):
                SimulationSession(scenario, max_steps=3, controller_registry=registry)

    def test_catalog_entries_are_omitted_when_torch_is_unavailable(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._catalog(root)
            with patch(
                "multi_agent_sim.saved_mlp_controllers.torch_available",
                return_value=False,
            ):
                result = create_registry_with_saved_controllers(
                    directory=root / "controllers"
                )

            self.assertEqual(result.registry.keys, ("random", "nearest_item"))
            self.assertTrue(any("PyTorch" in warning for warning in result.warnings))

    def test_empty_catalog_needs_no_torch_device_resolution(self) -> None:
        with TemporaryDirectory() as directory:
            with (
                patch(
                    "multi_agent_sim.saved_mlp_controllers.torch_available",
                    return_value=False,
                ),
                patch(
                    "multi_agent_sim.saved_mlp_controllers._resolved_device"
                ) as resolve_device,
            ):
                result = create_registry_with_saved_controllers(
                    directory=Path(directory) / "controllers",
                    device="auto",
                )

            self.assertEqual(result.registry.keys, ("random", "nearest_item"))
            resolve_device.assert_not_called()


if __name__ == "__main__":
    unittest.main()
