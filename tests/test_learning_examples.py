from __future__ import annotations

import importlib.util
from contextlib import redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch


class LearningExampleTests(unittest.TestCase):
    def _run(self, example: str, *arguments: str) -> subprocess.CompletedProcess[str]:
        project_root = Path(__file__).resolve().parents[1]
        environment = os.environ.copy()
        source_path = str(project_root / "src")
        existing_path = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            source_path
            if not existing_path
            else source_path + os.pathsep + existing_path
        )
        return subprocess.run(
            [
                sys.executable,
                str(project_root / "examples" / example),
                *arguments,
            ],
            cwd=project_root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_scored_episode_reports_success_and_all_cost_components(self) -> None:
        completed = self._run("scored_episode.py")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = json.loads(completed.stdout)
        metrics = payload["metrics"]
        self.assertTrue(metrics["success"])
        self.assertEqual(metrics["delivered_items"], 1)
        self.assertEqual(metrics["undelivered_items"], 0)
        self.assertEqual(metrics["elapsed_steps"], 4)
        self.assertEqual(metrics["battery_consumed"], 4.0)
        for component in (
            "failure_cost",
            "undelivered_cost",
            "time_cost",
            "energy_cost",
            "item_progress_cost",
            "delivery_cost",
            "total_cost",
            "cumulative_reward",
        ):
            self.assertIn(component, metrics)
        self.assertAlmostEqual(sum(payload["step_rewards"]), -metrics["total_cost"])
        self.assertAlmostEqual(
            sum(payload["step_item_progress_rewards"]),
            -metrics["item_progress_cost"],
        )
        self.assertAlmostEqual(
            sum(payload["step_delivery_rewards"]),
            -metrics["delivery_cost"],
        )

    def test_training_cli_help_does_not_require_starting_training(self) -> None:
        completed = self._run("train_mlp.py", "--help")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("train", completed.stdout)
        self.assertIn("evaluate", completed.stdout)

        train_help = self._run("train_mlp.py", "train", "--help")
        self.assertEqual(train_help.returncode, 0, train_help.stderr)
        for option in (
            "--initialize-from",
            "--resume-from",
            "--latest-checkpoint",
            "--item-progress-weight",
            "--delivery-weight",
            "--previous-stage-probability",
            "--retention-threshold",
        ):
            self.assertIn(option, train_help.stdout)

    def test_visual_controller_help_does_not_import_pygame(self) -> None:
        completed = self._run("visualize_mlp_controller.py", "--help")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--checkpoint", completed.stdout)
        self.assertIn("--headless", completed.stdout)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "PyTorch training extra is not installed",
    )
    def test_training_cli_writes_a_smoke_checkpoint_and_metadata(self) -> None:
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "smoke.pt"
            latest = Path(directory) / "latest.pt"
            controllers = Path(directory) / "controllers"
            completed = self._run(
                "train_mlp.py",
                "train",
                "--minutes",
                "0.0001",
                "--episodes-per-update",
                "1",
                "--evaluation-interval",
                "1",
                "--evaluation-episodes",
                "1",
                "--max-stages",
                "1",
                "--device",
                "cpu",
                "--checkpoint",
                str(checkpoint),
                "--latest-checkpoint",
                str(latest),
                "--controllers-dir",
                str(controllers),
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertTrue(checkpoint.exists())
            self.assertTrue(checkpoint.with_suffix(".json").exists())
            self.assertTrue(latest.exists())
            summary_start = completed.stdout.find('{\n  "checkpoint"')
            self.assertGreaterEqual(summary_start, 0)
            summary = json.loads(completed.stdout[summary_start:])
            self.assertEqual(summary["checkpoint"], str(checkpoint))
            self.assertEqual(summary["latest_checkpoint"], str(latest))
            self.assertIsNotNone(summary["published_controller"])
            self.assertEqual(len(tuple(controllers.glob("*.controller.json"))), 1)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "PyTorch training extra is not installed",
    )
    def test_saved_controller_example_runs_through_wrapper_headlessly(self) -> None:
        from multi_agent_sim.learning import SharedMLP, save_checkpoint

        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "controller.pt"
            trace_report = Path(directory) / "trace.json"
            save_checkpoint(SharedMLP(), checkpoint, {"highest_completed_stage": 1})
            completed = self._run(
                "visualize_mlp_controller.py",
                "--checkpoint",
                str(checkpoint),
                "--stage",
                "1",
                "--headless",
                "--stop-after",
                "1",
                "--device",
                "cpu",
                "--inspect-robot",
                "robot_1",
                "--trace-report",
                str(trace_report),
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary_start = completed.stdout.rfind('{\n  "checkpoint"')
            self.assertGreaterEqual(summary_start, 0)
            summary = json.loads(completed.stdout[summary_start:])
            self.assertEqual(summary["stage"], 1)
            self.assertEqual(summary["steps_run"], 1)
            self.assertEqual(summary["trace_steps"], 1)
            self.assertEqual(summary["metrics"]["elapsed_steps"], 1)
            trace = json.loads(trace_report.read_text(encoding="utf-8"))
            self.assertEqual(len(trace["steps"]), 1)
            step = trace["steps"][0]
            decision = step["decisions"]["robot_1"]
            self.assertAlmostEqual(sum(decision["probabilities"]), 1.0)
            self.assertEqual(len(decision["legal_mask"]), 7)
            self.assertIn("delivery", step["reward_components"])
            self.assertIn("item_progress", step["reward_components"])

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "PyTorch training extra is not installed",
    )
    def test_evaluation_cli_writes_full_episode_report(self) -> None:
        from multi_agent_sim.learning import SharedMLP, save_checkpoint

        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "controller.pt"
            report = Path(directory) / "evaluation.json"
            save_checkpoint(SharedMLP(), checkpoint, {"highest_completed_stage": 1})
            completed = self._run(
                "train_mlp.py",
                "evaluate",
                "--checkpoint",
                str(checkpoint),
                "--stage",
                "1",
                "--episodes",
                "1",
                "--device",
                "cpu",
                "--report",
                str(report),
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["episodes"]), 1)
            metrics = payload["episodes"][0]["metrics"]
            self.assertIn("initial_normalized_item_distance", metrics)
            self.assertIn("normalized_item_distance", metrics)
            self.assertIn("item_progress_cost", metrics)
            self.assertIn("delivery_cost", metrics)
            self.assertIn("delivered_fraction", metrics)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "PyTorch training extra is not installed",
    )
    def test_controller_management_cli_publishes_and_lists_existing_best(self) -> None:
        from multi_agent_sim.episode import EpisodeCostWeights
        from multi_agent_sim.learning import (
            DEFAULT_CURRICULUM,
            ReinforceConfig,
            SharedMLP,
            save_checkpoint,
        )

        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "best.pt"
            controllers = root / "controllers"
            stage = DEFAULT_CURRICULUM[0]
            save_checkpoint(
                SharedMLP(),
                checkpoint,
                {
                    "cost_weights": EpisodeCostWeights(),
                    "curriculum": (stage,),
                    "episodes_trained": 50,
                    "optimizer_updates": 2,
                    "highest_completed_stage": 0,
                    "hyperparameters": ReinforceConfig(),
                    "validation": [
                        {
                            "stage": {
                                "number": 1,
                                "width": 4,
                                "height": 4,
                                "num_robots": 1,
                                "num_items": 1,
                                "max_steps": 32,
                            },
                            "success_rate": 0.42,
                            "mean_cost": 7.0,
                            "episodes": [],
                        }
                    ],
                },
            )
            published = self._run(
                "train_mlp.py",
                "controllers",
                "publish",
                "--checkpoint",
                str(checkpoint),
                "--controllers-dir",
                str(controllers),
                "--name",
                "Candidate",
            )
            self.assertEqual(published.returncode, 0, published.stderr)
            published_payload = json.loads(published.stdout)
            self.assertFalse(published_payload["qualified"])

            listed = self._run(
                "train_mlp.py",
                "controllers",
                "list",
                "--controllers-dir",
                str(controllers),
            )
            self.assertEqual(listed.returncode, 0, listed.stderr)
            listed_payload = json.loads(listed.stdout)
            self.assertEqual(len(listed_payload["controllers"]), 1)
            self.assertEqual(
                listed_payload["controllers"][0]["controller_id"],
                published_payload["controller_id"],
            )

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "PyTorch training extra is not installed",
    )
    def test_graceful_interrupt_attempts_to_publish_the_existing_best(self) -> None:
        from examples import train_mlp

        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "best.pt"
            checkpoint.write_bytes(b"best")
            args = train_mlp.build_parser().parse_args(
                [
                    "train",
                    "--checkpoint",
                    str(checkpoint),
                    "--latest-checkpoint",
                    str(root / "latest.pt"),
                    "--controllers-dir",
                    str(root / "controllers"),
                    "--minutes",
                    "0.01",
                ]
            )
            with (
                patch.object(train_mlp, "train_curriculum", side_effect=KeyboardInterrupt),
                patch.object(train_mlp, "publish_checkpoint_controller", return_value=None) as publish,
                redirect_stdout(StringIO()) as output,
            ):
                status = train_mlp._train(args)

            self.assertEqual(status, 130)
            publish.assert_called_once()
            self.assertTrue(json.loads(output.getvalue())["interrupted"])


if __name__ == "__main__":
    unittest.main()
