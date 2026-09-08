from __future__ import annotations

import os
import importlib.util
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest


class DemoSmokeTests(unittest.TestCase):
    def _run_demo(self, *arguments: str) -> subprocess.CompletedProcess[str]:
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
                str(project_root / "examples" / "basic_simulation.py"),
                *arguments,
            ],
            cwd=project_root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_headless_demo_runs_without_pygame(self) -> None:
        completed = self._run_demo(
            "--headless",
            "--steps",
            "4",
            "--seed",
            "11",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Completed 4 steps.", completed.stdout)

    def test_headless_demo_stops_immediately_when_there_are_no_items(self) -> None:
        completed = self._run_demo(
            "--headless",
            "--steps",
            "10",
            "--robots",
            "1",
            "--items",
            "0",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Completed 0 steps.", completed.stdout)

    def test_visual_rate_must_be_between_one_and_thirty(self) -> None:
        completed = self._run_demo("--fps", "31")

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("--fps must be between 1 and 30", completed.stderr)

    def test_visual_rate_limit_does_not_change_headless_runs(self) -> None:
        completed = self._run_demo(
            "--headless",
            "--steps",
            "1",
            "--fps",
            "31",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Completed 1 steps.", completed.stdout)

    def test_headless_demo_accepts_action_cost_defaults(self) -> None:
        completed = self._run_demo(
            "--headless",
            "--steps",
            "2",
            "--move-cost",
            "2.5",
            "--pickup-cost",
            "3",
            "--drop-cost",
            "4",
            "--wait-cost",
            "0.25",
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Completed 2 steps.", completed.stdout)

    def test_action_cost_flags_reject_nonfinite_or_negative_values(self) -> None:
        for flag, value in (
            ("--move-cost", "nan"),
            ("--drop-cost", "-1"),
            ("--wait-cost", "-1"),
        ):
            with self.subTest(flag=flag, value=value):
                completed = self._run_demo("--headless", flag, value)

                self.assertNotEqual(completed.returncode, 0)
                self.assertIn("finite, non-negative", completed.stderr)

    @unittest.skipUnless(
        importlib.util.find_spec("torch") is not None,
        "PyTorch training extra is not installed",
    )
    def test_headless_demo_selects_a_saved_controller_by_id(self) -> None:
        import torch

        from multi_agent_sim.controller_catalog import publish_saved_controller
        from multi_agent_sim.learning import SharedMLP, save_checkpoint

        with TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "best.pt"
            model = SharedMLP()
            with torch.no_grad():
                model.layers[-1].weight.zero_()
                model.layers[-1].bias.zero_()
                model.layers[-1].bias[5] = 10.0
            save_checkpoint(model, checkpoint, {})
            published = publish_saved_controller(
                checkpoint,
                directory=root / "controllers",
                run_id="demo-run",
                controller_name="Demo",
                qualified_stage=1,
                evaluated_stage=1,
                success_rate=0.9,
                mean_cost=1.0,
                episodes_trained=100,
                optimizer_updates=4,
                promotion_success_rate=0.8,
                use_action_masks=True,
                torch_version=str(torch.__version__),
                curriculum=(
                    {
                        "number": 1,
                        "width": 4,
                        "height": 4,
                        "num_robots": 1,
                        "num_items": 1,
                        "max_steps": 32,
                    },
                ),
                cost_weights={},
                training_config={},
            )

            completed = self._run_demo(
                "--headless",
                "--steps",
                "1",
                "--robots",
                "1",
                "--items",
                "1",
                "--width",
                "4",
                "--height",
                "4",
                "--controllers-dir",
                str(root / "controllers"),
                "--controller",
                published.entry.manifest.controller_id,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("Completed 1 steps.", completed.stdout)


if __name__ == "__main__":
    unittest.main()
