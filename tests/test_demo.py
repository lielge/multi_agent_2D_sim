from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
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


if __name__ == "__main__":
    unittest.main()
