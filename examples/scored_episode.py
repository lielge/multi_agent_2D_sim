"""Run one headless delivery episode with external team scoring."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict
import json
import math

from multi_agent_sim import (
    Action,
    EpisodeCostWeights,
    EpisodeRunner,
    InitialScenario,
    RobotConfiguration,
)


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _finite_nonnegative(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a scripted episode scored outside SimulationWorld."
    )
    parser.add_argument("--max-steps", type=_positive_integer, default=4)
    parser.add_argument("--failure-weight", type=_finite_nonnegative, default=10.0)
    parser.add_argument("--undelivered-weight", type=_finite_nonnegative, default=2.0)
    parser.add_argument("--time-weight", type=_finite_nonnegative, default=0.5)
    parser.add_argument("--energy-weight", type=_finite_nonnegative, default=0.5)
    parser.add_argument(
        "--item-progress-weight", type=_finite_nonnegative, default=0.1
    )
    parser.add_argument("--delivery-weight", type=_finite_nonnegative, default=1.0)
    return parser


def run(args: argparse.Namespace) -> dict[str, object]:
    scenario = InitialScenario(
        width=4,
        height=1,
        robot_configurations=(RobotConfiguration((0, 0), battery_level=10),),
        item_positions=((1, 0),),
        delivery_destination_positions=((2, 0),),
        seed=1,
    )
    weights = EpisodeCostWeights(
        failure=args.failure_weight,
        undelivered=args.undelivered_weight,
        time=args.time_weight,
        energy=args.energy_weight,
        item_progress=args.item_progress_weight,
        delivery=args.delivery_weight,
    )
    runner = EpisodeRunner(scenario, max_steps=args.max_steps, cost_weights=weights)
    rewards: list[float] = []
    progress_rewards: list[float] = []
    delivery_rewards: list[float] = []
    for action in (
        Action.MOVE_RIGHT,
        Action.PICK_UP,
        Action.MOVE_RIGHT,
        Action.DROP,
    ):
        if runner.terminated:
            break
        transition = runner.step({"robot_1": action})
        rewards.append(transition.reward)
        progress_rewards.append(transition.item_progress_reward)
        delivery_rewards.append(transition.delivery_reward)

    return {
        "step_rewards": rewards,
        "step_item_progress_rewards": progress_rewards,
        "step_delivery_rewards": delivery_rewards,
        "metrics": asdict(runner.metrics),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(run(args), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
