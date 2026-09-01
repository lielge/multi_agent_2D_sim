"""Run a small seeded multi-agent simulation with optional visualization."""

from __future__ import annotations

import argparse
import math
from collections.abc import Sequence

from multi_agent_sim import (
    ActionBatteryCosts,
    Robot,
    SimulationSession,
    SimulationWorld,
    create_initial_scenario,
)


def _non_negative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _finite_non_negative_number(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError(
            "value must be a finite, non-negative number"
        )
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the basic multi-agent 2D simulation demonstration."
    )
    parser.add_argument(
        "--width", type=_positive_integer, default=20, help="initial grid width"
    )
    parser.add_argument(
        "--height", type=_positive_integer, default=20, help="initial grid height"
    )
    parser.add_argument(
        "--robots",
        type=_non_negative_integer,
        default=3,
        help="initial robot count",
    )
    parser.add_argument(
        "--items",
        type=_non_negative_integer,
        default=5,
        help="initial item count",
    )
    parser.add_argument(
        "--steps",
        type=_non_negative_integer,
        default=200,
        help="maximum visual step or number of headless steps",
    )
    parser.add_argument("--seed", type=int, default=42, help="initial random seed")
    parser.add_argument(
        "--fps",
        type=_positive_integer,
        default=5,
        help="initial visual step rate (1-30); ignored in headless mode",
    )
    parser.add_argument(
        "--move-cost",
        "--movement-cost",
        dest="move_cost",
        type=_finite_non_negative_number,
        default=1.0,
        help="battery spent by an attempted movement action",
    )
    parser.add_argument(
        "--pickup-cost",
        type=_finite_non_negative_number,
        default=1.0,
        help="battery spent by an attempted pickup action",
    )
    parser.add_argument(
        "--drop-cost",
        type=_finite_non_negative_number,
        default=1.0,
        help="battery spent by an attempted drop action",
    )
    parser.add_argument(
        "--wait-cost",
        type=_finite_non_negative_number,
        default=0.0,
        help="battery spent by a wait action",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run without importing or opening Pygame",
    )
    parser.add_argument(
        "--labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show or hide robot ID labels",
    )
    return parser


def _print_summary(world: SimulationWorld) -> None:
    robots = ", ".join(
        f"{robot.robot_id}={robot.position}"
        for robot in world.get_entities(Robot)
    )
    print(f"Completed {world.timestep} steps. {robots or 'No robots.'}")


def run_demo(args: argparse.Namespace) -> SimulationWorld | None:
    """Run the configured scenario and return its final world state."""

    action_battery_costs = ActionBatteryCosts(
        movement=args.move_cost,
        pickup=args.pickup_cost,
        wait=args.wait_cost,
        drop=args.drop_cost,
    )

    if not args.headless:
        if not 1 <= args.fps <= 30:
            raise SystemExit("--fps must be between 1 and 30 in visual mode")

        try:
            from multi_agent_sim.visualization import PygameSimulationApp
        except ImportError as exc:
            raise SystemExit(str(exc)) from exc

        world = PygameSimulationApp(
            width=args.width,
            height=args.height,
            num_robots=args.robots,
            num_items=args.items,
            seed=args.seed,
            max_steps=args.steps,
            step_rate=args.fps,
            show_labels=args.labels,
            movement_cost=action_battery_costs.movement,
            pickup_cost=action_battery_costs.pickup,
            wait_cost=action_battery_costs.wait,
            drop_cost=action_battery_costs.drop,
        ).run()
        if world is not None:
            _print_summary(world)
        return world

    scenario = create_initial_scenario(
        width=args.width,
        height=args.height,
        num_robots=args.robots,
        num_items=args.items,
        seed=args.seed,
        action_battery_costs=action_battery_costs,
    )
    session = SimulationSession(scenario, max_steps=args.steps)
    for _ in range(args.steps):
        session.step_forward()

    world = session.world
    _print_summary(world)
    return world


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run_demo(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
