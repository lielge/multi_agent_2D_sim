"""Run a small seeded multi-agent simulation with optional visualization."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from collections.abc import Sequence
import sys

from multi_agent_sim import (
    ActionBatteryCosts,
    Robot,
    SimulationSession,
    SimulationWorld,
    create_initial_scenario,
    create_world_observation,
    delivery_status,
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
    parser.add_argument(
        "--controllers-dir", type=Path, default=Path("controllers")
    )
    parser.add_argument(
        "--controller",
        help="controller key or saved controller ID to assign to every robot",
    )
    parser.add_argument(
        "--controller-device",
        choices=("cpu", "auto", "cuda"),
        default="cpu",
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
    from multi_agent_sim.saved_mlp_controllers import (
        create_registry_with_saved_controllers,
    )

    def load_registry():
        result = create_registry_with_saved_controllers(
            directory=args.controllers_dir,
            device=args.controller_device,
        )
        for warning in result.warnings:
            print(f"Controller catalog warning: {warning}", file=sys.stderr)
        return result

    registry_result = load_registry()
    controller_key = args.controller
    if (
        controller_key is not None
        and controller_key not in registry_result.registry.keys
    ):
        matches = tuple(
            entry.manifest.registry_key
            for entry in registry_result.catalog.entries
            if entry.manifest.controller_id == controller_key
        )
        if not matches:
            choices = ", ".join(registry_result.registry.keys)
            raise SystemExit(
                f"unknown --controller {controller_key!r}; available keys: {choices}"
            )
        controller_key = matches[0]
    if controller_key is not None:
        definition = registry_result.registry.definition(controller_key)
        if not definition.supports_scenario(args.robots, args.items):
            raise SystemExit(
                f"controller {definition.display_name!r} supports at most "
                f"{definition.max_robots} robots and {definition.max_items} items"
            )
        if definition.max_robots is not None and args.steps <= 0:
            raise SystemExit("saved neural controllers require --steps to be positive")

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
            controller_registry=registry_result.registry,
            controller_registry_loader=lambda: load_registry().registry,
            initial_controller_key=controller_key,
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
        controller_keys=(controller_key,) * args.robots if controller_key else None,
    )
    session = SimulationSession(
        scenario,
        max_steps=args.steps,
        controller_registry=registry_result.registry,
    )
    for _ in range(args.steps):
        if delivery_status(create_world_observation(session.world)).success:
            break
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
