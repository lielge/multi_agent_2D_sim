"""Run a saved shared MLP controller and render its scored episode."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from multi_agent_sim import FixedSlotTeamEnv, create_initial_scenario
from multi_agent_sim.learning import (
    DEFAULT_CURRICULUM,
    PolicyDecision,
    create_shared_controllers,
    load_checkpoint,
    torch_available,
)
from multi_agent_sim.training_env import ACTION_ORDER


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _metadata_stage(checkpoint: Path) -> int:
    metadata_path = checkpoint.with_suffix(".json")
    if not metadata_path.exists():
        return 1
    try:
        payload: Any = json.loads(metadata_path.read_text(encoding="utf-8"))
        stage = int(payload.get("highest_completed_stage", 0))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 1
    return max(stage, 1)


def _device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize a greedy or stochastic saved MLP controller."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/shared_mlp_best.pt"),
    )
    parser.add_argument(
        "--stage",
        type=_positive_integer,
        help="curriculum stage; defaults to checkpoint metadata",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="scenario seed; defaults to the first held-out seed for the stage",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fps", type=_positive_integer, default=5)
    parser.add_argument("--cell-size", type=_positive_integer, default=80)
    parser.add_argument(
        "--stop-after",
        type=_positive_integer,
        help="stop the demonstration after this many controller steps",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="sample commands instead of using greedy argmax evaluation",
    )
    parser.add_argument(
        "--action-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run the same controller loop without importing Pygame",
    )
    parser.add_argument(
        "--inspect-robot",
        help="robot ID shown in the diagnostic panel; defaults to the first robot",
    )
    parser.add_argument(
        "--trace-report",
        type=Path,
        help="optional JSON file receiving detached decisions and reward components",
    )
    return parser


def _diagnostic_lines(
    decision: PolicyDecision,
    transition: Any,
) -> tuple[str, ...]:
    components = transition.reward_components
    lines = [
        "Policy diagnostics",
        f"Robot: {decision.robot_id}",
        f"Mode: {decision.selection_mode}",
        f"Selected: {decision.selected_action}",
        f"Entropy: {decision.entropy:.4f}",
        "Action probabilities:",
    ]
    for index, action in enumerate(ACTION_ORDER):
        legal = "legal" if decision.legal_mask[index] else "masked"
        selected = " <" if index == decision.selected_action_index else ""
        lines.append(
            f"{index} {action.value:10s} {decision.probabilities[index]:6.1%} "
            f"{legal}{selected}"
        )
    lines.extend(
        (
            "Step reward:",
            f"time        {components.time:+.6f}",
            f"energy      {components.energy:+.6f}",
            f"route       {components.item_progress:+.6f}",
            f"delivery    {components.delivery:+.6f}",
            f"failure     {components.failure:+.6f}",
            f"undelivered {components.undelivered:+.6f}",
            f"total       {components.total:+.6f}",
            f"Route potential: {transition.metrics.route_potential:.4f}",
            f"Delivered: {transition.metrics.delivered_fraction:.1%}",
        )
    )
    return tuple(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch_available():
        raise SystemExit(
            "PyTorch is not installed. Run: "
            'python -m pip install -e ".[training]"'
        )
    if not 1 <= args.fps <= 30:
        raise SystemExit("--fps must be between 1 and 30")

    stage_number = args.stage or _metadata_stage(args.checkpoint)
    if not 1 <= stage_number <= len(DEFAULT_CURRICULUM):
        raise SystemExit(
            f"--stage must be between 1 and {len(DEFAULT_CURRICULUM)}"
        )
    stage = DEFAULT_CURRICULUM[stage_number - 1]
    seed = (
        args.seed
        if args.seed is not None
        else 1_000_000 + (stage.number - 1) * 100
    )

    def scenario_factory(scenario_seed: int):
        return create_initial_scenario(
            width=stage.width,
            height=stage.height,
            num_robots=stage.num_robots,
            num_items=stage.num_items,
            seed=scenario_seed,
            initially_delivered_items=stage.initially_delivered_items,
        )

    environment = FixedSlotTeamEnv(scenario_factory, max_steps=stage.max_steps)
    observation = environment.reset(seed)
    model = load_checkpoint(args.checkpoint, device=_device(args.device))
    model.eval()
    controllers = create_shared_controllers(
        observation.active_robot_ids,
        model,
        stochastic=args.stochastic,
        use_action_masks=args.action_masks,
    )
    inspect_robot = args.inspect_robot or observation.active_robot_ids[0]
    if inspect_robot not in observation.active_robot_ids:
        available = ", ".join(observation.active_robot_ids)
        raise SystemExit(
            f"--inspect-robot must be an active robot ID; available: {available}"
        )

    renderer = None
    window_open = True
    if not args.headless:
        try:
            from multi_agent_sim.visualization import PygameRenderer
        except ImportError as exc:
            raise SystemExit(str(exc)) from exc
        renderer = PygameRenderer(
            cell_size=args.cell_size,
            fps=args.fps,
            show_labels=True,
        )
        renderer.render(
            environment.world,
            ("Policy diagnostics", f"Robot: {inspect_robot}", "Waiting for step 1"),
        )

    steps_run = 0
    trace: list[dict[str, Any]] = []
    latest_diagnostic_lines: tuple[str, ...] = ()
    try:
        while not environment.terminated:
            if args.stop_after is not None and steps_run >= args.stop_after:
                break
            if renderer is not None and not renderer.process_events():
                window_open = False
                break

            import torch

            with torch.no_grad():
                commands = {
                    robot_id: controller.choose_command(observation)
                    for robot_id, controller in controllers.items()
                }
            decisions = {
                robot_id: controller.last_decision
                for robot_id, controller in controllers.items()
            }
            transition = environment.step(commands)
            steps_run += 1
            trace_step = {
                "step": transition.metrics.elapsed_steps,
                "decisions": {
                    robot_id: asdict(decision)
                    for robot_id, decision in decisions.items()
                },
                "reward_components": asdict(transition.reward_components),
                "metrics": asdict(transition.metrics),
            }
            trace.append(trace_step)
            print("policy_trace " + json.dumps(trace_step, sort_keys=True))
            latest_diagnostic_lines = _diagnostic_lines(
                decisions[inspect_robot], transition
            )
            observation = transition.observation
            if renderer is not None:
                renderer.render(environment.world, latest_diagnostic_lines)
                renderer.tick()

        if renderer is not None and window_open:
            print("Episode display complete. Close the window or press Escape.")
            while renderer.process_events():
                renderer.render(environment.world, latest_diagnostic_lines)
                renderer.tick()
    finally:
        if renderer is not None:
            renderer.close()

    summary = {
        "checkpoint": str(args.checkpoint),
        "stage": stage.number,
        "seed": seed,
        "selection": "stochastic" if args.stochastic else "greedy",
        "steps_run": steps_run,
        "metrics": asdict(environment.metrics),
        "trace_steps": len(trace),
    }
    if args.trace_report is not None:
        args.trace_report.parent.mkdir(parents=True, exist_ok=True)
        args.trace_report.write_text(
            json.dumps(
                {
                    "checkpoint": str(args.checkpoint),
                    "stage": stage.number,
                    "seed": seed,
                    "inspected_robot": inspect_robot,
                    "steps": trace,
                    "final_metrics": asdict(environment.metrics),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        summary["trace_report"] = str(args.trace_report)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
