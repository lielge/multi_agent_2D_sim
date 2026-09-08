"""Train or evaluate the optional shared PyTorch MLP policy."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import json
import math
from pathlib import Path
import sys
from typing import Any

from multi_agent_sim import (
    EpisodeCostWeights,
    FixedSlotTeamEnv,
    create_initial_scenario,
)
from multi_agent_sim.controller_catalog import discover_saved_controllers
from multi_agent_sim.learning import (
    DEFAULT_CURRICULUM,
    CurriculumStage,
    ReinforceConfig,
    evaluate_policy,
    is_training_checkpoint,
    load_checkpoint,
    load_training_checkpoint,
    publish_checkpoint_controller,
    torch_available,
    train_curriculum,
    validation_seeds,
)


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _positive_number(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def _probability(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("value must be between zero and one")
    return parsed


def _finite_nonnegative(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return parsed


def _add_weight_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--failure-weight", type=_finite_nonnegative, default=10.0)
    parser.add_argument("--undelivered-weight", type=_finite_nonnegative, default=2.0)
    parser.add_argument("--time-weight", type=_finite_nonnegative, default=0.5)
    parser.add_argument("--energy-weight", type=_finite_nonnegative, default=0.5)
    parser.add_argument(
        "--item-progress-weight", type=_finite_nonnegative, default=0.1
    )
    parser.add_argument("--delivery-weight", type=_finite_nonnegative, default=1.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the shared 155-128-128-7 MLP."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="run curriculum REINFORCE training")
    train.add_argument("--minutes", type=_positive_number, default=60.0)
    train.add_argument("--checkpoint", type=Path, default=Path("artifacts/shared_mlp_best.pt"))
    train.add_argument(
        "--latest-checkpoint",
        type=Path,
        default=Path("artifacts/shared_mlp_latest.pt"),
    )
    initialization = train.add_mutually_exclusive_group()
    initialization.add_argument(
        "--initialize-from",
        type=Path,
        help="warm-start model weights from a model-only checkpoint",
    )
    initialization.add_argument(
        "--resume-from",
        type=Path,
        help="exactly resume a full latest checkpoint; model-only files warm-start",
    )
    train.add_argument("--device", default="auto")
    train.add_argument("--seed", type=int, default=1729)
    train.add_argument("--gamma", type=_probability, default=1.0)
    train.add_argument("--episodes-per-update", type=_positive_integer, default=32)
    train.add_argument("--evaluation-interval", type=_positive_integer, default=250)
    train.add_argument("--evaluation-episodes", type=_positive_integer, default=100)
    train.add_argument(
        "--previous-stage-probability", type=_probability, default=0.4
    )
    train.add_argument("--retention-threshold", type=_probability, default=0.8)
    train.add_argument(
        "--max-stages", type=_positive_integer, default=len(DEFAULT_CURRICULUM)
    )
    train.add_argument(
        "--action-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    train.add_argument(
        "--controllers-dir", type=Path, default=Path("controllers")
    )
    train.add_argument("--controller-name")
    train.add_argument(
        "--publish-controller",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    _add_weight_arguments(train)

    evaluate = subparsers.add_parser(
        "evaluate", help="greedily evaluate a saved checkpoint"
    )
    evaluate.add_argument("--checkpoint", type=Path, default=Path("artifacts/shared_mlp_best.pt"))
    evaluate.add_argument("--stage", type=_positive_integer)
    evaluate.add_argument("--episodes", type=_positive_integer, default=100)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--report", type=Path, default=Path("artifacts/evaluation.json"))
    evaluate.add_argument(
        "--action-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    _add_weight_arguments(evaluate)

    controllers = subparsers.add_parser(
        "controllers", help="list or publish saved neural controllers"
    )
    controller_commands = controllers.add_subparsers(
        dest="controller_command", required=True
    )
    list_command = controller_commands.add_parser("list")
    list_command.add_argument(
        "--controllers-dir", type=Path, default=Path("controllers")
    )
    publish_command = controller_commands.add_parser("publish")
    publish_command.add_argument("--checkpoint", type=Path, required=True)
    publish_command.add_argument("--metadata", type=Path)
    publish_command.add_argument("--name")
    publish_command.add_argument(
        "--controllers-dir", type=Path, default=Path("controllers")
    )
    return parser


def _weights(args: argparse.Namespace) -> EpisodeCostWeights:
    return EpisodeCostWeights(
        failure=args.failure_weight,
        undelivered=args.undelivered_weight,
        time=args.time_weight,
        energy=args.energy_weight,
        item_progress=args.item_progress_weight,
        delivery=args.delivery_weight,
    )


def _environment_factory(
    weights: EpisodeCostWeights,
) -> Callable[[CurriculumStage, int], FixedSlotTeamEnv]:
    def create_environment(
        stage: CurriculumStage,
        _episode_seed: int,
    ) -> FixedSlotTeamEnv:
        def create_scenario(seed: int):
            return create_initial_scenario(
                width=stage.width,
                height=stage.height,
                num_robots=stage.num_robots,
                num_items=stage.num_items,
                seed=seed,
                initially_delivered_items=stage.initially_delivered_items,
            )

        return FixedSlotTeamEnv(
            create_scenario,
            max_steps=stage.max_steps,
            cost_weights=weights,
        )

    return create_environment


def _device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _require_training_extra() -> None:
    if not torch_available():
        raise SystemExit(
            "PyTorch is not installed. Run: "
            'python -m pip install -e ".[training]"'
        )


def _train(args: argparse.Namespace) -> int:
    _require_training_extra()
    stages = DEFAULT_CURRICULUM[: min(args.max_stages, len(DEFAULT_CURRICULUM))]
    weights = _weights(args)
    config = ReinforceConfig(
        episodes_per_update=args.episodes_per_update,
        gamma=args.gamma,
        evaluation_interval=args.evaluation_interval,
        evaluation_episodes=args.evaluation_episodes,
        time_budget_seconds=args.minutes * 60,
        training_seed=args.seed,
        previous_stage_probability=args.previous_stage_probability,
        retention_success_rate=args.retention_threshold,
        use_action_masks=args.action_masks,
    )
    device = _device(args.device)
    initial_model = (
        load_checkpoint(args.initialize_from, device=device)
        if args.initialize_from is not None
        else None
    )
    resume_checkpoint = None
    if args.resume_from is not None:
        if is_training_checkpoint(args.resume_from):
            resume_checkpoint = args.resume_from
        else:
            print(
                "warning: --resume-from received a model-only checkpoint; "
                "warm-starting with a new optimizer. Use --initialize-from "
                "for this behavior in future commands.",
                file=sys.stderr,
            )
            initial_model = load_checkpoint(args.resume_from, device=device)
    try:
        result = train_curriculum(
            _environment_factory(weights),
            model=initial_model,
            config=config,
            curriculum=stages,
            checkpoint_path=args.checkpoint,
            latest_checkpoint_path=args.latest_checkpoint,
            resume_checkpoint_path=resume_checkpoint,
            cost_weights=weights,
            device=device,
        )
    except KeyboardInterrupt:
        published = None
        if args.publish_controller and args.checkpoint.is_file():
            run_id = None
            try:
                state = load_training_checkpoint(
                    args.latest_checkpoint,
                    config=config,
                    curriculum=stages,
                    cost_weights=weights,
                    device="cpu",
                )
                run_id = state.run_id
            except ValueError:
                pass
            published = publish_checkpoint_controller(
                args.checkpoint,
                controllers_directory=args.controllers_dir,
                controller_name=args.controller_name,
                run_id=run_id,
            )
        print(
            json.dumps(
                {
                    "interrupted": True,
                    "latest_checkpoint": str(args.latest_checkpoint),
                    "published_controller": _published_dict(published),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 130
    published = (
        publish_checkpoint_controller(
            result.checkpoint_path,
            controllers_directory=args.controllers_dir,
            controller_name=args.controller_name,
            metadata_path=result.metadata_path,
            run_id=result.run_id,
        )
        if args.publish_controller
        else None
    )
    print(
        json.dumps(
            {
                "episodes_trained": result.episodes_trained,
                "optimizer_updates": result.optimizer_updates,
                "highest_completed_stage": result.highest_completed_stage,
                "elapsed_seconds": result.elapsed_seconds,
                "total_elapsed_seconds": result.total_elapsed_seconds,
                "checkpoint": str(result.checkpoint_path),
                "latest_checkpoint": str(result.latest_checkpoint_path),
                "metadata": str(result.metadata_path),
                "run_id": result.run_id,
                "interrupted": result.interrupted,
                "published_controller": _published_dict(published),
                "latest_validation": [
                    {
                        "stage": summary.stage.number,
                        "success_rate": summary.success_rate,
                        "mean_cost": summary.mean_cost,
                    }
                    for summary in result.evaluations
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _published_dict(published: Any) -> dict[str, Any] | None:
    if published is None:
        return None
    manifest = published.entry.manifest
    return {
        "controller_id": manifest.controller_id,
        "display_name": manifest.display_name,
        "model": str(published.entry.model_path),
        "manifest": str(published.entry.manifest_path),
        "created": published.created,
        "qualified": manifest.qualified,
        "qualified_stage": manifest.qualified_stage,
        "evaluated_stage": manifest.evaluated_stage,
        "success_rate": manifest.success_rate,
        "mean_cost": manifest.mean_cost,
    }


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


def _evaluate(args: argparse.Namespace) -> int:
    _require_training_extra()
    stage_number = args.stage or _metadata_stage(args.checkpoint)
    if stage_number > len(DEFAULT_CURRICULUM):
        raise SystemExit(f"--stage must be between 1 and {len(DEFAULT_CURRICULUM)}")
    stage = DEFAULT_CURRICULUM[stage_number - 1]
    weights = _weights(args)
    summary = evaluate_policy(
        _environment_factory(weights),
        load_checkpoint(args.checkpoint, device=_device(args.device)),
        stage,
        validation_seeds(stage, args.episodes),
        use_action_masks=args.action_masks,
    )
    report = summary.to_dict()
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "stage": stage.number,
                "success_rate": summary.success_rate,
                "mean_cost": summary.mean_cost,
                "episodes": len(summary.episodes),
                "report": str(args.report),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _controllers(args: argparse.Namespace) -> int:
    if args.controller_command == "list":
        catalog = discover_saved_controllers(args.controllers_dir)
        print(
            json.dumps(
                {
                    "directory": str(catalog.directory),
                    "controllers": [
                        {
                            "controller_id": entry.manifest.controller_id,
                            "registry_key": entry.manifest.registry_key,
                            "display_name": entry.manifest.display_name,
                            "qualified": entry.manifest.qualified,
                            "qualified_stage": entry.manifest.qualified_stage,
                            "evaluated_stage": entry.manifest.evaluated_stage,
                            "success_rate": entry.manifest.success_rate,
                            "mean_cost": entry.manifest.mean_cost,
                            "model": str(entry.model_path),
                        }
                        for entry in catalog.entries
                    ],
                    "warnings": list(catalog.warnings),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    _require_training_extra()
    published = publish_checkpoint_controller(
        args.checkpoint,
        controllers_directory=args.controllers_dir,
        controller_name=args.name,
        metadata_path=args.metadata,
    )
    print(json.dumps(_published_dict(published), indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        return _train(args)
    if args.command == "evaluate":
        return _evaluate(args)
    return _controllers(args)


if __name__ == "__main__":
    raise SystemExit(main())
