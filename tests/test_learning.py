from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from multi_agent_sim.episode import EpisodeCostWeights
from multi_agent_sim.episode import OBJECTIVE_SCHEMA_VERSION
from multi_agent_sim.learning import (
    ACTION_MAP_VERSION,
    ACTION_MASK_VERSION,
    DEFAULT_CURRICULUM,
    ENCODING_SCHEMA_VERSION,
    EvaluationSummary,
    INPUT_FEATURES,
    LEGACY_TRAINING_CHECKPOINT_VERSIONS,
    CurriculumStage,
    MLPRobotController,
    ReinforceConfig,
    ReinforceTrainer,
    SharedMLP,
    TrainingCheckpointState,
    collect_episode,
    create_shared_controllers,
    discounted_returns,
    evaluate_policy,
    load_checkpoint,
    load_training_checkpoint,
    save_checkpoint,
    save_training_checkpoint,
    is_training_checkpoint,
    torch_available,
    validation_seeds,
)
from multi_agent_sim.learning import (
    _per_timestep_advantages,
    _regressed_stage_index,
    _sample_training_stage_index,
)


TORCH_INSTALLED = importlib.util.find_spec("torch") is not None
if TORCH_INSTALLED:
    import torch


@dataclass(frozen=True)
class FakeMetrics:
    terminated: bool = True
    success: bool = True
    timed_out: bool = False
    total_items: int = 1
    delivered_items: int = 1
    undelivered_items: int = 0
    undelivered_fraction: float = 0.0
    elapsed_steps: int = 1
    max_steps: int = 2
    battery_consumed: float = 1.0
    max_energy: float = 100.0
    failure_cost: float = 0.0
    undelivered_cost: float = 0.0
    time_cost: float = 0.25
    energy_cost: float = 0.005
    initial_normalized_item_distance: float = 0.2
    normalized_item_distance: float = 0.1
    item_progress_cost: float = -0.01
    total_cost: float = 0.255
    cumulative_reward: float = -0.255


class FakeObservation:
    def __init__(self) -> None:
        self.active_robot_ids = ("robot_1", "robot_2")
        self._rows = {
            "robot_1": (0.0,) * INPUT_FEATURES,
            "robot_2": (1.0,) + (0.0,) * (INPUT_FEATURES - 1),
        }
        self._masks = {
            "robot_1": (1, 1, 1, 1, 1, 1, 1),
            "robot_2": (1, 1, 1, 1, 1, 1, 1),
        }

    def row_for(self, robot_id: str) -> tuple[float, ...]:
        return self._rows[robot_id]

    def action_mask_for(self, robot_id: str) -> tuple[int, ...]:
        return self._masks[robot_id]


class FakeEnvironment:
    """A one-transition environment implementing the structural contract."""

    def __init__(self, success: bool = True, total_cost: float = 0.255) -> None:
        self.terminated = False
        self.metrics = None
        self._success = success
        self._total_cost = total_cost
        self.commands: dict[str, tuple[int, ...]] | None = None

    def reset(self, seed: int) -> FakeObservation:
        self.terminated = False
        self.metrics = None
        return FakeObservation()

    def step(self, commands: dict[str, tuple[int, ...]]) -> SimpleNamespace:
        self.commands = commands
        self.terminated = True
        self.metrics = FakeMetrics(
            success=self._success,
            timed_out=not self._success,
            delivered_items=int(self._success),
            undelivered_items=int(not self._success),
            undelivered_fraction=float(not self._success),
            failure_cost=0.0 if self._success else 10.0,
            undelivered_cost=0.0 if self._success else 2.0,
            total_cost=self._total_cost,
            cumulative_reward=-self._total_cost,
        )
        return SimpleNamespace(
            observation=FakeObservation(),
            reward=-self._total_cost,
            terminated=True,
            metrics=self.metrics,
        )


class OptionalImportTests(unittest.TestCase):
    def test_learning_module_reports_torch_availability(self) -> None:
        self.assertEqual(torch_available(), TORCH_INSTALLED)

    @unittest.skipIf(TORCH_INSTALLED, "only relevant without the optional extra")
    def test_model_has_actionable_error_without_torch(self) -> None:
        with self.assertRaisesRegex(ImportError, "training.*extra"):
            SharedMLP()


@unittest.skipUnless(TORCH_INSTALLED, "PyTorch training extra is not installed")
class ModelAndControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)

    def test_model_has_exact_architecture_and_output_shape(self) -> None:
        model = SharedMLP()
        layers = tuple(model.layers)

        self.assertEqual(
            tuple(type(layer) for layer in layers),
            (torch.nn.Linear, torch.nn.ReLU, torch.nn.Linear, torch.nn.ReLU,
             torch.nn.Linear),
        )
        self.assertEqual((layers[0].in_features, layers[0].out_features), (155, 128))
        self.assertEqual((layers[2].in_features, layers[2].out_features), (128, 128))
        self.assertEqual((layers[4].in_features, layers[4].out_features), (128, 7))
        self.assertEqual(tuple(model(torch.zeros(3, 155)).shape), (3, 7))
        with self.assertRaises(ValueError):
            model(torch.zeros(154))

    def test_each_controller_returns_one_immutable_command_and_shares_model(self) -> None:
        model = SharedMLP()
        observation = FakeObservation()
        controllers = create_shared_controllers(
            observation.active_robot_ids,
            model,
            stochastic=True,
        )

        commands = {
            robot_id: controller.choose_command(observation)
            for robot_id, controller in controllers.items()
        }

        self.assertTrue(all(controller.model is model for controller in controllers.values()))
        self.assertEqual(tuple(commands), observation.active_robot_ids)
        for command in commands.values():
            self.assertIsInstance(command, tuple)
            self.assertEqual(len(command), 7)
            self.assertEqual(sum(command), 1)
            self.assertTrue(all(value in (0, 1) for value in command))
        for controller in controllers.values():
            self.assertTrue(controller.last_log_probability.requires_grad)
            self.assertTrue(controller.last_entropy.requires_grad)

    def test_greedy_controller_obeys_advisory_mask_when_enabled(self) -> None:
        model = SharedMLP()
        with torch.no_grad():
            final_layer = model.layers[-1]
            final_layer.weight.zero_()
            final_layer.bias.copy_(torch.arange(7, dtype=torch.float32))
        masked = MLPRobotController(
            "robot_1", model, stochastic=False, use_action_mask=True
        )
        unmasked = MLPRobotController(
            "robot_1", model, stochastic=False, use_action_mask=False
        )

        self.assertEqual(
            masked.choose_from_row((0.0,) * 155, (0, 1, 0, 0, 0, 0, 0)),
            (0, 1, 0, 0, 0, 0, 0),
        )
        self.assertEqual(
            unmasked.choose_from_row((0.0,) * 155, (0, 1, 0, 0, 0, 0, 0)),
            (0, 0, 0, 0, 0, 0, 1),
        )
        masked_decision = masked.last_decision
        self.assertEqual(masked_decision.selected_action, "move_down")
        self.assertEqual(masked_decision.selected_action_index, 1)
        self.assertEqual(masked_decision.legal_mask, (False, True) + (False,) * 5)
        self.assertAlmostEqual(sum(masked_decision.probabilities), 1.0)
        self.assertEqual(masked_decision.probabilities[1], 1.0)
        self.assertTrue(
            all(
                probability == 0.0
                for index, probability in enumerate(masked_decision.probabilities)
                if index != 1
            )
        )
        self.assertEqual(unmasked.last_decision.legal_mask, (True,) * 7)
        self.assertEqual(unmasked.last_decision.selection_mode, "greedy")

    def test_controller_rejects_bad_rows_masks_and_membership(self) -> None:
        model = SharedMLP()
        controller = MLPRobotController("robot_1", model, stochastic=False)
        with self.assertRaises(ValueError):
            controller.choose_from_row((0.0,) * 154)
        with self.assertRaises(ValueError):
            controller.choose_from_row((0.0,) * 155, (0,) * 7)
        with self.assertRaises(ValueError):
            controller.choose_from_row((0.0,) * 155, (1,) * 6)

        observation = FakeObservation()
        missing = MLPRobotController("missing", model)
        with self.assertRaises(ValueError):
            missing.choose_command(observation)


@unittest.skipUnless(TORCH_INSTALLED, "PyTorch training extra is not installed")
class TrainingTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)

    def test_rollout_uses_one_shared_reward_and_joint_log_probability(self) -> None:
        model = SharedMLP()
        environment = FakeEnvironment()
        stage = CurriculumStage(1, 4, 4, 2, 1, 2)

        rollout = collect_episode(
            environment, model, stage, 5, stochastic=True
        )

        self.assertEqual(rollout.rewards, (-0.255,))
        self.assertEqual(len(rollout.joint_log_probabilities), 1)
        self.assertEqual(len(rollout.joint_entropies), 1)
        self.assertEqual(set(environment.commands or ()), {"robot_1", "robot_2"})
        self.assertAlmostEqual(rollout.undiscounted_return, -0.255)

    def test_reinforce_update_is_finite_and_changes_parameters(self) -> None:
        model = SharedMLP()
        config = ReinforceConfig(episodes_per_update=1)
        trainer = ReinforceTrainer(model, config)
        rollout = collect_episode(
            FakeEnvironment(),
            model,
            CurriculumStage(1, 4, 4, 2, 1, 2),
            9,
            stochastic=True,
        )
        before = tuple(parameter.detach().clone() for parameter in model.parameters())

        result = trainer.update((rollout,))

        self.assertTrue(all(
            torch.isfinite(torch.tensor(value))
            for value in (
                result.loss,
                result.policy_loss,
                result.mean_entropy,
                result.gradient_norm,
            )
        ))
        self.assertEqual(result.transitions, 1)
        self.assertTrue(any(
            not torch.equal(old, new.detach())
            for old, new in zip(before, model.parameters(), strict=True)
        ))

    def test_batch_baseline_compares_only_matching_timesteps(self) -> None:
        advantages = _per_timestep_advantages(
            ((-12.5, -12.0), (-0.5,), (-12.25, -12.0))
        )

        self.assertAlmostEqual(sum(row[0] for row in advantages), 0.0)
        self.assertAlmostEqual(advantages[0][1] + advantages[2][1], 0.0)
        self.assertGreater(advantages[1][0], 0.0)

    def test_greedy_evaluation_retains_per_episode_metrics(self) -> None:
        stage = CurriculumStage(1, 4, 4, 2, 1, 2)
        calls: list[tuple[int, int]] = []

        def factory(stage_arg: CurriculumStage, seed: int) -> FakeEnvironment:
            calls.append((stage_arg.number, seed))
            return FakeEnvironment(success=seed % 2 == 0, total_cost=float(seed))

        summary = evaluate_policy(factory, SharedMLP(), stage, (2, 3))

        self.assertEqual(calls, [(1, 2), (1, 3)])
        self.assertEqual(summary.success_rate, 0.5)
        self.assertEqual(summary.mean_cost, 2.5)
        self.assertEqual(tuple(ep.seed for ep in summary.episodes), (2, 3))
        self.assertIn("battery_consumed", summary.to_dict()["episodes"][0]["metrics"])

    def test_model_state_and_metadata_round_trip(self) -> None:
        from tempfile import TemporaryDirectory

        model = SharedMLP()
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "best.pt"
            checkpoint_path, metadata_path = save_checkpoint(
                model,
                checkpoint,
                {"seed": 3, "nested": {"value": 4}},
            )
            restored = load_checkpoint(checkpoint_path)

            for expected, actual in zip(
                model.parameters(), restored.parameters(), strict=True
            ):
                self.assertTrue(torch.equal(expected, actual))
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["encoding_schema_version"], ENCODING_SCHEMA_VERSION)
            self.assertEqual(metadata["action_map_version"], ACTION_MAP_VERSION)
            self.assertEqual(metadata["action_mask_version"], ACTION_MASK_VERSION)
            self.assertEqual(metadata["objective_schema_version"], OBJECTIVE_SCHEMA_VERSION)
            self.assertEqual(metadata["seed"], 3)

    def test_full_checkpoint_resumes_model_optimizer_and_random_streams_exactly(
        self,
    ) -> None:
        from tempfile import TemporaryDirectory

        config = ReinforceConfig(episodes_per_update=1, evaluation_episodes=1)
        curriculum = (CurriculumStage(1, 4, 4, 2, 1, 2),)
        weights = EpisodeCostWeights()
        rng_a = random.Random(73)
        torch.manual_seed(73)
        trainer_a = ReinforceTrainer(SharedMLP(), config)
        first_seed = rng_a.randrange(-(2**63), 0)
        trainer_a.update((collect_episode(
            FakeEnvironment(), trainer_a.model, curriculum[0], first_seed,
            stochastic=True,
        ),))

        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "latest.pt"
            state = TrainingCheckpointState(
                model_state_dict=trainer_a.model.state_dict(),
                optimizer_state_dict=trainer_a.optimizer.state_dict(),
                python_rng_state=rng_a.getstate(),
                torch_rng_state=torch.get_rng_state(),
                cuda_rng_states=(),
                episodes_trained=1,
                optimizer_updates=1,
                current_stage_index=0,
                highest_completed_stage=0,
                next_evaluation=2,
                last_evaluation_episode=0,
                recovery_stage_index=None,
                best_score=(0, 0.0, -12.0),
                total_elapsed_seconds=1.25,
                saved_device_type="cpu",
            )
            save_training_checkpoint(
                checkpoint,
                state,
                config=config,
                curriculum=curriculum,
                cost_weights=weights,
            )
            self.assertTrue(is_training_checkpoint(checkpoint))
            self.assertFalse(any(Path(directory).glob("*.tmp")))

            # Continue once without stopping.
            second_seed_a = rng_a.randrange(-(2**63), 0)
            trainer_a.update((collect_episode(
                FakeEnvironment(), trainer_a.model, curriculum[0], second_seed_a,
                stochastic=True,
            ),))

            # Restore from the optimizer boundary and repeat the same update.
            restored = load_training_checkpoint(
                checkpoint,
                config=config,
                curriculum=curriculum,
                cost_weights=weights,
            )
            trainer_b = ReinforceTrainer(SharedMLP(), config)
            trainer_b.model.load_state_dict(restored.model_state_dict)
            trainer_b.optimizer.load_state_dict(restored.optimizer_state_dict)
            rng_b = random.Random()
            rng_b.setstate(restored.python_rng_state)
            torch.set_rng_state(restored.torch_rng_state)
            second_seed_b = rng_b.randrange(-(2**63), 0)
            trainer_b.update((collect_episode(
                FakeEnvironment(), trainer_b.model, curriculum[0], second_seed_b,
                stochastic=True,
            ),))

            self.assertEqual(second_seed_a, second_seed_b)
            for expected, actual in zip(
                trainer_a.model.parameters(), trainer_b.model.parameters(), strict=True
            ):
                self.assertTrue(torch.equal(expected, actual))
            self._assert_nested_equal(
                trainer_a.optimizer.state_dict(), trainer_b.optimizer.state_dict()
            )
            self.assertEqual(restored.episodes_trained, 1)
            self.assertEqual(restored.optimizer_updates, 1)
            self.assertEqual(restored.total_elapsed_seconds, 1.25)

    def test_latest_checkpoint_rejects_model_only_corrupt_and_incompatible_files(
        self,
    ) -> None:
        from tempfile import TemporaryDirectory

        config = ReinforceConfig(episodes_per_update=1)
        curriculum = (CurriculumStage(1, 4, 4, 1, 1, 2),)
        weights = EpisodeCostWeights()
        with TemporaryDirectory() as directory:
            directory_path = Path(directory)
            model_only = directory_path / "best.pt"
            save_checkpoint(SharedMLP(), model_only, {})
            with self.assertRaisesRegex(ValueError, "model-only"):
                load_training_checkpoint(
                    model_only,
                    config=config,
                    curriculum=curriculum,
                    cost_weights=weights,
                )

            corrupt = directory_path / "corrupt.pt"
            corrupt.write_text("not a torch checkpoint", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "could not load checkpoint"):
                load_training_checkpoint(
                    corrupt,
                    config=config,
                    curriculum=curriculum,
                    cost_weights=weights,
                )

            latest = directory_path / "latest.pt"
            save_training_checkpoint(
                latest,
                TrainingCheckpointState(
                    model_state_dict=SharedMLP().state_dict(),
                    optimizer_state_dict=ReinforceTrainer(
                        SharedMLP(), config
                    ).optimizer.state_dict(),
                    python_rng_state=random.Random(1).getstate(),
                    torch_rng_state=torch.get_rng_state(),
                    cuda_rng_states=(),
                    episodes_trained=0,
                    optimizer_updates=0,
                    current_stage_index=0,
                    highest_completed_stage=0,
                    next_evaluation=250,
                    last_evaluation_episode=-1,
                    recovery_stage_index=None,
                    best_score=None,
                    total_elapsed_seconds=0.0,
                    saved_device_type="cpu",
                ),
                config=config,
                curriculum=curriculum,
                cost_weights=weights,
            )
            with self.assertRaisesRegex(ValueError, "incompatible"):
                load_training_checkpoint(
                    latest,
                    config=ReinforceConfig(episodes_per_update=2),
                    curriculum=curriculum,
                    cost_weights=weights,
                )

            legacy_mask_path = directory_path / "legacy-mask.pt"
            legacy_mask = torch.load(
                latest, map_location="cpu", weights_only=True
            )
            mask_compatibility = dict(legacy_mask["compatibility"])
            mask_compatibility.pop("action_mask_version")
            legacy_mask["compatibility"] = mask_compatibility
            torch.save(legacy_mask, legacy_mask_path)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                load_training_checkpoint(
                    legacy_mask_path,
                    config=config,
                    curriculum=curriculum,
                    cost_weights=weights,
                )

            legacy_objective = torch.load(
                latest, map_location="cpu", weights_only=True
            )
            compatibility = dict(legacy_objective["compatibility"])
            compatibility.pop("objective_schema_version")
            legacy_weights = dict(compatibility["cost_weights"])
            legacy_weights.pop("delivery")
            compatibility["cost_weights"] = legacy_weights
            legacy_objective["compatibility"] = compatibility
            torch.save(legacy_objective, latest)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                load_training_checkpoint(
                    latest,
                    config=config,
                    curriculum=curriculum,
                    cost_weights=weights,
                )

    def test_version_one_latest_checkpoint_migrates_with_a_persistent_run_id(self) -> None:
        from tempfile import TemporaryDirectory

        config = ReinforceConfig(episodes_per_update=1)
        curriculum = (CurriculumStage(1, 4, 4, 1, 1, 2),)
        weights = EpisodeCostWeights()
        model = SharedMLP()
        trainer = ReinforceTrainer(model, config)
        state = TrainingCheckpointState(
            model.state_dict(), trainer.optimizer.state_dict(),
            random.Random(1).getstate(), torch.get_rng_state(), (),
            10, 2, 0, 0, 250, 0, None, None, 3.0, "cpu",
        )
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "latest.pt"
            save_training_checkpoint(
                checkpoint,
                state,
                config=config,
                curriculum=curriculum,
                cost_weights=weights,
            )
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            payload["checkpoint_version"] = LEGACY_TRAINING_CHECKPOINT_VERSIONS[0]
            payload.pop("run_id")
            payload.pop("current_stage_number")
            torch.save(payload, checkpoint)

            migrated = load_training_checkpoint(
                checkpoint,
                config=config,
                curriculum=curriculum,
                cost_weights=weights,
            )
            self.assertTrue(migrated.run_id)
            save_training_checkpoint(
                checkpoint,
                migrated,
                config=config,
                curriculum=curriculum,
                cost_weights=weights,
            )
            reloaded = load_training_checkpoint(
                checkpoint,
                config=config,
                curriculum=curriculum,
                cost_weights=weights,
            )
            self.assertEqual(reloaded.run_id, migrated.run_id)
            self.assertEqual(reloaded.episodes_trained, 10)
            self.assertEqual(reloaded.optimizer_updates, 2)

    def test_model_only_best_and_full_latest_are_independent(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            best = Path(directory) / "best.pt"
            latest = Path(directory) / "latest.pt"
            model = SharedMLP()
            save_checkpoint(model, best, {"highest_completed_stage": 1})
            best_bytes = best.read_bytes()
            config = ReinforceConfig(episodes_per_update=1)
            curriculum = (CurriculumStage(1, 4, 4, 1, 1, 2),)
            trainer = ReinforceTrainer(model, config)
            save_training_checkpoint(
                latest,
                TrainingCheckpointState(
                    model.state_dict(), trainer.optimizer.state_dict(),
                    random.Random(1).getstate(), torch.get_rng_state(), (),
                    0, 0, 0, 0, 1, -1, None, None, 0.0, "cpu",
                ),
                config=config,
                curriculum=curriculum,
                cost_weights=EpisodeCostWeights(),
            )
            self.assertEqual(best.read_bytes(), best_bytes)
            self.assertTrue(is_training_checkpoint(latest))
            self.assertFalse(is_training_checkpoint(best))

    def test_atomic_latest_save_retries_transient_permission_errors(self) -> None:
        from tempfile import TemporaryDirectory

        config = ReinforceConfig(episodes_per_update=1)
        curriculum = (CurriculumStage(1, 4, 4, 1, 1, 2),)
        model = SharedMLP()
        trainer = ReinforceTrainer(model, config)
        state = TrainingCheckpointState(
            model.state_dict(), trainer.optimizer.state_dict(),
            random.Random(1).getstate(), torch.get_rng_state(), (),
            0, 0, 0, 0, 1, -1, None, None, 0.0, "cpu",
        )
        real_replace = os.replace
        replace_calls = 0

        def transient_lock(source: Path, destination: Path) -> None:
            nonlocal replace_calls
            replace_calls += 1
            if replace_calls < 3:
                raise PermissionError(5, "simulated Windows file lock")
            real_replace(source, destination)

        with TemporaryDirectory() as directory:
            latest = Path(directory) / "latest.pt"
            with (
                patch(
                    "multi_agent_sim.learning.os.replace",
                    side_effect=transient_lock,
                ),
                patch("multi_agent_sim.learning.time.sleep"),
            ):
                save_training_checkpoint(
                    latest,
                    state,
                    config=config,
                    curriculum=curriculum,
                    cost_weights=EpisodeCostWeights(),
                )

            self.assertEqual(replace_calls, 3)
            self.assertTrue(is_training_checkpoint(latest))
            self.assertFalse(any(Path(directory).glob("*.tmp")))

    def _assert_nested_equal(self, expected: object, actual: object) -> None:
        if torch.is_tensor(expected):
            self.assertTrue(torch.equal(expected, actual))
        elif isinstance(expected, dict):
            self.assertEqual(expected.keys(), actual.keys())
            for key in expected:
                self._assert_nested_equal(expected[key], actual[key])
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(len(expected), len(actual))
            for expected_item, actual_item in zip(expected, actual, strict=True):
                self._assert_nested_equal(expected_item, actual_item)
        else:
            self.assertEqual(expected, actual)


class ConfigurationTests(unittest.TestCase):
    def test_default_curriculum_and_hyperparameters_match_contract(self) -> None:
        self.assertEqual(
            tuple(
                (
                    s.width,
                    s.height,
                    s.num_robots,
                    s.num_items,
                    s.max_steps,
                    s.initially_delivered_items,
                )
                for s in DEFAULT_CURRICULUM
            ),
            (
                (4, 4, 1, 1, 32, 0),
                (4, 4, 1, 2, 64, 1),
                (4, 4, 1, 2, 64, 0),
                (5, 5, 1, 2, 64, 0),
                (6, 6, 2, 2, 64, 0),
                (8, 8, 2, 4, 128, 0),
                (10, 10, 4, 4, 160, 0),
                (12, 12, 4, 8, 256, 0),
            ),
        )
        config = ReinforceConfig()
        self.assertEqual(config.learning_rate, 3e-4)
        self.assertEqual(config.episodes_per_update, 32)
        self.assertEqual(config.gamma, 1.0)
        self.assertEqual(config.entropy_coefficient, 0.01)
        self.assertEqual(config.gradient_clip_norm, 1.0)
        self.assertEqual(config.evaluation_interval, 250)
        self.assertEqual(config.evaluation_episodes, 100)
        self.assertEqual(config.previous_stage_probability, 0.4)
        self.assertEqual(config.retention_success_rate, 0.8)
        self.assertEqual(config.time_budget_seconds, 3600.0)

    def test_replay_sampling_and_regression_recovery(self) -> None:
        stages = DEFAULT_CURRICULUM[:3]
        healthy = tuple(
            EvaluationSummary(stage, (), 0.8, 1.0) for stage in stages
        )
        regressed = (
            EvaluationSummary(stages[0], (), 0.79, 1.0),
            EvaluationSummary(stages[1], (), 0.95, 1.0),
            EvaluationSummary(stages[2], (), 0.1, 1.0),
        )
        self.assertEqual(_regressed_stage_index(regressed, 2, 0.8), 0)
        self.assertIsNone(_regressed_stage_index(healthy, 2, 0.8))

        rng = random.Random(4)
        self.assertEqual(
            {_sample_training_stage_index(rng, 2, 0, 0.4) for _ in range(50)},
            {0},
        )
        rng = random.Random(4)
        samples = tuple(
            _sample_training_stage_index(rng, 2, None, 0.4)
            for _ in range(10_000)
        )
        replay_fraction = sum(index < 2 for index in samples) / len(samples)
        self.assertAlmostEqual(replay_fraction, 0.4, delta=0.02)
        self.assertGreater(samples.count(0), 0)
        self.assertGreater(samples.count(1), 0)

    def test_returns_and_validation_seed_sets_are_deterministic(self) -> None:
        self.assertEqual(discounted_returns((1.0, 2.0, 3.0)), (6.0, 5.0, 3.0))
        self.assertEqual(
            discounted_returns((1.0, 2.0, 3.0), gamma=0.5),
            (2.75, 3.5, 3.0),
        )
        first, second = DEFAULT_CURRICULUM[:2]
        self.assertTrue(
            set(validation_seeds(first, 100)).isdisjoint(
                validation_seeds(second, 100)
            )
        )

    def test_invalid_training_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ReinforceConfig(gamma=1.01)
        with self.assertRaises(ValueError):
            ReinforceConfig(episodes_per_update=0)
        with self.assertRaises(ValueError):
            CurriculumStage(1, 2, 2, 5, 1, 10)


if __name__ == "__main__":
    unittest.main()
