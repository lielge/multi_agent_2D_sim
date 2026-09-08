"""Optional PyTorch policies and REINFORCE training helpers.

This module deliberately depends only on the public, encoded training-
environment contract.  The simulation core does not import it, so importing
and using :mod:`multi_agent_sim` never requires PyTorch.  The module itself is
also safe to import without PyTorch; constructing a model or trainer then
raises an actionable :class:`ImportError`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import json
import math
import os
from pathlib import Path
import pickle
import random
import tempfile
import time
from typing import Any, Protocol, TypeAlias, cast, runtime_checkable
import uuid

from .episode import OBJECTIVE_SCHEMA_VERSION
from .training_env import (
    ACTION_MAP_VERSION,
    ACTION_MASK_VERSION,
    ACTION_ORDER,
    ENCODING_VERSION,
    FEATURE_DIM,
    OneHotCommand,
)

try:  # Keep the simulator usable without the optional training dependency.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised in installations without Torch.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    _TorchModuleBase = object
else:
    _TorchModuleBase = nn.Module


INPUT_FEATURES = FEATURE_DIM
HIDDEN_FEATURES = 128
NUM_ACTIONS = 7
# Keep checkpoint metadata exactly aligned with the framework-neutral encoder.
ENCODING_SCHEMA_VERSION = ENCODING_VERSION
TRAINING_CHECKPOINT_TYPE = "multi-agent-sim-training-state"
TRAINING_CHECKPOINT_VERSION = 2
LEGACY_TRAINING_CHECKPOINT_VERSIONS = (1,)
_CHECKPOINT_REPLACE_ATTEMPTS = 8

ProgressCallback: TypeAlias = Callable[[str], None]


def torch_available() -> bool:
    """Return whether the optional PyTorch dependency is installed."""

    return torch is not None


def _require_torch() -> None:
    if torch is None or nn is None:
        raise ImportError(
            "PyTorch is required for multi_agent_sim.learning; "
            "install the project with the 'training' extra"
        )


def _validate_positive_int(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_probability(value: float, name: str) -> float:
    normalized = _validate_finite_number(value, name)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return normalized


def _validate_nonnegative_number(value: float, name: str) -> float:
    normalized = _validate_finite_number(value, name)
    if normalized < 0.0:
        raise ValueError(f"{name} cannot be negative")
    return normalized


def _validate_positive_number(value: float, name: str) -> float:
    normalized = _validate_finite_number(value, name)
    if normalized <= 0.0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _validate_finite_number(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


class SharedMLP(_TorchModuleBase):  # type: ignore[misc]
    """The shared ``155 -> 128 -> 128 -> 7`` action-logit network."""

    input_features = INPUT_FEATURES
    hidden_features = HIDDEN_FEATURES
    output_features = NUM_ACTIONS

    def __init__(self) -> None:
        _require_torch()
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(INPUT_FEATURES, HIDDEN_FEATURES),
            nn.ReLU(),
            nn.Linear(HIDDEN_FEATURES, HIDDEN_FEATURES),
            nn.ReLU(),
            nn.Linear(HIDDEN_FEATURES, NUM_ACTIONS),
        )

    def forward(self, features: Any) -> Any:
        """Return seven unnormalized action logits for each feature row."""

        if getattr(features, "ndim", 0) < 1:
            raise ValueError("features must have at least one dimension")
        if features.shape[-1] != INPUT_FEATURES:
            raise ValueError(
                f"the last feature dimension must be {INPUT_FEATURES}, "
                f"got {features.shape[-1]}"
            )
        return self.layers(features)


@runtime_checkable
class EncodedObservation(Protocol):
    """Small structural interface consumed by learned robot controllers."""

    @property
    def active_robot_ids(self) -> tuple[str, ...]: ...

    def row_for(self, robot_id: str) -> Sequence[float]: ...

    def action_mask_for(self, robot_id: str) -> Sequence[int | bool]: ...


@runtime_checkable
class TrainingEnvironment(Protocol):
    """Structural environment interface used by rollout helpers."""

    def reset(self, seed: int) -> EncodedObservation: ...

    def step(self, commands: Mapping[str, Sequence[int]]) -> Any: ...


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Detached, plain-Python diagnostics for one policy selection."""

    robot_id: str
    probabilities: tuple[float, ...]
    legal_mask: tuple[bool, ...]
    selected_action_index: int
    selected_action: str
    entropy: float
    stochastic: bool

    def __post_init__(self) -> None:
        if not isinstance(self.robot_id, str) or not self.robot_id:
            raise ValueError("robot_id must be a non-empty string")
        probabilities = tuple(float(value) for value in self.probabilities)
        mask = tuple(self.legal_mask)
        if len(probabilities) != NUM_ACTIONS or any(
            not math.isfinite(value) or value < 0 or value > 1
            for value in probabilities
        ):
            raise ValueError("probabilities must contain seven values from zero to one")
        if not math.isclose(math.fsum(probabilities), 1.0, abs_tol=1e-6):
            raise ValueError("probabilities must sum to one")
        # Torch commonly supplies float32 probabilities whose detached Python
        # values differ from one by roughly 1e-7.  Normalize the diagnostic
        # copy (never the training distribution) so JSON consumers receive a
        # proper probability vector with a stable sum.
        probability_total = math.fsum(probabilities)
        probabilities = tuple(
            value / probability_total for value in probabilities
        )
        correction_index = max(range(NUM_ACTIONS), key=probabilities.__getitem__)
        corrected = list(probabilities)
        corrected[correction_index] += 1.0 - sum(corrected)
        probabilities = tuple(corrected)
        if len(mask) != NUM_ACTIONS or any(type(value) is not bool for value in mask):
            raise ValueError("legal_mask must contain seven booleans")
        if type(self.selected_action_index) is not int or not (
            0 <= self.selected_action_index < NUM_ACTIONS
        ):
            raise ValueError("selected_action_index must be from zero through six")
        expected_action = ACTION_ORDER[self.selected_action_index].value
        if self.selected_action != expected_action:
            raise ValueError("selected_action does not match selected_action_index")
        entropy = _validate_finite_number(self.entropy, "entropy")
        if entropy < 0:
            raise ValueError("entropy cannot be negative")
        if type(self.stochastic) is not bool:
            raise ValueError("stochastic must be a bool")
        object.__setattr__(self, "probabilities", probabilities)
        object.__setattr__(self, "legal_mask", mask)
        object.__setattr__(self, "entropy", entropy)

    @property
    def selection_mode(self) -> str:
        return "stochastic" if self.stochastic else "greedy"


class MLPRobotController:
    """One robot's policy view backed by a team-shared :class:`SharedMLP`.

    A controller emits plain immutable one-hot tuples.  The most recent
    selection's differentiable log-probability and entropy stay private to the
    learning block and are exposed read-only for the trainer.
    """

    def __init__(
        self,
        robot_id: str,
        model: SharedMLP,
        *,
        stochastic: bool = True,
        use_action_mask: bool = True,
    ) -> None:
        _require_torch()
        if not isinstance(robot_id, str) or not robot_id:
            raise ValueError("robot_id must be a non-empty string")
        if not isinstance(model, SharedMLP):
            raise ValueError("model must be a SharedMLP")
        if type(stochastic) is not bool:
            raise ValueError("stochastic must be a bool")
        if type(use_action_mask) is not bool:
            raise ValueError("use_action_mask must be a bool")
        self.robot_id = robot_id
        self.model = model
        self.stochastic = stochastic
        self.use_action_mask = use_action_mask
        self._last_log_probability: Any | None = None
        self._last_entropy: Any | None = None
        self._last_decision: PolicyDecision | None = None

    @property
    def last_log_probability(self) -> Any:
        """Differentiable log-probability of the last selected command."""

        if self._last_log_probability is None:
            raise RuntimeError("the controller has not selected a command yet")
        return self._last_log_probability

    @property
    def last_entropy(self) -> Any:
        """Differentiable categorical entropy from the last selection."""

        if self._last_entropy is None:
            raise RuntimeError("the controller has not selected a command yet")
        return self._last_entropy

    @property
    def last_decision(self) -> PolicyDecision:
        if self._last_decision is None:
            raise RuntimeError("the controller has not selected a command yet")
        return self._last_decision

    def choose_command(self, observation: EncodedObservation) -> OneHotCommand:
        """Select this controller's command from a shared encoded observation."""

        if not isinstance(observation, EncodedObservation):
            raise ValueError(
                "observation must provide active_robot_ids, row_for(), and "
                "action_mask_for()"
            )
        if self.robot_id not in observation.active_robot_ids:
            raise ValueError(
                f"robot {self.robot_id!r} is not active in this observation"
            )
        mask = (
            observation.action_mask_for(self.robot_id)
            if self.use_action_mask
            else None
        )
        return self.choose_from_row(observation.row_for(self.robot_id), mask)

    def choose_from_row(
        self,
        feature_row: Sequence[float],
        action_mask: Sequence[int | bool] | None = None,
    ) -> OneHotCommand:
        """Select a command directly from one 155-value feature row."""

        row = _validated_feature_row(feature_row)
        device = next(self.model.parameters()).device
        features = torch.tensor(row, dtype=torch.float32, device=device)
        logits = self.model(features)

        effective_mask = (True,) * NUM_ACTIONS
        if self.use_action_mask and action_mask is not None:
            mask = _validated_action_mask(action_mask)
            effective_mask = mask
            mask_tensor = torch.tensor(mask, dtype=torch.bool, device=device)
            logits = logits.masked_fill(~mask_tensor, -torch.inf)

        distribution = torch.distributions.Categorical(logits=logits)
        if self.stochastic:
            selected = distribution.sample()
        else:
            selected = torch.argmax(logits)

        self._last_log_probability = distribution.log_prob(selected)
        self._last_entropy = distribution.entropy()
        selected_index = int(selected.detach().cpu().item())
        probabilities = tuple(
            float(value)
            for value in distribution.probs.detach().cpu().tolist()
        )
        self._last_decision = PolicyDecision(
            robot_id=self.robot_id,
            probabilities=probabilities,
            legal_mask=effective_mask,
            selected_action_index=selected_index,
            selected_action=ACTION_ORDER[selected_index].value,
            entropy=float(self._last_entropy.detach().cpu().item()),
            stochastic=self.stochastic,
        )
        return cast(
            OneHotCommand,
            tuple(1 if index == selected_index else 0 for index in range(NUM_ACTIONS)),
        )


def _validated_feature_row(feature_row: Sequence[float]) -> tuple[float, ...]:
    if isinstance(feature_row, (str, bytes)):
        raise ValueError("feature_row must be a sequence of finite numbers")
    try:
        row = tuple(feature_row)
    except TypeError as exc:
        raise ValueError("feature_row must be a sequence of finite numbers") from exc
    if len(row) != INPUT_FEATURES:
        raise ValueError(
            f"feature_row must contain exactly {INPUT_FEATURES} values"
        )
    normalized: list[float] = []
    for index, value in enumerate(row):
        normalized.append(_validate_finite_number(value, f"feature_row[{index}]"))
    return tuple(normalized)


def _validated_action_mask(
    action_mask: Sequence[int | bool],
) -> tuple[bool, ...]:
    if isinstance(action_mask, (str, bytes)):
        raise ValueError("action_mask must be a sequence of seven binary values")
    try:
        values = tuple(action_mask)
    except TypeError as exc:
        raise ValueError(
            "action_mask must be a sequence of seven binary values"
        ) from exc
    if len(values) != NUM_ACTIONS:
        raise ValueError("action_mask must contain exactly seven values")
    if any(value not in (0, 1, False, True) for value in values):
        raise ValueError("action_mask values must be binary")
    normalized = tuple(bool(value) for value in values)
    if not any(normalized):
        raise ValueError("action_mask must allow at least one action")
    return normalized


def create_shared_controllers(
    robot_ids: Iterable[str],
    model: SharedMLP,
    *,
    stochastic: bool,
    use_action_masks: bool = True,
) -> dict[str, MLPRobotController]:
    """Create one controller per robot, all referencing the same model."""

    try:
        normalized_ids = tuple(robot_ids)
    except TypeError as exc:
        raise ValueError("robot_ids must be an iterable of strings") from exc
    if not normalized_ids:
        raise ValueError("robot_ids must contain at least one robot")
    if any(not isinstance(robot_id, str) or not robot_id for robot_id in normalized_ids):
        raise ValueError("robot_ids must contain non-empty strings")
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("robot_ids must be unique")
    return {
        robot_id: MLPRobotController(
            robot_id,
            model,
            stochastic=stochastic,
            use_action_mask=use_action_masks,
        )
        for robot_id in normalized_ids
    }


@dataclass(frozen=True, slots=True)
class CurriculumStage:
    """One progressively harder training scenario configuration."""

    number: int
    width: int
    height: int
    num_robots: int
    num_items: int
    max_steps: int
    initially_delivered_items: int = 0

    def __post_init__(self) -> None:
        _validate_positive_int(self.number, "number")
        _validate_positive_int(self.width, "width")
        _validate_positive_int(self.height, "height")
        _validate_positive_int(self.num_robots, "num_robots")
        _validate_positive_int(self.num_items, "num_items")
        _validate_positive_int(self.max_steps, "max_steps")
        if (
            type(self.initially_delivered_items) is not int
            or self.initially_delivered_items < 0
        ):
            raise ValueError(
                "initially_delivered_items must be a non-negative integer"
            )
        if self.initially_delivered_items > self.num_items:
            raise ValueError(
                "initially_delivered_items cannot exceed num_items"
            )
        if self.num_robots > 4:
            raise ValueError("num_robots cannot exceed four fixed slots")
        if self.num_items > 8:
            raise ValueError("num_items cannot exceed eight fixed slots")
        required_cells = (
            self.num_robots
            + 2 * self.num_items
            - self.initially_delivered_items
        )
        if required_cells > self.width * self.height:
            raise ValueError("the stage does not have enough cells for its entities")


DEFAULT_CURRICULUM: tuple[CurriculumStage, ...] = (
    CurriculumStage(1, 4, 4, 1, 1, 32),
    CurriculumStage(2, 4, 4, 1, 2, 64, 1),
    CurriculumStage(3, 4, 4, 1, 2, 64),
    CurriculumStage(4, 5, 5, 1, 2, 64),
    CurriculumStage(5, 6, 6, 2, 2, 64),
    CurriculumStage(6, 8, 8, 2, 4, 128),
    CurriculumStage(7, 10, 10, 4, 4, 160),
    CurriculumStage(8, 12, 12, 4, 8, 256),
)


@dataclass(frozen=True, slots=True)
class ReinforceConfig:
    """Validated defaults for parameter-sharing episodic REINFORCE."""

    learning_rate: float = 3e-4
    episodes_per_update: int = 32
    gamma: float = 1.0
    entropy_coefficient: float = 0.01
    gradient_clip_norm: float = 1.0
    evaluation_interval: int = 250
    evaluation_episodes: int = 100
    promotion_success_rate: float = 0.8
    previous_stage_probability: float = 0.4
    retention_success_rate: float = 0.8
    time_budget_seconds: float = 3600.0
    training_seed: int = 1729
    validation_seed_base: int = 1_000_000
    use_action_masks: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "learning_rate",
            _validate_positive_number(self.learning_rate, "learning_rate"),
        )
        _validate_positive_int(self.episodes_per_update, "episodes_per_update")
        object.__setattr__(self, "gamma", _validate_probability(self.gamma, "gamma"))
        object.__setattr__(
            self,
            "entropy_coefficient",
            _validate_nonnegative_number(
                self.entropy_coefficient, "entropy_coefficient"
            ),
        )
        object.__setattr__(
            self,
            "gradient_clip_norm",
            _validate_positive_number(
                self.gradient_clip_norm, "gradient_clip_norm"
            ),
        )
        _validate_positive_int(self.evaluation_interval, "evaluation_interval")
        _validate_positive_int(self.evaluation_episodes, "evaluation_episodes")
        object.__setattr__(
            self,
            "promotion_success_rate",
            _validate_probability(
                self.promotion_success_rate, "promotion_success_rate"
            ),
        )
        object.__setattr__(
            self,
            "previous_stage_probability",
            _validate_probability(
                self.previous_stage_probability, "previous_stage_probability"
            ),
        )
        object.__setattr__(
            self,
            "retention_success_rate",
            _validate_probability(
                self.retention_success_rate, "retention_success_rate"
            ),
        )
        object.__setattr__(
            self,
            "time_budget_seconds",
            _validate_positive_number(
                self.time_budget_seconds, "time_budget_seconds"
            ),
        )
        if type(self.training_seed) is not int:
            raise ValueError("training_seed must be an integer")
        if type(self.validation_seed_base) is not int or self.validation_seed_base < 0:
            raise ValueError("validation_seed_base must be a non-negative integer")
        if type(self.use_action_masks) is not bool:
            raise ValueError("use_action_masks must be a bool")


@dataclass(frozen=True, slots=True)
class EpisodeRollout:
    """One collected episode, including tensors needed for optimization."""

    seed: int
    stage_number: int
    rewards: tuple[float, ...]
    joint_log_probabilities: tuple[Any, ...]
    joint_entropies: tuple[Any, ...]
    metrics: Any

    def __post_init__(self) -> None:
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        _validate_positive_int(self.stage_number, "stage_number")
        if not (
            len(self.rewards)
            == len(self.joint_log_probabilities)
            == len(self.joint_entropies)
        ):
            raise ValueError("rollout transition sequences must have equal lengths")
        for index, reward in enumerate(self.rewards):
            _validate_finite_number(reward, f"rewards[{index}]")

    @property
    def undiscounted_return(self) -> float:
        return math.fsum(self.rewards)


@dataclass(frozen=True, slots=True)
class OptimizationMetrics:
    """Scalar diagnostics from one optimizer update."""

    loss: float
    policy_loss: float
    mean_entropy: float
    gradient_norm: float
    transitions: int


@dataclass(frozen=True, slots=True)
class EpisodeEvaluation:
    """Seed and external objective metrics for one greedy episode."""

    seed: int
    metrics: Any

    def to_dict(self) -> dict[str, Any]:
        return {"seed": self.seed, "metrics": _jsonable(self.metrics)}


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """Per-episode and aggregate greedy results for one curriculum stage."""

    stage: CurriculumStage
    episodes: tuple[EpisodeEvaluation, ...]
    success_rate: float
    mean_cost: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": _jsonable(self.stage),
            "success_rate": self.success_rate,
            "mean_cost": self.mean_cost,
            "episodes": [episode.to_dict() for episode in self.episodes],
        }


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Final curriculum outcome and selected checkpoint locations."""

    episodes_trained: int
    optimizer_updates: int
    highest_completed_stage: int
    elapsed_seconds: float
    total_elapsed_seconds: float
    checkpoint_path: Path
    latest_checkpoint_path: Path
    metadata_path: Path
    evaluations: tuple[EvaluationSummary, ...]
    run_id: str
    interrupted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes_trained": self.episodes_trained,
            "optimizer_updates": self.optimizer_updates,
            "highest_completed_stage": self.highest_completed_stage,
            "elapsed_seconds": self.elapsed_seconds,
            "total_elapsed_seconds": self.total_elapsed_seconds,
            "checkpoint_path": str(self.checkpoint_path),
            "latest_checkpoint_path": str(self.latest_checkpoint_path),
            "metadata_path": str(self.metadata_path),
            "evaluations": [evaluation.to_dict() for evaluation in self.evaluations],
            "run_id": self.run_id,
            "interrupted": self.interrupted,
        }


def discounted_returns(
    rewards: Sequence[float],
    gamma: float = 1.0,
) -> tuple[float, ...]:
    """Calculate a return-to-go for every timestep."""

    discount = _validate_probability(gamma, "gamma")
    normalized_rewards = tuple(
        _validate_finite_number(reward, f"rewards[{index}]")
        for index, reward in enumerate(rewards)
    )
    result = [0.0] * len(normalized_rewards)
    running = 0.0
    for index in range(len(normalized_rewards) - 1, -1, -1):
        running = normalized_rewards[index] + discount * running
        result[index] = running
    return tuple(result)


def collect_episode(
    environment: TrainingEnvironment,
    model: SharedMLP,
    stage: CurriculumStage,
    seed: int,
    *,
    stochastic: bool,
    use_action_masks: bool = True,
) -> EpisodeRollout:
    """Run one complete episode through independent per-robot controllers."""

    _require_torch()
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if not isinstance(stage, CurriculumStage):
        raise ValueError("stage must be a CurriculumStage")
    if not isinstance(environment, TrainingEnvironment):
        raise ValueError("environment must provide reset() and step()")

    observation = environment.reset(seed)
    controllers = create_shared_controllers(
        observation.active_robot_ids,
        model,
        stochastic=stochastic,
        use_action_masks=use_action_masks,
    )
    rewards: list[float] = []
    joint_log_probabilities: list[Any] = []
    joint_entropies: list[Any] = []
    transition: Any | None = None
    terminated = bool(getattr(environment, "terminated", False))

    while not terminated:
        commands = {
            robot_id: controller.choose_command(observation)
            for robot_id, controller in controllers.items()
        }
        log_probabilities = tuple(
            controller.last_log_probability for controller in controllers.values()
        )
        entropies = tuple(
            controller.last_entropy for controller in controllers.values()
        )
        joint_log_probabilities.append(torch.stack(log_probabilities).sum())
        joint_entropies.append(torch.stack(entropies).sum())

        transition = environment.step(commands)
        reward = _validate_finite_number(transition.reward, "transition.reward")
        rewards.append(reward)
        observation = transition.observation
        terminated = bool(transition.terminated)

    metrics = (
        transition.metrics
        if transition is not None
        else getattr(environment, "metrics", None)
    )
    if metrics is None:
        raise ValueError("a terminal environment must expose episode metrics")
    return EpisodeRollout(
        seed=seed,
        stage_number=stage.number,
        rewards=tuple(rewards),
        joint_log_probabilities=tuple(joint_log_probabilities),
        joint_entropies=tuple(joint_entropies),
        metrics=metrics,
    )


class ReinforceTrainer:
    """Optimizer for batches of parameter-sharing team rollouts."""

    def __init__(
        self,
        model: SharedMLP | None = None,
        config: ReinforceConfig | None = None,
        *,
        device: str = "cpu",
    ) -> None:
        _require_torch()
        self.config = config or ReinforceConfig()
        if not isinstance(self.config, ReinforceConfig):
            raise ValueError("config must be a ReinforceConfig")
        self.model = model or SharedMLP()
        if not isinstance(self.model, SharedMLP):
            raise ValueError("model must be a SharedMLP")
        try:
            self.device = torch.device(device)
            self.model.to(self.device)
        except (RuntimeError, TypeError) as exc:
            raise ValueError(f"invalid or unavailable Torch device: {device!r}") from exc
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.config.learning_rate
        )

    def update(self, rollouts: Sequence[EpisodeRollout]) -> OptimizationMetrics:
        """Apply one return-to-go REINFORCE update with a batch baseline.

        Returns are centered separately at each timestep across the episodes
        that reached that timestep.  This is a non-learned variance-reduction
        baseline: it does not alter the environment reward or rescale the
        configured cost terms, and it avoids treating a late failure penalty
        as preferable merely because it is numerically closer to zero than an
        earlier return-to-go.
        """

        try:
            rollout_batch = tuple(rollouts)
        except TypeError as exc:
            raise ValueError("rollouts must be a sequence") from exc
        if not rollout_batch:
            raise ValueError("rollouts must not be empty")
        if not all(isinstance(rollout, EpisodeRollout) for rollout in rollout_batch):
            raise ValueError("rollouts must contain EpisodeRollout values")

        returns_by_rollout: list[tuple[float, ...]] = []
        log_probabilities: list[Any] = []
        entropies: list[Any] = []
        for rollout in rollout_batch:
            returns_by_rollout.append(
                discounted_returns(rollout.rewards, self.config.gamma)
            )
            log_probabilities.extend(rollout.joint_log_probabilities)
            entropies.extend(rollout.joint_entropies)
        if not log_probabilities:
            raise ValueError("rollouts must contain at least one transition")

        advantages_by_rollout = _per_timestep_advantages(returns_by_rollout)
        flat_advantages = tuple(
            advantage
            for rollout_advantages in advantages_by_rollout
            for advantage in rollout_advantages
        )
        advantages_tensor = torch.tensor(
            flat_advantages, dtype=torch.float32, device=self.device
        )

        stacked_log_probabilities = torch.stack(log_probabilities)
        stacked_entropies = torch.stack(entropies)
        if stacked_log_probabilities.device != self.device:
            raise ValueError("rollout tensors are not on the trainer's device")
        policy_loss = -(stacked_log_probabilities * advantages_tensor).mean()
        mean_entropy = stacked_entropies.mean()
        loss = policy_loss - self.config.entropy_coefficient * mean_entropy

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.gradient_clip_norm
        )
        self.optimizer.step()
        return OptimizationMetrics(
            loss=float(loss.detach().cpu().item()),
            policy_loss=float(policy_loss.detach().cpu().item()),
            mean_entropy=float(mean_entropy.detach().cpu().item()),
            gradient_norm=float(torch.as_tensor(gradient_norm).detach().cpu().item()),
            transitions=len(flat_advantages),
        )


def _per_timestep_advantages(
    returns_by_rollout: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    """Center returns against episodes at the same timestep."""

    normalized = tuple(tuple(returns) for returns in returns_by_rollout)
    if not normalized:
        return ()
    advantages = [[0.0] * len(returns) for returns in normalized]
    longest = max((len(returns) for returns in normalized), default=0)
    for timestep in range(longest):
        active = tuple(
            (rollout_index, returns[timestep])
            for rollout_index, returns in enumerate(normalized)
            if timestep < len(returns)
        )
        values = torch.tensor(
            tuple(value for _, value in active),
            dtype=torch.float64,
        )
        centered = values - values.mean()
        for (rollout_index, _), advantage in zip(active, centered, strict=True):
            advantages[rollout_index][timestep] = float(advantage.item())
    return tuple(tuple(values) for values in advantages)


EnvironmentFactory: TypeAlias = Callable[[CurriculumStage, int], TrainingEnvironment]


def evaluate_policy(
    environment_factory: EnvironmentFactory,
    model: SharedMLP,
    stage: CurriculumStage,
    seeds: Iterable[int],
    *,
    use_action_masks: bool = True,
) -> EvaluationSummary:
    """Greedily evaluate a model and retain every episode's objective metrics."""

    _require_torch()
    seed_values = tuple(seeds)
    if not seed_values or any(type(seed) is not int for seed in seed_values):
        raise ValueError("seeds must contain at least one integer")
    was_training = model.training
    model.eval()
    episodes: list[EpisodeEvaluation] = []
    try:
        with torch.no_grad():
            for seed in seed_values:
                environment = environment_factory(stage, seed)
                rollout = collect_episode(
                    environment,
                    model,
                    stage,
                    seed,
                    stochastic=False,
                    use_action_masks=use_action_masks,
                )
                episodes.append(EpisodeEvaluation(seed, rollout.metrics))
    finally:
        model.train(was_training)

    successes = tuple(bool(_metric_value(ep.metrics, "success")) for ep in episodes)
    costs = tuple(float(_metric_value(ep.metrics, "total_cost")) for ep in episodes)
    return EvaluationSummary(
        stage=stage,
        episodes=tuple(episodes),
        success_rate=sum(successes) / len(successes),
        mean_cost=math.fsum(costs) / len(costs),
    )


def _metric_value(metrics: Any, name: str) -> Any:
    if isinstance(metrics, Mapping):
        try:
            return metrics[name]
        except KeyError:
            raise ValueError(f"episode metrics do not contain {name!r}") from None
    try:
        return getattr(metrics, name)
    except AttributeError:
        raise ValueError(f"episode metrics do not expose {name!r}") from None


def validation_seeds(
    stage: CurriculumStage,
    count: int,
    base: int = 1_000_000,
) -> tuple[int, ...]:
    """Return deterministic, stage-disjoint held-out evaluation seeds."""

    if not isinstance(stage, CurriculumStage):
        raise ValueError("stage must be a CurriculumStage")
    _validate_positive_int(count, "count")
    if type(base) is not int or base < 0:
        raise ValueError("base must be a non-negative integer")
    stage_base = base + (stage.number - 1) * count
    return tuple(stage_base + index for index in range(count))


@dataclass(frozen=True, slots=True)
class TrainingCheckpointState:
    """Complete optimizer-boundary state needed to continue training."""

    model_state_dict: Mapping[str, Any]
    optimizer_state_dict: Mapping[str, Any]
    python_rng_state: Any
    torch_rng_state: Any
    cuda_rng_states: tuple[Any, ...]
    episodes_trained: int
    optimizer_updates: int
    current_stage_index: int
    highest_completed_stage: int
    next_evaluation: int
    last_evaluation_episode: int
    recovery_stage_index: int | None
    best_score: tuple[int, float, float] | None
    total_elapsed_seconds: float
    saved_device_type: str
    run_id: str = ""


def _training_config_signature(config: ReinforceConfig) -> dict[str, Any]:
    payload = asdict(config)
    # Each invocation receives a fresh wall-clock budget; all parameters that
    # influence sampled episodes or optimizer updates remain compatibility-
    # checked.
    payload.pop("time_budget_seconds")
    return payload


def _training_compatibility_payload(
    config: ReinforceConfig,
    curriculum: Sequence[CurriculumStage],
    cost_weights: Any,
) -> dict[str, Any]:
    return {
        "encoding_schema_version": ENCODING_SCHEMA_VERSION,
        "action_map_version": ACTION_MAP_VERSION,
        "action_mask_version": ACTION_MASK_VERSION,
        "objective_schema_version": OBJECTIVE_SCHEMA_VERSION,
        "architecture": [INPUT_FEATURES, HIDDEN_FEATURES, HIDDEN_FEATURES, NUM_ACTIONS],
        "torch_version": str(torch.__version__),
        "training_config": _jsonable(_training_config_signature(config)),
        "curriculum": _jsonable(tuple(curriculum)),
        "cost_weights": _jsonable(cost_weights),
    }


def _load_torch_payload(path: Path, device: Any) -> Any:
    try:
        try:
            return torch.load(path, map_location=device, weights_only=True)
        except TypeError:  # PyTorch before the ``weights_only`` argument.
            return torch.load(path, map_location=device)
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        EOFError,
        pickle.UnpicklingError,
    ) as exc:
        raise ValueError(f"could not load checkpoint {str(path)!r}") from exc


def is_training_checkpoint(checkpoint_path: str | Path) -> bool:
    """Return whether a checkpoint is a full resumable training bundle."""

    _require_torch()
    checkpoint = Path(checkpoint_path)
    payload = _load_torch_payload(checkpoint, torch.device("cpu"))
    return (
        isinstance(payload, Mapping)
        and payload.get("checkpoint_type") == TRAINING_CHECKPOINT_TYPE
    )


def _replace_checkpoint_with_retry(source: Path, destination: Path) -> None:
    """Atomically replace a checkpoint despite short-lived Windows file locks."""

    for attempt in range(_CHECKPOINT_REPLACE_ATTEMPTS):
        try:
            os.replace(source, destination)
            return
        except PermissionError as exc:
            if attempt + 1 == _CHECKPOINT_REPLACE_ATTEMPTS:
                raise PermissionError(
                    f"could not atomically replace checkpoint {str(destination)!r} "
                    "because it remained locked; close programs scanning or "
                    "opening the file and resume from the previous checkpoint"
                ) from exc
            time.sleep(min(0.05 * (2**attempt), 0.5))


def save_training_checkpoint(
    checkpoint_path: str | Path,
    state: TrainingCheckpointState,
    *,
    config: ReinforceConfig,
    curriculum: Sequence[CurriculumStage],
    cost_weights: Any,
) -> Path:
    """Atomically save a complete training state at an optimizer boundary."""

    _require_torch()
    if not isinstance(state, TrainingCheckpointState):
        raise ValueError("state must be a TrainingCheckpointState")
    if not 0 <= state.current_stage_index < len(curriculum):
        raise ValueError("state has an invalid current stage")
    if not 0 <= state.highest_completed_stage <= len(curriculum):
        raise ValueError("state has an invalid highest completed stage")
    if state.recovery_stage_index is not None and not (
        0 <= state.recovery_stage_index < state.current_stage_index
    ):
        raise ValueError("state has an invalid recovery stage")
    if state.episodes_trained < 0 or state.optimizer_updates < 0:
        raise ValueError("state counters cannot be negative")
    if state.next_evaluation <= 0:
        raise ValueError("state next evaluation must be positive")
    if not -1 <= state.last_evaluation_episode <= state.episodes_trained:
        raise ValueError("state has an invalid last evaluation episode")
    if (
        not math.isfinite(state.total_elapsed_seconds)
        or state.total_elapsed_seconds < 0
    ):
        raise ValueError("state total elapsed seconds must be finite and non-negative")
    if not state.saved_device_type:
        raise ValueError("state saved device type must not be empty")
    run_id = state.run_id or uuid.uuid4().hex
    checkpoint = Path(checkpoint_path)
    if not checkpoint.name:
        raise ValueError("checkpoint_path must name a file")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_type": TRAINING_CHECKPOINT_TYPE,
        "checkpoint_version": TRAINING_CHECKPOINT_VERSION,
        "compatibility": _training_compatibility_payload(
            config, curriculum, cost_weights
        ),
        "model_state_dict": dict(state.model_state_dict),
        "optimizer_state_dict": dict(state.optimizer_state_dict),
        "python_rng_state": state.python_rng_state,
        "torch_rng_state": state.torch_rng_state,
        "cuda_rng_states": state.cuda_rng_states,
        "episodes_trained": state.episodes_trained,
        "optimizer_updates": state.optimizer_updates,
        "current_stage_index": state.current_stage_index,
        "current_stage_number": curriculum[state.current_stage_index].number,
        "highest_completed_stage": state.highest_completed_stage,
        "next_evaluation": state.next_evaluation,
        "last_evaluation_episode": state.last_evaluation_episode,
        "recovery_stage_index": state.recovery_stage_index,
        "best_score": state.best_score,
        "total_elapsed_seconds": state.total_elapsed_seconds,
        "saved_device_type": state.saved_device_type,
        "run_id": run_id,
    }

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{checkpoint.name}.",
        suffix=".tmp",
        dir=checkpoint.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        _replace_checkpoint_with_retry(temporary_path, checkpoint)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return checkpoint


def load_training_checkpoint(
    checkpoint_path: str | Path,
    *,
    config: ReinforceConfig,
    curriculum: Sequence[CurriculumStage],
    cost_weights: Any,
    device: str = "cpu",
) -> TrainingCheckpointState:
    """Load and compatibility-check a full resumable training bundle."""

    _require_torch()
    checkpoint = Path(checkpoint_path)
    target_device = torch.device(device)
    payload = _load_torch_payload(checkpoint, target_device)
    if not isinstance(payload, Mapping) or payload.get(
        "checkpoint_type"
    ) != TRAINING_CHECKPOINT_TYPE:
        raise ValueError("checkpoint is model-only and cannot be resumed exactly")
    checkpoint_version = payload.get("checkpoint_version")
    if checkpoint_version not in (
        TRAINING_CHECKPOINT_VERSION,
        *LEGACY_TRAINING_CHECKPOINT_VERSIONS,
    ):
        raise ValueError("unsupported training checkpoint version")
    expected = _training_compatibility_payload(config, curriculum, cost_weights)
    actual = payload.get("compatibility")
    if actual != expected:
        raise ValueError(
            "training checkpoint is incompatible with the requested "
            "architecture, dependencies, curriculum, weights, or configuration"
        )
    try:
        best_score_value = payload["best_score"]
        state = TrainingCheckpointState(
            model_state_dict=payload["model_state_dict"],
            optimizer_state_dict=payload["optimizer_state_dict"],
            python_rng_state=payload["python_rng_state"],
            torch_rng_state=payload["torch_rng_state"],
            cuda_rng_states=tuple(payload["cuda_rng_states"]),
            episodes_trained=int(payload["episodes_trained"]),
            optimizer_updates=int(payload["optimizer_updates"]),
            current_stage_index=int(payload["current_stage_index"]),
            highest_completed_stage=int(payload["highest_completed_stage"]),
            next_evaluation=int(payload["next_evaluation"]),
            last_evaluation_episode=int(payload["last_evaluation_episode"]),
            recovery_stage_index=(
                None
                if payload["recovery_stage_index"] is None
                else int(payload["recovery_stage_index"])
            ),
            best_score=(
                None
                if best_score_value is None
                else (
                    int(best_score_value[0]),
                    float(best_score_value[1]),
                    float(best_score_value[2]),
                )
            ),
            total_elapsed_seconds=float(payload["total_elapsed_seconds"]),
            saved_device_type=str(payload["saved_device_type"]),
            run_id=str(payload.get("run_id") or uuid.uuid4().hex),
        )
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError("training checkpoint state is incomplete or invalid") from exc
    if not 0 <= state.current_stage_index < len(curriculum):
        raise ValueError("training checkpoint has an invalid current stage")
    saved_stage_number = payload.get("current_stage_number")
    if saved_stage_number is not None and saved_stage_number != curriculum[
        state.current_stage_index
    ].number:
        raise ValueError("training checkpoint current stage is inconsistent")
    if not 0 <= state.highest_completed_stage <= len(curriculum):
        raise ValueError("training checkpoint has an invalid highest completed stage")
    if state.recovery_stage_index is not None and not (
        0 <= state.recovery_stage_index < state.current_stage_index
    ):
        raise ValueError("training checkpoint has an invalid recovery stage")
    if state.episodes_trained < 0 or state.optimizer_updates < 0:
        raise ValueError("training checkpoint counters cannot be negative")
    if state.next_evaluation <= 0:
        raise ValueError("training checkpoint next evaluation must be positive")
    if not -1 <= state.last_evaluation_episode <= state.episodes_trained:
        raise ValueError("training checkpoint has an invalid evaluation boundary")
    if (
        not math.isfinite(state.total_elapsed_seconds)
        or state.total_elapsed_seconds < 0
    ):
        raise ValueError("training checkpoint has invalid elapsed time")
    if not state.saved_device_type:
        raise ValueError("training checkpoint has an invalid device type")
    if not state.run_id.strip():
        raise ValueError("training checkpoint has an invalid run ID")
    return state


def save_checkpoint(
    model: SharedMLP,
    checkpoint_path: str | Path,
    metadata: Mapping[str, Any],
    metadata_path: str | Path | None = None,
) -> tuple[Path, Path]:
    """Save a model-only state dictionary and human-readable JSON metadata."""

    _require_torch()
    if not isinstance(model, SharedMLP):
        raise ValueError("model must be a SharedMLP")
    if not isinstance(metadata, Mapping):
        raise ValueError("metadata must be a mapping")
    checkpoint = Path(checkpoint_path)
    if not checkpoint.name:
        raise ValueError("checkpoint_path must name a file")
    metadata_file = (
        Path(metadata_path)
        if metadata_path is not None
        else checkpoint.with_suffix(".json")
    )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    payload = {
        "encoding_schema_version": ENCODING_SCHEMA_VERSION,
        "action_map_version": ACTION_MAP_VERSION,
        "action_mask_version": ACTION_MASK_VERSION,
        "objective_schema_version": OBJECTIVE_SCHEMA_VERSION,
        **dict(metadata),
    }
    metadata_file.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return checkpoint, metadata_file


def load_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str = "cpu",
) -> SharedMLP:
    """Construct the fixed architecture and load a saved state dictionary."""

    _require_torch()
    checkpoint = Path(checkpoint_path)
    model = SharedMLP()
    try:
        target_device = torch.device(device)
        state_dictionary = _load_torch_payload(checkpoint, target_device)
        if (
            isinstance(state_dictionary, Mapping)
            and state_dictionary.get("checkpoint_type")
            == TRAINING_CHECKPOINT_TYPE
        ):
            state_dictionary = state_dictionary["model_state_dict"]
        model.load_state_dict(state_dictionary)
        model.to(target_device)
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"could not load checkpoint {str(checkpoint)!r}") from exc
    return model


def publish_checkpoint_controller(
    checkpoint_path: str | Path,
    *,
    controllers_directory: str | Path = "controllers",
    controller_name: str | None = None,
    metadata_path: str | Path | None = None,
    run_id: str | None = None,
) -> Any:
    """Publish a model-only best checkpoint into the saved-controller catalog."""

    _require_torch()
    checkpoint = Path(checkpoint_path)
    if is_training_checkpoint(checkpoint):
        raise ValueError(
            "a resumable training bundle cannot be published directly; publish "
            "the model-only best checkpoint"
        )
    # Validate the fixed architecture before committing catalog files.
    load_checkpoint(checkpoint, device="cpu")
    metadata_file = (
        Path(metadata_path)
        if metadata_path is not None
        else checkpoint.with_suffix(".json")
    )
    try:
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"could not load checkpoint metadata {str(metadata_file)!r}"
        ) from exc
    if not isinstance(metadata, Mapping):
        raise ValueError("checkpoint metadata must be a JSON object")
    try:
        curriculum = tuple(dict(stage) for stage in metadata["curriculum"])
        validations = tuple(dict(summary) for summary in metadata["validation"])
        training_config = dict(metadata["hyperparameters"])
        cost_weights = dict(metadata["cost_weights"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "checkpoint metadata lacks curriculum, validation, hyperparameters, "
            "or cost weights required for publication"
        ) from exc
    if not validations:
        raise ValueError("checkpoint metadata has no held-out validation results")
    promotion_rate = float(training_config.get("promotion_success_rate", 0.8))
    qualified_stage = 0
    for summary in validations:
        if float(summary["success_rate"]) < promotion_rate:
            break
        qualified_stage += 1
    evaluated_stage = int(validations[-1]["stage"]["number"])
    score_summary = (
        validations[qualified_stage - 1] if qualified_stage else validations[-1]
    )
    from .controller_catalog import publish_saved_controller

    return publish_saved_controller(
        checkpoint,
        directory=controllers_directory,
        run_id=(run_id or str(metadata.get("run_id") or uuid.uuid4().hex)),
        controller_name=controller_name,
        qualified_stage=qualified_stage,
        evaluated_stage=evaluated_stage,
        success_rate=float(score_summary["success_rate"]),
        mean_cost=float(score_summary["mean_cost"]),
        episodes_trained=int(metadata.get("episodes_trained", 0)),
        optimizer_updates=int(metadata.get("optimizer_updates", 0)),
        promotion_success_rate=promotion_rate,
        use_action_masks=bool(training_config.get("use_action_masks", True)),
        torch_version=str(torch.__version__),
        curriculum=curriculum,
        cost_weights=cost_weights,
        training_config=training_config,
        objective_schema_version=str(
            metadata.get("objective_schema_version", "item-distance-v1")
        ),
    )


def _regressed_stage_index(
    summaries: Sequence[EvaluationSummary],
    current_stage_index: int,
    retention_success_rate: float,
) -> int | None:
    """Return the earliest prior stage below the retention threshold."""

    return next(
        (
            index
            for index, summary in enumerate(summaries[:current_stage_index])
            if summary.success_rate < retention_success_rate
        ),
        None,
    )


def _sample_training_stage_index(
    rng: random.Random,
    current_stage_index: int,
    recovery_stage_index: int | None,
    previous_stage_probability: float,
) -> int:
    """Select the recovery, replay, or current curriculum stage."""

    if recovery_stage_index is not None:
        return recovery_stage_index
    if (
        current_stage_index > 0
        and rng.random() < previous_stage_probability
    ):
        return rng.randrange(current_stage_index)
    return current_stage_index


def train_curriculum(
    environment_factory: EnvironmentFactory,
    *,
    model: SharedMLP | None = None,
    config: ReinforceConfig | None = None,
    curriculum: Sequence[CurriculumStage] = DEFAULT_CURRICULUM,
    checkpoint_path: str | Path = "artifacts/shared_mlp_best.pt",
    latest_checkpoint_path: str | Path = "artifacts/shared_mlp_latest.pt",
    resume_checkpoint_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    cost_weights: Any = None,
    device: str = "cpu",
    progress: ProgressCallback | None = print,
) -> TrainingResult:
    """Train, evaluate, promote, and checkpoint the progressive curriculum."""

    _require_torch()
    settings = config or ReinforceConfig()
    stages = tuple(curriculum)
    if not stages or not all(isinstance(stage, CurriculumStage) for stage in stages):
        raise ValueError("curriculum must contain CurriculumStage values")
    if tuple(stage.number for stage in stages) != tuple(range(1, len(stages) + 1)):
        raise ValueError("curriculum stage numbers must be contiguous starting at one")
    if progress is not None and not callable(progress):
        raise ValueError("progress must be callable or None")
    if resume_checkpoint_path is not None and model is not None:
        raise ValueError("model and resume_checkpoint_path are mutually exclusive")

    torch.manual_seed(settings.training_seed)
    rng = random.Random(settings.training_seed)
    trainer = ReinforceTrainer(model, settings, device=device)
    selected_model = trainer.model
    started_at = time.monotonic()
    deadline = started_at + settings.time_budget_seconds
    current_stage_index = 0
    highest_completed_stage = 0
    episodes_trained = 0
    optimizer_updates = 0
    next_evaluation = settings.evaluation_interval
    last_evaluation_episode = -1
    latest_evaluations: tuple[EvaluationSummary, ...] = ()
    best_score: tuple[int, float, float] | None = None
    recovery_stage_index: int | None = None
    elapsed_before = 0.0
    run_id = uuid.uuid4().hex
    checkpoint = Path(checkpoint_path)
    latest_checkpoint = Path(latest_checkpoint_path)
    metadata_file = (
        Path(metadata_path)
        if metadata_path is not None
        else checkpoint.with_suffix(".json")
    )

    def emit(message: str) -> None:
        if progress is not None:
            progress(message)

    if resume_checkpoint_path is not None:
        resume_state = load_training_checkpoint(
            resume_checkpoint_path,
            config=settings,
            curriculum=stages,
            cost_weights=cost_weights,
            device=device,
        )
        try:
            trainer.model.load_state_dict(resume_state.model_state_dict)
            trainer.optimizer.load_state_dict(resume_state.optimizer_state_dict)
            for optimizer_state in trainer.optimizer.state.values():
                for key, value in optimizer_state.items():
                    if torch.is_tensor(value):
                        optimizer_state[key] = value.to(trainer.device)
            rng.setstate(resume_state.python_rng_state)
            torch.set_rng_state(resume_state.torch_rng_state.cpu())
            if trainer.device.type == "cuda" and resume_state.cuda_rng_states:
                torch.cuda.set_rng_state_all(resume_state.cuda_rng_states)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError("could not restore training checkpoint state") from exc
        current_stage_index = resume_state.current_stage_index
        highest_completed_stage = resume_state.highest_completed_stage
        episodes_trained = resume_state.episodes_trained
        optimizer_updates = resume_state.optimizer_updates
        next_evaluation = resume_state.next_evaluation
        last_evaluation_episode = resume_state.last_evaluation_episode
        recovery_stage_index = resume_state.recovery_stage_index
        best_score = resume_state.best_score
        elapsed_before = resume_state.total_elapsed_seconds
        run_id = resume_state.run_id
        if resume_state.saved_device_type != trainer.device.type:
            emit(
                "warning: resumed on a different device type; numerical "
                "continuation may not be bit-for-bit identical"
            )
        emit(
            f"resumed {episodes_trained} episodes and {optimizer_updates} "
            f"updates from {resume_checkpoint_path}"
        )

    def save_latest() -> None:
        cuda_rng_states = (
            tuple(torch.cuda.get_rng_state_all())
            if torch.cuda.is_available()
            else ()
        )
        save_training_checkpoint(
            latest_checkpoint,
            TrainingCheckpointState(
                model_state_dict=selected_model.state_dict(),
                optimizer_state_dict=trainer.optimizer.state_dict(),
                python_rng_state=rng.getstate(),
                torch_rng_state=torch.get_rng_state(),
                cuda_rng_states=cuda_rng_states,
                episodes_trained=episodes_trained,
                optimizer_updates=optimizer_updates,
                current_stage_index=current_stage_index,
                highest_completed_stage=highest_completed_stage,
                next_evaluation=next_evaluation,
                last_evaluation_episode=last_evaluation_episode,
                recovery_stage_index=recovery_stage_index,
                best_score=best_score,
                total_elapsed_seconds=(
                    elapsed_before + time.monotonic() - started_at
                ),
                saved_device_type=trainer.device.type,
                run_id=run_id,
            ),
            config=settings,
            curriculum=stages,
            cost_weights=cost_weights,
        )

    def evaluate_and_checkpoint() -> bool:
        nonlocal current_stage_index, highest_completed_stage
        nonlocal best_score, latest_evaluations, last_evaluation_episode
        nonlocal recovery_stage_index
        summaries = tuple(
            evaluate_policy(
                environment_factory,
                selected_model,
                stage,
                validation_seeds(
                    stage,
                    settings.evaluation_episodes,
                    settings.validation_seed_base,
                ),
                use_action_masks=settings.use_action_masks,
            )
            for stage in stages[: current_stage_index + 1]
        )
        latest_evaluations = summaries
        last_evaluation_episode = episodes_trained
        regressed_stage_index = _regressed_stage_index(
            summaries,
            current_stage_index,
            settings.retention_success_rate,
        )
        previous_recovery_stage = recovery_stage_index
        recovery_stage_index = regressed_stage_index
        if recovery_stage_index is not None:
            if previous_recovery_stage != recovery_stage_index:
                emit(
                    f"retention recovery: training curriculum stage "
                    f"{recovery_stage_index + 1} exclusively until it returns "
                    f"to {settings.retention_success_rate:.0%} success"
                )
        elif previous_recovery_stage is not None:
            emit("retention recovered; returning to the current curriculum stage")

        previous_stages_retained = all(
            summary.success_rate >= settings.retention_success_rate
            for summary in summaries[:current_stage_index]
        )
        passed_through_current = (
            previous_stages_retained
            and summaries[current_stage_index].success_rate
            >= settings.promotion_success_rate
        )
        if passed_through_current:
            highest_completed_stage = max(
                highest_completed_stage, current_stage_index + 1
            )

        completed_prefix = 0
        for summary in summaries:
            if summary.success_rate < settings.promotion_success_rate:
                break
            completed_prefix += 1
        minimum_success = min(summary.success_rate for summary in summaries)
        mean_cost = math.fsum(summary.mean_cost for summary in summaries) / len(
            summaries
        )
        # Rank a checkpoint by the stages this exact model currently satisfies,
        # not by a historical promotion reached by an earlier set of weights.
        # This prevents later catastrophic forgetting from replacing a valid
        # earlier-stage checkpoint.
        score = (completed_prefix, minimum_success, -mean_cost)
        if best_score is None or score > best_score:
            best_score = score
            save_checkpoint(
                selected_model,
                checkpoint,
                {
                    "cost_weights": cost_weights,
                    "curriculum": stages,
                    "episodes_trained": episodes_trained,
                    "optimizer_updates": optimizer_updates,
                    "highest_completed_stage": highest_completed_stage,
                    "run_id": run_id,
                    "hyperparameters": settings,
                    "validation": [summary.to_dict() for summary in summaries],
                },
                metadata_file,
            )
        emit(
            f"evaluation after {episodes_trained} episodes: "
            f"stage {current_stage_index + 1} success "
            f"{summaries[-1].success_rate:.1%}, mean cost "
            f"{summaries[-1].mean_cost:.4f}"
        )
        if passed_through_current and current_stage_index + 1 < len(stages):
            current_stage_index += 1
            emit(f"promoted to curriculum stage {current_stage_index + 1}")
        return highest_completed_stage == len(stages)

    pending_rollouts: list[EpisodeRollout] = []
    if resume_checkpoint_path is None:
        if evaluate_and_checkpoint():
            save_latest()
            elapsed = time.monotonic() - started_at
            return TrainingResult(
                episodes_trained=episodes_trained,
                optimizer_updates=optimizer_updates,
                highest_completed_stage=highest_completed_stage,
                elapsed_seconds=elapsed,
                total_elapsed_seconds=elapsed_before + elapsed,
                checkpoint_path=checkpoint,
                latest_checkpoint_path=latest_checkpoint,
                metadata_path=metadata_file,
                evaluations=latest_evaluations,
                run_id=run_id,
            )
        save_latest()

    try:
        while (
            time.monotonic() < deadline
            and highest_completed_stage < len(stages)
        ):
            if episodes_trained >= next_evaluation:
                if evaluate_and_checkpoint():
                    break
                while next_evaluation <= episodes_trained:
                    next_evaluation += settings.evaluation_interval
                if time.monotonic() >= deadline:
                    break

            sampled_stage = stages[
                _sample_training_stage_index(
                    rng,
                    current_stage_index,
                    recovery_stage_index,
                    settings.previous_stage_probability,
                )
            ]
            # Training and validation seed ranges are disjoint by construction.
            training_seed = rng.randrange(-(2**63), 0)
            environment = environment_factory(sampled_stage, training_seed)
            selected_model.train()
            pending_rollouts.append(
                collect_episode(
                    environment,
                    selected_model,
                    sampled_stage,
                    training_seed,
                    stochastic=True,
                    use_action_masks=settings.use_action_masks,
                )
            )
            episodes_trained += 1

            update_due = len(pending_rollouts) >= settings.episodes_per_update
            budget_expired = time.monotonic() >= deadline
            if update_due or budget_expired:
                optimization = trainer.update(pending_rollouts)
                optimizer_updates += 1
                pending_rollouts.clear()
                save_latest()
                emit(
                    f"update {optimizer_updates}: loss {optimization.loss:.5f}, "
                    f"{optimization.transitions} transitions"
                )
    except KeyboardInterrupt:
        emit(
            "training interrupted; the latest checkpoint remains at the last "
            "completed optimizer boundary"
        )
        raise

    if pending_rollouts:
        trainer.update(pending_rollouts)
        optimizer_updates += 1
        pending_rollouts.clear()
        last_evaluation_episode = -1
        save_latest()
    # Always produce a final held-out report and checkpoint candidate, including
    # short time-budget smoke runs that never reached the normal interval.
    if not latest_evaluations or last_evaluation_episode != episodes_trained:
        evaluate_and_checkpoint()
    save_latest()

    elapsed = time.monotonic() - started_at
    return TrainingResult(
        episodes_trained=episodes_trained,
        optimizer_updates=optimizer_updates,
        highest_completed_stage=highest_completed_stage,
        elapsed_seconds=elapsed,
        total_elapsed_seconds=elapsed_before + elapsed,
        checkpoint_path=checkpoint,
        latest_checkpoint_path=latest_checkpoint,
        metadata_path=metadata_file,
        evaluations=latest_evaluations,
        run_id=run_id,
    )


def _jsonable(value: Any) -> Any:
    """Recursively convert public metadata and metrics to JSON-safe values."""

    if is_dataclass(value) and not isinstance(value, type):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "__dict__"):
        return {
            str(key): _jsonable(item)
            for key, item in vars(value).items()
            if not str(key).startswith("_")
        }
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


__all__ = [
    "ACTION_MAP_VERSION",
    "ACTION_MASK_VERSION",
    "DEFAULT_CURRICULUM",
    "ENCODING_SCHEMA_VERSION",
    "HIDDEN_FEATURES",
    "INPUT_FEATURES",
    "NUM_ACTIONS",
    "TRAINING_CHECKPOINT_TYPE",
    "TRAINING_CHECKPOINT_VERSION",
    "LEGACY_TRAINING_CHECKPOINT_VERSIONS",
    "CurriculumStage",
    "EpisodeEvaluation",
    "EpisodeRollout",
    "EvaluationSummary",
    "MLPRobotController",
    "OneHotCommand",
    "OptimizationMetrics",
    "PolicyDecision",
    "ReinforceConfig",
    "ReinforceTrainer",
    "SharedMLP",
    "TrainingResult",
    "TrainingCheckpointState",
    "collect_episode",
    "create_shared_controllers",
    "discounted_returns",
    "evaluate_policy",
    "load_checkpoint",
    "load_training_checkpoint",
    "publish_checkpoint_controller",
    "save_checkpoint",
    "save_training_checkpoint",
    "is_training_checkpoint",
    "torch_available",
    "train_curriculum",
    "validation_seeds",
]
