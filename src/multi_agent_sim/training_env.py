"""Framework-neutral fixed-slot observations and team-action environment.

This module is the boundary between the simulator/episode runner and learning
code.  It intentionally contains no tensor-library dependency: observations,
legal-action masks, commands, rewards, and results are immutable Python data.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Final, TypeAlias

from .actions import ACTION_DELTAS, Action, ActionResult
from .controllers import WorldObservation
from .episode import (
    EpisodeCostWeights,
    EpisodeMetrics,
    EpisodeRunner,
    StepRewardComponents,
)
from .session import InitialScenario
from .world import SimulationWorld


MAX_ROBOT_SLOTS: Final = 4
MAX_ITEM_SLOTS: Final = 8
ROBOT_SLOT_WIDTH: Final = 12
ITEM_SLOT_WIDTH: Final = 12
WORLD_FEATURE_WIDTH: Final = 7
CONTROLLED_ROBOT_IDENTITY_WIDTH: Final = MAX_ROBOT_SLOTS
FEATURE_DIM: Final = (
    MAX_ROBOT_SLOTS * ROBOT_SLOT_WIDTH
    + MAX_ITEM_SLOTS * ITEM_SLOT_WIDTH
    + WORLD_FEATURE_WIDTH
    + CONTROLLED_ROBOT_IDENTITY_WIDTH
)

ENCODING_VERSION: Final = "fixed-slots-4r-8i-v1"
ACTION_MAP_VERSION: Final = "one-hot-7-v1"
ACTION_MASK_VERSION: Final = "protect-delivered-pickup-v1"

ACTION_ORDER: Final[tuple[Action, ...]] = (
    Action.MOVE_UP,
    Action.MOVE_DOWN,
    Action.MOVE_LEFT,
    Action.MOVE_RIGHT,
    Action.PICK_UP,
    Action.WAIT,
    Action.DROP,
)
_ACTION_TO_INDEX: Final = MappingProxyType(
    {action: index for index, action in enumerate(ACTION_ORDER)}
)

OneHotCommand: TypeAlias = tuple[int, int, int, int, int, int, int]
FeatureRow: TypeAlias = tuple[float, ...]
ActionMask: TypeAlias = tuple[bool, bool, bool, bool, bool, bool, bool]
ScenarioFactory: TypeAlias = Callable[[int], InitialScenario]


def validate_one_hot_command(command: Sequence[int]) -> OneHotCommand:
    """Return ``command`` as an immutable, validated seven-way one-hot value."""

    if isinstance(command, (str, bytes)):
        raise ValueError("command must be a sequence of seven binary integers")
    try:
        normalized = tuple(command)
    except TypeError as exc:
        raise ValueError(
            "command must be a sequence of seven binary integers"
        ) from exc
    if len(normalized) != len(ACTION_ORDER):
        raise ValueError("command must contain exactly seven values")
    if any(type(value) is not int or value not in (0, 1) for value in normalized):
        raise ValueError("command values must be binary integers (0 or 1)")
    if sum(normalized) != 1:
        raise ValueError("command must contain exactly one selected action")
    return normalized  # type: ignore[return-value]


def action_index(action: Action) -> int:
    """Return the stable command index for an :class:`~.actions.Action`."""

    if not isinstance(action, Action):
        raise ValueError("action must be an Action value")
    return _ACTION_TO_INDEX[action]


def action_from_index(index: int) -> Action:
    """Return the action at a stable command index."""

    if type(index) is not int or not 0 <= index < len(ACTION_ORDER):
        raise ValueError("action index must be an integer from 0 through 6")
    return ACTION_ORDER[index]


def action_to_one_hot(action: Action) -> OneHotCommand:
    """Encode one simulator action as a length-seven one-hot command."""

    index = action_index(action)
    return tuple(
        1 if candidate == index else 0 for candidate in range(len(ACTION_ORDER))
    )  # type: ignore[return-value]


def one_hot_to_action(command: Sequence[int]) -> Action:
    """Validate and decode a one-hot command into a simulator action."""

    return ACTION_ORDER[validate_one_hot_command(command).index(1)]


# Symmetric spellings are convenient when command conversion is used directly.
action_to_index = action_index
index_to_action = action_from_index


@dataclass(frozen=True, slots=True)
class EncodedTeamObservation:
    """Four fixed robot rows plus their identities and advisory masks."""

    rows: tuple[FeatureRow, ...]
    robot_ids: tuple[str | None, ...]
    robot_presence: tuple[int, ...]
    action_masks: tuple[ActionMask, ...]

    def __post_init__(self) -> None:
        try:
            rows = tuple(tuple(row) for row in self.rows)
            robot_ids = tuple(self.robot_ids)
            robot_presence = tuple(self.robot_presence)
            masks = tuple(tuple(mask) for mask in self.action_masks)
        except TypeError as exc:
            raise ValueError("encoded observation fields must be iterable") from exc

        if len(rows) != MAX_ROBOT_SLOTS:
            raise ValueError("rows must contain exactly four feature rows")
        for row in rows:
            if len(row) != FEATURE_DIM:
                raise ValueError(f"every feature row must contain {FEATURE_DIM} values")
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in row
            ):
                raise ValueError("feature rows must contain only finite numbers")

        if len(robot_ids) != MAX_ROBOT_SLOTS:
            raise ValueError("robot_ids must contain exactly four values")
        active_ids = tuple(robot_id for robot_id in robot_ids if robot_id is not None)
        if any(
            not isinstance(robot_id, str) or not robot_id.strip()
            for robot_id in active_ids
        ):
            raise ValueError("active robot IDs must be non-empty strings")
        if len(set(active_ids)) != len(active_ids):
            raise ValueError("active robot IDs must be unique")

        if (
            len(robot_presence) != MAX_ROBOT_SLOTS
            or any(type(value) is not int or value not in (0, 1) for value in robot_presence)
        ):
            raise ValueError("robot_presence must contain four binary integers")
        expected_presence = tuple(
            1 if robot_id is not None else 0 for robot_id in robot_ids
        )
        if robot_presence != expected_presence:
            raise ValueError("robot_presence must match robot_ids")
        seen_padding = False
        for robot_id in robot_ids:
            if robot_id is None:
                seen_padding = True
            elif seen_padding:
                raise ValueError("active robot slots must precede padded slots")

        if len(masks) != MAX_ROBOT_SLOTS:
            raise ValueError("action_masks must contain exactly four masks")
        for mask in masks:
            if len(mask) != len(ACTION_ORDER) or any(
                type(value) is not bool for value in mask
            ):
                raise ValueError("every action mask must contain seven booleans")

        zero_row = (0.0,) * FEATURE_DIM
        zero_mask = (False,) * len(ACTION_ORDER)
        for index, present in enumerate(robot_presence):
            if not present and rows[index] != zero_row:
                raise ValueError("padded feature rows must be zero-filled")
            if not present and masks[index] != zero_mask:
                raise ValueError("padded action masks must be false-filled")

        object.__setattr__(
            self,
            "rows",
            tuple(tuple(float(value) for value in row) for row in rows),
        )
        object.__setattr__(self, "robot_ids", robot_ids)
        object.__setattr__(self, "robot_presence", robot_presence)
        object.__setattr__(self, "action_masks", masks)

    @property
    def active_robot_ids(self) -> tuple[str, ...]:
        """Return active IDs in their stable slot order."""

        return tuple(
            robot_id for robot_id in self.robot_ids if robot_id is not None
        )

    def row_for(self, robot_id: str) -> FeatureRow:
        """Return the robot-specific encoded row for ``robot_id``."""

        try:
            index = self.robot_ids.index(robot_id)
        except ValueError:
            raise KeyError(f"unknown robot ID: {robot_id!r}") from None
        return self.rows[index]

    def action_mask_for(self, robot_id: str) -> ActionMask:
        """Return the advisory legal-action mask for ``robot_id``."""

        try:
            index = self.robot_ids.index(robot_id)
        except ValueError:
            raise KeyError(f"unknown robot ID: {robot_id!r}") from None
        return self.action_masks[index]


@dataclass(frozen=True, slots=True)
class TrainingTransition:
    """One environment transition expressed without tensor dependencies."""

    observation: EncodedTeamObservation
    reward: float
    item_progress_reward: float
    delivery_reward: float
    reward_components: StepRewardComponents
    terminated: bool
    action_results: Mapping[str, ActionResult]
    metrics: EpisodeMetrics

    def __post_init__(self) -> None:
        if not isinstance(self.observation, EncodedTeamObservation):
            raise ValueError("observation must be an EncodedTeamObservation")
        if (
            isinstance(self.reward, bool)
            or not isinstance(self.reward, (int, float))
            or not math.isfinite(self.reward)
        ):
            raise ValueError("reward must be a finite number")
        if type(self.terminated) is not bool:
            raise ValueError("terminated must be a boolean")
        for name in ("item_progress_reward", "delivery_reward"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
            object.__setattr__(self, name, float(value))
        if not isinstance(self.reward_components, StepRewardComponents):
            raise ValueError("reward_components must be a StepRewardComponents value")
        if not isinstance(self.action_results, Mapping):
            raise ValueError("action_results must be a mapping")
        results = dict(self.action_results)
        if any(
            not isinstance(robot_id, str) or not isinstance(result, ActionResult)
            for robot_id, result in results.items()
        ):
            raise ValueError("action_results must map robot IDs to ActionResult values")
        if not isinstance(self.metrics, EpisodeMetrics):
            raise ValueError("metrics must be an EpisodeMetrics value")
        if self.terminated != self.metrics.terminated:
            raise ValueError("terminated must match metrics.terminated")
        object.__setattr__(self, "reward", float(self.reward))
        object.__setattr__(self, "action_results", MappingProxyType(results))

    @property
    def next_observation(self) -> EncodedTeamObservation:
        """Alias emphasizing that this is the post-action observation."""

        return self.observation


class FixedSlotEncoder:
    """Encode one episode's observations using stable four/eight slot rosters."""

    def __init__(
        self,
        initial_observation: WorldObservation,
        max_steps: int,
    ) -> None:
        if not isinstance(initial_observation, WorldObservation):
            raise ValueError("initial_observation must be a WorldObservation")
        if type(max_steps) is not int or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")

        robot_ids = tuple(robot.robot_id for robot in initial_observation.robots)
        if not 1 <= len(robot_ids) <= MAX_ROBOT_SLOTS:
            raise ValueError("fixed-slot encoding requires between one and four robots")

        destinations_by_item = {
            destination.target_item_id: destination
            for destination in initial_observation.destinations
        }
        item_ids = tuple(sorted(destinations_by_item))
        if len(item_ids) > MAX_ITEM_SLOTS:
            raise ValueError("fixed-slot encoding supports at most eight items")

        self._initial_observation = initial_observation
        self._max_steps = max_steps
        self._robot_ids = robot_ids
        self._item_ids = item_ids
        self._destinations_by_item = destinations_by_item
        self._initial_timestep = initial_observation.timestep
        self._energy_max = sum(
            robot.battery_level for robot in initial_observation.robots
        )
        self._validate_observation(initial_observation, initial=True)

    @property
    def robot_ids(self) -> tuple[str, ...]:
        return self._robot_ids

    @property
    def item_ids(self) -> tuple[str, ...]:
        return self._item_ids

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def energy_max(self) -> float:
        return self._energy_max

    def encode(self, observation: WorldObservation) -> EncodedTeamObservation:
        """Encode a compatible observation into four robot-specific rows."""

        self._validate_observation(observation)
        robots_by_id = {robot.robot_id: robot for robot in observation.robots}
        grid_items_by_id = {item.item_id: item for item in observation.items}
        carriers_by_item = {
            robot.carried_item_id: robot
            for robot in observation.robots
            if robot.carried_item_id is not None
        }

        coordinate_scale_x = _coordinate_scale(observation.width)
        coordinate_scale_y = _coordinate_scale(observation.height)
        energy_scale = max(self._energy_max, 1.0)

        shared_features: list[float] = []
        for slot_index in range(MAX_ROBOT_SLOTS):
            if slot_index >= len(self._robot_ids):
                shared_features.extend((0.0,) * ROBOT_SLOT_WIDTH)
                continue
            robot = robots_by_id[self._robot_ids[slot_index]]
            carried_item = [0.0] * MAX_ITEM_SLOTS
            if robot.carried_item_id is not None:
                carried_item[self._item_ids.index(robot.carried_item_id)] = 1.0
            shared_features.extend(
                (
                    1.0,
                    robot.position[0] * coordinate_scale_x,
                    robot.position[1] * coordinate_scale_y,
                    robot.battery_level / energy_scale,
                    *carried_item,
                )
            )

        for slot_index in range(MAX_ITEM_SLOTS):
            if slot_index >= len(self._item_ids):
                shared_features.extend((0.0,) * ITEM_SLOT_WIDTH)
                continue

            item_id = self._item_ids[slot_index]
            destination = self._destinations_by_item[item_id]
            carrier = carriers_by_item.get(item_id)
            if carrier is not None:
                position = carrier.position
                state = (0.0, 0.0, 1.0)
                carrier_identity = [0.0] * MAX_ROBOT_SLOTS
                carrier_identity[self._robot_ids.index(carrier.robot_id)] = 1.0
            else:
                position = grid_items_by_id[item_id].position
                state = (
                    (0.0, 1.0, 0.0)
                    if position == destination.position
                    else (1.0, 0.0, 0.0)
                )
                carrier_identity = [0.0] * MAX_ROBOT_SLOTS
            shared_features.extend(
                (
                    1.0,
                    position[0] * coordinate_scale_x,
                    position[1] * coordinate_scale_y,
                    destination.position[0] * coordinate_scale_x,
                    destination.position[1] * coordinate_scale_y,
                    *state,
                    *carrier_identity,
                )
            )

        elapsed_steps = observation.timestep - self._initial_timestep
        costs = observation.action_battery_costs
        shared_features.extend(
            (
                elapsed_steps / self._max_steps,
                coordinate_scale_x,
                coordinate_scale_y,
                costs.movement / energy_scale,
                costs.pickup / energy_scale,
                costs.wait / energy_scale,
                costs.drop / energy_scale,
            )
        )
        if len(shared_features) != FEATURE_DIM - CONTROLLED_ROBOT_IDENTITY_WIDTH:
            raise RuntimeError("fixed-slot feature layout has an inconsistent width")

        rows: list[FeatureRow] = []
        masks: list[ActionMask] = []
        padded_robot_ids: list[str | None] = []
        presence: list[int] = []
        for slot_index in range(MAX_ROBOT_SLOTS):
            if slot_index >= len(self._robot_ids):
                rows.append((0.0,) * FEATURE_DIM)
                masks.append((False,) * len(ACTION_ORDER))  # type: ignore[arg-type]
                padded_robot_ids.append(None)
                presence.append(0)
                continue
            identity = tuple(
                1.0 if index == slot_index else 0.0
                for index in range(MAX_ROBOT_SLOTS)
            )
            rows.append(tuple((*shared_features, *identity)))
            robot_id = self._robot_ids[slot_index]
            masks.append(_legal_action_mask(observation, robot_id))
            padded_robot_ids.append(robot_id)
            presence.append(1)

        return EncodedTeamObservation(
            rows=tuple(rows),
            robot_ids=tuple(padded_robot_ids),
            robot_presence=tuple(presence),
            action_masks=tuple(masks),
        )

    def _validate_observation(
        self,
        observation: WorldObservation,
        *,
        initial: bool = False,
    ) -> None:
        if not isinstance(observation, WorldObservation):
            raise ValueError("observation must be a WorldObservation")
        expected = self._initial_observation
        if observation.width != expected.width or observation.height != expected.height:
            raise ValueError("world dimensions changed during the episode")
        if observation.action_battery_costs != expected.action_battery_costs:
            raise ValueError("action battery costs changed during the episode")
        if observation.timestep < self._initial_timestep:
            raise ValueError("observation precedes the episode's initial timestep")
        if observation.timestep - self._initial_timestep > self._max_steps:
            raise ValueError("observation exceeds the episode's maximum timestep")

        robot_ids = tuple(robot.robot_id for robot in observation.robots)
        if robot_ids != self._robot_ids:
            raise ValueError("robot roster changed during the episode")
        destination_roster = tuple(
            (
                destination.destination_id,
                destination.target_item_id,
                destination.position,
            )
            for destination in observation.destinations
        )
        expected_destinations = tuple(
            (
                destination.destination_id,
                destination.target_item_id,
                destination.position,
            )
            for destination in expected.destinations
        )
        if destination_roster != expected_destinations:
            raise ValueError("destination roster changed during the episode")

        grid_item_ids = tuple(item.item_id for item in observation.items)
        carried_item_ids = tuple(
            robot.carried_item_id
            for robot in observation.robots
            if robot.carried_item_id is not None
        )
        if len(set(carried_item_ids)) != len(carried_item_ids):
            raise ValueError("an item cannot be carried by more than one robot")
        located_item_ids = (*grid_item_ids, *carried_item_ids)
        if len(set(located_item_ids)) != len(located_item_ids):
            raise ValueError("an item cannot be both on-grid and carried")
        if set(located_item_ids) != set(self._item_ids):
            message = (
                "every item must have exactly one associated destination"
                if initial
                else "item roster changed during the episode"
            )
            raise ValueError(message)


class FixedSlotTeamEnv:
    """Run scored episodes using per-robot one-hot action commands."""

    def __init__(
        self,
        scenario_factory: ScenarioFactory,
        max_steps: int,
        cost_weights: EpisodeCostWeights | None = None,
    ) -> None:
        if not callable(scenario_factory):
            raise ValueError("scenario_factory must be callable")
        if type(max_steps) is not int or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        if cost_weights is not None and not isinstance(
            cost_weights, EpisodeCostWeights
        ):
            raise ValueError("cost_weights must be an EpisodeCostWeights value")
        self._scenario_factory = scenario_factory
        self._max_steps = max_steps
        self._cost_weights = cost_weights or EpisodeCostWeights()
        self._runner: EpisodeRunner | None = None
        self._encoder: FixedSlotEncoder | None = None
        self._observation: EncodedTeamObservation | None = None

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def cost_weights(self) -> EpisodeCostWeights:
        return self._cost_weights

    @property
    def observation(self) -> EncodedTeamObservation:
        if self._observation is None:
            raise RuntimeError("environment must be reset before use")
        return self._observation

    @property
    def raw_observation(self) -> WorldObservation:
        if self._runner is None:
            raise RuntimeError("environment must be reset before use")
        return self._runner.observation

    @property
    def world(self) -> SimulationWorld:
        """Expose the authoritative world for read-only rendering."""

        if self._runner is None:
            raise RuntimeError("environment must be reset before use")
        return self._runner.world

    @property
    def metrics(self) -> EpisodeMetrics:
        if self._runner is None:
            raise RuntimeError("environment must be reset before use")
        return self._runner.metrics

    @property
    def terminated(self) -> bool:
        return self._runner.terminated if self._runner is not None else False

    def reset(self, seed: int) -> EncodedTeamObservation:
        """Create a fresh seeded scenario and return its encoded observation."""

        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        scenario = self._scenario_factory(seed)
        if not isinstance(scenario, InitialScenario):
            raise ValueError("scenario_factory must return an InitialScenario")
        runner = EpisodeRunner(
            scenario,
            max_steps=self._max_steps,
            cost_weights=self._cost_weights,
        )
        encoder = FixedSlotEncoder(runner.observation, self._max_steps)
        observation = encoder.encode(runner.observation)
        self._runner = runner
        self._encoder = encoder
        self._observation = observation
        return observation

    def step(
        self,
        commands: Mapping[str, Sequence[int]],
    ) -> TrainingTransition:
        """Decode one complete team command batch and advance one world step."""

        if self._runner is None or self._encoder is None:
            raise RuntimeError("environment must be reset before use")
        if self._runner.terminated:
            raise RuntimeError("cannot step a terminated episode")
        if not isinstance(commands, Mapping):
            raise ValueError("commands must be a mapping of robot IDs to one-hot values")

        expected_ids = self._encoder.robot_ids
        supplied_ids = set(commands)
        missing_ids = set(expected_ids).difference(supplied_ids)
        unknown_ids = supplied_ids.difference(expected_ids)
        if missing_ids or unknown_ids:
            details: list[str] = []
            if missing_ids:
                details.append("missing robot(s): " + ", ".join(sorted(missing_ids)))
            if unknown_ids:
                details.append(
                    "unknown robot(s): "
                    + ", ".join(sorted(str(value) for value in unknown_ids))
                )
            raise ValueError(
                "commands must cover exactly the active robots ("
                + "; ".join(details)
                + ")"
            )

        actions = {
            robot_id: one_hot_to_action(commands[robot_id])
            for robot_id in expected_ids
        }
        episode_transition = self._runner.step(actions)
        observation = self._encoder.encode(episode_transition.observation)
        self._observation = observation
        return TrainingTransition(
            observation=observation,
            reward=episode_transition.reward,
            item_progress_reward=episode_transition.item_progress_reward,
            delivery_reward=episode_transition.delivery_reward,
            reward_components=episode_transition.reward_components,
            terminated=episode_transition.terminated,
            action_results=episode_transition.action_results,
            metrics=episode_transition.metrics,
        )


# A descriptive alias for callers that prefer the longer name.
FixedSlotTrainingEnvironment = FixedSlotTeamEnv


def _coordinate_scale(size: int) -> float:
    return 0.0 if size == 1 else 1.0 / (size - 1)


def _legal_action_mask(
    observation: WorldObservation,
    robot_id: str,
) -> ActionMask:
    robot = observation.get_robot(robot_id)
    costs = observation.action_battery_costs
    robot_positions = {
        candidate.position
        for candidate in observation.robots
        if candidate.robot_id != robot_id
    }
    items_by_id = {item.item_id: item for item in observation.items}
    items_by_position = {item.position: item for item in observation.items}
    delivered_item_ids = {
        destination.target_item_id
        for destination in observation.destinations
        if destination.target_item_id in items_by_id
        and items_by_id[destination.target_item_id].position
        == destination.position
    }
    item_positions = set(items_by_position)
    mask: list[bool] = []

    for action in ACTION_ORDER:
        affordable = robot.battery_level >= costs.cost_for(action)
        legal = affordable
        if action in {
            Action.MOVE_UP,
            Action.MOVE_DOWN,
            Action.MOVE_LEFT,
            Action.MOVE_RIGHT,
        }:
            dx, dy = ACTION_DELTAS[action]
            target = (robot.position[0] + dx, robot.position[1] + dy)
            legal = affordable and (
                0 <= target[0] < observation.width
                and 0 <= target[1] < observation.height
                and target not in robot_positions
            )
        elif action is Action.PICK_UP:
            colocated_item = items_by_position.get(robot.position)
            legal = (
                affordable
                and robot.carried_item_id is None
                and colocated_item is not None
                and colocated_item.item_id not in delivered_item_ids
            )
        elif action is Action.DROP:
            legal = (
                affordable
                and robot.carried_item_id is not None
                and robot.position not in item_positions
            )
        mask.append(legal)

    if not any(mask):
        mask[action_index(Action.WAIT)] = True
    return tuple(mask)  # type: ignore[return-value]


__all__ = [
    "ACTION_MAP_VERSION",
    "ACTION_MASK_VERSION",
    "ACTION_ORDER",
    "CONTROLLED_ROBOT_IDENTITY_WIDTH",
    "ENCODING_VERSION",
    "FEATURE_DIM",
    "ITEM_SLOT_WIDTH",
    "MAX_ITEM_SLOTS",
    "MAX_ROBOT_SLOTS",
    "ROBOT_SLOT_WIDTH",
    "WORLD_FEATURE_WIDTH",
    "ActionMask",
    "EncodedTeamObservation",
    "FeatureRow",
    "FixedSlotEncoder",
    "FixedSlotTeamEnv",
    "FixedSlotTrainingEnvironment",
    "OneHotCommand",
    "ScenarioFactory",
    "TrainingTransition",
    "action_from_index",
    "action_index",
    "action_to_index",
    "action_to_one_hot",
    "index_to_action",
    "one_hot_to_action",
    "validate_one_hot_command",
]
