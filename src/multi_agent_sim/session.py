"""Immutable scenario specifications and reversible simulation sessions.

This module deliberately has no visualization dependency.  A UI can create an
initial scenario, construct a :class:`SimulationSession`, and render the
session's current ``world`` without needing to know how rewind is implemented.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
import random
from types import MappingProxyType

from .actions import Action, ActionBatteryCosts, ActionResult
from .controllers import (
    ControllerFactoryContext,
    ControllerRegistry,
    RobotController,
    WorldObservation,
    create_default_controller_registry,
)
from .entities import (
    DeliveryDestination,
    Item,
    Position,
    Robot,
    validate_position_shape,
)
from .world import SimulationWorld


def _validate_positive_integer(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_non_negative_integer(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_seed(seed: int) -> None:
    if type(seed) is not int:
        raise ValueError("seed must be an integer")


@dataclass(frozen=True, slots=True)
class RobotConfiguration:
    """The user-configurable part of one robot's initial state.

    Robot IDs are intentionally omitted: tuple order determines the automatic
    IDs ``robot_1``, ``robot_2``, and so on.
    """

    position: Position
    battery_level: float = 100.0
    controller_key: str = "random"

    def __post_init__(self) -> None:
        validate_position_shape(self.position)
        battery_level = self.battery_level
        if (
            isinstance(battery_level, bool)
            or not isinstance(battery_level, (int, float))
            or not math.isfinite(battery_level)
        ):
            raise ValueError("battery_level must be a finite number")
        if battery_level < 0:
            raise ValueError("battery_level cannot be negative")
        if not isinstance(self.controller_key, str) or not self.controller_key.strip():
            raise ValueError("controller_key must be a non-empty string")
        object.__setattr__(self, "battery_level", float(battery_level))
        object.__setattr__(self, "controller_key", self.controller_key.strip())


@dataclass(frozen=True, slots=True)
class InitialScenario:
    """A validated, immutable description of a world's initial state."""

    width: int
    height: int
    robot_configurations: tuple[RobotConfiguration, ...]
    item_positions: tuple[Position, ...]
    seed: int
    action_battery_costs: ActionBatteryCosts = field(
        default_factory=ActionBatteryCosts
    )
    delivery_destination_positions: tuple[Position, ...] = ()

    def __post_init__(self) -> None:
        _validate_positive_integer(self.width, "width")
        _validate_positive_integer(self.height, "height")
        _validate_seed(self.seed)
        if not isinstance(self.action_battery_costs, ActionBatteryCosts):
            raise ValueError(
                "action_battery_costs must be an ActionBatteryCosts instance"
            )

        try:
            robot_configurations = tuple(self.robot_configurations)
        except TypeError as exc:
            raise ValueError(
                "robot_configurations must be a sequence of RobotConfiguration values"
            ) from exc
        try:
            item_positions = tuple(self.item_positions)
        except TypeError as exc:
            raise ValueError("item_positions must be a sequence of positions") from exc
        try:
            delivery_destination_positions = tuple(
                self.delivery_destination_positions
            )
        except TypeError as exc:
            raise ValueError(
                "delivery_destination_positions must be a sequence of positions"
            ) from exc

        if not all(
            isinstance(configuration, RobotConfiguration)
            for configuration in robot_configurations
        ):
            raise ValueError(
                "robot_configurations must contain only RobotConfiguration values"
            )
        for position in item_positions:
            validate_position_shape(position)
        for position in delivery_destination_positions:
            validate_position_shape(position)

        if len(delivery_destination_positions) != len(item_positions):
            raise ValueError(
                "delivery_destination_positions must contain exactly one "
                "position for each item"
            )

        object.__setattr__(self, "robot_configurations", robot_configurations)
        object.__setattr__(self, "item_positions", item_positions)
        object.__setattr__(
            self,
            "delivery_destination_positions",
            delivery_destination_positions,
        )

        capacity = self.width * self.height
        entity_count = (
            len(robot_configurations)
            + len(item_positions)
            + len(delivery_destination_positions)
        )
        if entity_count > capacity:
            raise ValueError(
                f"cannot place {entity_count} entities in a world with {capacity} cells"
            )

        robot_positions = tuple(
            configuration.position for configuration in robot_configurations
        )
        if len(set(robot_positions)) != len(robot_positions):
            raise ValueError("robot positions must be unique")
        if len(set(item_positions)) != len(item_positions):
            raise ValueError("item positions must be unique")
        if len(set(delivery_destination_positions)) != len(
            delivery_destination_positions
        ):
            raise ValueError("delivery destination positions must be unique")

        for position in (
            *robot_positions,
            *item_positions,
            *delivery_destination_positions,
        ):
            if not self._is_in_bounds(position):
                raise ValueError(
                    f"position {position!r} is outside world bounds "
                    f"0 <= x < {self.width}, 0 <= y < {self.height}"
                )

        all_positions = (
            *robot_positions,
            *item_positions,
            *delivery_destination_positions,
        )
        if len(set(all_positions)) != len(all_positions):
            raise ValueError(
                "initial robot, item, and delivery destination positions "
                "must not overlap"
            )

    @property
    def num_robots(self) -> int:
        return len(self.robot_configurations)

    @property
    def num_items(self) -> int:
        return len(self.item_positions)

    @property
    def num_destinations(self) -> int:
        return len(self.delivery_destination_positions)

    def create_world(self) -> SimulationWorld:
        """Build a fresh world at timestep zero from this specification."""

        world = SimulationWorld(
            self.width,
            self.height,
            action_battery_costs=self.action_battery_costs,
        )
        for index, configuration in enumerate(self.robot_configurations, start=1):
            world.add_entity(
                Robot(
                    robot_id=f"robot_{index}",
                    position=configuration.position,
                    battery_level=configuration.battery_level,
                )
            )
        for index, position in enumerate(self.item_positions, start=1):
            world.add_entity(Item(item_id=f"item_{index}", position=position))
        for index, position in enumerate(
            self.delivery_destination_positions,
            start=1,
        ):
            world.add_entity(
                DeliveryDestination(
                    destination_id=f"destination_{index}",
                    position=position,
                    target_item_id=f"item_{index}",
                )
            )
        return world

    def _is_in_bounds(self, position: Position) -> bool:
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height


def create_initial_scenario(
    width: int,
    height: int,
    num_robots: int,
    num_items: int,
    seed: int,
    robot_configurations: Sequence[RobotConfiguration] | None = None,
    *,
    action_battery_costs: ActionBatteryCosts | None = None,
    controller_keys: Sequence[str] | None = None,
) -> InitialScenario:
    """Create a reproducible random or manually configured initial scenario.

    When ``robot_configurations`` is ``None``, robot positions are sampled
    along with item positions and every robot starts with battery level 100.
    When configurations are supplied, their positions and batteries are kept
    exactly and only the item positions are sampled. In both cases, one
    destination is sampled for every item after robot and item placement.
    """

    _validate_positive_integer(width, "width")
    _validate_positive_integer(height, "height")
    _validate_non_negative_integer(num_robots, "num_robots")
    _validate_non_negative_integer(num_items, "num_items")
    _validate_seed(seed)
    costs = action_battery_costs or ActionBatteryCosts()
    if not isinstance(costs, ActionBatteryCosts):
        raise ValueError(
            "action_battery_costs must be an ActionBatteryCosts instance"
        )

    if controller_keys is None:
        normalized_controller_keys: tuple[str, ...] | None = None
    else:
        try:
            normalized_controller_keys = tuple(controller_keys)
        except TypeError as exc:
            raise ValueError("controller_keys must be a sequence of strings") from exc
        if len(normalized_controller_keys) != num_robots:
            raise ValueError(
                f"controller_keys must contain exactly {num_robots} values"
            )
        if any(
            not isinstance(key, str) or not key.strip()
            for key in normalized_controller_keys
        ):
            raise ValueError("controller_keys must contain non-empty strings")
        normalized_controller_keys = tuple(
            key.strip() for key in normalized_controller_keys
        )

    capacity = width * height
    entity_count = num_robots + 2 * num_items
    if entity_count > capacity:
        raise ValueError(
            f"cannot place {entity_count} entities in a world with {capacity} cells"
        )

    rng = random.Random(seed)
    if robot_configurations is None:
        initial_entity_count = num_robots + num_items
        sampled_indices = rng.sample(range(capacity), initial_entity_count)
        sampled_positions = tuple(
            (cell_index % width, cell_index // width)
            for cell_index in sampled_indices
        )
        keys = normalized_controller_keys or ("random",) * num_robots
        configurations = tuple(
            RobotConfiguration(position=position, controller_key=keys[index])
            for index, position in enumerate(sampled_positions[:num_robots])
        )
        item_positions = sampled_positions[num_robots:]
    else:
        try:
            configurations = tuple(robot_configurations)
        except TypeError as exc:
            raise ValueError(
                "robot_configurations must be a sequence of RobotConfiguration values"
            ) from exc
        if len(configurations) != num_robots:
            raise ValueError(
                "robot_configurations must contain exactly "
                f"{num_robots} configurations"
            )
        if not all(
            isinstance(configuration, RobotConfiguration)
            for configuration in configurations
        ):
            raise ValueError(
                "robot_configurations must contain only RobotConfiguration values"
            )
        if normalized_controller_keys is not None:
            configurations = tuple(
                RobotConfiguration(
                    position=configuration.position,
                    battery_level=configuration.battery_level,
                    controller_key=normalized_controller_keys[index],
                )
                for index, configuration in enumerate(configurations)
            )

        # Validate manual positions before sampling so malformed input produces
        # a useful validation error rather than an incidental sampling error.
        manual_scenario = InitialScenario(
            width=width,
            height=height,
            robot_configurations=configurations,
            item_positions=(),
            seed=seed,
            action_battery_costs=costs,
        )
        occupied_indices = {
            configuration.position[1] * width + configuration.position[0]
            for configuration in manual_scenario.robot_configurations
        }
        available_indices = tuple(
            index for index in range(capacity) if index not in occupied_indices
        )
        sampled_indices = rng.sample(available_indices, num_items)
        item_positions = tuple(
            (cell_index % width, cell_index // width)
            for cell_index in sampled_indices
        )

    occupied_indices = {
        configuration.position[1] * width + configuration.position[0]
        for configuration in configurations
    }
    occupied_indices.update(
        position[1] * width + position[0] for position in item_positions
    )
    destination_indices = rng.sample(
        tuple(index for index in range(capacity) if index not in occupied_indices),
        num_items,
    )
    delivery_destination_positions = tuple(
        (cell_index % width, cell_index // width)
        for cell_index in destination_indices
    )

    return InitialScenario(
        width=width,
        height=height,
        robot_configurations=configurations,
        item_positions=item_positions,
        seed=seed,
        action_battery_costs=costs,
        delivery_destination_positions=delivery_destination_positions,
    )


class SimulationSession:
    """Own playback state and exact action history for one scenario."""

    MIN_STEP_RATE = 1
    MAX_STEP_RATE = 30

    def __init__(
        self,
        scenario: InitialScenario,
        max_steps: int = 200,
        step_rate: int = 5,
        *,
        controller_registry: ControllerRegistry | None = None,
        controller_overrides: Mapping[str, RobotController] | None = None,
    ) -> None:
        if not isinstance(scenario, InitialScenario):
            raise ValueError("scenario must be an InitialScenario")
        _validate_non_negative_integer(max_steps, "max_steps")

        self._scenario = scenario
        self._max_steps = max_steps
        self._step_rate = self.MIN_STEP_RATE
        self.step_rate = step_rate
        self._world = scenario.create_world()
        self._action_history: list[dict[str, Action]] = []
        self._controller_error_history: list[dict[str, str]] = []
        self._controller_errors: dict[str, str] = {}
        self._cursor = 0
        self._playing = False

        if controller_registry is None:
            controller_registry = create_default_controller_registry()
        if not isinstance(controller_registry, ControllerRegistry):
            raise ValueError("controller_registry must be a ControllerRegistry")
        self._controller_registry = controller_registry

        robot_ids = tuple(
            f"robot_{index}"
            for index in range(1, len(scenario.robot_configurations) + 1)
        )
        self._controller_keys = {
            robot_id: configuration.controller_key
            for robot_id, configuration in zip(
                robot_ids, scenario.robot_configurations, strict=True
            )
        }

        if controller_overrides is None:
            overrides: dict[str, RobotController] = {}
        else:
            if not isinstance(controller_overrides, Mapping):
                raise ValueError("controller_overrides must be a mapping")
            overrides = dict(controller_overrides)
        unknown_override_ids = set(overrides).difference(robot_ids)
        if unknown_override_ids:
            unknown = ", ".join(sorted(map(str, unknown_override_ids)))
            raise ValueError(f"controller override refers to unknown robot(s): {unknown}")
        self._overridden_robot_ids = frozenset(overrides)

        self._controllers: dict[str, RobotController] = {}
        for robot_id in robot_ids:
            if robot_id in overrides:
                controller = overrides[robot_id]
            else:
                controller = controller_registry.create(
                    self._controller_keys[robot_id],
                    ControllerFactoryContext(seed=scenario.seed, robot_id=robot_id),
                )
            if not callable(getattr(controller, "choose_action", None)):
                raise ValueError(
                    f"controller for {robot_id!r} must define choose_action()"
                )
            self._controllers[robot_id] = controller

    @property
    def scenario(self) -> InitialScenario:
        return self._scenario

    @property
    def world(self) -> SimulationWorld:
        return self._world

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def step_rate(self) -> int:
        return self._step_rate

    @step_rate.setter
    def step_rate(self, value: int) -> None:
        if (
            type(value) is not int
            or not self.MIN_STEP_RATE <= value <= self.MAX_STEP_RATE
        ):
            raise ValueError(
                f"step_rate must be an integer from {self.MIN_STEP_RATE} "
                f"to {self.MAX_STEP_RATE}"
            )
        self._step_rate = value

    @property
    def playing(self) -> bool:
        return self._playing

    @property
    def cursor(self) -> int:
        return self._cursor

    @property
    def current_step(self) -> int:
        """Alias for ``cursor`` suitable for status displays."""

        return self._cursor

    @property
    def history_length(self) -> int:
        return len(self._action_history)

    @property
    def action_history(self) -> tuple[Mapping[str, Action], ...]:
        """Return immutable snapshots of all generated action batches."""

        return tuple(
            MappingProxyType(dict(actions)) for actions in self._action_history
        )

    @property
    def controller_registry(self) -> ControllerRegistry:
        """The registry used to construct this session's controllers."""

        return self._controller_registry

    def controller_key_for(self, robot_id: str) -> str:
        """Return the scenario's stable configured key for ``robot_id``."""

        try:
            return self._controller_keys[robot_id]
        except KeyError as exc:
            raise KeyError(f"unknown robot ID: {robot_id!r}") from exc

    def controller_display_name_for(self, robot_id: str) -> str:
        """Describe the controller instance actually used for ``robot_id``."""

        controller_key = self.controller_key_for(robot_id)
        if robot_id in self._overridden_robot_ids:
            controller_type = type(self._controllers[robot_id]).__name__
            return f"{controller_type} (override)"
        return self._controller_registry.display_name(controller_key)

    @property
    def controller_errors(self) -> Mapping[str, str]:
        """Warnings produced by the action batch at the current cursor."""

        return MappingProxyType(dict(self._controller_errors))

    @property
    def latest_controller_errors(self) -> Mapping[str, str]:
        """Compatibility alias for :attr:`controller_errors`."""

        return self.controller_errors

    @property
    def controller_error_history(self) -> tuple[Mapping[str, str], ...]:
        """Immutable controller-warning snapshots aligned with action history."""

        return tuple(
            MappingProxyType(dict(errors))
            for errors in self._controller_error_history
        )

    @property
    def can_step_back(self) -> bool:
        return self._cursor > 0

    @property
    def can_step_forward(self) -> bool:
        return self._cursor < self._max_steps

    @property
    def at_history_tip(self) -> bool:
        return self._cursor == len(self._action_history)

    def play(self) -> None:
        """Start automatic playback unless the configured maximum was reached."""

        self._playing = self.can_step_forward

    def pause(self) -> None:
        self._playing = False

    def toggle_playing(self) -> bool:
        """Toggle playback and return the resulting state."""

        if self._playing:
            self.pause()
        else:
            self.play()
        return self._playing

    def step_forward(self) -> dict[str, ActionResult] | None:
        """Advance once, replaying history before generating any new actions."""

        if not self.can_step_forward:
            self.pause()
            return None

        is_new_action_batch = self.at_history_tip
        if is_new_action_batch:
            actions, controller_errors = self._generate_action_batch()
        else:
            actions = self._action_history[self._cursor]
            controller_errors = self._controller_error_history[self._cursor]

        results = self._world.step(actions)
        if is_new_action_batch:
            self._action_history.append(actions)
            self._controller_error_history.append(controller_errors)
        self._cursor += 1
        self._controller_errors = dict(controller_errors)

        if not self.can_step_forward:
            self.pause()
        return results

    def step_back(self) -> bool:
        """Rebuild and replay to one step earlier, returning whether it moved."""

        self.pause()
        if not self.can_step_back:
            return False

        target_cursor = self._cursor - 1
        rebuilt_world = self._scenario.create_world()
        for actions in self._action_history[:target_cursor]:
            rebuilt_world.step(actions)
        self._world = rebuilt_world
        self._cursor = target_cursor
        self._controller_errors = (
            dict(self._controller_error_history[target_cursor - 1])
            if target_cursor > 0
            else {}
        )
        return True

    def _generate_action_batch(self) -> tuple[dict[str, Action], dict[str, str]]:
        observation = WorldObservation.from_world(self._world)
        actions: dict[str, Action] = {}
        errors: dict[str, str] = {}
        for robot in sorted(
            self._world.get_entities(Robot), key=lambda entity: entity.robot_id
        ):
            robot_id = robot.robot_id
            controller = self._controllers[robot_id]
            try:
                action = controller.choose_action(observation, robot_id)
                if not isinstance(action, Action):
                    raise TypeError(
                        "controller returned "
                        f"{type(action).__name__}, expected an Action"
                    )
            except Exception as exc:  # Controllers are intentionally isolated.
                action = Action.WAIT
                message = str(exc).strip()
                errors[robot_id] = (
                    f"{type(exc).__name__}: {message}"
                    if message
                    else type(exc).__name__
                )
            actions[robot_id] = action
        return actions, errors
