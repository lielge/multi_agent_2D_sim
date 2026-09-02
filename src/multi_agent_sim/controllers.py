"""Controller-facing observations and pluggable robot policies.

Controllers deliberately operate on immutable snapshots rather than on a
``SimulationWorld`` itself.  This keeps planning, learned policies, and remote
or multi-agent adapters outside the state-transition layer: only the session
asks for actions, and only the world applies them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
import random
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from .actions import Action, ActionBatteryCosts
from .entities import (
    DeliveryDestination,
    Item,
    Position,
    Robot,
    validate_position_shape,
)
from .world import SimulationWorld


def _validate_identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{name} cannot have leading or trailing whitespace")


def _validate_entity_identifier(value: str, name: str) -> None:
    """Match the entity layer's accepted ID contract exactly."""

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class RobotObservation:
    """Read-only controller view of one robot."""

    robot_id: str
    position: Position
    battery_level: float
    carried_item_id: str | None = None

    def __post_init__(self) -> None:
        _validate_entity_identifier(self.robot_id, "robot_id")
        validate_position_shape(self.position)
        if (
            isinstance(self.battery_level, bool)
            or not isinstance(self.battery_level, (int, float))
            or not math.isfinite(self.battery_level)
        ):
            raise ValueError("battery_level must be a finite number")
        if self.battery_level < 0:
            raise ValueError("battery_level cannot be negative")
        if self.carried_item_id is not None:
            _validate_entity_identifier(self.carried_item_id, "carried_item_id")
        object.__setattr__(self, "battery_level", float(self.battery_level))


@dataclass(frozen=True, slots=True)
class ItemObservation:
    """Read-only controller view of one uncollected item."""

    item_id: str
    position: Position

    def __post_init__(self) -> None:
        _validate_entity_identifier(self.item_id, "item_id")
        validate_position_shape(self.position)


@dataclass(frozen=True, slots=True)
class DeliveryDestinationObservation:
    """Read-only controller view of one delivery destination."""

    destination_id: str
    position: Position
    target_item_id: str

    def __post_init__(self) -> None:
        _validate_entity_identifier(self.destination_id, "destination_id")
        _validate_entity_identifier(self.target_item_id, "target_item_id")
        if self.destination_id == self.target_item_id:
            raise ValueError("destination_id and target_item_id must be different")
        validate_position_shape(self.position)


@dataclass(frozen=True, slots=True)
class WorldObservation:
    """An immutable, deterministically ordered snapshot of a world."""

    width: int
    height: int
    timestep: int
    robots: tuple[RobotObservation, ...]
    items: tuple[ItemObservation, ...]
    action_battery_costs: ActionBatteryCosts
    destinations: tuple[DeliveryDestinationObservation, ...] = ()

    def __post_init__(self) -> None:
        if type(self.width) is not int or self.width <= 0:
            raise ValueError("width must be a positive integer")
        if type(self.height) is not int or self.height <= 0:
            raise ValueError("height must be a positive integer")
        if type(self.timestep) is not int or self.timestep < 0:
            raise ValueError("timestep must be a non-negative integer")
        if not isinstance(self.action_battery_costs, ActionBatteryCosts):
            raise ValueError(
                "action_battery_costs must be an ActionBatteryCosts instance"
            )

        try:
            robots = tuple(self.robots)
        except TypeError as exc:
            raise ValueError(
                "robots must be an iterable of RobotObservation values"
            ) from exc
        try:
            items = tuple(self.items)
        except TypeError as exc:
            raise ValueError(
                "items must be an iterable of ItemObservation values"
            ) from exc
        try:
            destinations = tuple(self.destinations)
        except TypeError as exc:
            raise ValueError(
                "destinations must be an iterable of "
                "DeliveryDestinationObservation values"
            ) from exc
        if not all(isinstance(robot, RobotObservation) for robot in robots):
            raise ValueError("robots must contain only RobotObservation values")
        if not all(isinstance(item, ItemObservation) for item in items):
            raise ValueError("items must contain only ItemObservation values")
        if not all(
            isinstance(destination, DeliveryDestinationObservation)
            for destination in destinations
        ):
            raise ValueError(
                "destinations must contain only "
                "DeliveryDestinationObservation values"
            )

        robots = tuple(sorted(robots, key=lambda robot: robot.robot_id))
        items = tuple(sorted(items, key=lambda item: item.item_id))
        destinations = tuple(
            sorted(destinations, key=lambda destination: destination.destination_id)
        )
        robot_ids = tuple(robot.robot_id for robot in robots)
        item_ids = tuple(item.item_id for item in items)
        destination_ids = tuple(
            destination.destination_id for destination in destinations
        )
        if len(set(robot_ids)) != len(robot_ids):
            raise ValueError("robot IDs must be unique")
        if len(set(item_ids)) != len(item_ids):
            raise ValueError("item IDs must be unique")
        if len(set(destination_ids)) != len(destination_ids):
            raise ValueError("destination IDs must be unique")
        target_item_ids = tuple(
            destination.target_item_id for destination in destinations
        )
        if len(set(target_item_ids)) != len(target_item_ids):
            raise ValueError("destination target item IDs must be unique")
        for observation in (*robots, *items, *destinations):
            x, y = observation.position
            if not 0 <= x < self.width or not 0 <= y < self.height:
                raise ValueError(
                    f"position {observation.position!r} is outside observation bounds"
                )

        object.__setattr__(self, "robots", robots)
        object.__setattr__(self, "items", items)
        object.__setattr__(self, "destinations", destinations)

    @classmethod
    def create(
        cls,
        width: int,
        height: int,
        timestep: int,
        robots: Iterable[RobotObservation],
        items: Iterable[ItemObservation],
        action_battery_costs: ActionBatteryCosts,
        destinations: Iterable[DeliveryDestinationObservation] = (),
    ) -> WorldObservation:
        """Create a validated snapshot from arbitrary observation iterables."""

        return cls(
            width=width,
            height=height,
            timestep=timestep,
            robots=tuple(robots),
            items=tuple(items),
            action_battery_costs=action_battery_costs,
            destinations=tuple(destinations),
        )

    @classmethod
    def from_world(cls, world: SimulationWorld) -> WorldObservation:
        """Copy the public state of ``world`` into an immutable snapshot."""

        if not isinstance(world, SimulationWorld):
            raise ValueError("world must be a SimulationWorld")
        return cls.create(
            width=world.width,
            height=world.height,
            timestep=world.timestep,
            robots=(
                RobotObservation(
                    robot_id=robot.robot_id,
                    position=robot.position,
                    battery_level=robot.battery_level,
                    carried_item_id=robot.carried_item_id,
                )
                for robot in world.get_entities(Robot)
            ),
            items=(
                ItemObservation(item_id=item.item_id, position=item.position)
                for item in world.get_entities(Item)
            ),
            destinations=(
                DeliveryDestinationObservation(
                    destination_id=destination.destination_id,
                    position=destination.position,
                    target_item_id=destination.target_item_id,
                )
                for destination in world.get_entities(DeliveryDestination)
            ),
            action_battery_costs=world.action_battery_costs,
        )

    def get_robot(self, robot_id: str) -> RobotObservation:
        """Return a robot observation or raise ``KeyError`` for an unknown ID."""

        for robot in self.robots:
            if robot.robot_id == robot_id:
                return robot
        raise KeyError(f"unknown robot ID: {robot_id!r}")

    @property
    def action_costs(self) -> ActionBatteryCosts:
        """Concise alias for the immutable action battery costs."""

        return self.action_battery_costs

    def get_item(self, item_id: str) -> ItemObservation:
        """Return an item observation or raise ``KeyError`` for an unknown ID."""

        for item in self.items:
            if item.item_id == item_id:
                return item
        raise KeyError(f"unknown item ID: {item_id!r}")

    def get_destination(
        self,
        destination_id: str,
    ) -> DeliveryDestinationObservation:
        """Return a destination observation or raise ``KeyError``."""

        for destination in self.destinations:
            if destination.destination_id == destination_id:
                return destination
        raise KeyError(f"unknown destination ID: {destination_id!r}")


def create_world_observation(world: SimulationWorld) -> WorldObservation:
    """Convenience wrapper for :meth:`WorldObservation.from_world`."""

    return WorldObservation.from_world(world)


@runtime_checkable
class RobotController(Protocol):
    """Minimal interface implemented by all per-robot controllers."""

    def choose_action(
        self,
        observation: WorldObservation,
        robot_id: str,
    ) -> Action:
        """Choose one action without mutating the observation or world."""


@dataclass(frozen=True, slots=True)
class ControllerFactoryContext:
    """Stable inputs supplied when constructing a robot controller."""

    seed: int
    robot_id: str

    def __post_init__(self) -> None:
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        _validate_entity_identifier(self.robot_id, "robot_id")


ControllerFactory = Callable[[ControllerFactoryContext], RobotController]


@dataclass(frozen=True, slots=True)
class ControllerDefinition:
    """One discoverable controller type registered under a stable key."""

    key: str
    display_name: str
    factory: ControllerFactory

    def __post_init__(self) -> None:
        _validate_identifier(self.key, "controller key")
        _validate_identifier(self.display_name, "controller display_name")
        if not callable(self.factory):
            raise ValueError("controller factory must be callable")


class ControllerRegistry:
    """Insertion-ordered registry used by sessions and setup UIs."""

    def __init__(
        self,
        definitions: Iterable[ControllerDefinition] = (),
    ) -> None:
        self._definitions: dict[str, ControllerDefinition] = {}
        try:
            initial_definitions = tuple(definitions)
        except TypeError as exc:
            raise ValueError(
                "definitions must be an iterable of ControllerDefinition values"
            ) from exc
        for definition in initial_definitions:
            self.register(definition)

    @property
    def definitions(self) -> tuple[ControllerDefinition, ...]:
        return tuple(self._definitions.values())

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(self._definitions)

    def register(
        self,
        definition: ControllerDefinition | str,
        display_name: str | None = None,
        factory: ControllerFactory | None = None,
    ) -> ControllerDefinition:
        """Register a definition, accepting either object or field arguments."""

        if isinstance(definition, ControllerDefinition):
            if display_name is not None or factory is not None:
                raise ValueError(
                    "display_name and factory cannot accompany a ControllerDefinition"
                )
            normalized = definition
        elif isinstance(definition, str):
            if display_name is None or factory is None:
                raise ValueError(
                    "display_name and factory are required when registering by key"
                )
            normalized = ControllerDefinition(definition, display_name, factory)
        else:
            raise ValueError("definition must be a ControllerDefinition or key string")

        if normalized.key in self._definitions:
            raise ValueError(f"duplicate controller key: {normalized.key!r}")
        self._definitions[normalized.key] = normalized
        return normalized

    def create(
        self,
        key: str,
        context: ControllerFactoryContext,
    ) -> RobotController:
        """Construct a fresh controller instance for one robot."""

        if not isinstance(context, ControllerFactoryContext):
            raise ValueError("context must be a ControllerFactoryContext")
        definition = self._get_definition(key)
        controller = definition.factory(context)
        if not isinstance(controller, RobotController) or not callable(
            getattr(controller, "choose_action", None)
        ):
            raise ValueError(
                f"factory for controller {key!r} did not return a RobotController"
            )
        return controller

    def display_name(self, key: str) -> str:
        return self._get_definition(key).display_name

    def _get_definition(self, key: str) -> ControllerDefinition:
        try:
            return self._definitions[key]
        except (KeyError, TypeError):
            raise KeyError(f"unknown controller key: {key!r}") from None


class RandomController:
    """A reproducible policy that chooses uniformly from all actions."""

    def __init__(self, seed: int) -> None:
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self._random = random.Random(seed)
        self._actions = tuple(Action)

    def choose_action(
        self,
        observation: WorldObservation,
        robot_id: str,
    ) -> Action:
        observation.get_robot(robot_id)
        return self._random.choice(self._actions)


class NearestItemController:
    """Deterministically approach and pick up the closest available item."""

    def choose_action(
        self,
        observation: WorldObservation,
        robot_id: str,
    ) -> Action:
        robot = observation.get_robot(robot_id)
        if robot.carried_item_id is not None or not observation.items:
            return Action.WAIT

        robot_x, robot_y = robot.position
        target = min(
            observation.items,
            key=lambda item: (
                abs(item.position[0] - robot_x) + abs(item.position[1] - robot_y),
                item.item_id,
            ),
        )
        target_x, target_y = target.position
        if target_x < robot_x:
            desired_action = Action.MOVE_LEFT
        elif target_x > robot_x:
            desired_action = Action.MOVE_RIGHT
        elif target_y < robot_y:
            desired_action = Action.MOVE_UP
        elif target_y > robot_y:
            desired_action = Action.MOVE_DOWN
        else:
            desired_action = Action.PICK_UP

        if robot.battery_level < observation.action_battery_costs.cost_for(
            desired_action
        ):
            return Action.WAIT
        return desired_action


@runtime_checkable
class MultiAgentController(Protocol):
    """Interface for one policy that chooses actions for a robot team."""

    def choose_actions(
        self,
        observation: WorldObservation,
        robot_ids: tuple[str, ...],
    ) -> Mapping[str, Action]:
        """Return exactly one valid action for every requested robot ID."""


class MultiAgentControllerAdapter:
    """Expose a joint team policy through the per-robot controller protocol.

    A session can share one adapter instance as the override for every team
    member.  The joint policy is then evaluated once for a given immutable
    timestep snapshot and its validated results are served from the cache.
    """

    def __init__(
        self,
        controller: MultiAgentController,
        robot_ids: Sequence[str],
    ) -> None:
        if not isinstance(controller, MultiAgentController) or not callable(
            getattr(controller, "choose_actions", None)
        ):
            raise ValueError("controller must provide a choose_actions() method")
        if isinstance(robot_ids, (str, bytes)):
            raise ValueError("robot_ids must be a sequence of robot ID strings")
        try:
            normalized_ids = tuple(robot_ids)
        except TypeError as exc:
            raise ValueError(
                "robot_ids must be a sequence of robot ID strings"
            ) from exc
        if not normalized_ids:
            raise ValueError("robot_ids cannot be empty")
        for robot_id in normalized_ids:
            _validate_entity_identifier(robot_id, "robot_id")
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("robot_ids must be unique")

        self._controller = controller
        self._robot_ids = normalized_ids
        self._cached_observation: WorldObservation | None = None
        self._cached_actions: Mapping[str, Action] | None = None
        self._cached_error: Exception | None = None

    @property
    def robot_ids(self) -> tuple[str, ...]:
        return self._robot_ids

    def choose_action(
        self,
        observation: WorldObservation,
        robot_id: str,
    ) -> Action:
        if robot_id not in self._robot_ids:
            raise ValueError(f"robot {robot_id!r} is not a member of this controller")
        for team_robot_id in self._robot_ids:
            observation.get_robot(team_robot_id)

        if self._cached_observation != observation:
            self._evaluate(observation)
        if self._cached_error is not None:
            raise self._cached_error
        assert self._cached_actions is not None
        return self._cached_actions[robot_id]

    def _evaluate(self, observation: WorldObservation) -> None:
        self._cached_observation = observation
        self._cached_actions = None
        self._cached_error = None
        try:
            selected = self._controller.choose_actions(
                observation,
                self._robot_ids,
            )
            if not isinstance(selected, Mapping):
                raise ValueError("multi-agent controller output must be a mapping")
            selected_actions = dict(selected)
            expected_ids = set(self._robot_ids)
            actual_ids = set(selected_actions)
            if actual_ids != expected_ids:
                missing = sorted(expected_ids - actual_ids)
                extra = sorted(actual_ids - expected_ids)
                raise ValueError(
                    "multi-agent controller output must contain exactly the team "
                    f"robot IDs (missing={missing!r}, extra={extra!r})"
                )
            for selected_robot_id, action in selected_actions.items():
                if not isinstance(action, Action):
                    raise ValueError(
                        "multi-agent controller returned an invalid action for "
                        f"{selected_robot_id!r}: {action!r}"
                    )
            self._cached_actions = MappingProxyType(selected_actions)
        except Exception as exc:
            # Cache failures too, so a shared adapter never repeatedly invokes a
            # failing joint policy for each robot during the same timestep.
            self._cached_error = exc


def _seed_for_context(context: ControllerFactoryContext) -> int:
    """Derive a stable per-robot seed without Python's randomized ``hash``."""

    material = f"{context.seed}\0{context.robot_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def create_default_controller_registry() -> ControllerRegistry:
    """Return a fresh registry containing the built-in controllers."""

    registry = ControllerRegistry()
    registry.register(
        "random",
        "Random",
        lambda context: RandomController(_seed_for_context(context)),
    )
    registry.register(
        "nearest_item",
        "Nearest Item",
        lambda context: NearestItemController(),
    )
    return registry
