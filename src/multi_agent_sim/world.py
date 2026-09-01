"""Central world state and deterministic batched state transitions."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Protocol, TypeVar, runtime_checkable

from .actions import (
    ACTION_DELTAS,
    Action,
    ActionBatteryCosts,
    ActionFailureReason,
    ActionResult,
)
from .entities import Entity, Item, Position, Robot, validate_position_shape

EntityT = TypeVar("EntityT", bound=Entity)


@runtime_checkable
class OccupancyPolicy(Protocol):
    """Decide whether an entity may share a cell with current occupants."""

    def allows(self, entity: Entity, occupants: Sequence[Entity]) -> bool:
        """Return ``True`` when ``entity`` may enter the occupied cell."""


class TypeExclusiveOccupancyPolicy:
    """Allow robot-item overlap while keeping equal entity types exclusive."""

    def allows(self, entity: Entity, occupants: Sequence[Entity]) -> bool:
        if isinstance(entity, Robot):
            return not any(isinstance(occupant, Robot) for occupant in occupants)
        if isinstance(entity, Item):
            return not any(isinstance(occupant, Item) for occupant in occupants)
        return not any(type(occupant) is type(entity) for occupant in occupants)


class SimulationWorld:
    """Own all global simulation state for a rectangular discrete grid."""

    def __init__(
        self,
        width: int,
        height: int,
        occupancy_policy: OccupancyPolicy | None = None,
        action_battery_costs: ActionBatteryCosts | None = None,
    ) -> None:
        if type(width) is not int or width <= 0:
            raise ValueError("width must be a positive integer")
        if type(height) is not int or height <= 0:
            raise ValueError("height must be a positive integer")

        policy = occupancy_policy or TypeExclusiveOccupancyPolicy()
        if not isinstance(policy, OccupancyPolicy):
            raise ValueError("occupancy_policy must provide an allows() method")
        if action_battery_costs is not None and not isinstance(
            action_battery_costs, ActionBatteryCosts
        ):
            raise ValueError("action_battery_costs must be an ActionBatteryCosts value")

        self._width = width
        self._height = height
        self._occupancy_policy = policy
        self._action_battery_costs = action_battery_costs or ActionBatteryCosts()
        self._entities: dict[str, Entity] = {}
        self._occupancy: dict[Position, set[str]] = {}
        self._reserved_item_ids: dict[str, str] = {}
        self._timestep = 0

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def timestep(self) -> int:
        return self._timestep

    @property
    def action_battery_costs(self) -> ActionBatteryCosts:
        return self._action_battery_costs

    def is_valid_position(self, position: Position) -> bool:
        try:
            validate_position_shape(position)
        except ValueError:
            return False
        x, y = position
        return 0 <= x < self.width and 0 <= y < self.height

    def add_entity(self, entity: Entity) -> None:
        if not isinstance(entity, Entity):
            raise ValueError("entity must be an Entity instance")
        if (
            entity.entity_id in self._entities
            or entity.entity_id in self._reserved_item_ids
        ):
            raise ValueError(f"duplicate entity ID: {entity.entity_id!r}")
        carried_item_id = (
            entity.carried_item_id if isinstance(entity, Robot) else None
        )
        if carried_item_id is not None and (
            carried_item_id == entity.entity_id
            or carried_item_id in self._entities
            or carried_item_id in self._reserved_item_ids
        ):
            raise ValueError(f"duplicate entity ID: {carried_item_id!r}")
        self._require_valid_position(entity.position)

        occupants = self._entities_at_unchecked(entity.position)
        if not self._occupancy_policy.allows(entity, occupants):
            raise ValueError(
                f"occupancy policy rejects {entity.entity_id!r} at {entity.position!r}"
            )

        self._register_entity_unchecked(entity)
        if carried_item_id is not None:
            self._reserved_item_ids[carried_item_id] = entity.entity_id

    def remove_entity(self, entity_id: str) -> Entity:
        entity = self.get_entity(entity_id)
        cell = self._occupancy[entity.position]
        cell.remove(entity_id)
        if not cell:
            del self._occupancy[entity.position]
        del self._entities[entity_id]
        entity._unbind_from_world(self)
        if isinstance(entity, Robot) and entity.carried_item_id is not None:
            reserved_for = self._reserved_item_ids.get(entity.carried_item_id)
            if reserved_for != entity.entity_id:
                raise RuntimeError("carried-item ID reservation is inconsistent")
            del self._reserved_item_ids[entity.carried_item_id]
        return entity

    def get_entity(self, entity_id: str) -> Entity:
        try:
            return self._entities[entity_id]
        except KeyError:
            raise KeyError(f"unknown entity ID: {entity_id!r}") from None

    def get_entities(self, entity_type: type[EntityT] | None = None) -> tuple[EntityT, ...] | tuple[Entity, ...]:
        entities = sorted(self._entities.values(), key=lambda entity: entity.entity_id)
        if entity_type is None:
            return tuple(entities)
        if not isinstance(entity_type, type) or not issubclass(entity_type, Entity):
            raise ValueError("entity_type must be an Entity subclass")
        return tuple(entity for entity in entities if isinstance(entity, entity_type))

    def get_entities_at(self, position: Position) -> tuple[Entity, ...]:
        self._require_valid_position(position)
        return self._entities_at_unchecked(position)

    def move_entity(self, entity_id: str, new_position: Position) -> None:
        """Administratively relocate an entity while preserving world invariants."""

        entity = self.get_entity(entity_id)
        self._require_valid_position(new_position)
        if new_position == entity.position:
            return

        occupants = tuple(
            occupant
            for occupant in self._entities_at_unchecked(new_position)
            if occupant.entity_id != entity_id
        )
        if not self._occupancy_policy.allows(entity, occupants):
            raise ValueError(
                f"occupancy policy rejects {entity.entity_id!r} at {new_position!r}"
            )
        self._relocate_entity(entity, new_position)

    def step(
        self,
        actions: Mapping[str, Action] | None = None,
    ) -> dict[str, ActionResult]:
        """Apply one snapshot-resolved action per robot and advance simulation time."""

        supplied_actions = self._validate_actions(actions)
        robots = self.get_entities(Robot)
        selected_actions = {
            robot.entity_id: supplied_actions.get(robot.entity_id, Action.WAIT)
            for robot in robots
        }
        starts = {robot.entity_id: robot.position for robot in robots}
        batteries_before = {
            robot.entity_id: robot.battery_level for robot in robots
        }
        action_costs = {
            robot.entity_id: self.action_battery_costs.cost_for(
                selected_actions[robot.entity_id]
            )
            for robot in robots
        }
        targets = {
            robot.entity_id: self._target_for(robot.position, selected_actions[robot.entity_id])
            for robot in robots
        }

        failures: dict[str, ActionFailureReason] = {}
        moving_targets: list[Position] = []

        for robot in robots:
            robot_id = robot.entity_id
            action = selected_actions[robot_id]
            target = targets[robot_id]
            if robot.battery_level < action_costs[robot_id]:
                failures[robot_id] = ActionFailureReason.INSUFFICIENT_BATTERY
                continue
            if action in {Action.WAIT, Action.PICK_UP, Action.DROP}:
                continue
            if not self.is_valid_position(target):
                failures[robot_id] = ActionFailureReason.OUT_OF_BOUNDS
                continue

            occupants = tuple(
                occupant
                for occupant in self._entities_at_unchecked(target)
                if occupant.entity_id != robot_id
            )
            if not self._occupancy_policy.allows(robot, occupants):
                failures[robot_id] = ActionFailureReason.OCCUPIED
                continue
            moving_targets.append(target)

        target_counts = Counter(moving_targets)
        for robot in robots:
            robot_id = robot.entity_id
            if robot_id in failures or selected_actions[robot_id] in {
                Action.WAIT,
                Action.PICK_UP,
                Action.DROP,
            }:
                continue
            if target_counts[targets[robot_id]] > 1:
                failures[robot_id] = ActionFailureReason.CONFLICT

        results: dict[str, ActionResult] = {}
        for robot in robots:
            robot_id = robot.entity_id
            action = selected_actions[robot_id]
            start = starts[robot_id]
            failure = failures.get(robot_id)
            picked_up_item_id: str | None = None
            dropped_item_id: str | None = None

            if failure is None and action is Action.PICK_UP:
                items = self._items_at(robot.position)
                if not items:
                    failure = ActionFailureReason.NO_ITEM
                elif robot.carried_item_id is not None:
                    failure = ActionFailureReason.INVENTORY_FULL
                else:
                    item = items[0]
                    picked_up_item_id = item.item_id
                    self.remove_entity(item.item_id)
                    robot._store_item(self, item.item_id)
                    self._reserved_item_ids[item.item_id] = robot.entity_id
            elif failure is None and action is Action.DROP:
                carried_item_id = robot.carried_item_id
                if carried_item_id is None:
                    failure = ActionFailureReason.NO_CARRIED_ITEM
                else:
                    item = Item(carried_item_id, robot.position)
                    occupants = self._entities_at_unchecked(robot.position)
                    if any(isinstance(occupant, Item) for occupant in occupants):
                        failure = ActionFailureReason.OCCUPIED
                    elif not self._occupancy_policy.allows(item, occupants):
                        failure = ActionFailureReason.OCCUPIED
                    else:
                        reserved_for = self._reserved_item_ids.get(carried_item_id)
                        if (
                            reserved_for != robot.entity_id
                            or carried_item_id in self._entities
                        ):
                            raise RuntimeError(
                                "carried-item ID reservation is inconsistent"
                            )
                        self._register_entity_unchecked(item)
                        released_item_id = robot._release_item(self)
                        if released_item_id != carried_item_id:
                            raise RuntimeError("robot released an unexpected item")
                        del self._reserved_item_ids[carried_item_id]
                        dropped_item_id = carried_item_id
            elif failure is None and action is not Action.WAIT:
                self._relocate_entity(robot, targets[robot_id])

            battery_spent = 0.0
            if failure is not ActionFailureReason.INSUFFICIENT_BATTERY:
                battery_spent = action_costs[robot_id]
                robot._spend_battery(self, battery_spent)
            results[robot_id] = ActionResult(
                action=action,
                success=failure is None,
                start_position=start,
                end_position=robot.position,
                failure_reason=failure,
                battery_before=batteries_before[robot_id],
                battery_after=robot.battery_level,
                battery_spent=battery_spent,
                picked_up_item_id=picked_up_item_id,
                dropped_item_id=dropped_item_id,
            )

        self._timestep += 1
        return results

    def _validate_actions(
        self,
        actions: Mapping[str, Action] | None,
    ) -> dict[str, Action]:
        if actions is None:
            return {}
        if not isinstance(actions, Mapping):
            raise ValueError("actions must be a mapping of robot IDs to Action values")

        validated: dict[str, Action] = {}
        for entity_id, action in actions.items():
            entity = self.get_entity(entity_id)
            if not isinstance(entity, Robot):
                raise ValueError(f"entity {entity_id!r} is not a robot")
            if not isinstance(action, Action):
                raise ValueError(f"invalid action for {entity_id!r}: {action!r}")
            validated[entity_id] = action
        return validated

    def _require_valid_position(self, position: Position) -> None:
        if not self.is_valid_position(position):
            raise ValueError(
                f"position {position!r} is outside world bounds "
                f"0 <= x < {self.width}, 0 <= y < {self.height}"
            )

    def _entities_at_unchecked(self, position: Position) -> tuple[Entity, ...]:
        entity_ids = self._occupancy.get(position, set())
        return tuple(self._entities[entity_id] for entity_id in sorted(entity_ids))

    def _items_at(self, position: Position) -> tuple[Item, ...]:
        return tuple(
            entity
            for entity in self._entities_at_unchecked(position)
            if isinstance(entity, Item)
        )

    def _relocate_entity(self, entity: Entity, new_position: Position) -> None:
        old_position = entity.position
        old_cell = self._occupancy[old_position]
        old_cell.remove(entity.entity_id)
        if not old_cell:
            del self._occupancy[old_position]
        entity._set_position(self, new_position)
        self._occupancy.setdefault(new_position, set()).add(entity.entity_id)

    def _register_entity_unchecked(self, entity: Entity) -> None:
        """Register an entity after ID, bounds, and occupancy preflight checks."""

        entity._bind_to_world(self)
        self._entities[entity.entity_id] = entity
        self._occupancy.setdefault(entity.position, set()).add(entity.entity_id)

    @staticmethod
    def _target_for(position: Position, action: Action) -> Position:
        dx, dy = ACTION_DELTAS[action]
        return position[0] + dx, position[1] + dy
