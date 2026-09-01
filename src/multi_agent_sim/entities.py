"""Entity definitions shared by the simulation world."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import TypeAlias

Position: TypeAlias = tuple[int, int]


def validate_position_shape(position: Position) -> None:
    """Validate the shape and coordinate types of a grid position.

    Bounds are deliberately checked by :class:`SimulationWorld`, because an
    entity does not know which world it belongs to.
    """

    if (
        not isinstance(position, tuple)
        or len(position) != 2
        or any(type(coordinate) is not int for coordinate in position)
    ):
        raise ValueError("position must be a tuple of exactly two integers")


def _validate_entity_id(entity_id: str, field_name: str) -> None:
    if not isinstance(entity_id, str) or not entity_id.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


class Entity(ABC):
    """Base abstraction for anything that can be registered in a world.

    Positions are read-only to callers once constructed. The world uses the
    package-private methods below to keep its occupancy index synchronized.
    """

    __slots__ = ("_position", "_world_owner")

    def __init__(self, position: Position) -> None:
        validate_position_shape(position)
        self._position = position
        self._world_owner: object | None = None

    @property
    @abstractmethod
    def entity_id(self) -> str:
        """Return the ID used by the world's global entity registry."""

    @property
    def position(self) -> Position:
        return self._position

    def _bind_to_world(self, world: object) -> None:
        if self._world_owner is not None and self._world_owner is not world:
            raise ValueError("entity is already registered in another world")
        self._world_owner = world

    def _unbind_from_world(self, world: object) -> None:
        if self._world_owner is world:
            self._world_owner = None

    def _set_position(self, world: object, position: Position) -> None:
        if self._world_owner is not world:
            raise RuntimeError("only the owning world may change entity position")
        validate_position_shape(position)
        self._position = position


@dataclass(slots=True, init=False, eq=False)
class Robot(Entity):
    """A robot agent with a stable ID and an initial battery value."""

    robot_id: str
    _battery_level: float
    _carried_item_id: str | None

    def __init__(
        self,
        robot_id: str,
        position: Position,
        battery_level: float = 100.0,
    ) -> None:
        _validate_entity_id(robot_id, "robot_id")
        if (
            isinstance(battery_level, bool)
            or not isinstance(battery_level, (int, float))
            or not math.isfinite(battery_level)
        ):
            raise ValueError("battery_level must be a finite number")
        if battery_level < 0:
            raise ValueError("battery_level cannot be negative")
        Entity.__init__(self, position)
        self.robot_id = robot_id
        self._battery_level = float(battery_level)
        self._carried_item_id = None

    @property
    def entity_id(self) -> str:
        return self.robot_id

    @property
    def battery_level(self) -> float:
        return self._battery_level

    @property
    def carried_item_id(self) -> str | None:
        return self._carried_item_id

    def _spend_battery(self, world: object, amount: float) -> None:
        if self._world_owner is not world:
            raise RuntimeError("only the owning world may spend robot battery")
        if amount < 0 or amount > self._battery_level:
            raise ValueError("battery spend must be nonnegative and affordable")
        self._battery_level = max(0.0, self._battery_level - amount)

    def _store_item(self, world: object, item_id: str) -> None:
        if self._world_owner is not world:
            raise RuntimeError("only the owning world may change robot inventory")
        if self._carried_item_id is not None:
            raise RuntimeError("robot inventory is already full")
        _validate_entity_id(item_id, "item_id")
        self._carried_item_id = item_id

    def _release_item(self, world: object) -> str:
        """Remove and return the carried item under owning-world control."""

        if self._world_owner is not world:
            raise RuntimeError("only the owning world may change robot inventory")
        if self._carried_item_id is None:
            raise RuntimeError("robot inventory is empty")
        item_id = self._carried_item_id
        self._carried_item_id = None
        return item_id

    def __repr__(self) -> str:
        return (
            f"Robot(robot_id={self.robot_id!r}, position={self.position!r}, "
            f"battery_level={self.battery_level!r}, "
            f"carried_item_id={self.carried_item_id!r})"
        )


@dataclass(slots=True, init=False, eq=False)
class Item(Entity):
    """A passive item placed in the simulation world."""

    item_id: str

    def __init__(self, item_id: str, position: Position) -> None:
        _validate_entity_id(item_id, "item_id")
        Entity.__init__(self, position)
        self.item_id = item_id

    @property
    def entity_id(self) -> str:
        return self.item_id

    def __repr__(self) -> str:
        return f"Item(item_id={self.item_id!r}, position={self.position!r})"
