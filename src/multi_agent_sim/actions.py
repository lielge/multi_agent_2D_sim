"""Robot actions and state-transition results."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from types import MappingProxyType
from typing import Final, Mapping

from .entities import Position


class Action(str, Enum):
    """Actions supported by the simulation world."""

    MOVE_UP = "move_up"
    MOVE_DOWN = "move_down"
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"
    PICK_UP = "pick_up"
    WAIT = "wait"
    DROP = "drop"


ACTION_DELTAS: Final[Mapping[Action, Position]] = MappingProxyType(
    {
        Action.MOVE_UP: (0, -1),
        Action.MOVE_DOWN: (0, 1),
        Action.MOVE_LEFT: (-1, 0),
        Action.MOVE_RIGHT: (1, 0),
        Action.PICK_UP: (0, 0),
        Action.WAIT: (0, 0),
        Action.DROP: (0, 0),
    }
)


@dataclass(frozen=True, slots=True)
class ActionBatteryCosts:
    """Battery charged for each category of robot action."""

    movement: float = 1.0
    pickup: float = 1.0
    wait: float = 0.0
    drop: float = 1.0

    def __post_init__(self) -> None:
        for field_name in ("movement", "pickup", "wait", "drop"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{field_name} battery cost must be a finite number")
            if value < 0:
                raise ValueError(f"{field_name} battery cost cannot be negative")
            object.__setattr__(self, field_name, float(value))

    def cost_for(self, action: Action) -> float:
        """Return the configured cost for ``action``."""

        if not isinstance(action, Action):
            raise ValueError("action must be an Action value")
        if action in {
            Action.MOVE_UP,
            Action.MOVE_DOWN,
            Action.MOVE_LEFT,
            Action.MOVE_RIGHT,
        }:
            return self.movement
        if action is Action.PICK_UP:
            return self.pickup
        if action is Action.DROP:
            return self.drop
        return self.wait


class ActionFailureReason(str, Enum):
    """Stable failure categories returned by :meth:`SimulationWorld.step`."""

    OUT_OF_BOUNDS = "out_of_bounds"
    OCCUPIED = "occupied"
    CONFLICT = "conflict"
    INSUFFICIENT_BATTERY = "insufficient_battery"
    NO_ITEM = "no_item"
    INVENTORY_FULL = "inventory_full"
    NO_CARRIED_ITEM = "no_carried_item"


@dataclass(frozen=True, slots=True)
class ActionResult:
    """The outcome of one robot action in a simulation step."""

    action: Action
    success: bool
    start_position: Position
    end_position: Position
    failure_reason: ActionFailureReason | None = None
    battery_before: float = 0.0
    battery_after: float = 0.0
    battery_spent: float = 0.0
    picked_up_item_id: str | None = None
    dropped_item_id: str | None = None

    @property
    def picked_item_id(self) -> str | None:
        """Alias for the ID collected by a successful pickup action."""

        return self.picked_up_item_id
