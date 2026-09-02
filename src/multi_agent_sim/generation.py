"""Deterministic helpers for creating simple simulation scenarios."""

from __future__ import annotations

import random

from .actions import ActionBatteryCosts
from .entities import DeliveryDestination, Item, Position, Robot
from .world import SimulationWorld


def generate_random_world(
    width: int,
    height: int,
    num_robots: int,
    num_items: int,
    seed: int | None = None,
    action_battery_costs: ActionBatteryCosts | None = None,
) -> SimulationWorld:
    """Create a seeded world whose initial entities occupy unique cells."""

    _validate_count(num_robots, "num_robots")
    _validate_count(num_items, "num_items")
    world = SimulationWorld(
        width=width,
        height=height,
        action_battery_costs=action_battery_costs,
    )

    initial_entity_count = num_robots + num_items
    entity_count = num_robots + 2 * num_items
    capacity = width * height
    if entity_count > capacity:
        raise ValueError(
            f"cannot place {entity_count} entities in a world with {capacity} cells"
        )

    rng = random.Random(seed)
    cell_indices = rng.sample(range(capacity), initial_entity_count)
    positions: list[Position] = [
        (cell_index % width, cell_index // width) for cell_index in cell_indices
    ]

    for index in range(num_robots):
        world.add_entity(
            Robot(robot_id=f"robot_{index + 1}", position=positions[index])
        )

    for index in range(num_items):
        world.add_entity(
            Item(
                item_id=f"item_{index + 1}",
                position=positions[num_robots + index],
            )
        )

    occupied_indices = set(cell_indices)
    destination_indices = rng.sample(
        tuple(index for index in range(capacity) if index not in occupied_indices),
        num_items,
    )
    for index, cell_index in enumerate(destination_indices, start=1):
        world.add_entity(
            DeliveryDestination(
                destination_id=f"destination_{index}",
                position=(cell_index % width, cell_index // width),
                target_item_id=f"item_{index}",
            )
        )

    return world


def _validate_count(count: int, name: str) -> None:
    if type(count) is not int or count < 0:
        raise ValueError(f"{name} must be a non-negative integer")
