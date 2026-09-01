from __future__ import annotations

import random
import unittest

from multi_agent_sim import ActionBatteryCosts, Item, Robot, generate_random_world


def snapshot(world: object) -> list[tuple[str, str, tuple[int, int]]]:
    return [
        (type(entity).__name__, entity.entity_id, entity.position)
        for entity in world.get_entities()  # type: ignore[attr-defined]
    ]


class RandomGenerationTests(unittest.TestCase):
    def test_same_seed_produces_same_world(self) -> None:
        first = generate_random_world(8, 6, 4, 7, seed=42)
        second = generate_random_world(8, 6, 4, 7, seed=42)

        self.assertEqual(snapshot(first), snapshot(second))

    def test_generation_uses_stable_ids_and_unique_positions(self) -> None:
        world = generate_random_world(5, 4, 3, 5, seed=7)
        robots = world.get_entities(Robot)
        items = world.get_entities(Item)

        self.assertEqual(
            tuple(robot.robot_id for robot in robots),
            ("robot_1", "robot_2", "robot_3"),
        )
        self.assertEqual(
            tuple(item.item_id for item in items),
            ("item_1", "item_2", "item_3", "item_4", "item_5"),
        )
        positions = [entity.position for entity in world.get_entities()]
        self.assertEqual(len(positions), len(set(positions)))

    def test_generation_rejects_invalid_counts_and_over_capacity(self) -> None:
        with self.assertRaises(ValueError):
            generate_random_world(2, 2, -1, 0)
        with self.assertRaises(ValueError):
            generate_random_world(2, 2, 1, True)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            generate_random_world(2, 2, 3, 2)

    def test_generation_does_not_mutate_global_random_state(self) -> None:
        random.seed(12345)
        expected = random.Random(12345).random()

        generate_random_world(4, 4, 2, 2, seed=99)

        self.assertEqual(random.random(), expected)

    def test_generation_passes_action_costs_to_the_world(self) -> None:
        costs = ActionBatteryCosts(movement=2, pickup=3, wait=0.5)

        world = generate_random_world(
            3,
            3,
            1,
            1,
            seed=2,
            action_battery_costs=costs,
        )

        self.assertIs(world.action_battery_costs, costs)


if __name__ == "__main__":
    unittest.main()
