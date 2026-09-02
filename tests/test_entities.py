from __future__ import annotations

import math
import unittest

from multi_agent_sim import DeliveryDestination, Item, Robot, SimulationWorld


class EntityTests(unittest.TestCase):
    def test_robot_and_item_expose_common_entity_ids(self) -> None:
        robot = Robot("robot_1", (2, 3), battery_level=75)
        item = Item("item_1", (4, 5))

        self.assertEqual(robot.entity_id, "robot_1")
        self.assertEqual(item.entity_id, "item_1")
        self.assertEqual(robot.battery_level, 75.0)
        self.assertEqual(robot.position, (2, 3))

    def test_entity_validates_id_position_and_battery(self) -> None:
        with self.assertRaises(ValueError):
            Robot("", (0, 0))
        with self.assertRaises(ValueError):
            Item("item_1", [0, 0])  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            Robot("robot_1", (0, 0), battery_level=-1)
        with self.assertRaises(ValueError):
            Robot("robot_1", (0, 0), battery_level=True)
        for invalid_battery in (math.nan, math.inf, -math.inf):
            with self.subTest(battery_level=invalid_battery):
                with self.assertRaises(ValueError):
                    Robot("robot_1", (0, 0), battery_level=invalid_battery)

    def test_delivery_destination_exposes_validated_assignment(self) -> None:
        destination = DeliveryDestination(
            "destination_1",
            (4, 5),
            "item_1",
        )

        self.assertEqual(destination.entity_id, "destination_1")
        self.assertEqual(destination.position, (4, 5))
        self.assertEqual(destination.target_item_id, "item_1")
        self.assertEqual(
            repr(destination),
            "DeliveryDestination(destination_id='destination_1', "
            "position=(4, 5), target_item_id='item_1')",
        )

        for arguments in (
            ("", (0, 0), "item_1"),
            ("destination_1", (0, 0), ""),
            ("item_1", (0, 0), "item_1"),
            ("destination_1", [0, 0], "item_1"),
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    DeliveryDestination(*arguments)  # type: ignore[arg-type]

    def test_position_is_read_only_to_callers(self) -> None:
        robot = Robot("robot_1", (0, 0))

        with self.assertRaises(AttributeError):
            robot.position = (1, 0)  # type: ignore[misc]

    def test_registered_entity_cannot_be_shared_between_worlds(self) -> None:
        robot = Robot("robot_1", (0, 0))
        first = SimulationWorld(2, 2)
        second = SimulationWorld(2, 2)
        first.add_entity(robot)

        with self.assertRaises(ValueError):
            second.add_entity(robot)
        self.assertEqual(second.get_entities(), ())


if __name__ == "__main__":
    unittest.main()
