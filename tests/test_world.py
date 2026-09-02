from __future__ import annotations

import unittest

from multi_agent_sim import (
    Action,
    ActionFailureReason,
    DeliveryDestination,
    Item,
    Robot,
    SimulationWorld,
)


class AllowAllOccupancy:
    def allows(self, entity: object, occupants: object) -> bool:
        return True


class WorldConstructionTests(unittest.TestCase):
    def test_world_dimensions_and_initial_state(self) -> None:
        world = SimulationWorld(width=5, height=4)

        self.assertEqual((world.width, world.height), (5, 4))
        self.assertEqual(world.timestep, 0)
        self.assertEqual(world.get_entities(), ())

    def test_dimensions_must_be_positive_integers(self) -> None:
        invalid_dimensions = [(0, 2), (-1, 2), (2, 0), (True, 2), (2.5, 2)]
        for width, height in invalid_dimensions:
            with self.subTest(width=width, height=height):
                with self.assertRaises(ValueError):
                    SimulationWorld(width, height)  # type: ignore[arg-type]

    def test_position_validation(self) -> None:
        world = SimulationWorld(3, 2)

        self.assertTrue(world.is_valid_position((0, 0)))
        self.assertTrue(world.is_valid_position((2, 1)))
        self.assertFalse(world.is_valid_position((-1, 0)))
        self.assertFalse(world.is_valid_position((3, 1)))
        self.assertFalse(world.is_valid_position([0, 0]))  # type: ignore[arg-type]

        with self.assertRaises(ValueError):
            world.get_entities_at((3, 1))

    def test_occupancy_policy_must_provide_allows(self) -> None:
        with self.assertRaises(ValueError):
            SimulationWorld(2, 2, occupancy_policy=object())  # type: ignore[arg-type]


class EntityRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.world = SimulationWorld(4, 4)

    def test_add_get_query_remove_and_readd(self) -> None:
        robot = Robot("robot_1", (1, 1))
        item = Item("item_1", (1, 1))
        self.world.add_entity(robot)
        self.world.add_entity(item)

        self.assertIs(self.world.get_entity("robot_1"), robot)
        self.assertEqual(
            tuple(entity.entity_id for entity in self.world.get_entities_at((1, 1))),
            ("item_1", "robot_1"),
        )
        self.assertEqual(self.world.get_entities(Robot), (robot,))

        removed = self.world.remove_entity("robot_1")
        self.assertIs(removed, robot)
        self.assertEqual(self.world.get_entities_at((1, 1)), (item,))

        another_world = SimulationWorld(2, 2)
        another_world.add_entity(robot)
        self.assertIs(another_world.get_entity("robot_1"), robot)

    def test_duplicate_global_id_is_rejected_without_partial_mutation(self) -> None:
        self.world.add_entity(Robot("shared_id", (0, 0)))

        with self.assertRaises(ValueError):
            self.world.add_entity(Item("shared_id", (1, 1)))

        self.assertEqual(len(self.world.get_entities()), 1)
        self.assertEqual(self.world.get_entities_at((1, 1)), ())

    def test_duplicate_destination_target_is_rejected_and_removal_releases_it(self) -> None:
        first = DeliveryDestination("destination_1", (0, 0), "item_1")
        duplicate = DeliveryDestination("destination_2", (1, 0), "item_1")
        self.world.add_entity(first)

        with self.assertRaisesRegex(ValueError, "duplicate delivery target"):
            self.world.add_entity(duplicate)
        self.assertEqual(self.world.get_entities(DeliveryDestination), (first,))

        self.world.remove_entity(first.destination_id)
        self.world.add_entity(duplicate)
        self.assertEqual(self.world.get_entities(DeliveryDestination), (duplicate,))

    def test_unknown_id_raises_key_error(self) -> None:
        with self.assertRaises(KeyError):
            self.world.get_entity("missing")
        with self.assertRaises(KeyError):
            self.world.remove_entity("missing")

    def test_invalid_and_occupied_placement_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.world.add_entity(Robot("outside", (4, 0)))
        self.world.add_entity(Robot("robot_1", (1, 1)))
        self.world.add_entity(Item("item_1", (1, 1)))

        with self.assertRaises(ValueError):
            self.world.add_entity(Robot("robot_2", (1, 1)))
        with self.assertRaises(ValueError):
            self.world.add_entity(Item("item_2", (1, 1)))

    def test_custom_policy_can_replace_default_placement_rules(self) -> None:
        world = SimulationWorld(2, 2, occupancy_policy=AllowAllOccupancy())
        world.add_entity(Robot("robot_1", (0, 0)))
        world.add_entity(Robot("robot_2", (0, 0)))

        self.assertEqual(len(world.get_entities_at((0, 0))), 2)

    def test_administrative_move_updates_index(self) -> None:
        robot = Robot("robot_1", (0, 0))
        item = Item("item_1", (1, 0))
        self.world.add_entity(robot)
        self.world.add_entity(item)

        self.world.move_entity("robot_1", (1, 0))

        self.assertEqual(robot.position, (1, 0))
        self.assertEqual(self.world.get_entities_at((0, 0)), ())
        self.assertEqual(len(self.world.get_entities_at((1, 0))), 2)

    def test_failed_administrative_move_preserves_state(self) -> None:
        first = Robot("robot_1", (0, 0))
        second = Robot("robot_2", (1, 0))
        self.world.add_entity(first)
        self.world.add_entity(second)

        with self.assertRaises(ValueError):
            self.world.move_entity("robot_1", (1, 0))
        with self.assertRaises(ValueError):
            self.world.move_entity("robot_1", (-1, 0))

        self.assertEqual(first.position, (0, 0))
        self.assertEqual(self.world.get_entities_at((0, 0)), (first,))


class StepTests(unittest.TestCase):
    def test_all_direction_actions_and_wait(self) -> None:
        cases = [
            (Action.MOVE_UP, (1, 0)),
            (Action.MOVE_DOWN, (1, 2)),
            (Action.MOVE_LEFT, (0, 1)),
            (Action.MOVE_RIGHT, (2, 1)),
            (Action.WAIT, (1, 1)),
        ]
        for action, expected in cases:
            with self.subTest(action=action):
                world = SimulationWorld(3, 3)
                robot = Robot("robot_1", (1, 1))
                world.add_entity(robot)
                result = world.step({"robot_1": action})["robot_1"]

                self.assertTrue(result.success)
                self.assertIsNone(result.failure_reason)
                self.assertEqual(result.start_position, (1, 1))
                self.assertEqual(result.end_position, expected)
                self.assertEqual(robot.position, expected)
                self.assertEqual(world.timestep, 1)

    def test_omitted_action_becomes_wait(self) -> None:
        world = SimulationWorld(2, 2)
        robot = Robot("robot_1", (0, 0))
        world.add_entity(robot)

        result = world.step()["robot_1"]

        self.assertEqual(result.action, Action.WAIT)
        self.assertTrue(result.success)
        self.assertEqual(robot.position, (0, 0))

    def test_boundary_failure_preserves_position_and_advances_time(self) -> None:
        world = SimulationWorld(2, 2)
        robot = Robot("robot_1", (0, 0))
        world.add_entity(robot)

        result = world.step({"robot_1": Action.MOVE_LEFT})["robot_1"]

        self.assertFalse(result.success)
        self.assertEqual(result.failure_reason, ActionFailureReason.OUT_OF_BOUNDS)
        self.assertEqual(result.end_position, (0, 0))
        self.assertEqual(world.timestep, 1)

    def test_robot_can_move_onto_item(self) -> None:
        world = SimulationWorld(3, 1)
        robot = Robot("robot_1", (0, 0))
        item = Item("item_1", (1, 0))
        world.add_entity(robot)
        world.add_entity(item)

        result = world.step({"robot_1": Action.MOVE_RIGHT})["robot_1"]

        self.assertTrue(result.success)
        self.assertEqual(len(world.get_entities_at((1, 0))), 2)

    def test_destination_allows_robot_and_item_overlap_and_reversible_delivery(self) -> None:
        world = SimulationWorld(3, 1)
        robot = Robot("robot_1", (0, 0))
        item = Item("item_1", (1, 0))
        destination = DeliveryDestination("destination_1", (2, 0), "item_1")
        world.add_entity(robot)
        world.add_entity(item)
        world.add_entity(destination)

        self.assertTrue(world.step({"robot_1": Action.MOVE_RIGHT})["robot_1"].success)
        self.assertTrue(world.step({"robot_1": Action.PICK_UP})["robot_1"].success)
        self.assertTrue(world.step({"robot_1": Action.MOVE_RIGHT})["robot_1"].success)
        dropped = world.step({"robot_1": Action.DROP})["robot_1"]

        self.assertTrue(dropped.success)
        self.assertEqual(dropped.dropped_item_id, "item_1")
        self.assertEqual(
            tuple(type(entity) for entity in world.get_entities_at((2, 0))),
            (DeliveryDestination, Item, Robot),
        )

        picked_up = world.step({"robot_1": Action.PICK_UP})["robot_1"]
        self.assertTrue(picked_up.success)
        self.assertEqual(picked_up.picked_up_item_id, "item_1")
        self.assertEqual(world.get_entities_at((2, 0)), (destination, robot))

    def test_wrong_item_can_be_dropped_but_blocks_the_target_item(self) -> None:
        world = SimulationWorld(3, 1)
        first = Robot("robot_1", (0, 0))
        second = Robot("robot_2", (2, 0))
        destination = DeliveryDestination("destination_1", (1, 0), "item_1")
        world.add_entity(first)
        world.add_entity(second)
        world.add_entity(Item("item_2", (0, 0)))
        world.add_entity(Item("item_1", (2, 0)))
        world.add_entity(destination)

        world.step({"robot_1": Action.PICK_UP, "robot_2": Action.PICK_UP})
        world.step({"robot_1": Action.MOVE_RIGHT, "robot_2": Action.WAIT})
        wrong_drop = world.step(
            {"robot_1": Action.DROP, "robot_2": Action.WAIT}
        )["robot_1"]
        self.assertTrue(wrong_drop.success)

        world.step({"robot_1": Action.MOVE_LEFT, "robot_2": Action.WAIT})
        world.step({"robot_1": Action.WAIT, "robot_2": Action.MOVE_LEFT})
        blocked = world.step({"robot_2": Action.DROP})["robot_2"]

        self.assertFalse(blocked.success)
        self.assertEqual(blocked.failure_reason, ActionFailureReason.OCCUPIED)
        self.assertEqual(second.carried_item_id, "item_1")
        self.assertEqual(
            tuple(
                entity.entity_id for entity in world.get_entities_at((1, 0))
            ),
            ("destination_1", "item_2", "robot_2"),
        )

    def test_contested_empty_target_blocks_all_contenders(self) -> None:
        world = SimulationWorld(3, 1)
        left = Robot("robot_left", (0, 0))
        right = Robot("robot_right", (2, 0))
        world.add_entity(left)
        world.add_entity(right)

        results = world.step(
            {
                "robot_left": Action.MOVE_RIGHT,
                "robot_right": Action.MOVE_LEFT,
            }
        )

        self.assertEqual(
            results["robot_left"].failure_reason,
            ActionFailureReason.CONFLICT,
        )
        self.assertEqual(
            results["robot_right"].failure_reason,
            ActionFailureReason.CONFLICT,
        )
        self.assertEqual(left.position, (0, 0))
        self.assertEqual(right.position, (2, 0))

    def test_snapshot_occupied_cell_stays_blocked(self) -> None:
        world = SimulationWorld(3, 1)
        trailing = Robot("robot_a", (0, 0))
        leading = Robot("robot_b", (1, 0))
        world.add_entity(trailing)
        world.add_entity(leading)

        results = world.step(
            {
                "robot_a": Action.MOVE_RIGHT,
                "robot_b": Action.MOVE_RIGHT,
            }
        )

        self.assertEqual(
            results["robot_a"].failure_reason,
            ActionFailureReason.OCCUPIED,
        )
        self.assertTrue(results["robot_b"].success)
        self.assertEqual(trailing.position, (0, 0))
        self.assertEqual(leading.position, (2, 0))

    def test_independent_moves_succeed_in_same_step(self) -> None:
        world = SimulationWorld(3, 2)
        first = Robot("robot_a", (0, 0))
        second = Robot("robot_b", (2, 1))
        world.add_entity(first)
        world.add_entity(second)

        results = world.step(
            {
                "robot_a": Action.MOVE_RIGHT,
                "robot_b": Action.MOVE_LEFT,
            }
        )

        self.assertTrue(all(result.success for result in results.values()))
        self.assertEqual(first.position, (1, 0))
        self.assertEqual(second.position, (1, 1))

    def test_invalid_actions_are_transactional(self) -> None:
        world = SimulationWorld(2, 2)
        robot = Robot("robot_1", (0, 0))
        item = Item("item_1", (1, 1))
        world.add_entity(robot)
        world.add_entity(item)

        invalid_cases = [
            {"missing": Action.WAIT},
            {"item_1": Action.WAIT},
            {"robot_1": "move_right"},
        ]
        for actions in invalid_cases:
            with self.subTest(actions=actions):
                with self.assertRaises((KeyError, ValueError)):
                    world.step(actions)  # type: ignore[arg-type]
                self.assertEqual(world.timestep, 0)
                self.assertEqual(robot.position, (0, 0))

    def test_empty_world_step_advances_time(self) -> None:
        world = SimulationWorld(2, 2)

        self.assertEqual(world.step(), {})
        self.assertEqual(world.timestep, 1)


if __name__ == "__main__":
    unittest.main()
