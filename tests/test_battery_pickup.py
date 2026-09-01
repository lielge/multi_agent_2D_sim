from __future__ import annotations

from dataclasses import FrozenInstanceError
import math
import unittest

from multi_agent_sim.actions import (
    Action,
    ActionBatteryCosts,
    ActionFailureReason,
    ActionResult,
)
from multi_agent_sim.entities import Item, Robot
from multi_agent_sim.world import SimulationWorld


class ActionBatteryCostsTests(unittest.TestCase):
    def test_defaults_and_action_mapping(self) -> None:
        costs = ActionBatteryCosts()

        self.assertEqual(
            (costs.movement, costs.pickup, costs.wait, costs.drop),
            (1.0, 1.0, 0.0, 1.0),
        )
        for action in (
            Action.MOVE_UP,
            Action.MOVE_DOWN,
            Action.MOVE_LEFT,
            Action.MOVE_RIGHT,
        ):
            with self.subTest(action=action):
                self.assertEqual(costs.cost_for(action), 1.0)
        self.assertEqual(costs.cost_for(Action.PICK_UP), 1.0)
        self.assertEqual(costs.cost_for(Action.WAIT), 0.0)
        self.assertEqual(costs.cost_for(Action.DROP), 1.0)

    def test_custom_values_are_normalized_and_configuration_is_frozen(self) -> None:
        costs = ActionBatteryCosts(movement=2, pickup=3.5, wait=0, drop=4)

        self.assertEqual(
            (costs.movement, costs.pickup, costs.wait, costs.drop),
            (2.0, 3.5, 0.0, 4.0),
        )
        with self.assertRaises(FrozenInstanceError):
            costs.movement = 10  # type: ignore[misc]

    def test_drop_is_appended_after_existing_positional_cost_fields(self) -> None:
        old_style = ActionBatteryCosts(2, 3, 0.5)
        all_positional = ActionBatteryCosts(2, 3, 0.5, 4)

        self.assertEqual(
            (old_style.movement, old_style.pickup, old_style.wait, old_style.drop),
            (2.0, 3.0, 0.5, 1.0),
        )
        self.assertEqual(all_positional.cost_for(Action.DROP), 4.0)

    def test_costs_must_be_finite_nonnegative_numbers(self) -> None:
        invalid_values = (-1, True, "1", math.inf, -math.inf, math.nan)
        for field_name in ("movement", "pickup", "wait", "drop"):
            for value in invalid_values:
                with self.subTest(field_name=field_name, value=value):
                    kwargs = {field_name: value}
                    with self.assertRaises(ValueError):
                        ActionBatteryCosts(**kwargs)  # type: ignore[arg-type]

    def test_cost_for_rejects_non_action_values(self) -> None:
        with self.assertRaises(ValueError):
            ActionBatteryCosts().cost_for("wait")  # type: ignore[arg-type]

    def test_action_result_new_fields_have_compatible_defaults(self) -> None:
        result = ActionResult(Action.WAIT, True, (0, 0), (0, 0))

        self.assertEqual(result.battery_before, 0.0)
        self.assertEqual(result.battery_after, 0.0)
        self.assertEqual(result.battery_spent, 0.0)
        self.assertIsNone(result.picked_up_item_id)
        self.assertIsNone(result.picked_item_id)
        self.assertIsNone(result.dropped_item_id)


class BatteryAccountingTests(unittest.TestCase):
    def test_world_exposes_immutable_action_costs(self) -> None:
        costs = ActionBatteryCosts(movement=2, pickup=3, wait=0.5)
        world = SimulationWorld(2, 2, action_battery_costs=costs)

        self.assertIs(world.action_battery_costs, costs)
        with self.assertRaises(ValueError):
            SimulationWorld(2, 2, action_battery_costs=object())  # type: ignore[arg-type]

    def test_successful_move_charges_configured_cost_and_allows_exact_battery(self) -> None:
        world = SimulationWorld(
            2,
            1,
            action_battery_costs=ActionBatteryCosts(movement=2.5),
        )
        robot = Robot("robot_1", (0, 0), battery_level=2.5)
        world.add_entity(robot)

        result = world.step({robot.robot_id: Action.MOVE_RIGHT})[robot.robot_id]

        self.assertTrue(result.success)
        self.assertEqual(robot.position, (1, 0))
        self.assertEqual(robot.battery_level, 0.0)
        self.assertEqual(result.battery_before, 2.5)
        self.assertEqual(result.battery_after, 0.0)
        self.assertEqual(result.battery_spent, 2.5)

    def test_affordable_failed_action_is_still_charged(self) -> None:
        world = SimulationWorld(
            2,
            1,
            action_battery_costs=ActionBatteryCosts(movement=2),
        )
        robot = Robot("robot_1", (0, 0), battery_level=5)
        world.add_entity(robot)

        result = world.step({robot.robot_id: Action.MOVE_LEFT})[robot.robot_id]

        self.assertFalse(result.success)
        self.assertEqual(result.failure_reason, ActionFailureReason.OUT_OF_BOUNDS)
        self.assertEqual(robot.battery_level, 3.0)
        self.assertEqual(result.battery_spent, 2.0)

    def test_occupied_and_conflicting_moves_are_charged(self) -> None:
        costs = ActionBatteryCosts(movement=1.5)
        occupied_world = SimulationWorld(2, 1, action_battery_costs=costs)
        blocked = Robot("robot_1", (0, 0), battery_level=5)
        occupant = Robot("robot_2", (1, 0), battery_level=5)
        occupied_world.add_entity(blocked)
        occupied_world.add_entity(occupant)

        occupied = occupied_world.step(
            {"robot_1": Action.MOVE_RIGHT, "robot_2": Action.WAIT}
        )["robot_1"]
        self.assertEqual(occupied.failure_reason, ActionFailureReason.OCCUPIED)
        self.assertEqual(occupied.battery_spent, 1.5)
        self.assertEqual(blocked.battery_level, 3.5)

        conflict_world = SimulationWorld(3, 1, action_battery_costs=costs)
        left = Robot("robot_left", (0, 0), battery_level=5)
        right = Robot("robot_right", (2, 0), battery_level=5)
        conflict_world.add_entity(left)
        conflict_world.add_entity(right)
        conflicted = conflict_world.step(
            {
                "robot_left": Action.MOVE_RIGHT,
                "robot_right": Action.MOVE_LEFT,
            }
        )

        self.assertTrue(
            all(
                result.failure_reason is ActionFailureReason.CONFLICT
                for result in conflicted.values()
            )
        )
        self.assertTrue(
            all(result.battery_spent == 1.5 for result in conflicted.values())
        )
        self.assertEqual((left.battery_level, right.battery_level), (3.5, 3.5))

    def test_unaffordable_action_takes_precedence_and_spends_nothing(self) -> None:
        world = SimulationWorld(
            1,
            1,
            action_battery_costs=ActionBatteryCosts(movement=2),
        )
        robot = Robot("robot_1", (0, 0), battery_level=1)
        world.add_entity(robot)

        result = world.step({robot.robot_id: Action.MOVE_LEFT})[robot.robot_id]

        self.assertFalse(result.success)
        self.assertEqual(
            result.failure_reason,
            ActionFailureReason.INSUFFICIENT_BATTERY,
        )
        self.assertEqual(robot.position, (0, 0))
        self.assertEqual(robot.battery_level, 1.0)
        self.assertEqual(result.battery_before, 1.0)
        self.assertEqual(result.battery_after, 1.0)
        self.assertEqual(result.battery_spent, 0.0)

    def test_wait_uses_configured_cost_including_when_omitted(self) -> None:
        world = SimulationWorld(
            1,
            1,
            action_battery_costs=ActionBatteryCosts(wait=0.75),
        )
        robot = Robot("robot_1", (0, 0), battery_level=2)
        world.add_entity(robot)

        result = world.step()[robot.robot_id]

        self.assertTrue(result.success)
        self.assertEqual(result.action, Action.WAIT)
        self.assertEqual(result.battery_spent, 0.75)
        self.assertEqual(robot.battery_level, 1.25)

    def test_unaffordable_wait_fails_without_spending(self) -> None:
        world = SimulationWorld(
            1,
            1,
            action_battery_costs=ActionBatteryCosts(wait=2),
        )
        robot = Robot("robot_1", (0, 0), battery_level=1)
        world.add_entity(robot)

        result = world.step()[robot.robot_id]

        self.assertEqual(
            result.failure_reason,
            ActionFailureReason.INSUFFICIENT_BATTERY,
        )
        self.assertEqual(result.battery_spent, 0.0)
        self.assertEqual(robot.battery_level, 1.0)

    def test_invalid_action_batch_does_not_charge_or_advance(self) -> None:
        world = SimulationWorld(2, 1)
        robot = Robot("robot_1", (0, 0), battery_level=5)
        world.add_entity(robot)

        with self.assertRaises(ValueError):
            world.step({robot.robot_id: "move_right"})  # type: ignore[dict-item]

        self.assertEqual(robot.battery_level, 5.0)
        self.assertEqual(world.timestep, 0)


class PickupTests(unittest.TestCase):
    def test_pickup_removes_item_and_stores_its_id(self) -> None:
        world = SimulationWorld(
            2,
            1,
            action_battery_costs=ActionBatteryCosts(pickup=1.5),
        )
        robot = Robot("robot_1", (0, 0), battery_level=4)
        item = Item("item_1", (0, 0))
        world.add_entity(robot)
        world.add_entity(item)

        result = world.step({robot.robot_id: Action.PICK_UP})[robot.robot_id]

        self.assertTrue(result.success)
        self.assertEqual(result.picked_up_item_id, "item_1")
        self.assertEqual(result.picked_item_id, "item_1")
        self.assertEqual(robot.carried_item_id, "item_1")
        self.assertEqual(robot.battery_level, 2.5)
        self.assertEqual(world.get_entities(Item), ())
        self.assertEqual(world.get_entities_at((0, 0)), (robot,))

    def test_no_item_is_checked_before_full_inventory_and_attempts_are_charged(self) -> None:
        costs = ActionBatteryCosts(pickup=1)
        world = SimulationWorld(1, 1, action_battery_costs=costs)
        robot = Robot("robot_1", (0, 0), battery_level=5)
        world.add_entity(robot)
        world.add_entity(Item("item_1", (0, 0)))
        world.step({robot.robot_id: Action.PICK_UP})

        no_item = world.step({robot.robot_id: Action.PICK_UP})[robot.robot_id]
        self.assertEqual(no_item.failure_reason, ActionFailureReason.NO_ITEM)
        self.assertEqual(no_item.battery_spent, 1.0)

        world.add_entity(Item("item_2", (0, 0)))
        full = world.step({robot.robot_id: Action.PICK_UP})[robot.robot_id]
        self.assertEqual(full.failure_reason, ActionFailureReason.INVENTORY_FULL)
        self.assertEqual(full.battery_spent, 1.0)
        self.assertEqual(world.get_entity("item_2").entity_id, "item_2")
        self.assertEqual(robot.battery_level, 2.0)

    def test_unaffordable_pickup_does_not_remove_item(self) -> None:
        world = SimulationWorld(
            1,
            1,
            action_battery_costs=ActionBatteryCosts(pickup=2),
        )
        robot = Robot("robot_1", (0, 0), battery_level=1)
        item = Item("item_1", (0, 0))
        world.add_entity(robot)
        world.add_entity(item)

        result = world.step({robot.robot_id: Action.PICK_UP})[robot.robot_id]

        self.assertEqual(
            result.failure_reason,
            ActionFailureReason.INSUFFICIENT_BATTERY,
        )
        self.assertIs(world.get_entity("item_1"), item)
        self.assertIsNone(robot.carried_item_id)
        self.assertEqual(robot.battery_level, 1.0)

    def test_inventory_is_read_only_to_callers(self) -> None:
        robot = Robot("robot_1", (0, 0))

        with self.assertRaises(AttributeError):
            robot.carried_item_id = "item_1"  # type: ignore[misc]


class DropTests(unittest.TestCase):
    def test_drop_restores_same_item_id_and_supports_pickup_round_trip(self) -> None:
        world = SimulationWorld(
            1,
            1,
            action_battery_costs=ActionBatteryCosts(pickup=1, drop=2),
        )
        robot = Robot("robot_1", (0, 0), battery_level=10)
        original_item = Item("item_1", (0, 0))
        world.add_entity(robot)
        world.add_entity(original_item)
        world.step({robot.robot_id: Action.PICK_UP})

        result = world.step({robot.robot_id: Action.DROP})[robot.robot_id]

        self.assertTrue(result.success)
        self.assertEqual(result.dropped_item_id, "item_1")
        self.assertIsNone(result.picked_up_item_id)
        self.assertIsNone(robot.carried_item_id)
        self.assertEqual(result.battery_spent, 2.0)
        self.assertEqual(robot.battery_level, 7.0)
        dropped = world.get_entity("item_1")
        self.assertIsInstance(dropped, Item)
        self.assertIsNot(dropped, original_item)
        self.assertEqual(dropped.position, robot.position)

        pickup_again = world.step({robot.robot_id: Action.PICK_UP})[robot.robot_id]
        self.assertTrue(pickup_again.success)
        self.assertEqual(pickup_again.picked_up_item_id, "item_1")
        self.assertEqual(robot.carried_item_id, "item_1")

    def test_empty_inventory_drop_fails_and_is_charged(self) -> None:
        world = SimulationWorld(
            1,
            1,
            action_battery_costs=ActionBatteryCosts(drop=1.5),
        )
        robot = Robot("robot_1", (0, 0), battery_level=3)
        world.add_entity(robot)

        result = world.step({robot.robot_id: Action.DROP})[robot.robot_id]

        self.assertFalse(result.success)
        self.assertIs(result.failure_reason, ActionFailureReason.NO_CARRIED_ITEM)
        self.assertEqual(result.battery_spent, 1.5)
        self.assertEqual(robot.battery_level, 1.5)
        self.assertIsNone(result.dropped_item_id)

    def test_existing_item_blocks_drop_even_when_inventory_is_full(self) -> None:
        world = SimulationWorld(
            1,
            1,
            action_battery_costs=ActionBatteryCosts(pickup=0, drop=2),
        )
        robot = Robot("robot_1", (0, 0), battery_level=5)
        world.add_entity(robot)
        world.add_entity(Item("item_1", (0, 0)))
        world.step({robot.robot_id: Action.PICK_UP})
        blocking_item = Item("item_2", (0, 0))
        world.add_entity(blocking_item)

        result = world.step({robot.robot_id: Action.DROP})[robot.robot_id]

        self.assertIs(result.failure_reason, ActionFailureReason.OCCUPIED)
        self.assertEqual(result.battery_spent, 2.0)
        self.assertEqual(robot.battery_level, 3.0)
        self.assertEqual(robot.carried_item_id, "item_1")
        self.assertIs(world.get_entity("item_2"), blocking_item)
        with self.assertRaises(KeyError):
            world.get_entity("item_1")

    def test_occupancy_policy_can_reject_drop_without_partial_mutation(self) -> None:
        class RejectItemOnRobotPolicy:
            def allows(self, entity: object, occupants: object) -> bool:
                if isinstance(entity, Item):
                    return not any(
                        isinstance(occupant, Robot) for occupant in occupants  # type: ignore[union-attr]
                    )
                return True

        world = SimulationWorld(
            1,
            1,
            occupancy_policy=RejectItemOnRobotPolicy(),
            action_battery_costs=ActionBatteryCosts(pickup=0, drop=1),
        )
        item = Item("item_1", (0, 0))
        robot = Robot("robot_1", (0, 0), battery_level=2)
        world.add_entity(item)
        world.add_entity(robot)
        world.step({robot.robot_id: Action.PICK_UP})

        result = world.step({robot.robot_id: Action.DROP})[robot.robot_id]

        self.assertIs(result.failure_reason, ActionFailureReason.OCCUPIED)
        self.assertEqual(result.battery_spent, 1.0)
        self.assertEqual(robot.carried_item_id, "item_1")
        self.assertEqual(world.get_entities(Item), ())

    def test_exact_battery_drop_succeeds_and_unaffordable_drop_is_inert(self) -> None:
        exact_world = SimulationWorld(
            1,
            1,
            action_battery_costs=ActionBatteryCosts(pickup=0, drop=2),
        )
        exact_robot = Robot("robot_exact", (0, 0), battery_level=2)
        exact_world.add_entity(exact_robot)
        exact_world.add_entity(Item("item_exact", (0, 0)))
        exact_world.step({exact_robot.robot_id: Action.PICK_UP})

        exact = exact_world.step({exact_robot.robot_id: Action.DROP})[
            exact_robot.robot_id
        ]
        self.assertTrue(exact.success)
        self.assertEqual(exact_robot.battery_level, 0.0)
        self.assertEqual(exact.dropped_item_id, "item_exact")

        insufficient_world = SimulationWorld(
            1,
            1,
            action_battery_costs=ActionBatteryCosts(pickup=0, drop=2),
        )
        insufficient_robot = Robot("robot_low", (0, 0), battery_level=1)
        insufficient_world.add_entity(insufficient_robot)
        insufficient_world.add_entity(Item("item_low", (0, 0)))
        insufficient_world.step({insufficient_robot.robot_id: Action.PICK_UP})

        insufficient = insufficient_world.step(
            {insufficient_robot.robot_id: Action.DROP}
        )[insufficient_robot.robot_id]
        self.assertIs(
            insufficient.failure_reason,
            ActionFailureReason.INSUFFICIENT_BATTERY,
        )
        self.assertEqual(insufficient.battery_spent, 0.0)
        self.assertEqual(insufficient_robot.battery_level, 1.0)
        self.assertEqual(insufficient_robot.carried_item_id, "item_low")
        self.assertEqual(insufficient_world.get_entities(Item), ())

    def test_carried_item_id_is_reserved_until_drop_or_carrier_removal(self) -> None:
        world = SimulationWorld(
            1,
            1,
            action_battery_costs=ActionBatteryCosts(pickup=0),
        )
        robot = Robot("robot_1", (0, 0))
        world.add_entity(robot)
        world.add_entity(Item("item_1", (0, 0)))
        world.step({robot.robot_id: Action.PICK_UP})

        with self.assertRaises(ValueError):
            world.add_entity(Item("item_1", (0, 0)))
        with self.assertRaises(ValueError):
            world.add_entity(Robot("item_1", (0, 0)))

        self.assertIs(world.remove_entity(robot.robot_id), robot)
        replacement = Item("item_1", (0, 0))
        world.add_entity(replacement)
        with self.assertRaises(ValueError):
            world.add_entity(robot)

        world.remove_entity(replacement.item_id)
        world.add_entity(robot)
        with self.assertRaises(ValueError):
            world.add_entity(Item("item_1", (0, 0)))

        dropped = world.step({robot.robot_id: Action.DROP})[robot.robot_id]
        self.assertTrue(dropped.success)
        self.assertEqual(dropped.dropped_item_id, "item_1")
        self.assertIsNone(robot.carried_item_id)
        self.assertEqual(world.get_entity("item_1").position, (0, 0))


if __name__ == "__main__":
    unittest.main()
