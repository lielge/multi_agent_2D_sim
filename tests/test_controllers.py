from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest

from multi_agent_sim.actions import Action, ActionBatteryCosts
from multi_agent_sim.controllers import (
    ControllerDefinition,
    ControllerFactoryContext,
    ControllerRegistry,
    ItemObservation,
    MultiAgentControllerAdapter,
    NearestItemController,
    RandomController,
    RobotObservation,
    WorldObservation,
    create_default_controller_registry,
    create_world_observation,
)
from multi_agent_sim.entities import Item, Robot
from multi_agent_sim.world import SimulationWorld


def make_observation(
    *,
    timestep: int = 0,
    robots: tuple[RobotObservation, ...] | None = None,
    items: tuple[ItemObservation, ...] = (),
    costs: ActionBatteryCosts | None = None,
) -> WorldObservation:
    return WorldObservation(
        width=6,
        height=5,
        timestep=timestep,
        robots=robots
        or (RobotObservation("robot_1", (2, 2), battery_level=100),),
        items=items,
        action_battery_costs=costs or ActionBatteryCosts(),
    )


class ObservationTests(unittest.TestCase):
    def test_create_normalizes_order_and_exposes_lookup_helpers(self) -> None:
        observation = WorldObservation.create(
            width=4,
            height=3,
            timestep=7,
            robots=[
                RobotObservation("robot_2", (2, 0), 20),
                RobotObservation("robot_1", (1, 0), 10, "collected"),
            ],
            items=[
                ItemObservation("item_2", (3, 2)),
                ItemObservation("item_1", (0, 2)),
            ],
            action_battery_costs=ActionBatteryCosts(2, 3, 0.5, 4),
        )

        self.assertEqual(
            tuple(robot.robot_id for robot in observation.robots),
            ("robot_1", "robot_2"),
        )
        self.assertEqual(
            tuple(item.item_id for item in observation.items),
            ("item_1", "item_2"),
        )
        self.assertEqual(observation.get_robot("robot_1").carried_item_id, "collected")
        self.assertEqual(observation.get_item("item_2").position, (3, 2))
        self.assertIs(observation.action_costs, observation.action_battery_costs)
        self.assertEqual(observation.action_costs.drop, 4.0)
        self.assertEqual(observation.action_costs.cost_for(Action.DROP), 4.0)
        with self.assertRaises(KeyError):
            observation.get_robot("missing")
        with self.assertRaises(KeyError):
            observation.get_item("missing")

    def test_observations_are_frozen_and_validate_unique_in_bounds_data(self) -> None:
        robot = RobotObservation("robot_1", (0, 0), 5)
        with self.assertRaises(FrozenInstanceError):
            robot.position = (1, 1)  # type: ignore[misc]

        with self.assertRaises(ValueError):
            make_observation(robots=(robot, robot))
        with self.assertRaises(ValueError):
            make_observation(robots=(RobotObservation("robot_1", (6, 0), 5),))
        with self.assertRaises(ValueError):
            RobotObservation("robot_1", (0, 0), float("nan"))

    def test_from_world_copies_public_state_without_aliasing_entities(self) -> None:
        costs = ActionBatteryCosts(movement=2, pickup=4, wait=0.25, drop=5)
        world = SimulationWorld(4, 3, action_battery_costs=costs)
        world.add_entity(Robot("robot_2", (2, 1), battery_level=8))
        world.add_entity(Robot("robot_1", (1, 1), battery_level=6))
        world.add_entity(Item("item_1", (0, 0)))

        observation = create_world_observation(world)

        self.assertEqual(observation.action_battery_costs, costs)
        self.assertEqual(observation.action_battery_costs.drop, 5.0)
        self.assertEqual(observation.timestep, 0)
        self.assertEqual(
            tuple(robot.robot_id for robot in observation.robots),
            ("robot_1", "robot_2"),
        )
        world.move_entity("robot_1", (3, 2))
        self.assertEqual(observation.get_robot("robot_1").position, (1, 1))

    def test_from_world_accepts_every_entity_id_the_world_accepts(self) -> None:
        world = SimulationWorld(2, 1)
        world.add_entity(Robot(" robot 1 ", (0, 0)))
        world.add_entity(Item(" item 1 ", (1, 0)))

        observation = WorldObservation.from_world(world)

        self.assertEqual(observation.robots[0].robot_id, " robot 1 ")
        self.assertEqual(observation.items[0].item_id, " item 1 ")


class ControllerRegistryTests(unittest.TestCase):
    class WaitController:
        def choose_action(
            self,
            observation: WorldObservation,
            robot_id: str,
        ) -> Action:
            return Action.WAIT

    def test_registry_preserves_order_and_constructs_with_context(self) -> None:
        contexts: list[ControllerFactoryContext] = []

        def factory(context: ControllerFactoryContext) -> object:
            contexts.append(context)
            return self.WaitController()

        registry = ControllerRegistry()
        first = registry.register(
            "wait", "Always Wait", factory  # type: ignore[arg-type]
        )
        second = registry.register(
            ControllerDefinition(
                "wait_2", "Wait Again", factory  # type: ignore[arg-type]
            )
        )

        context = ControllerFactoryContext(seed=4, robot_id="robot_1")
        controller = registry.create("wait", context)

        self.assertIsInstance(controller, self.WaitController)
        self.assertEqual(contexts, [context])
        self.assertEqual(registry.keys, ("wait", "wait_2"))
        self.assertEqual(registry.definitions, (first, second))
        self.assertEqual(registry.display_name("wait_2"), "Wait Again")

    def test_registry_rejects_duplicates_unknowns_and_invalid_factories(self) -> None:
        registry = ControllerRegistry()
        registry.register("wait", "Always Wait", lambda context: self.WaitController())
        with self.assertRaises(ValueError):
            registry.register(
                "wait", "Duplicate", lambda context: self.WaitController()
            )
        with self.assertRaises(KeyError):
            registry.create(
                "missing",
                ControllerFactoryContext(seed=1, robot_id="robot_1"),
            )

        registry.register(
            "invalid", "Invalid", lambda context: object()  # type: ignore[arg-type]
        )
        with self.assertRaises(ValueError):
            registry.create(
                "invalid",
                ControllerFactoryContext(seed=1, robot_id="robot_1"),
            )

    def test_default_registry_has_builtins_and_per_robot_determinism(self) -> None:
        first_registry = create_default_controller_registry()
        second_registry = create_default_controller_registry()
        self.assertEqual(first_registry.keys, ("random", "nearest_item"))
        self.assertEqual(first_registry.display_name("nearest_item"), "Nearest Item")

        context = ControllerFactoryContext(seed=123, robot_id="robot_1")
        first = first_registry.create("random", context)
        second = second_registry.create("random", context)
        observation = make_observation()
        self.assertEqual(
            [first.choose_action(observation, "robot_1") for _ in range(20)],
            [second.choose_action(observation, "robot_1") for _ in range(20)],
        )


class BuiltinControllerTests(unittest.TestCase):
    def test_random_controller_is_seeded_and_returns_actions(self) -> None:
        first = RandomController(19)
        second = RandomController(19)
        observation = make_observation()
        first_actions = [first.choose_action(observation, "robot_1") for _ in range(30)]
        second_actions = [
            second.choose_action(observation, "robot_1") for _ in range(30)
        ]

        self.assertEqual(first_actions, second_actions)
        self.assertTrue(all(isinstance(action, Action) for action in first_actions))

    def test_random_controller_includes_drop_deterministically(self) -> None:
        first = RandomController(47)
        second = RandomController(47)
        observation = make_observation()

        first_actions = [first.choose_action(observation, "robot_1") for _ in range(100)]
        second_actions = [
            second.choose_action(observation, "robot_1") for _ in range(100)
        ]

        self.assertEqual(first_actions, second_actions)
        self.assertIn(Action.DROP, first_actions)

    def test_nearest_item_ties_by_id_and_moves_horizontally_first(self) -> None:
        controller = NearestItemController()
        tie = make_observation(
            items=(
                ItemObservation("item_z", (1, 2)),
                ItemObservation("item_a", (3, 2)),
            )
        )
        diagonal = make_observation(items=(ItemObservation("item_1", (0, 0)),))

        self.assertIs(controller.choose_action(tie, "robot_1"), Action.MOVE_RIGHT)
        self.assertIs(
            controller.choose_action(diagonal, "robot_1"),
            Action.MOVE_LEFT,
        )

    def test_nearest_item_picks_up_and_checks_intended_action_cost(self) -> None:
        controller = NearestItemController()
        colocated = make_observation(
            robots=(RobotObservation("robot_1", (2, 2), 3),),
            items=(ItemObservation("item_1", (2, 2)),),
            costs=ActionBatteryCosts(movement=2, pickup=3, wait=1),
        )
        insufficient = make_observation(
            robots=(RobotObservation("robot_1", (2, 2), 2.5),),
            items=(ItemObservation("item_1", (2, 2)),),
            costs=ActionBatteryCosts(movement=2, pickup=3, wait=1),
        )

        self.assertIs(controller.choose_action(colocated, "robot_1"), Action.PICK_UP)
        self.assertIs(controller.choose_action(insufficient, "robot_1"), Action.WAIT)

    def test_nearest_item_waits_when_loaded_or_no_items_remain(self) -> None:
        controller = NearestItemController()
        loaded = make_observation(
            robots=(RobotObservation("robot_1", (2, 2), 100, "item_old"),),
            items=(ItemObservation("item_1", (3, 2)),),
            costs=ActionBatteryCosts(drop=0),
        )
        empty = make_observation()

        self.assertIs(controller.choose_action(loaded, "robot_1"), Action.WAIT)
        self.assertIs(controller.choose_action(empty, "robot_1"), Action.WAIT)


class CountingTeamController:
    def __init__(self) -> None:
        self.call_count = 0

    def choose_actions(
        self,
        observation: WorldObservation,
        robot_ids: tuple[str, ...],
    ) -> dict[str, Action]:
        self.call_count += 1
        return {
            robot_id: (Action.MOVE_LEFT if index == 0 else Action.MOVE_RIGHT)
            for index, robot_id in enumerate(robot_ids)
        }


class MultiAgentAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.robots = (
            RobotObservation("robot_1", (1, 1), 100),
            RobotObservation("robot_2", (3, 1), 100),
        )

    def test_joint_controller_runs_once_for_each_snapshot(self) -> None:
        joint = CountingTeamController()
        adapter = MultiAgentControllerAdapter(joint, ("robot_1", "robot_2"))
        initial = make_observation(robots=self.robots)

        self.assertIs(adapter.choose_action(initial, "robot_2"), Action.MOVE_RIGHT)
        self.assertIs(adapter.choose_action(initial, "robot_1"), Action.MOVE_LEFT)
        self.assertIs(adapter.choose_action(initial, "robot_1"), Action.MOVE_LEFT)
        self.assertEqual(joint.call_count, 1)

        next_step = make_observation(timestep=1, robots=self.robots)
        adapter.choose_action(next_step, "robot_1")
        self.assertEqual(joint.call_count, 2)

    def test_adapter_validates_membership_and_complete_output(self) -> None:
        observation = make_observation(robots=self.robots)
        adapter = MultiAgentControllerAdapter(
            CountingTeamController(),
            ("robot_1", "robot_2"),
        )
        with self.assertRaises(ValueError):
            adapter.choose_action(observation, "robot_3")

        class MissingController:
            def __init__(self) -> None:
                self.calls = 0

            def choose_actions(
                self,
                observation: WorldObservation,
                robot_ids: tuple[str, ...],
            ) -> dict[str, Action]:
                self.calls += 1
                return {"robot_1": Action.WAIT}

        incomplete = MissingController()
        invalid_adapter = MultiAgentControllerAdapter(
            incomplete,
            ("robot_1", "robot_2"),
        )
        with self.assertRaises(ValueError):
            invalid_adapter.choose_action(observation, "robot_1")
        with self.assertRaises(ValueError):
            invalid_adapter.choose_action(observation, "robot_2")
        self.assertEqual(incomplete.calls, 1)


if __name__ == "__main__":
    unittest.main()
