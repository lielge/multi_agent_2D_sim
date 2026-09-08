from __future__ import annotations

from dataclasses import FrozenInstanceError
import random
import unittest

from multi_agent_sim import (
    Action,
    ActionBatteryCosts,
    ControllerFactoryContext,
    ControllerRegistry,
    DeliveryDestination,
    InitialScenario,
    Item,
    MultiAgentControllerAdapter,
    Robot,
    RobotConfiguration,
    SimulationSession,
    WorldObservation,
    create_initial_scenario,
)


def world_snapshot(session: SimulationSession) -> tuple[object, ...]:
    return tuple(
        (
            type(entity).__name__,
            entity.entity_id,
            entity.position,
            getattr(entity, "battery_level", None),
            getattr(entity, "carried_item_id", None),
            getattr(entity, "target_item_id", None),
        )
        for entity in session.world.get_entities()
    )


class RobotConfigurationTests(unittest.TestCase):
    def test_configuration_is_frozen_and_normalizes_battery(self) -> None:
        configuration = RobotConfiguration((2, 3), 75, "nearest_item")

        self.assertEqual(configuration.battery_level, 75.0)
        self.assertEqual(configuration.controller_key, "nearest_item")
        with self.assertRaises(FrozenInstanceError):
            configuration.position = (0, 0)  # type: ignore[misc]

    def test_configuration_validates_position_and_battery(self) -> None:
        invalid_positions = [(1,), [1, 2], (1.0, 2)]
        for position in invalid_positions:
            with self.subTest(position=position):
                with self.assertRaises(ValueError):
                    RobotConfiguration(position)  # type: ignore[arg-type]

        invalid_batteries = [-1, True, "100", float("inf"), float("nan")]
        for battery in invalid_batteries:
            with self.subTest(battery=battery):
                with self.assertRaises(ValueError):
                    RobotConfiguration((0, 0), battery)  # type: ignore[arg-type]
        for controller_key in ("", "   ", 3):
            with self.subTest(controller_key=controller_key):
                with self.assertRaises(ValueError):
                    RobotConfiguration(
                        (0, 0), controller_key=controller_key  # type: ignore[arg-type]
                    )


class InitialScenarioTests(unittest.TestCase):
    def test_random_scenario_is_reproducible_unique_and_auto_named(self) -> None:
        first = create_initial_scenario(5, 4, 3, 6, seed=91)
        second = create_initial_scenario(5, 4, 3, 6, seed=91)

        self.assertEqual(first, second)
        self.assertEqual(first.num_robots, 3)
        self.assertEqual(first.num_items, 6)
        self.assertEqual(first.num_destinations, 6)
        positions = tuple(configuration.position for configuration in first.robot_configurations)
        positions += first.item_positions
        positions += first.delivery_destination_positions
        self.assertEqual(len(positions), len(set(positions)))
        self.assertEqual(
            tuple(configuration.battery_level for configuration in first.robot_configurations),
            (100.0, 100.0, 100.0),
        )

        world = first.create_world()
        self.assertEqual(
            tuple(robot.robot_id for robot in world.get_entities(Robot)),
            ("robot_1", "robot_2", "robot_3"),
        )
        self.assertEqual(
            tuple(item.item_id for item in world.get_entities(Item)),
            ("item_1", "item_2", "item_3", "item_4", "item_5", "item_6"),
        )
        self.assertEqual(
            tuple(
                (destination.destination_id, destination.target_item_id)
                for destination in world.get_entities(DeliveryDestination)
            ),
            tuple(
                (f"destination_{index}", f"item_{index}")
                for index in range(1, 7)
            ),
        )

    def test_manual_configuration_preserves_positions_and_batteries(self) -> None:
        robots = (
            RobotConfiguration((0, 0), 12.5),
            RobotConfiguration((2, 1), 0),
        )
        scenario = create_initial_scenario(
            4, 3, 2, 5, seed=8, robot_configurations=robots
        )

        self.assertEqual(scenario.robot_configurations, robots)
        self.assertTrue(
            set(scenario.item_positions).isdisjoint(
                configuration.position for configuration in robots
            )
        )
        all_positions = tuple(configuration.position for configuration in robots)
        all_positions += scenario.item_positions
        all_positions += scenario.delivery_destination_positions
        self.assertEqual(len(all_positions), len(set(all_positions)))
        world = scenario.create_world()
        created_robots = world.get_entities(Robot)
        self.assertEqual(
            tuple((robot.position, robot.battery_level) for robot in created_robots),
            (((0, 0), 12.5), ((2, 1), 0.0)),
        )

    def test_seeded_scenario_can_start_with_leading_items_delivered(self) -> None:
        scenario = create_initial_scenario(
            4,
            4,
            1,
            2,
            seed=17,
            initially_delivered_items=1,
        )

        self.assertEqual(scenario.initially_delivered_items, 1)
        self.assertEqual(
            scenario.item_positions[0],
            scenario.delivery_destination_positions[0],
        )
        self.assertNotEqual(
            scenario.item_positions[1],
            scenario.delivery_destination_positions[1],
        )
        world = scenario.create_world()
        self.assertEqual(
            world.get_entity("item_1").position,
            world.get_entity("destination_1").position,
        )

    def test_initial_delivery_configuration_is_explicit_and_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            create_initial_scenario(
                3,
                3,
                1,
                1,
                seed=4,
                initially_delivered_items=2,
            )
        with self.assertRaisesRegex(ValueError, "configured as initially delivered"):
            InitialScenario(
                3,
                3,
                (),
                ((0, 0),),
                2,
                delivery_destination_positions=((0, 0),),
            )

    def test_action_costs_and_controller_keys_are_preserved(self) -> None:
        costs = ActionBatteryCosts(movement=2, pickup=3, wait=0.5)
        scenario = create_initial_scenario(
            3,
            2,
            2,
            1,
            seed=5,
            action_battery_costs=costs,
            controller_keys=("nearest_item", "random"),
        )

        self.assertIs(scenario.action_battery_costs, costs)
        self.assertEqual(
            tuple(
                configuration.controller_key
                for configuration in scenario.robot_configurations
            ),
            ("nearest_item", "random"),
        )
        self.assertIs(scenario.create_world().action_battery_costs, costs)

    def test_create_world_always_returns_fresh_entities_at_timestep_zero(self) -> None:
        scenario = create_initial_scenario(3, 3, 1, 1, seed=4)
        first = scenario.create_world()
        first_robot = first.get_entities(Robot)[0]
        first.step({first_robot.robot_id: Action.WAIT})
        second = scenario.create_world()

        self.assertIsNot(first, second)
        self.assertIsNot(first_robot, second.get_entities(Robot)[0])
        self.assertEqual(second.timestep, 0)

    def test_factory_does_not_mutate_global_random_state(self) -> None:
        random.seed(1234)
        expected = random.Random(1234).random()

        create_initial_scenario(4, 4, 2, 3, seed=20)

        self.assertEqual(random.random(), expected)

    def test_destination_sampling_preserves_seeded_robot_and_item_cells(self) -> None:
        seed = 27
        sampled = random.Random(seed).sample(range(20), 5)
        expected_positions = tuple((index % 5, index // 5) for index in sampled)

        scenario = create_initial_scenario(5, 4, 2, 3, seed=seed)
        actual_positions = tuple(
            configuration.position
            for configuration in scenario.robot_configurations
        ) + scenario.item_positions

        self.assertEqual(actual_positions, expected_positions)

    def test_factory_validates_dimensions_counts_seed_and_capacity(self) -> None:
        invalid_calls = [
            (0, 2, 0, 0, 1),
            (2, True, 0, 0, 1),
            (2, 2, -1, 0, 1),
            (2, 2, 0, True, 1),
            (2, 2, 0, 0, True),
            (2, 2, 1, 2, 1),
            (2, 2, 3, 2, 1),
        ]
        for args in invalid_calls:
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    create_initial_scenario(*args)  # type: ignore[arg-type]

    def test_manual_configuration_count_and_values_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 2"):
            create_initial_scenario(
                3,
                3,
                2,
                0,
                1,
                robot_configurations=(RobotConfiguration((0, 0)),),
            )
        with self.assertRaises(ValueError):
            create_initial_scenario(
                3, 3, 1, 0, 1, robot_configurations=((0, 0),)  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            create_initial_scenario(
                3,
                3,
                2,
                0,
                1,
                robot_configurations=(
                    RobotConfiguration((1, 1)),
                    RobotConfiguration((1, 1)),
                ),
            )
        with self.assertRaisesRegex(ValueError, "outside world bounds"):
            create_initial_scenario(
                3,
                3,
                1,
                0,
                1,
                robot_configurations=(RobotConfiguration((3, 0)),),
            )

    def test_direct_scenario_rejects_duplicate_or_overlapping_items(self) -> None:
        robot = RobotConfiguration((0, 0))
        with self.assertRaisesRegex(ValueError, "item positions must be unique"):
            InitialScenario(
                3,
                3,
                (robot,),
                ((1, 1), (1, 1)),
                2,
                delivery_destination_positions=((2, 1), (2, 2)),
            )
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            InitialScenario(
                3,
                3,
                (robot,),
                ((0, 0),),
                2,
                delivery_destination_positions=((2, 2),),
            )

    def test_direct_scenario_requires_one_destination_per_item(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one position"):
            InitialScenario(
                3,
                3,
                (),
                ((0, 0),),
                2,
            )

    def test_direct_scenario_validates_destination_positions(self) -> None:
        with self.assertRaisesRegex(ValueError, "destination positions must be unique"):
            InitialScenario(
                3,
                3,
                (),
                ((0, 0), (1, 0)),
                2,
                delivery_destination_positions=((2, 0), (2, 0)),
            )
        with self.assertRaisesRegex(ValueError, "must not overlap"):
            InitialScenario(
                3,
                3,
                (),
                ((0, 0),),
                2,
                delivery_destination_positions=((0, 0),),
            )
        with self.assertRaisesRegex(ValueError, "outside world bounds"):
            InitialScenario(
                3,
                3,
                (),
                ((0, 0),),
                2,
                delivery_destination_positions=((3, 0),),
            )


class SimulationSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scenario = InitialScenario(
            width=5,
            height=5,
            robot_configurations=(
                RobotConfiguration((1, 1), 42),
                RobotConfiguration((3, 3), 7.5),
            ),
            item_positions=((0, 4),),
            seed=19,
            delivery_destination_positions=((4, 0),),
        )

    def test_initial_state_and_boundaries(self) -> None:
        session = SimulationSession(self.scenario, max_steps=3, step_rate=7)

        self.assertEqual(session.cursor, 0)
        self.assertEqual(session.current_step, 0)
        self.assertEqual(session.history_length, 0)
        self.assertFalse(session.playing)
        self.assertFalse(session.can_step_back)
        self.assertTrue(session.can_step_forward)
        self.assertTrue(session.at_history_tip)
        self.assertEqual(session.step_rate, 7)
        self.assertEqual(session.max_steps, 3)

    def test_rate_and_constructor_validation(self) -> None:
        for invalid_rate in [0, 31, 1.5, True]:
            with self.subTest(rate=invalid_rate):
                with self.assertRaises(ValueError):
                    SimulationSession(
                        self.scenario, step_rate=invalid_rate  # type: ignore[arg-type]
                    )
        with self.assertRaises(ValueError):
            SimulationSession(self.scenario, max_steps=-1)

        session = SimulationSession(self.scenario)
        session.step_rate = 30
        self.assertEqual(session.step_rate, 30)
        with self.assertRaises(ValueError):
            session.step_rate = 0

    def test_play_pause_toggle_and_zero_step_maximum(self) -> None:
        session = SimulationSession(self.scenario, max_steps=2)
        session.play()
        self.assertTrue(session.playing)
        self.assertFalse(session.toggle_playing())
        self.assertFalse(session.playing)
        self.assertTrue(session.toggle_playing())
        session.pause()
        self.assertFalse(session.playing)

        empty_session = SimulationSession(self.scenario, max_steps=0)
        empty_session.play()
        self.assertFalse(empty_session.playing)
        self.assertIsNone(empty_session.step_forward())

    def test_forward_records_actions_and_stops_at_maximum(self) -> None:
        session = SimulationSession(self.scenario, max_steps=2)
        session.play()

        first_results = session.step_forward()
        second_results = session.step_forward()

        self.assertIsNotNone(first_results)
        self.assertIsNotNone(second_results)
        self.assertEqual(session.cursor, 2)
        self.assertEqual(session.world.timestep, 2)
        self.assertEqual(session.history_length, 2)
        self.assertFalse(session.can_step_forward)
        self.assertFalse(session.playing)
        self.assertIsNone(session.step_forward())
        self.assertEqual(session.cursor, 2)

    def test_back_rebuilds_and_forward_replays_exact_history(self) -> None:
        session = SimulationSession(self.scenario, max_steps=5)
        session.step_forward()
        expected_at_one = world_snapshot(session)
        session.step_forward()
        expected_at_two = world_snapshot(session)
        recorded_history = tuple(dict(actions) for actions in session.action_history)
        session.play()

        self.assertTrue(session.step_back())
        self.assertFalse(session.playing)
        self.assertEqual(session.cursor, 1)
        self.assertEqual(session.world.timestep, 1)
        self.assertEqual(session.history_length, 2)
        self.assertFalse(session.at_history_tip)
        self.assertEqual(world_snapshot(session), expected_at_one)

        session.step_forward()

        self.assertEqual(world_snapshot(session), expected_at_two)
        self.assertEqual(
            tuple(dict(actions) for actions in session.action_history),
            recorded_history,
        )
        self.assertTrue(session.at_history_tip)

    def test_back_at_zero_is_a_safe_noop_and_pauses(self) -> None:
        session = SimulationSession(self.scenario)
        session.play()

        self.assertFalse(session.step_back())
        self.assertFalse(session.playing)
        self.assertEqual(session.cursor, 0)

    def test_resuming_generation_after_replay_preserves_random_stream(self) -> None:
        uninterrupted = SimulationSession(self.scenario, max_steps=5)
        rewound = SimulationSession(self.scenario, max_steps=5)
        for _ in range(3):
            uninterrupted.step_forward()
            rewound.step_forward()

        rewound.step_back()
        rewound.step_back()
        rewound.step_forward()
        rewound.step_forward()
        uninterrupted.step_forward()
        rewound.step_forward()

        self.assertEqual(
            tuple(dict(actions) for actions in rewound.action_history),
            tuple(dict(actions) for actions in uninterrupted.action_history),
        )
        self.assertEqual(world_snapshot(rewound), world_snapshot(uninterrupted))

    def test_action_history_cannot_be_mutated_by_callers(self) -> None:
        session = SimulationSession(self.scenario)
        session.step_forward()
        history = session.action_history

        with self.assertRaises(TypeError):
            history[0]["robot_1"] = Action.WAIT  # type: ignore[index]

    def test_controller_failure_falls_back_to_wait_and_replays_warning(self) -> None:
        class RaisingController:
            def __init__(self) -> None:
                self.calls = 0

            def choose_action(
                self, observation: WorldObservation, robot_id: str
            ) -> Action:
                self.calls += 1
                raise RuntimeError("policy unavailable")

        costs = ActionBatteryCosts(wait=0.5)
        scenario = InitialScenario(
            width=2,
            height=1,
            robot_configurations=(RobotConfiguration((0, 0), 2),),
            item_positions=(),
            seed=1,
            action_battery_costs=costs,
        )
        controller = RaisingController()
        session = SimulationSession(
            scenario,
            max_steps=2,
            controller_overrides={"robot_1": controller},
        )

        result = session.step_forward()["robot_1"]  # type: ignore[index]
        self.assertIs(result.action, Action.WAIT)
        self.assertEqual(result.battery_spent, 0.5)
        self.assertEqual(controller.calls, 1)
        self.assertIn("policy unavailable", session.controller_errors["robot_1"])
        self.assertEqual(len(session.controller_error_history), 1)

        session.step_back()
        self.assertEqual(session.controller_errors, {})
        session.step_forward()
        self.assertEqual(controller.calls, 1)
        self.assertIn("policy unavailable", session.controller_errors["robot_1"])

    def test_invalid_controller_result_also_becomes_wait(self) -> None:
        class InvalidController:
            def choose_action(
                self, observation: WorldObservation, robot_id: str
            ) -> Action:
                return "move_right"  # type: ignore[return-value]

        scenario = InitialScenario(
            2,
            1,
            (RobotConfiguration((0, 0)),),
            (),
            1,
        )
        session = SimulationSession(
            scenario,
            controller_overrides={"robot_1": InvalidController()},
        )

        result = session.step_forward()["robot_1"]  # type: ignore[index]
        self.assertIs(result.action, Action.WAIT)
        self.assertIn("expected an Action", session.controller_errors["robot_1"])

    def test_registry_constructs_selected_controller_with_robot_context(self) -> None:
        contexts: list[ControllerFactoryContext] = []

        class WaitController:
            def choose_action(
                self, observation: WorldObservation, robot_id: str
            ) -> Action:
                return Action.WAIT

        def factory(context: ControllerFactoryContext) -> WaitController:
            contexts.append(context)
            return WaitController()

        registry = ControllerRegistry()
        registry.register("custom", "Custom", factory)
        scenario = InitialScenario(
            2,
            1,
            (RobotConfiguration((0, 0), controller_key="custom"),),
            (),
            73,
        )
        session = SimulationSession(scenario, controller_registry=registry)

        self.assertEqual(
            contexts,
            [ControllerFactoryContext(seed=73, robot_id="robot_1", max_steps=200)],
        )
        self.assertEqual(session.controller_key_for("robot_1"), "custom")
        self.assertEqual(session.controller_registry.display_name("custom"), "Custom")
        self.assertEqual(session.controller_display_name_for("robot_1"), "Custom")

    def test_pickup_inventory_and_battery_are_restored_by_rewind(self) -> None:
        scenario = InitialScenario(
            width=3,
            height=1,
            robot_configurations=(
                RobotConfiguration((0, 0), 7, controller_key="nearest_item"),
            ),
            item_positions=((2, 0),),
            seed=4,
            action_battery_costs=ActionBatteryCosts(movement=2, pickup=3),
            delivery_destination_positions=((1, 0),),
        )
        session = SimulationSession(scenario, max_steps=3)

        for _ in range(3):
            session.step_forward()
        robot = session.world.get_entities(Robot)[0]
        self.assertEqual(robot.position, (2, 0))
        self.assertEqual(robot.battery_level, 0.0)
        self.assertEqual(robot.carried_item_id, "item_1")
        self.assertEqual(session.world.get_entities(Item), ())

        session.step_back()
        restored = session.world.get_entities(Robot)[0]
        self.assertEqual(restored.position, (2, 0))
        self.assertEqual(restored.battery_level, 3.0)
        self.assertIsNone(restored.carried_item_id)
        self.assertEqual(session.world.get_entities(Item)[0].position, (2, 0))

        session.step_forward()
        replayed = session.world.get_entities(Robot)[0]
        self.assertEqual(replayed.carried_item_id, "item_1")
        self.assertEqual(replayed.battery_level, 0.0)

    def test_drop_inventory_item_and_battery_are_replayed_without_controller(self) -> None:
        class ScriptedController:
            def __init__(self) -> None:
                self.calls = 0
                self.actions = (
                    Action.MOVE_RIGHT,
                    Action.PICK_UP,
                    Action.MOVE_RIGHT,
                    Action.DROP,
                )

            def choose_action(
                self, observation: WorldObservation, robot_id: str
            ) -> Action:
                action = self.actions[self.calls]
                self.calls += 1
                return action

        scenario = InitialScenario(
            width=4,
            height=1,
            robot_configurations=(RobotConfiguration((0, 0), 15),),
            item_positions=((1, 0),),
            seed=9,
            action_battery_costs=ActionBatteryCosts(
                movement=2,
                pickup=3,
                wait=0,
                drop=4,
            ),
            delivery_destination_positions=((2, 0),),
        )
        controller = ScriptedController()
        session = SimulationSession(
            scenario,
            max_steps=4,
            controller_overrides={"robot_1": controller},
        )

        session.step_forward()
        pickup_results = session.step_forward()
        after_pickup = session.world.get_entities(Robot)[0]
        self.assertEqual(
            pickup_results["robot_1"].picked_up_item_id,  # type: ignore[index]
            "item_1",
        )
        self.assertEqual(after_pickup.position, (1, 0))
        self.assertEqual(after_pickup.battery_level, 10.0)
        self.assertEqual(after_pickup.carried_item_id, "item_1")
        self.assertEqual(session.world.get_entities(Item), ())

        session.step_forward()
        drop_results = session.step_forward()
        dropped_robot = session.world.get_entities(Robot)[0]
        dropped_items = session.world.get_entities(Item)
        recorded_history = tuple(dict(actions) for actions in session.action_history)

        self.assertEqual(controller.calls, 4)
        self.assertEqual(dropped_robot.position, (2, 0))
        self.assertEqual(dropped_robot.battery_level, 4.0)
        self.assertIsNone(dropped_robot.carried_item_id)
        self.assertEqual(
            tuple((item.item_id, item.position) for item in dropped_items),
            (("item_1", (2, 0)),),
        )
        destination = session.world.get_entities(DeliveryDestination)[0]
        self.assertEqual(
            (destination.position, destination.target_item_id),
            ((2, 0), "item_1"),
        )
        self.assertEqual(
            drop_results["robot_1"].dropped_item_id,  # type: ignore[index]
            "item_1",
        )
        self.assertEqual(
            recorded_history,
            (
                {"robot_1": Action.MOVE_RIGHT},
                {"robot_1": Action.PICK_UP},
                {"robot_1": Action.MOVE_RIGHT},
                {"robot_1": Action.DROP},
            ),
        )

        self.assertTrue(session.step_back())
        before_drop = session.world.get_entities(Robot)[0]
        self.assertEqual(session.cursor, 3)
        self.assertEqual(before_drop.position, (2, 0))
        self.assertEqual(before_drop.battery_level, 8.0)
        self.assertEqual(before_drop.carried_item_id, "item_1")
        self.assertEqual(session.world.get_entities(Item), ())
        self.assertEqual(
            session.world.get_entities(DeliveryDestination)[0].position,
            (2, 0),
        )

        replayed_results = session.step_forward()
        replayed_robot = session.world.get_entities(Robot)[0]
        replayed_items = session.world.get_entities(Item)
        self.assertEqual(controller.calls, 4)
        self.assertEqual(replayed_robot.battery_level, 4.0)
        self.assertIsNone(replayed_robot.carried_item_id)
        self.assertEqual(
            tuple((item.item_id, item.position) for item in replayed_items),
            (("item_1", (2, 0)),),
        )
        self.assertEqual(
            replayed_results["robot_1"].dropped_item_id,  # type: ignore[index]
            "item_1",
        )
        self.assertEqual(
            tuple(dict(actions) for actions in session.action_history),
            recorded_history,
        )

    def test_shared_multi_agent_override_runs_once_per_new_timestep(self) -> None:
        class TeamController:
            def __init__(self) -> None:
                self.calls = 0

            def choose_actions(
                self,
                observation: WorldObservation,
                robot_ids: tuple[str, ...],
            ) -> dict[str, Action]:
                self.calls += 1
                return {robot_id: Action.WAIT for robot_id in robot_ids}

        scenario = InitialScenario(
            3,
            1,
            (
                RobotConfiguration((0, 0)),
                RobotConfiguration((2, 0)),
            ),
            (),
            8,
        )
        joint = TeamController()
        adapter = MultiAgentControllerAdapter(
            joint,
            ("robot_1", "robot_2"),
        )
        session = SimulationSession(
            scenario,
            max_steps=3,
            controller_overrides={
                "robot_1": adapter,
                "robot_2": adapter,
            },
        )

        self.assertEqual(
            session.controller_display_name_for("robot_1"),
            "MultiAgentControllerAdapter (override)",
        )

        session.step_forward()
        session.step_forward()
        self.assertEqual(joint.calls, 2)
        session.step_back()
        session.step_forward()
        self.assertEqual(joint.calls, 2)
        session.step_forward()
        self.assertEqual(joint.calls, 3)


if __name__ == "__main__":
    unittest.main()
