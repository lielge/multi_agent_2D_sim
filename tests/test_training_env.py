from __future__ import annotations

import unittest
from dataclasses import replace

from multi_agent_sim.actions import Action, ActionBatteryCosts, ActionFailureReason
from multi_agent_sim.controllers import (
    DeliveryDestinationObservation,
    ItemObservation,
    RobotObservation,
    WorldObservation,
)
from multi_agent_sim.session import InitialScenario, RobotConfiguration
from multi_agent_sim.training_env import (
    ACTION_ORDER,
    FEATURE_DIM,
    FixedSlotEncoder,
    FixedSlotTeamEnv,
    action_from_index,
    action_index,
    action_to_one_hot,
    one_hot_to_action,
    validate_one_hot_command,
)


def _scenario(seed: int = 7) -> InitialScenario:
    return InitialScenario(
        width=3,
        height=2,
        robot_configurations=(
            RobotConfiguration(position=(0, 0), battery_level=20),
        ),
        item_positions=((1, 0), (2, 0)),
        delivery_destination_positions=((0, 1), (1, 1)),
        action_battery_costs=ActionBatteryCosts(
            movement=2,
            pickup=3,
            wait=0,
            drop=4,
        ),
        seed=seed,
    )


class OneHotCommandTests(unittest.TestCase):
    def test_action_order_and_round_trip_are_stable(self) -> None:
        self.assertEqual(
            ACTION_ORDER,
            (
                Action.MOVE_UP,
                Action.MOVE_DOWN,
                Action.MOVE_LEFT,
                Action.MOVE_RIGHT,
                Action.PICK_UP,
                Action.WAIT,
                Action.DROP,
            ),
        )
        for expected_index, action in enumerate(ACTION_ORDER):
            command = action_to_one_hot(action)
            self.assertIsInstance(command, tuple)
            self.assertEqual(len(command), 7)
            self.assertEqual(sum(command), 1)
            self.assertEqual(command[expected_index], 1)
            self.assertEqual(action_index(action), expected_index)
            self.assertIs(action_from_index(expected_index), action)
            self.assertIs(one_hot_to_action(command), action)

    def test_command_validation_rejects_non_one_hot_values(self) -> None:
        invalid_commands = (
            (1, 0),
            (0, 0, 0, 0, 0, 0, 0),
            (1, 1, 0, 0, 0, 0, 0),
            (True, 0, 0, 0, 0, 0, 0),
            (1.0, 0, 0, 0, 0, 0, 0),
            (2, 0, 0, 0, 0, 0, 0),
        )
        for command in invalid_commands:
            with self.subTest(command=command), self.assertRaises(ValueError):
                validate_one_hot_command(command)


class FixedSlotEncoderTests(unittest.TestCase):
    def test_exact_shape_padding_presence_and_controlled_identity(self) -> None:
        scenario = _scenario()
        observation = WorldObservation.from_world(scenario.create_world())
        encoded = FixedSlotEncoder(observation, max_steps=20).encode(observation)

        self.assertEqual(FEATURE_DIM, 155)
        self.assertEqual(len(encoded.rows), 4)
        self.assertTrue(all(len(row) == FEATURE_DIM for row in encoded.rows))
        self.assertEqual(encoded.robot_ids, ("robot_1", None, None, None))
        self.assertEqual(encoded.robot_presence, (1, 0, 0, 0))
        self.assertEqual(encoded.active_robot_ids, ("robot_1",))
        self.assertEqual(encoded.rows[0][-4:], (1.0, 0.0, 0.0, 0.0))
        self.assertEqual(encoded.rows[1], (0.0,) * FEATURE_DIM)
        self.assertEqual(encoded.action_masks[1], (False,) * 7)
        self.assertIs(encoded.row_for("robot_1"), encoded.rows[0])
        self.assertIs(encoded.action_mask_for("robot_1"), encoded.action_masks[0])
        with self.assertRaises(KeyError):
            encoded.row_for("unknown")

    def test_feature_layout_contains_normalized_slots_world_and_identity(self) -> None:
        scenario = _scenario()
        observation = WorldObservation.from_world(scenario.create_world())
        encoded = FixedSlotEncoder(observation, max_steps=20).encode(observation)
        row = encoded.rows[0]

        # First robot slot: presence, normalized position, team-normalized
        # battery, then the eight-way carried-item indicator.
        self.assertEqual(row[0:4], (1.0, 0.0, 0.0, 1.0))
        self.assertEqual(row[4:12], (0.0,) * 8)
        self.assertEqual(row[12:48], (0.0,) * 36)

        # Item 1 occupies the first logical item slot and points to (0, 1).
        self.assertEqual(
            row[48:60],
            (1.0, 0.5, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
        # Two item slots are present; the remaining six are zero padding.
        self.assertEqual(row[72:144], (0.0,) * 72)
        # T/T_max, coordinate scales, and four team-energy-normalized costs.
        self.assertEqual(
            row[144:151],
            (0.0, 0.5, 1.0, 0.1, 0.15, 0.0, 0.2),
        )
        self.assertEqual(row[151:155], (1.0, 0.0, 0.0, 0.0))

    def test_item_slot_stays_stable_and_carried_item_is_not_padding(self) -> None:
        env = FixedSlotTeamEnv(lambda seed: _scenario(seed), max_steps=10)
        initial = env.reset(seed=11)
        moved = env.step({"robot_1": action_to_one_hot(Action.MOVE_RIGHT)})
        carried = env.step({"robot_1": action_to_one_hot(Action.PICK_UP)})
        dropped = env.step({"robot_1": action_to_one_hot(Action.DROP)})

        self.assertEqual(initial.rows[0][48], 1.0)
        self.assertEqual(moved.observation.rows[0][48], 1.0)
        row = carried.observation.rows[0]
        self.assertEqual(row[4:12], (1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        self.assertEqual(row[48], 1.0)
        self.assertEqual(row[49:51], (0.5, 0.0))
        self.assertEqual(row[53:56], (0.0, 0.0, 1.0))
        self.assertEqual(row[56:60], (1.0, 0.0, 0.0, 0.0))
        self.assertEqual(row[60], 1.0)  # Item 2 remains in logical slot 2.
        self.assertEqual(row[72:84], (0.0,) * 12)  # Slot 3 is actual padding.
        dropped_row = dropped.observation.rows[0]
        self.assertEqual(dropped_row[48], 1.0)
        self.assertEqual(dropped_row[49:51], (0.5, 0.0))
        self.assertEqual(dropped_row[53:56], (1.0, 0.0, 0.0))
        self.assertEqual(dropped_row[60], 1.0)

    def test_action_mask_is_local_advice_and_wait_is_available(self) -> None:
        env = FixedSlotTeamEnv(lambda seed: _scenario(seed), max_steps=10)
        observation = env.reset(seed=1)
        mask = observation.action_mask_for("robot_1")

        self.assertFalse(mask[action_index(Action.MOVE_UP)])
        self.assertTrue(mask[action_index(Action.MOVE_DOWN)])
        self.assertFalse(mask[action_index(Action.MOVE_LEFT)])
        self.assertTrue(mask[action_index(Action.MOVE_RIGHT)])
        self.assertFalse(mask[action_index(Action.PICK_UP)])
        self.assertTrue(mask[action_index(Action.WAIT)])
        self.assertFalse(mask[action_index(Action.DROP)])

        # The mask is advisory: an unmasked out-of-bounds command reaches the
        # simulator and is charged according to normal world semantics.
        transition = env.step({"robot_1": action_to_one_hot(Action.MOVE_LEFT)})
        result = transition.action_results["robot_1"]
        self.assertFalse(result.success)
        self.assertIs(result.failure_reason, ActionFailureReason.OUT_OF_BOUNDS)
        self.assertEqual(result.battery_spent, 2.0)

    def test_action_mask_protects_a_correctly_delivered_item(self) -> None:
        observation = WorldObservation(
            width=3,
            height=2,
            timestep=0,
            robots=(RobotObservation("robot_1", (1, 0), 10),),
            items=(
                ItemObservation("item_1", (1, 0)),
                ItemObservation("item_2", (2, 0)),
            ),
            destinations=(
                DeliveryDestinationObservation(
                    "destination_1", (1, 0), "item_1"
                ),
                DeliveryDestinationObservation(
                    "destination_2", (2, 1), "item_2"
                ),
            ),
            action_battery_costs=ActionBatteryCosts(),
        )

        encoded = FixedSlotEncoder(observation, max_steps=10).encode(observation)

        self.assertFalse(
            encoded.action_mask_for("robot_1")[action_index(Action.PICK_UP)]
        )

    def test_rejects_an_observation_beyond_the_episode_horizon(self) -> None:
        observation = WorldObservation.from_world(_scenario().create_world())
        encoder = FixedSlotEncoder(observation, max_steps=2)

        with self.assertRaisesRegex(ValueError, "maximum timestep"):
            encoder.encode(replace(observation, timestep=3))

    def test_rejects_unsupported_robot_and_item_counts(self) -> None:
        five_robot_scenario = InitialScenario(
            width=5,
            height=1,
            robot_configurations=tuple(
                RobotConfiguration(position=(index, 0)) for index in range(5)
            ),
            item_positions=(),
            delivery_destination_positions=(),
            seed=1,
        )
        with self.assertRaisesRegex(ValueError, "one and four robots"):
            FixedSlotEncoder(
                WorldObservation.from_world(five_robot_scenario.create_world()),
                max_steps=10,
            )

        nine_item_scenario = InitialScenario(
            width=7,
            height=4,
            robot_configurations=(RobotConfiguration(position=(0, 0)),),
            item_positions=tuple((index + 1, 0) for index in range(6))
            + tuple((index, 1) for index in range(3)),
            delivery_destination_positions=tuple((index, 2) for index in range(7))
            + ((0, 3), (1, 3)),
            seed=1,
        )
        with self.assertRaisesRegex(ValueError, "at most eight items"):
            FixedSlotEncoder(
                WorldObservation.from_world(nine_item_scenario.create_world()),
                max_steps=10,
            )


class FixedSlotTeamEnvTests(unittest.TestCase):
    def test_reset_is_seeded_and_each_batch_is_one_world_timestep(self) -> None:
        seen_seeds: list[int] = []

        def factory(seed: int) -> InitialScenario:
            seen_seeds.append(seed)
            return _scenario(seed)

        env = FixedSlotTeamEnv(factory, max_steps=5)
        first = env.reset(seed=101)
        self.assertEqual(env.world.timestep, 0)
        transition = env.step({"robot_1": action_to_one_hot(Action.WAIT)})
        self.assertEqual(env.world.timestep, 1)
        second = env.reset(seed=101)

        self.assertEqual(seen_seeds, [101, 101])
        self.assertEqual(first, second)
        self.assertEqual(transition.metrics.elapsed_steps, 1)
        self.assertEqual(transition.item_progress_reward, 0.0)
        self.assertEqual(transition.observation.rows[0][144], 0.2)
        self.assertEqual(env.raw_observation.timestep, 0)

    def test_step_requires_exactly_one_command_per_active_robot(self) -> None:
        scenario = InitialScenario(
            width=3,
            height=2,
            robot_configurations=(
                RobotConfiguration(position=(0, 0)),
                RobotConfiguration(position=(1, 0)),
            ),
            item_positions=((2, 0),),
            delivery_destination_positions=((2, 1),),
            seed=1,
        )
        env = FixedSlotTeamEnv(lambda seed: scenario, max_steps=5)
        env.reset(seed=1)
        wait = action_to_one_hot(Action.WAIT)

        with self.assertRaisesRegex(ValueError, "missing robot"):
            env.step({"robot_1": wait})
        with self.assertRaisesRegex(ValueError, "unknown robot"):
            env.step({"robot_1": wait, "robot_2": wait, "robot_3": wait})
        with self.assertRaises(ValueError):
            env.step({"robot_1": wait, "robot_2": (0,) * 7})

        transition = env.step({"robot_1": wait, "robot_2": wait})
        self.assertEqual(transition.metrics.elapsed_steps, 1)
        self.assertEqual(set(transition.action_results), {"robot_1", "robot_2"})

    def test_terminal_and_zero_item_episodes_cannot_be_stepped(self) -> None:
        timeout_env = FixedSlotTeamEnv(lambda seed: _scenario(seed), max_steps=1)
        timeout_env.reset(seed=1)
        timeout = timeout_env.step({"robot_1": action_to_one_hot(Action.WAIT)})
        self.assertTrue(timeout.terminated)
        with self.assertRaisesRegex(RuntimeError, "terminated"):
            timeout_env.step({"robot_1": action_to_one_hot(Action.WAIT)})

        empty_scenario = InitialScenario(
            width=1,
            height=1,
            robot_configurations=(RobotConfiguration(position=(0, 0)),),
            item_positions=(),
            delivery_destination_positions=(),
            seed=1,
        )
        empty_env = FixedSlotTeamEnv(lambda seed: empty_scenario, max_steps=5)
        observation = empty_env.reset(seed=1)
        self.assertEqual(observation.active_robot_ids, ("robot_1",))
        self.assertTrue(empty_env.terminated)
        self.assertTrue(empty_env.metrics.success)
        with self.assertRaisesRegex(RuntimeError, "terminated"):
            empty_env.step({"robot_1": action_to_one_hot(Action.WAIT)})

    def test_reset_rejects_a_factory_returning_wrong_type(self) -> None:
        env = FixedSlotTeamEnv(lambda seed: object(), max_steps=5)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "InitialScenario"):
            env.reset(seed=1)


if __name__ == "__main__":
    unittest.main()
