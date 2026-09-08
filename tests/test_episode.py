from __future__ import annotations

from dataclasses import FrozenInstanceError
import unittest

from multi_agent_sim.actions import (
    Action,
    ActionBatteryCosts,
    ActionFailureReason,
    ActionResult,
)
from multi_agent_sim.controllers import (
    DeliveryDestinationObservation,
    ItemObservation,
    RobotObservation,
    WorldObservation,
)
from multi_agent_sim.episode import (
    EpisodeCostWeights,
    EpisodeRunner,
    TeamCostEvaluator,
    delivery_status,
)
from multi_agent_sim.session import InitialScenario, RobotConfiguration


def make_observation(
    *,
    timestep: int = 0,
    robots: tuple[RobotObservation, ...] = (
        RobotObservation("robot_1", (0, 0), 10),
    ),
    items: tuple[ItemObservation, ...] = (
        ItemObservation("item_1", (1, 0)),
    ),
    destinations: tuple[DeliveryDestinationObservation, ...] = (
        DeliveryDestinationObservation("destination_1", (2, 0), "item_1"),
    ),
    costs: ActionBatteryCosts | None = None,
) -> WorldObservation:
    return WorldObservation(
        width=5,
        height=3,
        timestep=timestep,
        robots=robots,
        items=items,
        action_battery_costs=costs or ActionBatteryCosts(),
        destinations=destinations,
    )


def make_result(
    previous_robot: RobotObservation,
    next_robot: RobotObservation,
    *,
    action: Action = Action.WAIT,
    success: bool = True,
    failure_reason: ActionFailureReason | None = None,
    picked_up_item_id: str | None = None,
    dropped_item_id: str | None = None,
) -> ActionResult:
    return ActionResult(
        action=action,
        success=success,
        start_position=previous_robot.position,
        end_position=next_robot.position,
        failure_reason=failure_reason,
        battery_before=previous_robot.battery_level,
        battery_after=next_robot.battery_level,
        battery_spent=previous_robot.battery_level - next_robot.battery_level,
        picked_up_item_id=picked_up_item_id,
        dropped_item_id=dropped_item_id,
    )


class EpisodeCostWeightsTests(unittest.TestCase):
    def test_defaults_custom_values_and_validation(self) -> None:
        defaults = EpisodeCostWeights()
        self.assertEqual(
            (
                defaults.failure,
                defaults.undelivered,
                defaults.time,
                defaults.energy,
                defaults.item_progress,
                defaults.delivery,
            ),
            (10.0, 2.0, 0.5, 0.5, 0.1, 1.0),
        )
        custom = EpisodeCostWeights(8, 3, 1, 0, 0.25, 0.75)
        self.assertEqual(custom.failure_weight, 8.0)
        self.assertEqual(custom.undelivered_weight, 3.0)
        self.assertEqual(custom.time_weight, 1.0)
        self.assertEqual(custom.energy_weight, 0.0)
        self.assertEqual(custom.item_progress_weight, 0.25)
        self.assertEqual(custom.delivery_weight, 0.75)
        with self.assertRaises(FrozenInstanceError):
            custom.failure = 1  # type: ignore[misc]

        for field_name in (
            "failure",
            "undelivered",
            "time",
            "energy",
            "item_progress",
            "delivery",
        ):
            for value in (-1, float("inf"), float("nan"), True, "1"):
                with self.subTest(field_name=field_name, value=value):
                    arguments = {field_name: value}
                    with self.assertRaises(ValueError):
                        EpisodeCostWeights(**arguments)  # type: ignore[arg-type]


class TeamCostEvaluatorTests(unittest.TestCase):
    def test_early_success_has_time_cost_and_no_terminal_penalty(self) -> None:
        previous = make_observation()
        next_observation = make_observation(
            timestep=1,
            items=(ItemObservation("item_1", (2, 0)),),
        )
        evaluator = TeamCostEvaluator(previous, max_steps=4)

        evaluation = evaluator.evaluate_transition(
            previous,
            next_observation,
            {
                "robot_1": make_result(
                    previous.robots[0],
                    next_observation.robots[0],
                )
            },
        )

        self.assertTrue(evaluation.terminated)
        self.assertTrue(evaluation.metrics.success)
        self.assertFalse(evaluation.metrics.timed_out)
        self.assertEqual(evaluation.metrics.delivered_items, 1)
        self.assertEqual(evaluation.metrics.undelivered_items, 0)
        self.assertEqual(evaluation.metrics.failure_cost, 0.0)
        self.assertEqual(evaluation.metrics.undelivered_cost, 0.0)
        self.assertAlmostEqual(evaluation.item_progress_reward, 1 / 30)
        self.assertEqual(evaluation.delivery_reward, 1.0)
        self.assertAlmostEqual(
            evaluation.metrics.initial_route_potential, 1 / 3
        )
        self.assertEqual(evaluation.metrics.normalized_item_distance, 0.0)
        self.assertAlmostEqual(evaluation.metrics.route_progress_cost, -1 / 30)
        self.assertEqual(evaluation.metrics.delivery_cost, -1.0)
        self.assertAlmostEqual(evaluation.reward, -0.125 + 1 / 30 + 1.0)
        self.assertAlmostEqual(evaluation.metrics.total_cost, 0.125 - 1 / 30 - 1.0)
        self.assertAlmostEqual(
            evaluation.metrics.cumulative_reward,
            -evaluation.metrics.total_cost,
        )

    def test_timeout_with_items_at_incorrect_destinations(self) -> None:
        destinations = (
            DeliveryDestinationObservation("destination_1", (3, 0), "item_1"),
            DeliveryDestinationObservation("destination_2", (4, 0), "item_2"),
        )
        previous = make_observation(
            items=(
                ItemObservation("item_1", (1, 0)),
                ItemObservation("item_2", (2, 0)),
            ),
            destinations=destinations,
        )
        swapped = make_observation(
            timestep=1,
            items=(
                ItemObservation("item_1", (4, 0)),
                ItemObservation("item_2", (3, 0)),
            ),
            destinations=destinations,
        )
        evaluator = TeamCostEvaluator(previous, max_steps=1)

        evaluation = evaluator.evaluate_transition(
            previous,
            swapped,
            {"robot_1": make_result(previous.robots[0], swapped.robots[0])},
        )

        metrics = evaluation.metrics
        self.assertTrue(metrics.terminated)
        self.assertFalse(metrics.success)
        self.assertTrue(metrics.timed_out)
        self.assertEqual(metrics.delivered_items, 0)
        self.assertEqual(metrics.undelivered_items, 2)
        self.assertEqual(metrics.undelivered_fraction, 1.0)
        self.assertEqual(metrics.failure, 1)
        self.assertEqual(metrics.failure_cost, 10.0)
        self.assertEqual(metrics.undelivered_cost, 2.0)
        self.assertEqual(metrics.time_cost, 0.5)
        self.assertAlmostEqual(metrics.item_progress_cost, 1 / 60)
        self.assertAlmostEqual(metrics.total_cost, 12.5 + 1 / 60)
        self.assertAlmostEqual(evaluation.reward, -12.5 - 1 / 60)

    def test_picking_up_a_delivered_item_removes_its_delivery_credit(self) -> None:
        destinations = (
            DeliveryDestinationObservation("destination_1", (1, 0), "item_1"),
            DeliveryDestinationObservation("destination_2", (4, 0), "item_2"),
        )
        previous_robot = RobotObservation("robot_1", (1, 0), 10)
        previous = make_observation(
            robots=(previous_robot,),
            items=(
                ItemObservation("item_1", (1, 0)),
                ItemObservation("item_2", (2, 0)),
            ),
            destinations=destinations,
        )
        next_robot = RobotObservation("robot_1", (1, 0), 9, "item_1")
        after_pickup = make_observation(
            timestep=1,
            robots=(next_robot,),
            items=(ItemObservation("item_2", (2, 0)),),
            destinations=destinations,
        )
        evaluator = TeamCostEvaluator(previous, max_steps=2)
        self.assertEqual(evaluator.metrics.delivered_items, 1)

        evaluation = evaluator.evaluate_transition(
            previous,
            after_pickup,
            {
                "robot_1": make_result(
                    previous_robot,
                    next_robot,
                    action=Action.PICK_UP,
                    picked_up_item_id="item_1",
                )
            },
        )

        self.assertFalse(evaluation.terminated)
        self.assertEqual(evaluation.metrics.delivered_items, 0)
        self.assertEqual(evaluation.metrics.undelivered_items, 2)
        self.assertEqual(evaluation.item_progress_reward, 0.0)
        self.assertEqual(evaluation.delivery_reward, -0.5)
        self.assertEqual(evaluation.metrics.delivery_cost, 0.5)
        self.assertAlmostEqual(evaluation.reward, -0.8)

    def test_custom_weights_and_partial_undelivered_fraction(self) -> None:
        destinations = (
            DeliveryDestinationObservation("destination_1", (3, 0), "item_1"),
            DeliveryDestinationObservation("destination_2", (4, 0), "item_2"),
        )
        previous = make_observation(
            items=(
                ItemObservation("item_1", (1, 0)),
                ItemObservation("item_2", (2, 0)),
            ),
            destinations=destinations,
        )
        partial = make_observation(
            timestep=1,
            items=(
                ItemObservation("item_1", (3, 0)),
                ItemObservation("item_2", (2, 0)),
            ),
            destinations=destinations,
        )
        evaluator = TeamCostEvaluator(
            previous,
            max_steps=1,
            cost_weights=EpisodeCostWeights(7, 4, 2, 0, 0.3),
        )

        evaluation = evaluator.evaluate_transition(
            previous,
            partial,
            {"robot_1": make_result(previous.robots[0], partial.robots[0])},
        )

        self.assertEqual(evaluation.metrics.undelivered_fraction, 0.5)
        self.assertEqual(evaluation.metrics.failure_cost, 7.0)
        self.assertEqual(evaluation.metrics.undelivered_cost, 2.0)
        self.assertEqual(evaluation.metrics.time_cost, 2.0)
        self.assertAlmostEqual(evaluation.metrics.item_progress_cost, -0.075)
        self.assertEqual(evaluation.metrics.delivery_cost, -0.5)
        self.assertAlmostEqual(evaluation.metrics.total_cost, 10.425)

    def test_signed_progress_rewards_closer_and_penalizes_farther_without_cycles(
        self,
    ) -> None:
        destination = (
            DeliveryDestinationObservation("destination_1", (4, 0), "item_1"),
        )
        initial = make_observation(
            items=(ItemObservation("item_1", (2, 0)),),
            destinations=destination,
        )
        closer = make_observation(
            timestep=1,
            robots=(RobotObservation("robot_1", (1, 0), 10),),
            items=(ItemObservation("item_1", (2, 0)),),
            destinations=destination,
        )
        restored = make_observation(
            timestep=2,
            robots=(RobotObservation("robot_1", (0, 0), 10),),
            items=(ItemObservation("item_1", (2, 0)),),
            destinations=destination,
        )
        evaluator = TeamCostEvaluator(initial, max_steps=3)

        closer_step = evaluator.evaluate_transition(
            initial,
            closer,
            {
                "robot_1": make_result(
                    initial.robots[0], closer.robots[0], action=Action.MOVE_RIGHT
                )
            },
        )
        farther_step = evaluator.evaluate_transition(
            closer,
            restored,
            {
                "robot_1": make_result(
                    closer.robots[0], restored.robots[0], action=Action.MOVE_LEFT
                )
            },
        )

        self.assertAlmostEqual(closer_step.item_progress_reward, 1 / 60)
        self.assertAlmostEqual(farther_step.item_progress_reward, -1 / 60)
        self.assertEqual(farther_step.metrics.item_progress_cost, 0.0)
        self.assertAlmostEqual(farther_step.metrics.total_cost, 1 / 3)
        self.assertAlmostEqual(
            farther_step.metrics.cumulative_reward,
            -farther_step.metrics.total_cost,
        )

    def test_carried_item_uses_its_carrier_position_for_progress(self) -> None:
        destination = (
            DeliveryDestinationObservation("destination_1", (4, 0), "item_1"),
        )
        previous_robot = RobotObservation("robot_1", (1, 0), 10, "item_1")
        next_robot = RobotObservation("robot_1", (2, 0), 10, "item_1")
        previous = make_observation(
            robots=(previous_robot,), items=(), destinations=destination
        )
        next_observation = make_observation(
            timestep=1,
            robots=(next_robot,),
            items=(),
            destinations=destination,
        )
        evaluation = TeamCostEvaluator(previous, max_steps=2).evaluate_transition(
            previous,
            next_observation,
            {"robot_1": make_result(previous_robot, next_robot)},
        )

        self.assertAlmostEqual(evaluation.item_progress_reward, 1 / 60)
        self.assertAlmostEqual(evaluation.metrics.item_progress_cost, -1 / 60)

    def test_pickup_and_wrong_drop_are_continuous_in_route_potential(self) -> None:
        destination = (
            DeliveryDestinationObservation("destination_1", (4, 0), "item_1"),
        )
        on_item_robot = RobotObservation("robot_1", (2, 0), 10)
        initial = make_observation(
            robots=(on_item_robot,),
            items=(ItemObservation("item_1", (2, 0)),),
            destinations=destination,
        )
        carrying_robot = RobotObservation("robot_1", (2, 0), 9, "item_1")
        carrying = make_observation(
            timestep=1,
            robots=(carrying_robot,),
            items=(),
            destinations=destination,
        )
        dropped_robot = RobotObservation("robot_1", (2, 0), 8)
        wrong_drop = make_observation(
            timestep=2,
            robots=(dropped_robot,),
            items=(ItemObservation("item_1", (2, 0)),),
            destinations=destination,
        )
        evaluator = TeamCostEvaluator(initial, max_steps=3)

        pickup = evaluator.evaluate_transition(
            initial,
            carrying,
            {
                "robot_1": make_result(
                    on_item_robot,
                    carrying_robot,
                    action=Action.PICK_UP,
                    picked_up_item_id="item_1",
                )
            },
        )
        drop = evaluator.evaluate_transition(
            carrying,
            wrong_drop,
            {
                "robot_1": make_result(
                    carrying_robot,
                    dropped_robot,
                    action=Action.DROP,
                    dropped_item_id="item_1",
                )
            },
        )

        self.assertEqual(pickup.item_progress_reward, 0.0)
        self.assertEqual(drop.item_progress_reward, 0.0)
        self.assertEqual(pickup.delivery_reward, 0.0)
        self.assertEqual(drop.delivery_reward, 0.0)

    def test_delivery_removal_and_redelivery_cycle_has_zero_net_bonus(self) -> None:
        destinations = (
            DeliveryDestinationObservation("destination_1", (1, 0), "item_1"),
            DeliveryDestinationObservation("destination_2", (4, 0), "item_2"),
        )
        initial_robot = RobotObservation("robot_1", (1, 0), 10)
        initial = make_observation(
            robots=(initial_robot,),
            items=(
                ItemObservation("item_1", (1, 0)),
                ItemObservation("item_2", (3, 0)),
            ),
            destinations=destinations,
        )
        carrying_robot = RobotObservation("robot_1", (1, 0), 9, "item_1")
        removed = make_observation(
            timestep=1,
            robots=(carrying_robot,),
            items=(ItemObservation("item_2", (3, 0)),),
            destinations=destinations,
        )
        dropped_robot = RobotObservation("robot_1", (1, 0), 8)
        restored = make_observation(
            timestep=2,
            robots=(dropped_robot,),
            items=(
                ItemObservation("item_1", (1, 0)),
                ItemObservation("item_2", (3, 0)),
            ),
            destinations=destinations,
        )
        evaluator = TeamCostEvaluator(initial, max_steps=3)

        pickup = evaluator.evaluate_transition(
            initial,
            removed,
            {
                "robot_1": make_result(
                    initial_robot,
                    carrying_robot,
                    action=Action.PICK_UP,
                    picked_up_item_id="item_1",
                )
            },
        )
        drop = evaluator.evaluate_transition(
            removed,
            restored,
            {
                "robot_1": make_result(
                    carrying_robot,
                    dropped_robot,
                    action=Action.DROP,
                    dropped_item_id="item_1",
                )
            },
        )

        self.assertEqual(pickup.delivery_reward, -0.5)
        self.assertEqual(drop.delivery_reward, 0.5)
        self.assertEqual(pickup.delivery_reward + drop.delivery_reward, 0.0)
        self.assertEqual(drop.metrics.delivery_cost, 0.0)

    def test_route_potential_uses_the_nearest_robot(self) -> None:
        costs = ActionBatteryCosts(movement=0)
        destination = (
            DeliveryDestinationObservation("destination_1", (0, 0), "item_1"),
        )
        initial_robots = (
            RobotObservation("robot_1", (0, 0), 10),
            RobotObservation("robot_2", (4, 0), 10),
        )
        initial = make_observation(
            robots=initial_robots,
            items=(ItemObservation("item_1", (3, 0)),),
            destinations=destination,
            costs=costs,
        )
        farther_robot_moved = (
            RobotObservation("robot_1", (1, 0), 10),
            initial_robots[1],
        )
        unchanged_nearest = make_observation(
            timestep=1,
            robots=farther_robot_moved,
            items=initial.items,
            destinations=destination,
            costs=costs,
        )
        nearest_robot_moved = (
            farther_robot_moved[0],
            RobotObservation("robot_2", (3, 0), 10),
        )
        approached = make_observation(
            timestep=2,
            robots=nearest_robot_moved,
            items=initial.items,
            destinations=destination,
            costs=costs,
        )
        evaluator = TeamCostEvaluator(initial, max_steps=3)

        first = evaluator.evaluate_transition(
            initial,
            unchanged_nearest,
            {
                robot.robot_id: make_result(robot, next_robot)
                for robot, next_robot in zip(
                    initial_robots, farther_robot_moved, strict=True
                )
            },
        )
        second = evaluator.evaluate_transition(
            unchanged_nearest,
            approached,
            {
                robot.robot_id: make_result(robot, next_robot)
                for robot, next_robot in zip(
                    farther_robot_moved, nearest_robot_moved, strict=True
                )
            },
        )

        self.assertEqual(first.item_progress_reward, 0.0)
        self.assertAlmostEqual(second.item_progress_reward, 1 / 60)

    def test_delivery_status_is_observation_only_and_zero_items_succeed(self) -> None:
        status = delivery_status(make_observation())
        self.assertEqual(status.total_items, 1)
        self.assertEqual(status.delivered_items, 0)
        self.assertFalse(status.success)

        empty = make_observation(items=(), destinations=())
        empty_status = delivery_status(empty)
        self.assertTrue(empty_status.success)
        self.assertEqual(empty_status.delivered_fraction, 0.0)

    def test_degenerate_grid_has_zero_progress_denominator(self) -> None:
        observation = WorldObservation(
            width=1,
            height=1,
            timestep=0,
            robots=(RobotObservation("robot_1", (0, 0), 0),),
            items=(ItemObservation("item_1", (0, 0)),),
            action_battery_costs=ActionBatteryCosts(),
            destinations=(
                DeliveryDestinationObservation(
                    "destination_1", (0, 0), "item_1"
                ),
            ),
        )
        metrics = TeamCostEvaluator(observation, max_steps=1).metrics

        self.assertTrue(metrics.success)
        self.assertEqual(metrics.initial_normalized_item_distance, 0.0)
        self.assertEqual(metrics.normalized_item_distance, 0.0)
        self.assertEqual(metrics.item_progress_cost, 0.0)

    def test_rejects_changed_rosters_and_inconsistent_battery_accounting(self) -> None:
        previous = make_observation()
        evaluator = TeamCostEvaluator(previous, max_steps=2)
        missing_item = make_observation(timestep=1, items=())
        with self.assertRaisesRegex(ValueError, "logical item roster"):
            evaluator.evaluate_transition(
                previous,
                missing_item,
                {
                    "robot_1": make_result(
                        previous.robots[0],
                        missing_item.robots[0],
                    )
                },
            )

        next_robot = RobotObservation("robot_1", (0, 0), 9)
        next_observation = make_observation(timestep=1, robots=(next_robot,))
        inconsistent = ActionResult(
            action=Action.WAIT,
            success=True,
            start_position=(0, 0),
            end_position=(0, 0),
            battery_before=10,
            battery_after=9,
            battery_spent=0,
        )
        with self.assertRaisesRegex(ValueError, "battery_spent"):
            evaluator.evaluate_transition(
                previous,
                next_observation,
                {"robot_1": inconsistent},
            )


class EpisodeRunnerTests(unittest.TestCase):
    @staticmethod
    def delivery_scenario(
        *,
        battery: float = 10,
        costs: ActionBatteryCosts | None = None,
    ) -> InitialScenario:
        return InitialScenario(
            width=4,
            height=2,
            robot_configurations=(RobotConfiguration((0, 0), battery),),
            item_positions=((1, 0),),
            seed=1,
            action_battery_costs=costs or ActionBatteryCosts(),
            delivery_destination_positions=((2, 0),),
        )

    def run_delivery(self, max_steps: int) -> tuple[EpisodeRunner, list[float]]:
        runner = EpisodeRunner(self.delivery_scenario(), max_steps=max_steps)
        rewards = []
        for action in (
            Action.MOVE_RIGHT,
            Action.PICK_UP,
            Action.MOVE_RIGHT,
            Action.DROP,
        ):
            transition = runner.step({"robot_1": action})
            rewards.append(transition.reward)
        return runner, rewards

    def test_success_on_final_allowed_step_and_return_equals_negative_cost(self) -> None:
        runner, rewards = self.run_delivery(max_steps=4)

        self.assertTrue(runner.terminated)
        self.assertTrue(runner.metrics.success)
        self.assertFalse(runner.metrics.timed_out)
        self.assertEqual(runner.metrics.elapsed_steps, 4)
        self.assertEqual(runner.metrics.battery_consumed, 4.0)
        self.assertEqual(runner.metrics.max_energy, 10.0)
        self.assertEqual(runner.metrics.time_cost, 0.5)
        self.assertEqual(runner.metrics.energy_cost, 0.2)
        self.assertAlmostEqual(sum(rewards), -runner.metrics.total_cost)
        with self.assertRaisesRegex(RuntimeError, "terminated"):
            runner.step({"robot_1": Action.WAIT})

    def test_success_before_timeout(self) -> None:
        runner, _ = self.run_delivery(max_steps=10)

        self.assertTrue(runner.metrics.success)
        self.assertEqual(runner.metrics.elapsed_steps, 4)
        self.assertFalse(runner.metrics.timed_out)

    def test_charged_failures_and_simultaneous_battery_use_are_summed(self) -> None:
        scenario = InitialScenario(
            width=3,
            height=2,
            robot_configurations=(
                RobotConfiguration((0, 0), 5),
                RobotConfiguration((2, 0), 5),
            ),
            item_positions=((0, 1),),
            seed=3,
            action_battery_costs=ActionBatteryCosts(movement=2),
            delivery_destination_positions=((2, 1),),
        )
        runner = EpisodeRunner(scenario, max_steps=1)

        transition = runner.step(
            {
                "robot_1": Action.MOVE_LEFT,
                "robot_2": Action.MOVE_RIGHT,
            }
        )

        self.assertEqual(
            tuple(
                result.failure_reason
                for result in transition.action_results.values()
            ),
            (
                ActionFailureReason.OUT_OF_BOUNDS,
                ActionFailureReason.OUT_OF_BOUNDS,
            ),
        )
        self.assertEqual(transition.metrics.elapsed_steps, 1)
        self.assertEqual(transition.metrics.battery_consumed, 4.0)
        self.assertEqual(transition.metrics.energy_cost, 0.2)
        self.assertAlmostEqual(transition.metrics.total_cost, 12.7)

    def test_insufficient_battery_failure_consumes_nothing(self) -> None:
        costs = ActionBatteryCosts(movement=2)
        runner = EpisodeRunner(
            self.delivery_scenario(battery=1, costs=costs),
            max_steps=1,
        )

        transition = runner.step({"robot_1": Action.MOVE_RIGHT})

        result = transition.action_results["robot_1"]
        self.assertIs(result.failure_reason, ActionFailureReason.INSUFFICIENT_BATTERY)
        self.assertEqual(result.battery_spent, 0.0)
        self.assertEqual(transition.metrics.battery_consumed, 0.0)
        self.assertEqual(transition.metrics.energy_cost, 0.0)

    def test_zero_items_is_immediately_successful(self) -> None:
        scenario = InitialScenario(
            width=2,
            height=1,
            robot_configurations=(RobotConfiguration((0, 0), 3),),
            item_positions=(),
            seed=1,
        )
        runner = EpisodeRunner(scenario, max_steps=5)

        self.assertTrue(runner.terminated)
        self.assertTrue(runner.metrics.success)
        self.assertEqual(runner.metrics.total_items, 0)
        self.assertEqual(runner.metrics.elapsed_steps, 0)
        self.assertEqual(runner.metrics.total_cost, 0.0)
        self.assertEqual(runner.metrics.cumulative_reward, 0.0)
        self.assertEqual(runner.metrics.initial_normalized_item_distance, 0.0)
        self.assertEqual(runner.metrics.normalized_item_distance, 0.0)
        self.assertEqual(runner.metrics.item_progress_cost, 0.0)

    def test_zero_total_battery_has_zero_energy_component(self) -> None:
        runner = EpisodeRunner(self.delivery_scenario(battery=0), max_steps=1)

        transition = runner.step({"robot_1": Action.WAIT})

        self.assertTrue(transition.metrics.timed_out)
        self.assertEqual(transition.metrics.max_energy, 0.0)
        self.assertEqual(transition.metrics.battery_consumed, 0.0)
        self.assertEqual(transition.metrics.energy_cost, 0.0)
        self.assertEqual(transition.metrics.total_cost, 12.5)

    def test_requires_a_complete_valid_action_mapping(self) -> None:
        runner = EpisodeRunner(self.delivery_scenario(), max_steps=3)
        with self.assertRaisesRegex(ValueError, "missing robot IDs"):
            runner.step({})
        with self.assertRaisesRegex(ValueError, "unknown robot IDs"):
            runner.step(
                {
                    "robot_1": Action.WAIT,
                    "unknown": Action.WAIT,
                }
            )
        with self.assertRaisesRegex(ValueError, "invalid action"):
            runner.step({"robot_1": "wait"})  # type: ignore[dict-item]


if __name__ == "__main__":
    unittest.main()
