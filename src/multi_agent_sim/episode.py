"""External episode scoring and headless execution.

The simulation world remains the sole owner of state changes.  This module
observes immutable before/after snapshots and action results to score those
changes, and deliberately keeps all objective and termination policy outside
``SimulationWorld`` and ``SimulationSession``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType

from .actions import Action, ActionResult
from .controllers import WorldObservation
from .session import InitialScenario
from .world import SimulationWorld


_FLOAT_ABS_TOLERANCE = 1e-9
_FLOAT_REL_TOLERANCE = 1e-9
OBJECTIVE_SCHEMA_VERSION = "route-delivery-v2"


def _is_close(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=_FLOAT_REL_TOLERANCE,
        abs_tol=_FLOAT_ABS_TOLERANCE,
    )


def _manhattan_distance(
    left: tuple[int, int],
    right: tuple[int, int],
) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def _validate_finite_nonnegative(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
    return float(value)


@dataclass(frozen=True, slots=True)
class EpisodeCostWeights:
    """Configurable coefficients for the shared team episode objective."""

    failure: float = 10.0
    undelivered: float = 2.0
    time: float = 0.5
    energy: float = 0.5
    item_progress: float = 0.1
    delivery: float = 1.0

    def __post_init__(self) -> None:
        for field_name in (
            "failure",
            "undelivered",
            "time",
            "energy",
            "item_progress",
            "delivery",
        ):
            object.__setattr__(
                self,
                field_name,
                _validate_finite_nonnegative(
                    getattr(self, field_name),
                    f"{field_name} cost weight",
                ),
            )

    @property
    def failure_weight(self) -> float:
        return self.failure

    @property
    def undelivered_weight(self) -> float:
        return self.undelivered

    @property
    def time_weight(self) -> float:
        return self.time

    @property
    def energy_weight(self) -> float:
        return self.energy

    @property
    def item_progress_weight(self) -> float:
        return self.item_progress

    @property
    def delivery_weight(self) -> float:
        return self.delivery


@dataclass(frozen=True, slots=True)
class DeliveryStatus:
    """Current observation-only delivery status for the team objective."""

    total_items: int
    delivered_items: int

    def __post_init__(self) -> None:
        for name in ("total_items", "delivered_items"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.delivered_items > self.total_items:
            raise ValueError("delivered_items cannot exceed total_items")

    @property
    def undelivered_items(self) -> int:
        return self.total_items - self.delivered_items

    @property
    def delivered_fraction(self) -> float:
        return self.delivered_items / self.total_items if self.total_items else 0.0

    @property
    def success(self) -> bool:
        return self.delivered_items == self.total_items


def delivery_status(observation: WorldObservation) -> DeliveryStatus:
    """Return simultaneous target-item delivery status from an immutable snapshot."""

    if not isinstance(observation, WorldObservation):
        raise ValueError("observation must be a WorldObservation")
    item_positions = {item.item_id: item.position for item in observation.items}
    destinations = observation.destinations
    delivered = sum(
        item_positions.get(destination.target_item_id) == destination.position
        for destination in destinations
    )
    return DeliveryStatus(len(destinations), delivered)


@dataclass(frozen=True, slots=True)
class StepRewardComponents:
    """Plain immutable decomposition of one shared team reward."""

    time: float
    energy: float
    item_progress: float
    delivery: float
    failure: float
    undelivered: float
    total: float

    def __post_init__(self) -> None:
        for name in (
            "time",
            "energy",
            "item_progress",
            "delivery",
            "failure",
            "undelivered",
            "total",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} reward component must be finite")
            object.__setattr__(self, name, float(value))
        component_sum = math.fsum(
            (
                self.time,
                self.energy,
                self.item_progress,
                self.delivery,
                self.failure,
                self.undelivered,
            )
        )
        if not _is_close(self.total, component_sum):
            raise ValueError("total reward is inconsistent with its components")


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    """Immutable current or terminal metrics for one scored episode."""

    terminated: bool
    success: bool
    timed_out: bool
    total_items: int
    delivered_items: int
    undelivered_items: int
    undelivered_fraction: float
    delivered_fraction: float
    elapsed_steps: int
    max_steps: int
    battery_consumed: float
    max_energy: float
    initial_normalized_item_distance: float
    normalized_item_distance: float
    failure_cost: float
    undelivered_cost: float
    time_cost: float
    energy_cost: float
    item_progress_cost: float
    delivery_cost: float
    total_cost: float
    cumulative_reward: float

    def __post_init__(self) -> None:
        for field_name in ("terminated", "success", "timed_out"):
            if type(getattr(self, field_name)) is not bool:
                raise ValueError(f"{field_name} must be a boolean")
        for field_name in (
            "total_items",
            "delivered_items",
            "undelivered_items",
            "elapsed_steps",
            "max_steps",
        ):
            value = getattr(self, field_name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        if self.elapsed_steps > self.max_steps:
            raise ValueError("elapsed_steps cannot exceed max_steps")
        if self.delivered_items + self.undelivered_items != self.total_items:
            raise ValueError("delivered and undelivered counts must sum to total_items")

        expected_fraction = (
            self.undelivered_items / self.total_items
            if self.total_items
            else 0.0
        )
        fraction = _validate_finite_nonnegative(
            self.undelivered_fraction,
            "undelivered_fraction",
        )
        if fraction > 1 or not _is_close(fraction, expected_fraction):
            raise ValueError("undelivered_fraction is inconsistent with item counts")
        object.__setattr__(self, "undelivered_fraction", fraction)
        delivered_fraction = _validate_finite_nonnegative(
            self.delivered_fraction,
            "delivered_fraction",
        )
        expected_delivered_fraction = (
            self.delivered_items / self.total_items if self.total_items else 0.0
        )
        if (
            delivered_fraction > 1
            or not _is_close(delivered_fraction, expected_delivered_fraction)
        ):
            raise ValueError("delivered_fraction is inconsistent with item counts")
        object.__setattr__(self, "delivered_fraction", delivered_fraction)

        for field_name in (
            "battery_consumed",
            "max_energy",
            "initial_normalized_item_distance",
            "normalized_item_distance",
            "failure_cost",
            "undelivered_cost",
            "time_cost",
            "energy_cost",
        ):
            object.__setattr__(
                self,
                field_name,
                _validate_finite_nonnegative(getattr(self, field_name), field_name),
            )
        for field_name in ("item_progress_cost", "delivery_cost", "total_cost"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{field_name} must be a finite number")
            object.__setattr__(self, field_name, float(value))
        if self.initial_normalized_item_distance > 2:
            raise ValueError("initial route potential cannot exceed two")
        if self.normalized_item_distance > 2:
            raise ValueError("route potential cannot exceed two")
        cumulative_reward = self.cumulative_reward
        if (
            isinstance(cumulative_reward, bool)
            or not isinstance(cumulative_reward, (int, float))
            or not math.isfinite(cumulative_reward)
        ):
            raise ValueError("cumulative_reward must be a finite number")
        object.__setattr__(self, "cumulative_reward", float(cumulative_reward))

        if self.success and not self.terminated:
            raise ValueError("a successful episode must be terminated")
        if self.success and self.undelivered_items:
            raise ValueError("a successful episode cannot have undelivered items")
        if self.success and self.timed_out:
            raise ValueError("success takes precedence over timeout")
        if self.timed_out and not self.terminated:
            raise ValueError("a timed-out episode must be terminated")
        if self.terminated and not self.success and not self.timed_out:
            raise ValueError("an unsuccessful terminated episode must be timed out")

        component_total = (
            self.failure_cost
            + self.undelivered_cost
            + self.time_cost
            + self.energy_cost
            + self.item_progress_cost
            + self.delivery_cost
        )
        if not _is_close(self.total_cost, component_total):
            raise ValueError("total_cost is inconsistent with its components")
        if not _is_close(self.cumulative_reward, -self.total_cost):
            raise ValueError("cumulative_reward must equal negative accrued cost")

    @property
    def failure(self) -> int:
        """The objective's binary failure term ``F``."""

        return int(self.terminated and not self.success)

    @property
    def delivered_item_count(self) -> int:
        return self.delivered_items

    @property
    def undelivered_item_count(self) -> int:
        return self.undelivered_items

    @property
    def elapsed_timesteps(self) -> int:
        return self.elapsed_steps

    @property
    def t_max(self) -> int:
        return self.max_steps

    @property
    def total_battery_consumed(self) -> float:
        return self.battery_consumed

    @property
    def e_max(self) -> float:
        return self.max_energy

    @property
    def cumulative_item_progress_reward(self) -> float:
        return -self.item_progress_cost

    @property
    def initial_route_potential(self) -> float:
        return self.initial_normalized_item_distance

    @property
    def route_potential(self) -> float:
        return self.normalized_item_distance

    @property
    def route_progress_cost(self) -> float:
        return self.item_progress_cost

    @property
    def cumulative_delivery_reward(self) -> float:
        return -self.delivery_cost


@dataclass(frozen=True, slots=True)
class EpisodeEvaluation:
    """Reward and updated metrics produced for one observed transition."""

    reward: float
    metrics: EpisodeMetrics
    item_progress_reward: float = 0.0
    delivery_reward: float = 0.0
    reward_components: StepRewardComponents | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.reward, bool)
            or not isinstance(self.reward, (int, float))
            or not math.isfinite(self.reward)
        ):
            raise ValueError("reward must be a finite number")
        if not isinstance(self.metrics, EpisodeMetrics):
            raise ValueError("metrics must be an EpisodeMetrics instance")
        for name in ("item_progress_reward", "delivery_reward"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
            object.__setattr__(self, name, float(value))
        if self.reward_components is not None:
            if not isinstance(self.reward_components, StepRewardComponents):
                raise ValueError(
                    "reward_components must be a StepRewardComponents value or None"
                )
            if not _is_close(self.reward, self.reward_components.total):
                raise ValueError("reward must match reward_components.total")
        object.__setattr__(self, "reward", float(self.reward))

    @property
    def terminated(self) -> bool:
        return self.metrics.terminated


@dataclass(frozen=True, slots=True)
class EpisodeTransition:
    """One externally commanded world transition and its team score."""

    previous_observation: WorldObservation
    observation: WorldObservation
    action_results: Mapping[str, ActionResult]
    reward: float
    item_progress_reward: float
    delivery_reward: float
    reward_components: StepRewardComponents
    terminated: bool
    metrics: EpisodeMetrics

    def __post_init__(self) -> None:
        if not isinstance(self.previous_observation, WorldObservation):
            raise ValueError("previous_observation must be a WorldObservation")
        if not isinstance(self.observation, WorldObservation):
            raise ValueError("observation must be a WorldObservation")
        if not isinstance(self.action_results, Mapping):
            raise ValueError("action_results must be a mapping")
        results = dict(self.action_results)
        if not all(
            isinstance(robot_id, str) and isinstance(result, ActionResult)
            for robot_id, result in results.items()
        ):
            raise ValueError("action_results must map robot IDs to ActionResult values")
        if (
            isinstance(self.reward, bool)
            or not isinstance(self.reward, (int, float))
            or not math.isfinite(self.reward)
        ):
            raise ValueError("reward must be a finite number")
        if type(self.terminated) is not bool:
            raise ValueError("terminated must be a boolean")
        for name in ("item_progress_reward", "delivery_reward"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite number")
            object.__setattr__(self, name, float(value))
        if not isinstance(self.reward_components, StepRewardComponents):
            raise ValueError("reward_components must be a StepRewardComponents value")
        if not _is_close(self.reward, self.reward_components.total):
            raise ValueError("reward must match reward_components.total")
        if not isinstance(self.metrics, EpisodeMetrics):
            raise ValueError("metrics must be an EpisodeMetrics instance")
        if self.terminated != self.metrics.terminated:
            raise ValueError("terminated must match metrics.terminated")
        object.__setattr__(self, "action_results", MappingProxyType(results))
        object.__setattr__(self, "reward", float(self.reward))

    @property
    def next_observation(self) -> WorldObservation:
        return self.observation

    @property
    def done(self) -> bool:
        return self.terminated


class TeamCostEvaluator:
    """Score immutable team transitions without access to a simulation world."""

    def __init__(
        self,
        initial_observation: WorldObservation,
        max_steps: int,
        cost_weights: EpisodeCostWeights | None = None,
    ) -> None:
        if not isinstance(initial_observation, WorldObservation):
            raise ValueError("initial_observation must be a WorldObservation")
        if type(max_steps) is not int or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        if cost_weights is not None and not isinstance(
            cost_weights, EpisodeCostWeights
        ):
            raise ValueError("cost_weights must be an EpisodeCostWeights instance")

        self._initial_observation = initial_observation
        self._observation = initial_observation
        self._max_steps = max_steps
        self._cost_weights = cost_weights or EpisodeCostWeights()
        self._initial_timestep = initial_observation.timestep
        self._robot_ids = tuple(
            robot.robot_id for robot in initial_observation.robots
        )
        self._destinations = initial_observation.destinations
        self._item_ids = frozenset(
            destination.target_item_id for destination in self._destinations
        )
        self._max_energy = sum(
            robot.battery_level for robot in initial_observation.robots
        )
        self._battery_consumed = 0.0
        self._cumulative_reward = 0.0

        self._validate_logical_item_roster(initial_observation)
        self._initial_normalized_item_distance = self._route_potential(
            initial_observation
        )
        delivered_items = self._count_delivered(initial_observation)
        self._initial_delivered_fraction = (
            delivered_items / len(self._item_ids) if self._item_ids else 0.0
        )
        success = delivered_items == len(self._item_ids)
        self._metrics = self._build_metrics(
            observation=initial_observation,
            delivered_items=delivered_items,
            success=success,
            timed_out=False,
        )

    @property
    def initial_observation(self) -> WorldObservation:
        return self._initial_observation

    @property
    def observation(self) -> WorldObservation:
        return self._observation

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def t_max(self) -> int:
        return self._max_steps

    @property
    def cost_weights(self) -> EpisodeCostWeights:
        return self._cost_weights

    @property
    def metrics(self) -> EpisodeMetrics:
        return self._metrics

    @property
    def terminated(self) -> bool:
        return self._metrics.terminated

    def evaluate_transition(
        self,
        previous_observation: WorldObservation,
        next_observation: WorldObservation,
        action_results: Mapping[str, ActionResult],
    ) -> EpisodeEvaluation:
        """Validate and score exactly one sequential world transition."""

        if self.terminated:
            raise RuntimeError("cannot evaluate a transition after episode termination")
        if not isinstance(previous_observation, WorldObservation):
            raise ValueError("previous_observation must be a WorldObservation")
        if not isinstance(next_observation, WorldObservation):
            raise ValueError("next_observation must be a WorldObservation")
        if previous_observation != self._observation:
            raise ValueError(
                "previous_observation does not match the evaluator's current state"
            )
        if next_observation.timestep != previous_observation.timestep + 1:
            raise ValueError("observations must describe consecutive timesteps")

        self._validate_world_identity(previous_observation, next_observation)
        self._validate_logical_item_roster(next_observation)
        results = self._validate_action_results(
            previous_observation,
            next_observation,
            action_results,
        )

        delta_energy = sum(result.battery_spent for result in results.values())
        elapsed_steps = next_observation.timestep - self._initial_timestep
        if elapsed_steps > self._max_steps:
            raise ValueError("transition advances beyond the maximum episode length")

        delivered_items = self._count_delivered(next_observation)
        success = delivered_items == len(self._item_ids)
        timed_out = not success and elapsed_steps == self._max_steps
        previous_distance = self._route_potential(previous_observation)
        next_distance = self._route_potential(next_observation)
        item_progress_reward = self._cost_weights.item_progress * (
            previous_distance - next_distance
        )
        previous_delivered_fraction = (
            self._count_delivered(previous_observation) / len(self._item_ids)
            if self._item_ids
            else 0.0
        )
        next_delivered_fraction = (
            delivered_items / len(self._item_ids) if self._item_ids else 0.0
        )
        delivery_reward = self._cost_weights.delivery * (
            next_delivered_fraction - previous_delivered_fraction
        )

        time_reward = -self._cost_weights.time / self._max_steps
        energy_reward = 0.0
        if self._max_energy > 0:
            energy_reward = -(
                self._cost_weights.energy * delta_energy / self._max_energy
            )
        failure_reward = 0.0
        undelivered_reward = 0.0
        if timed_out:
            undelivered_fraction = (
                (len(self._item_ids) - delivered_items) / len(self._item_ids)
                if self._item_ids
                else 0.0
            )
            failure_reward = -self._cost_weights.failure
            undelivered_reward = -(
                self._cost_weights.undelivered * undelivered_fraction
            )
        reward_components = StepRewardComponents(
            time=time_reward,
            energy=energy_reward,
            item_progress=item_progress_reward,
            delivery=delivery_reward,
            failure=failure_reward,
            undelivered=undelivered_reward,
            total=math.fsum(
                (
                    time_reward,
                    energy_reward,
                    item_progress_reward,
                    delivery_reward,
                    failure_reward,
                    undelivered_reward,
                )
            ),
        )
        reward = reward_components.total

        self._battery_consumed += delta_energy
        self._cumulative_reward += reward
        self._observation = next_observation
        self._metrics = self._build_metrics(
            observation=next_observation,
            delivered_items=delivered_items,
            success=success,
            timed_out=timed_out,
        )
        return EpisodeEvaluation(
            reward=reward,
            metrics=self._metrics,
            item_progress_reward=item_progress_reward,
            delivery_reward=delivery_reward,
            reward_components=reward_components,
        )

    def _validate_world_identity(
        self,
        previous_observation: WorldObservation,
        next_observation: WorldObservation,
    ) -> None:
        if (
            next_observation.width != self._initial_observation.width
            or next_observation.height != self._initial_observation.height
        ):
            raise ValueError("world dimensions cannot change during an episode")
        if (
            next_observation.action_battery_costs
            != self._initial_observation.action_battery_costs
        ):
            raise ValueError("action battery costs cannot change during an episode")
        if next_observation.destinations != self._destinations:
            raise ValueError("destination roster cannot change during an episode")

        previous_robot_ids = tuple(
            robot.robot_id for robot in previous_observation.robots
        )
        next_robot_ids = tuple(robot.robot_id for robot in next_observation.robots)
        if previous_robot_ids != self._robot_ids or next_robot_ids != self._robot_ids:
            raise ValueError("robot roster cannot change during an episode")

    def _validate_logical_item_roster(
        self,
        observation: WorldObservation,
    ) -> None:
        visible_item_ids = [item.item_id for item in observation.items]
        carried_item_ids = [
            robot.carried_item_id
            for robot in observation.robots
            if robot.carried_item_id is not None
        ]
        logical_item_ids = visible_item_ids + carried_item_ids
        if len(set(logical_item_ids)) != len(logical_item_ids):
            raise ValueError("an item cannot be both on-grid and carried, or carried twice")
        if frozenset(logical_item_ids) != self._item_ids:
            raise ValueError(
                "logical item roster must match destination target item IDs"
            )

    def _validate_action_results(
        self,
        previous_observation: WorldObservation,
        next_observation: WorldObservation,
        action_results: Mapping[str, ActionResult],
    ) -> dict[str, ActionResult]:
        if not isinstance(action_results, Mapping):
            raise ValueError("action_results must be a mapping")
        results = dict(action_results)
        if set(results) != set(self._robot_ids):
            raise ValueError(
                "action_results must contain exactly one result for every robot"
            )
        if not all(isinstance(result, ActionResult) for result in results.values()):
            raise ValueError("action_results must contain only ActionResult values")

        previous_robots = {
            robot.robot_id: robot for robot in previous_observation.robots
        }
        next_robots = {robot.robot_id: robot for robot in next_observation.robots}
        observed_delta_energy = 0.0
        result_delta_energy = 0.0
        for robot_id in self._robot_ids:
            previous_robot = previous_robots[robot_id]
            next_robot = next_robots[robot_id]
            result = results[robot_id]
            if not isinstance(result.action, Action):
                raise ValueError(f"action result for {robot_id!r} has an invalid action")
            if type(result.success) is not bool:
                raise ValueError(f"action result for {robot_id!r} has invalid success")
            for value, name in (
                (result.battery_before, "battery_before"),
                (result.battery_after, "battery_after"),
                (result.battery_spent, "battery_spent"),
            ):
                _validate_finite_nonnegative(value, f"{robot_id} {name}")
            if result.start_position != previous_robot.position:
                raise ValueError(
                    f"action result start position does not match {robot_id!r}"
                )
            if result.end_position != next_robot.position:
                raise ValueError(
                    f"action result end position does not match {robot_id!r}"
                )
            if not _is_close(result.battery_before, previous_robot.battery_level):
                raise ValueError(
                    f"action result battery_before does not match {robot_id!r}"
                )
            if not _is_close(result.battery_after, next_robot.battery_level):
                raise ValueError(
                    f"action result battery_after does not match {robot_id!r}"
                )
            if next_robot.battery_level > (
                previous_robot.battery_level + _FLOAT_ABS_TOLERANCE
            ):
                raise ValueError("robot battery cannot increase during an episode")

            observed_delta = previous_robot.battery_level - next_robot.battery_level
            if not _is_close(observed_delta, result.battery_spent):
                raise ValueError(
                    f"battery_spent is inconsistent with observations for {robot_id!r}"
                )
            if not _is_close(
                result.battery_before - result.battery_after,
                result.battery_spent,
            ):
                raise ValueError(
                    f"battery_spent is inconsistent within result for {robot_id!r}"
                )
            observed_delta_energy += observed_delta
            result_delta_energy += result.battery_spent

        if not _is_close(observed_delta_energy, result_delta_energy):
            raise ValueError("total battery accounting is inconsistent")
        return results

    def _count_delivered(self, observation: WorldObservation) -> int:
        return delivery_status(observation).delivered_items

    def _route_potential(self, observation: WorldObservation) -> float:
        if not self._destinations:
            return 0.0
        maximum_distance = observation.width + observation.height - 2
        if maximum_distance <= 0:
            return 0.0

        item_positions = {item.item_id: item.position for item in observation.items}
        carriers = {
            robot.carried_item_id: robot
            for robot in observation.robots
            if robot.carried_item_id is not None
        }
        route_distances: list[int] = []
        for destination in self._destinations:
            item_id = destination.target_item_id
            item_position = item_positions.get(item_id)
            if item_position == destination.position:
                route_distances.append(0)
                continue
            carrier = carriers.get(item_id)
            if carrier is not None:
                route_distances.append(
                    _manhattan_distance(carrier.position, destination.position)
                )
                continue
            if item_position is None:
                raise ValueError(f"logical item {item_id!r} has no position")
            approach_distance = (
                min(
                    _manhattan_distance(robot.position, item_position)
                    for robot in observation.robots
                )
                if observation.robots
                else maximum_distance
            )
            route_distances.append(
                approach_distance
                + _manhattan_distance(item_position, destination.position)
            )
        distance_sum = math.fsum(route_distances)
        return distance_sum / (len(self._destinations) * maximum_distance)

    def _build_metrics(
        self,
        *,
        observation: WorldObservation,
        delivered_items: int,
        success: bool,
        timed_out: bool,
    ) -> EpisodeMetrics:
        total_items = len(self._item_ids)
        undelivered_items = total_items - delivered_items
        undelivered_fraction = (
            undelivered_items / total_items if total_items else 0.0
        )
        elapsed_steps = observation.timestep - self._initial_timestep
        terminated = success or timed_out
        failure_cost = self._cost_weights.failure if timed_out else 0.0
        undelivered_cost = (
            self._cost_weights.undelivered * undelivered_fraction
            if timed_out
            else 0.0
        )
        time_cost = self._cost_weights.time * elapsed_steps / self._max_steps
        energy_cost = (
            self._cost_weights.energy * self._battery_consumed / self._max_energy
            if self._max_energy > 0
            else 0.0
        )
        normalized_item_distance = self._route_potential(observation)
        item_progress_cost = self._cost_weights.item_progress * (
            normalized_item_distance - self._initial_normalized_item_distance
        )
        delivered_fraction = (
            delivered_items / total_items if total_items else 0.0
        )
        delivery_cost = self._cost_weights.delivery * (
            self._initial_delivered_fraction - delivered_fraction
        )
        total_cost = (
            failure_cost
            + undelivered_cost
            + time_cost
            + energy_cost
            + item_progress_cost
            + delivery_cost
        )
        return EpisodeMetrics(
            terminated=terminated,
            success=success,
            timed_out=timed_out,
            total_items=total_items,
            delivered_items=delivered_items,
            undelivered_items=undelivered_items,
            undelivered_fraction=undelivered_fraction,
            delivered_fraction=delivered_fraction,
            elapsed_steps=elapsed_steps,
            max_steps=self._max_steps,
            battery_consumed=self._battery_consumed,
            max_energy=self._max_energy,
            initial_normalized_item_distance=(
                self._initial_normalized_item_distance
            ),
            normalized_item_distance=normalized_item_distance,
            failure_cost=failure_cost,
            undelivered_cost=undelivered_cost,
            time_cost=time_cost,
            energy_cost=energy_cost,
            item_progress_cost=item_progress_cost,
            delivery_cost=delivery_cost,
            total_cost=total_cost,
            cumulative_reward=self._cumulative_reward,
        )


class EpisodeRunner:
    """Run externally selected team actions in a fresh scenario world."""

    def __init__(
        self,
        scenario: InitialScenario,
        max_steps: int,
        cost_weights: EpisodeCostWeights | None = None,
    ) -> None:
        if not isinstance(scenario, InitialScenario):
            raise ValueError("scenario must be an InitialScenario")
        if type(max_steps) is not int or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        if cost_weights is not None and not isinstance(
            cost_weights, EpisodeCostWeights
        ):
            raise ValueError("cost_weights must be an EpisodeCostWeights instance")

        self._scenario = scenario
        self._world = scenario.create_world()
        initial_observation = WorldObservation.from_world(self._world)
        self._evaluator = TeamCostEvaluator(
            initial_observation,
            max_steps=max_steps,
            cost_weights=cost_weights,
        )
        self._latest_reward = 0.0

    @property
    def scenario(self) -> InitialScenario:
        return self._scenario

    @property
    def world(self) -> SimulationWorld:
        return self._world

    @property
    def observation(self) -> WorldObservation:
        return self._evaluator.observation

    @property
    def metrics(self) -> EpisodeMetrics:
        return self._evaluator.metrics

    @property
    def terminated(self) -> bool:
        return self._evaluator.terminated

    @property
    def max_steps(self) -> int:
        return self._evaluator.max_steps

    @property
    def t_max(self) -> int:
        return self._evaluator.max_steps

    @property
    def cost_weights(self) -> EpisodeCostWeights:
        return self._evaluator.cost_weights

    @property
    def latest_reward(self) -> float:
        return self._latest_reward

    def step(self, actions: Mapping[str, Action]) -> EpisodeTransition:
        """Apply one complete simultaneous team action batch and score it."""

        if self.terminated:
            raise RuntimeError("cannot step a terminated episode")
        validated_actions = self._validate_actions(actions)
        previous_observation = self.observation
        action_results = self._world.step(validated_actions)
        next_observation = WorldObservation.from_world(self._world)
        evaluation = self._evaluator.evaluate_transition(
            previous_observation,
            next_observation,
            action_results,
        )
        self._latest_reward = evaluation.reward
        return EpisodeTransition(
            previous_observation=previous_observation,
            observation=next_observation,
            action_results=action_results,
            reward=evaluation.reward,
            item_progress_reward=evaluation.item_progress_reward,
            delivery_reward=evaluation.delivery_reward,
            reward_components=evaluation.reward_components,
            terminated=evaluation.terminated,
            metrics=evaluation.metrics,
        )

    def _validate_actions(self, actions: Mapping[str, Action]) -> dict[str, Action]:
        if not isinstance(actions, Mapping):
            raise ValueError("actions must be a mapping of robot IDs to Action values")
        actions_copy = dict(actions)
        robot_ids = tuple(robot.robot_id for robot in self.observation.robots)
        if set(actions_copy) != set(robot_ids):
            missing = sorted(set(robot_ids) - set(actions_copy), key=str)
            unknown = sorted(set(actions_copy) - set(robot_ids), key=str)
            details = []
            if missing:
                details.append(f"missing robot IDs: {missing!r}")
            if unknown:
                details.append(f"unknown robot IDs: {unknown!r}")
            raise ValueError(
                "actions must contain exactly one action for every robot"
                + (f" ({'; '.join(details)})" if details else "")
            )
        for robot_id, action in actions_copy.items():
            if not isinstance(robot_id, str) or not isinstance(action, Action):
                raise ValueError(
                    f"invalid action for robot {robot_id!r}: {action!r}"
                )
        return actions_copy
