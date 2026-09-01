"""Public interface for the multi-agent simulation core."""

from .actions import Action, ActionBatteryCosts, ActionFailureReason, ActionResult
from .controllers import (
    ControllerDefinition,
    ControllerFactory,
    ControllerFactoryContext,
    ControllerRegistry,
    ItemObservation,
    MultiAgentController,
    MultiAgentControllerAdapter,
    NearestItemController,
    RandomController,
    RobotController,
    RobotObservation,
    WorldObservation,
    create_default_controller_registry,
    create_world_observation,
)
from .entities import Entity, Item, Position, Robot
from .generation import generate_random_world
from .session import (
    InitialScenario,
    RobotConfiguration,
    SimulationSession,
    create_initial_scenario,
)
from .world import (
    OccupancyPolicy,
    SimulationWorld,
    TypeExclusiveOccupancyPolicy,
)

__all__ = [
    "Action",
    "ActionBatteryCosts",
    "ActionFailureReason",
    "ActionResult",
    "ControllerDefinition",
    "ControllerFactory",
    "ControllerFactoryContext",
    "ControllerRegistry",
    "Entity",
    "Item",
    "InitialScenario",
    "ItemObservation",
    "MultiAgentController",
    "MultiAgentControllerAdapter",
    "NearestItemController",
    "OccupancyPolicy",
    "Position",
    "Robot",
    "RobotController",
    "RobotConfiguration",
    "RobotObservation",
    "RandomController",
    "SimulationSession",
    "SimulationWorld",
    "TypeExclusiveOccupancyPolicy",
    "WorldObservation",
    "create_default_controller_registry",
    "generate_random_world",
    "create_initial_scenario",
    "create_world_observation",
]
