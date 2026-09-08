"""Optional bridge from saved one-hot MLP policies to session controllers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .actions import Action
from .controller_catalog import (
    ControllerCatalog,
    DEFAULT_CONTROLLERS_DIRECTORY,
    SavedControllerEntry,
    discover_saved_controllers,
)
from .controllers import (
    ControllerDefinition,
    ControllerFactoryContext,
    ControllerRegistry,
    WorldObservation,
    create_default_controller_registry,
)
from .learning import MLPRobotController, load_checkpoint, torch_available
from .training_env import FixedSlotEncoder, one_hot_to_action


class _SharedModelRuntime:
    def __init__(self, model_path: Path, device: str) -> None:
        self.model_path = model_path
        self.device = device
        self._model: Any | None = None

    @property
    def model(self) -> Any:
        if self._model is None:
            self._model = load_checkpoint(self.model_path, device=self.device)
            self._model.eval()
        return self._model


class SavedMLPSessionController:
    """Decode one robot's greedy one-hot policy output for SimulationSession."""

    def __init__(
        self,
        context: ControllerFactoryContext,
        runtime: _SharedModelRuntime,
        *,
        use_action_masks: bool,
    ) -> None:
        if context.max_steps is None or context.max_steps <= 0:
            raise ValueError("saved MLP controllers require a positive session max_steps")
        self.robot_id = context.robot_id
        self.max_steps = context.max_steps
        self.runtime = runtime
        self.use_action_masks = use_action_masks
        self._encoder: FixedSlotEncoder | None = None
        self._controller: MLPRobotController | None = None

    def choose_action(
        self,
        observation: WorldObservation,
        robot_id: str,
    ) -> Action:
        if robot_id != self.robot_id:
            raise ValueError(
                f"controller for {self.robot_id!r} cannot control {robot_id!r}"
            )
        if self._encoder is None:
            self._encoder = FixedSlotEncoder(observation, self.max_steps)
        encoded = self._encoder.encode(observation)
        if self._controller is None:
            self._controller = MLPRobotController(
                self.robot_id,
                self.runtime.model,
                stochastic=False,
                use_action_mask=self.use_action_masks,
            )
        import torch

        with torch.no_grad():
            command = self._controller.choose_command(encoded)
        return one_hot_to_action(command)


@dataclass(frozen=True, slots=True)
class SavedControllerRegistryResult:
    registry: ControllerRegistry
    catalog: ControllerCatalog
    warnings: tuple[str, ...]


def _resolved_device(requested: str) -> str:
    if requested not in ("cpu", "auto", "cuda"):
        raise ValueError("controller device must be 'cpu', 'auto', or 'cuda'")
    if requested != "auto":
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def create_registry_with_saved_controllers(
    *,
    directory: str | Path = DEFAULT_CONTROLLERS_DIRECTORY,
    base_registry: ControllerRegistry | None = None,
    device: str = "cpu",
) -> SavedControllerRegistryResult:
    """Append valid saved MLP definitions to a fresh controller registry."""

    base = base_registry or create_default_controller_registry()
    if not isinstance(base, ControllerRegistry):
        raise ValueError("base_registry must be a ControllerRegistry")
    registry = ControllerRegistry(base.definitions)
    catalog = discover_saved_controllers(directory)
    warnings = list(catalog.warnings)
    if not catalog.entries:
        return SavedControllerRegistryResult(registry, catalog, tuple(warnings))
    if not torch_available():
        warnings.append(
            "saved neural controllers are unavailable because PyTorch is not "
            "installed; install the training extra"
        )
        return SavedControllerRegistryResult(registry, catalog, tuple(warnings))

    resolved_device = _resolved_device(device)
    for entry in catalog.entries:
        runtime = _SharedModelRuntime(entry.model_path, resolved_device)

        def factory(
            context: ControllerFactoryContext,
            *,
            runtime: _SharedModelRuntime = runtime,
            entry: SavedControllerEntry = entry,
        ) -> SavedMLPSessionController:
            return SavedMLPSessionController(
                context,
                runtime,
                use_action_masks=entry.manifest.use_action_masks,
            )

        registry.register(
            ControllerDefinition(
                entry.manifest.registry_key,
                entry.manifest.display_name,
                factory,
                max_robots=4,
                max_items=8,
            )
        )
    return SavedControllerRegistryResult(registry, catalog, tuple(warnings))


__all__ = [
    "SavedControllerRegistryResult",
    "SavedMLPSessionController",
    "create_registry_with_saved_controllers",
]
