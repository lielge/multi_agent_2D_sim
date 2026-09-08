"""Dependency-free catalog for immutable saved neural controllers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from types import MappingProxyType
from typing import Any
import uuid

from .episode import OBJECTIVE_SCHEMA_VERSION
from .training_env import ACTION_MAP_VERSION, ENCODING_VERSION, FEATURE_DIM


CONTROLLER_MANIFEST_VERSION = 2
LEGACY_CONTROLLER_MANIFEST_VERSIONS = (1,)
LEGACY_OBJECTIVE_SCHEMA_VERSION = "item-distance-v1"
CONTROLLER_KIND = "shared_mlp"
CONTROLLER_ARCHITECTURE = (FEATURE_DIM, 128, 128, 7)
DEFAULT_CONTROLLERS_DIRECTORY = Path("controllers")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_SAFE_NAME_PATTERN = re.compile(r"[^a-z0-9]+")
_REPLACE_ATTEMPTS = 8


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _nonnegative_integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _nonempty_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{name} cannot have surrounding whitespace")
    return value


def _plain_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    try:
        # JSON round-tripping rejects non-serializable or non-string-key data
        # and prevents callers from mutating a manifest through shared values.
        normalized = json.loads(json.dumps(dict(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain JSON-compatible values") from exc
    if not isinstance(normalized, dict):
        raise ValueError(f"{name} must be an object")
    return normalized


@dataclass(frozen=True, slots=True)
class SavedControllerManifest:
    """Validated metadata paired with one immutable inference state dict."""

    controller_id: str
    display_name: str
    model_file: str
    model_sha256: str
    created_utc: str
    run_id: str
    qualified: bool
    qualified_stage: int
    evaluated_stage: int
    success_rate: float
    mean_cost: float
    episodes_trained: int
    optimizer_updates: int
    use_action_masks: bool
    torch_version: str
    curriculum: tuple[Mapping[str, Any], ...]
    cost_weights: Mapping[str, Any]
    training_config: Mapping[str, Any]
    source_checkpoint: str
    objective_schema_version: str = OBJECTIVE_SCHEMA_VERSION
    architecture: tuple[int, ...] = CONTROLLER_ARCHITECTURE
    encoding_schema_version: str = ENCODING_VERSION
    action_map_version: str = ACTION_MAP_VERSION
    controller_kind: str = CONTROLLER_KIND
    manifest_version: int = CONTROLLER_MANIFEST_VERSION

    def __post_init__(self) -> None:
        for name in (
            "controller_id",
            "display_name",
            "model_file",
            "created_utc",
            "run_id",
            "torch_version",
            "source_checkpoint",
        ):
            object.__setattr__(self, name, _nonempty_text(getattr(self, name), name))
        if Path(self.model_file).name != self.model_file:
            raise ValueError("model_file must be a filename within the catalog")
        checksum = _nonempty_text(self.model_sha256, "model_sha256").lower()
        if _SHA256_PATTERN.fullmatch(checksum) is None:
            raise ValueError("model_sha256 must contain 64 lowercase hex characters")
        object.__setattr__(self, "model_sha256", checksum)
        try:
            parsed_time = datetime.fromisoformat(self.created_utc.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("created_utc must be an ISO-8601 timestamp") from exc
        if parsed_time.tzinfo is None:
            raise ValueError("created_utc must contain a timezone")
        if type(self.qualified) is not bool:
            raise ValueError("qualified must be a bool")
        for name in (
            "qualified_stage",
            "evaluated_stage",
            "episodes_trained",
            "optimizer_updates",
        ):
            object.__setattr__(
                self, name, _nonnegative_integer(getattr(self, name), name)
            )
        success_rate = _finite(self.success_rate, "success_rate")
        if not 0 <= success_rate <= 1:
            raise ValueError("success_rate must be between zero and one")
        object.__setattr__(self, "success_rate", success_rate)
        object.__setattr__(self, "mean_cost", _finite(self.mean_cost, "mean_cost"))
        if type(self.use_action_masks) is not bool:
            raise ValueError("use_action_masks must be a bool")
        architecture = tuple(self.architecture)
        if architecture != CONTROLLER_ARCHITECTURE:
            raise ValueError("controller architecture is incompatible")
        object.__setattr__(self, "architecture", architecture)
        if self.encoding_schema_version != ENCODING_VERSION:
            raise ValueError("controller encoding version is incompatible")
        if self.action_map_version != ACTION_MAP_VERSION:
            raise ValueError("controller action-map version is incompatible")
        if self.controller_kind != CONTROLLER_KIND:
            raise ValueError("controller kind is unsupported")
        if self.manifest_version not in (
            CONTROLLER_MANIFEST_VERSION,
            *LEGACY_CONTROLLER_MANIFEST_VERSIONS,
        ):
            raise ValueError("controller manifest version is unsupported")
        objective_version = _nonempty_text(
            self.objective_schema_version,
            "objective_schema_version",
        )
        allowed_objectives = {
            OBJECTIVE_SCHEMA_VERSION,
            LEGACY_OBJECTIVE_SCHEMA_VERSION,
        }
        if objective_version not in allowed_objectives:
            raise ValueError("controller objective version is unsupported")
        if (
            self.manifest_version == CONTROLLER_MANIFEST_VERSION
            and objective_version != OBJECTIVE_SCHEMA_VERSION
        ):
            raise ValueError("current controller manifests require the current objective")
        try:
            curriculum = tuple(
                MappingProxyType(_plain_mapping(stage, "curriculum stage"))
                for stage in self.curriculum
            )
        except TypeError as exc:
            raise ValueError("curriculum must be a sequence") from exc
        object.__setattr__(self, "curriculum", curriculum)
        object.__setattr__(
            self,
            "cost_weights",
            MappingProxyType(_plain_mapping(self.cost_weights, "cost_weights")),
        )
        object.__setattr__(
            self,
            "training_config",
            MappingProxyType(_plain_mapping(self.training_config, "training_config")),
        )

    @property
    def registry_key(self) -> str:
        return f"saved_mlp:{self.controller_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "controller_id": self.controller_id,
            "display_name": self.display_name,
            "model_file": self.model_file,
            "model_sha256": self.model_sha256,
            "created_utc": self.created_utc,
            "run_id": self.run_id,
            "qualified": self.qualified,
            "qualified_stage": self.qualified_stage,
            "evaluated_stage": self.evaluated_stage,
            "success_rate": self.success_rate,
            "mean_cost": self.mean_cost,
            "episodes_trained": self.episodes_trained,
            "optimizer_updates": self.optimizer_updates,
            "use_action_masks": self.use_action_masks,
            "torch_version": self.torch_version,
            "curriculum": [dict(stage) for stage in self.curriculum],
            "cost_weights": dict(self.cost_weights),
            "training_config": dict(self.training_config),
            "source_checkpoint": self.source_checkpoint,
            "objective_schema_version": self.objective_schema_version,
            "architecture": list(self.architecture),
            "encoding_schema_version": self.encoding_schema_version,
            "action_map_version": self.action_map_version,
            "controller_kind": self.controller_kind,
            "manifest_version": self.manifest_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SavedControllerManifest:
        if not isinstance(payload, Mapping):
            raise ValueError("controller manifest must be an object")
        try:
            values = dict(payload)
            manifest_version = int(values.get("manifest_version", 1))
            values.setdefault(
                "objective_schema_version",
                (
                    LEGACY_OBJECTIVE_SCHEMA_VERSION
                    if manifest_version in LEGACY_CONTROLLER_MANIFEST_VERSIONS
                    else OBJECTIVE_SCHEMA_VERSION
                ),
            )
            values["architecture"] = tuple(values["architecture"])
            values["curriculum"] = tuple(values["curriculum"])
            return cls(**values)
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, ValueError):
                raise
            raise ValueError("controller manifest is incomplete") from exc


@dataclass(frozen=True, slots=True)
class SavedControllerEntry:
    manifest: SavedControllerManifest
    model_path: Path
    manifest_path: Path


@dataclass(frozen=True, slots=True)
class ControllerCatalog:
    directory: Path
    entries: tuple[SavedControllerEntry, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PublishedController:
    entry: SavedControllerEntry
    created: bool


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_saved_controllers(
    directory: str | Path = DEFAULT_CONTROLLERS_DIRECTORY,
) -> ControllerCatalog:
    """Return valid committed controller packages and non-fatal warnings."""

    catalog_directory = Path(directory)
    if not catalog_directory.exists():
        return ControllerCatalog(catalog_directory, (), ())
    if not catalog_directory.is_dir():
        return ControllerCatalog(
            catalog_directory, (), (f"controller catalog is not a directory: {directory}",)
        )

    entries: list[SavedControllerEntry] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()
    root = catalog_directory.resolve()
    for manifest_path in sorted(
        catalog_directory.glob("*.controller.json"), key=lambda path: path.name
    ):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = SavedControllerManifest.from_dict(payload)
            if manifest.controller_id in seen_ids:
                raise ValueError(f"duplicate controller ID {manifest.controller_id!r}")
            model_path = (catalog_directory / manifest.model_file).resolve()
            if model_path.parent != root:
                raise ValueError("model path escapes the controller catalog")
            if not model_path.is_file():
                raise ValueError(f"model file is missing: {manifest.model_file}")
            if file_sha256(model_path) != manifest.model_sha256:
                raise ValueError("model checksum does not match its manifest")
            seen_ids.add(manifest.controller_id)
            entries.append(SavedControllerEntry(manifest, model_path, manifest_path))
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            warnings.append(f"ignored {manifest_path.name}: {exc}")
    entries.sort(
        key=lambda entry: (
            entry.manifest.display_name.casefold(),
            entry.manifest.controller_id,
        )
    )
    return ControllerCatalog(catalog_directory, tuple(entries), tuple(warnings))


def _slug(value: str | None) -> str:
    if value is None:
        return "mlp"
    normalized = _SAFE_NAME_PATTERN.sub("-", value.strip().lower()).strip("-")
    return normalized[:32] or "mlp"


def _replace_with_retry(source: Path, destination: Path) -> None:
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt + 1 == _REPLACE_ATTEMPTS:
                raise
            time.sleep(min(0.05 * (2**attempt), 0.5))


def _atomic_copy(source: Path, destination: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        shutil.copyfile(source, temporary)
        _replace_with_retry(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(payload: Mapping[str, Any], destination: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _replace_with_retry(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def publish_saved_controller(
    checkpoint_path: str | Path,
    *,
    directory: str | Path = DEFAULT_CONTROLLERS_DIRECTORY,
    run_id: str,
    controller_name: str | None,
    qualified_stage: int,
    evaluated_stage: int,
    success_rate: float,
    mean_cost: float,
    episodes_trained: int,
    optimizer_updates: int,
    promotion_success_rate: float,
    use_action_masks: bool,
    torch_version: str,
    curriculum: Sequence[Mapping[str, Any]],
    cost_weights: Mapping[str, Any],
    training_config: Mapping[str, Any],
    objective_schema_version: str = OBJECTIVE_SCHEMA_VERSION,
    created_at: datetime | None = None,
) -> PublishedController:
    """Publish one model-only checkpoint, deduplicating identical model bytes."""

    source = Path(checkpoint_path)
    if not source.is_file():
        raise ValueError(f"controller checkpoint does not exist: {checkpoint_path}")
    normalized_qualified_stage = _nonnegative_integer(
        qualified_stage, "qualified_stage"
    )
    normalized_evaluated_stage = _nonnegative_integer(
        evaluated_stage, "evaluated_stage"
    )
    if normalized_evaluated_stage <= 0:
        raise ValueError("evaluated_stage must be positive")
    if normalized_qualified_stage > normalized_evaluated_stage:
        raise ValueError("qualified_stage cannot exceed evaluated_stage")
    normalized_success_rate = _finite(success_rate, "success_rate")
    if not 0 <= normalized_success_rate <= 1:
        raise ValueError("success_rate must be between zero and one")
    normalized_mean_cost = _finite(mean_cost, "mean_cost")
    normalized_episodes = _nonnegative_integer(
        episodes_trained, "episodes_trained"
    )
    normalized_updates = _nonnegative_integer(
        optimizer_updates, "optimizer_updates"
    )
    normalized_promotion_rate = _finite(
        promotion_success_rate, "promotion_success_rate"
    )
    if not 0 <= normalized_promotion_rate <= 1:
        raise ValueError("promotion_success_rate must be between zero and one")
    if (
        normalized_qualified_stage > 0
        and normalized_success_rate < normalized_promotion_rate
    ):
        raise ValueError(
            "a qualified controller's success_rate must meet promotion_success_rate"
        )
    if type(use_action_masks) is not bool:
        raise ValueError("use_action_masks must be a bool")
    normalized_controller_name = (
        None
        if controller_name is None
        else _nonempty_text(controller_name, "controller_name")
    )
    normalized_objective_version = _nonempty_text(
        objective_schema_version,
        "objective_schema_version",
    )
    if normalized_objective_version not in {
        OBJECTIVE_SCHEMA_VERSION,
        LEGACY_OBJECTIVE_SCHEMA_VERSION,
    }:
        raise ValueError("objective_schema_version is unsupported")
    target_directory = Path(directory)
    target_directory.mkdir(parents=True, exist_ok=True)
    checksum = file_sha256(source)
    existing = discover_saved_controllers(target_directory)
    for entry in existing.entries:
        if entry.manifest.model_sha256 == checksum:
            return PublishedController(entry, False)

    normalized_run_id = _nonempty_text(run_id, "run_id")
    timestamp = created_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ValueError("created_at must contain a timezone")
    timestamp = timestamp.astimezone(timezone.utc)
    created_utc = timestamp.isoformat(timespec="seconds").replace("+00:00", "Z")
    stage_details = next(
        (
            dict(stage)
            for stage in curriculum
            if int(stage.get("number", -1)) == normalized_evaluated_stage
        ),
        {},
    )
    if not stage_details:
        raise ValueError("curriculum does not contain the evaluated stage")
    stage_fragment = (
        f"s{normalized_evaluated_stage:02d}-"
        f"{int(stage_details.get('width', 0))}x{int(stage_details.get('height', 0))}-"
        f"r{int(stage_details.get('num_robots', 0))}-"
        f"i{int(stage_details.get('num_items', 0))}"
    )
    timestamp_fragment = timestamp.strftime("%Y%m%dT%H%M%SZ")
    unique_fragment = f"{_slug(normalized_run_id)[:8]}-{uuid.uuid4().hex[:8]}"
    stem = (
        f"{_slug(normalized_controller_name)}-{stage_fragment}-"
        f"sr{round(normalized_success_rate * 100):03d}-"
        f"ep{normalized_episodes:06d}-{timestamp_fragment}-{unique_fragment}"
    )
    controller_id = stem
    model_path = target_directory / f"{stem}.pt"
    manifest_path = target_directory / f"{stem}.controller.json"
    while model_path.exists() or manifest_path.exists():
        stem = f"{stem}-{uuid.uuid4().hex[:8]}"
        controller_id = stem
        model_path = target_directory / f"{stem}.pt"
        manifest_path = target_directory / f"{stem}.controller.json"

    qualified = normalized_qualified_stage > 0
    label_prefix = normalized_controller_name or "MLP"
    display_name = (
        f"{label_prefix} · Stage {normalized_qualified_stage} · "
        f"{normalized_success_rate:.0%} · ep {normalized_episodes:,}"
        if qualified
        else (
            f"{label_prefix} · Unqualified · Stage {normalized_evaluated_stage} · "
            f"{normalized_success_rate:.0%} · ep {normalized_episodes:,}"
        )
    )
    manifest = SavedControllerManifest(
        controller_id=controller_id,
        display_name=display_name,
        model_file=model_path.name,
        model_sha256=checksum,
        created_utc=created_utc,
        run_id=normalized_run_id,
        qualified=qualified,
        qualified_stage=normalized_qualified_stage,
        evaluated_stage=normalized_evaluated_stage,
        success_rate=normalized_success_rate,
        mean_cost=normalized_mean_cost,
        episodes_trained=normalized_episodes,
        optimizer_updates=normalized_updates,
        use_action_masks=use_action_masks,
        torch_version=torch_version,
        curriculum=tuple(curriculum),
        cost_weights=cost_weights,
        training_config=training_config,
        source_checkpoint=str(source),
        objective_schema_version=normalized_objective_version,
        manifest_version=(
            CONTROLLER_MANIFEST_VERSION
            if normalized_objective_version == OBJECTIVE_SCHEMA_VERSION
            else LEGACY_CONTROLLER_MANIFEST_VERSIONS[0]
        ),
    )
    _atomic_copy(source, model_path)
    try:
        _atomic_json(manifest.to_dict(), manifest_path)
    except Exception:
        # An uncommitted model is invisible to discovery. Remove it when safe.
        try:
            model_path.unlink()
        except OSError:
            pass
        raise
    return PublishedController(
        SavedControllerEntry(manifest, model_path.resolve(), manifest_path), True
    )


__all__ = [
    "CONTROLLER_ARCHITECTURE",
    "CONTROLLER_KIND",
    "CONTROLLER_MANIFEST_VERSION",
    "LEGACY_CONTROLLER_MANIFEST_VERSIONS",
    "LEGACY_OBJECTIVE_SCHEMA_VERSION",
    "DEFAULT_CONTROLLERS_DIRECTORY",
    "ControllerCatalog",
    "PublishedController",
    "SavedControllerEntry",
    "SavedControllerManifest",
    "discover_saved_controllers",
    "file_sha256",
    "publish_saved_controller",
]
