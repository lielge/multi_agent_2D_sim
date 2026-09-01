"""Same-window setup and playback application for the Pygame demo."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Literal, Sequence

import pygame

from multi_agent_sim.actions import ActionBatteryCosts
from multi_agent_sim.controllers import (
    ControllerRegistry,
    create_default_controller_registry,
)
from multi_agent_sim.entities import Item, Robot
from multi_agent_sim.session import (
    InitialScenario,
    RobotConfiguration,
    SimulationSession,
    create_initial_scenario,
)
from multi_agent_sim.world import SimulationWorld

from .widgets import (
    PALETTE,
    Button,
    ChoiceSelector,
    IntegerSlider,
    TextInput,
    draw_text,
    draw_wrapped_text,
)


RobotMode = Literal["random", "manual"]


def _integer_characters(value: str) -> bool:
    return all(character in "+-0123456789" for character in value)


def _number_characters(value: str) -> bool:
    return all(character in "+-0123456789.eE" for character in value)


@dataclass(slots=True)
class RobotDraft:
    """Editable values for one manually configured robot."""

    x: TextInput
    y: TextInput
    battery: TextInput

    @classmethod
    def create(cls, position: tuple[int, int], battery: float = 100.0) -> RobotDraft:
        battery_text = str(int(battery)) if float(battery).is_integer() else str(battery)
        return cls(
            x=TextInput(str(position[0]), character_filter=_integer_characters),
            y=TextInput(str(position[1]), character_filter=_integer_characters),
            battery=TextInput(battery_text, character_filter=_number_characters),
        )


@dataclass(frozen=True, slots=True)
class SetupConfiguration:
    """Validated values needed to start a simulation session."""

    width: int
    height: int
    num_robots: int
    num_items: int
    seed: int
    max_steps: int
    action_battery_costs: ActionBatteryCosts
    robot_mode: RobotMode
    controller_keys: tuple[str, ...]
    robot_configurations: tuple[RobotConfiguration, ...] | None


@dataclass(slots=True)
class SetupValidation:
    """Validation result, including field locations for inline feedback."""

    configuration: SetupConfiguration | None = None
    field_errors: dict[str, str] = field(default_factory=dict)
    robot_errors: dict[int, dict[str, str]] = field(default_factory=dict)
    general_errors: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return self.configuration is not None

    @property
    def messages(self) -> tuple[str, ...]:
        messages = list(dict.fromkeys(self.field_errors.values()))
        for errors in self.robot_errors.values():
            messages.extend(
                message for message in errors.values() if message not in messages
            )
        messages.extend(
            message for message in self.general_errors if message not in messages
        )
        return tuple(messages)


class InitializationState:
    """Form state and validation independent from the application's event loop."""

    _INTEGER_FIELDS = (
        "width",
        "height",
        "num_robots",
        "num_items",
        "seed",
        "max_steps",
    )
    _COST_FIELDS = (
        "movement_cost",
        "pickup_cost",
        "drop_cost",
        "wait_cost",
    )

    def __init__(
        self,
        width: int,
        height: int,
        num_robots: int,
        num_items: int,
        seed: int,
        max_steps: int,
        movement_cost: float = 1,
        pickup_cost: float = 1,
        wait_cost: float = 0,
        controller_choices: Sequence[tuple[str, str]] = (
            ("random", "Random"),
            ("nearest_item", "Nearest Item"),
        ),
        drop_cost: float = 1,
    ) -> None:
        self.fields: dict[str, TextInput] = {
            "width": TextInput(str(width), character_filter=_integer_characters),
            "height": TextInput(str(height), character_filter=_integer_characters),
            "num_robots": TextInput(
                str(num_robots), character_filter=_integer_characters
            ),
            "num_items": TextInput(str(num_items), character_filter=_integer_characters),
            "seed": TextInput(str(seed), character_filter=_integer_characters),
            "max_steps": TextInput(
                str(max_steps), character_filter=_integer_characters
            ),
            "movement_cost": TextInput(
                self._format_number(movement_cost),
                character_filter=_number_characters,
            ),
            "pickup_cost": TextInput(
                self._format_number(pickup_cost),
                character_filter=_number_characters,
            ),
            "drop_cost": TextInput(
                self._format_number(drop_cost),
                character_filter=_number_characters,
            ),
            "wait_cost": TextInput(
                self._format_number(wait_cost),
                character_filter=_number_characters,
            ),
        }
        self.controller_choices = tuple(controller_choices)
        if not self.controller_choices:
            raise ValueError("controller_choices cannot be empty")
        choice_keys = tuple(key for key, _ in self.controller_choices)
        self._default_controller_key = (
            "random" if "random" in choice_keys else choice_keys[0]
        )
        # Selectors are retained when the count shrinks so growing it again
        # restores prior per-robot assignments instead of silently resetting.
        self.controller_selectors: list[ChoiceSelector] = []
        self.robot_mode: RobotMode = "random"
        self.robot_drafts: list[RobotDraft] = []
        self.selected_robot = 0
        self.scroll_offset = 0
        self._manual_initialized = False
        self._preview_key: tuple[object, ...] | None = None
        self._preview_scenario: InitialScenario | None = None
        self._preview_error: str | None = None
        self.sync_robot_drafts()

    def invalidate_preview(self) -> None:
        self._preview_key = None
        self._preview_scenario = None
        self._preview_error = None

    def sync_robot_drafts(self) -> None:
        """Keep manual rows aligned with a currently valid robot count."""

        try:
            count = int(self.fields["num_robots"].text)
        except ValueError:
            return
        if count < 0:
            return

        # A bad pasted value should not freeze the GUI by allocating an
        # unbounded list.  Counts beyond the known capacity remain invalid and
        # rows are created only once the dimensions can support them.
        capacity: int | None = None
        try:
            width = int(self.fields["width"].text)
            height = int(self.fields["height"].text)
            if width > 0 and height > 0:
                capacity = width * height
        except ValueError:
            pass
        if capacity is not None and count > capacity:
            return
        if count > 100_000:
            return

        while len(self.robot_drafts) < count:
            index = len(self.robot_drafts)
            width = self._positive_width_or_one()
            self.robot_drafts.append(
                RobotDraft.create((index % width, index // width))
            )
        while len(self.controller_selectors) < count:
            self.controller_selectors.append(
                ChoiceSelector(
                    self.controller_choices,
                    value=self._default_controller_key,
                )
            )
        if len(self.robot_drafts) > count:
            del self.robot_drafts[count:]
        self.selected_robot = max(0, min(self.selected_robot, max(0, count - 1)))
        self.invalidate_preview()

    def set_robot_mode(self, mode: RobotMode) -> None:
        if mode not in ("random", "manual"):
            raise ValueError("mode must be 'random' or 'manual'")
        if mode == self.robot_mode:
            return
        if mode == "manual" and not self._manual_initialized:
            self._seed_manual_rows_from_random_preview()
            self._manual_initialized = True
        self.robot_mode = mode
        self.invalidate_preview()

    def reroll(self) -> int:
        """Choose a fresh seed, update the field, and return it."""

        try:
            previous = int(self.fields["seed"].text)
        except ValueError:
            previous = None
        generator = random.SystemRandom()
        seed = generator.randrange(-(2**31), 2**31)
        while seed == previous:
            seed = generator.randrange(-(2**31), 2**31)
        self.fields["seed"].set_text(str(seed))
        self.invalidate_preview()
        return seed

    def validate(self) -> SetupValidation:
        result = SetupValidation()
        values: dict[str, int] = {}
        costs: dict[str, float] = {}

        specifications = {
            "width": ("Grid width", True),
            "height": ("Grid height", True),
            "num_robots": ("Robot count", False),
            "num_items": ("Item count", False),
            "seed": ("Seed", None),
            "max_steps": ("Maximum steps", False),
        }
        for name, (label, positive) in specifications.items():
            text = self.fields[name].text
            try:
                value = int(text)
            except ValueError:
                result.field_errors[name] = f"{label} must be an integer."
                continue
            if positive is True and value <= 0:
                result.field_errors[name] = f"{label} must be positive."
                continue
            if positive is False and value < 0:
                result.field_errors[name] = f"{label} cannot be negative."
                continue
            values[name] = value

        cost_specifications = {
            "movement_cost": "Move cost",
            "pickup_cost": "Pickup cost",
            "drop_cost": "Drop cost",
            "wait_cost": "Wait cost",
        }
        for name, label in cost_specifications.items():
            text = self.fields[name].text
            try:
                value = float(text)
            except ValueError:
                result.field_errors[name] = f"{label} must be a number."
                continue
            if not math.isfinite(value) or value < 0:
                result.field_errors[name] = f"{label} must be nonnegative and finite."
                continue
            costs[name] = value

        dimensions_ready = "width" in values and "height" in values
        counts_ready = "num_robots" in values and "num_items" in values
        if dimensions_ready and counts_ready:
            capacity = values["width"] * values["height"]
            total = values["num_robots"] + values["num_items"]
            if total > capacity:
                message = f"{total} entities do not fit in {capacity} grid cells."
                result.field_errors["num_robots"] = message
                result.field_errors["num_items"] = message

        configurations: tuple[RobotConfiguration, ...] | None = None
        controller_keys: tuple[str, ...] = ()
        if "num_robots" in values:
            count = values["num_robots"]
            if len(self.controller_selectors) < count:
                result.general_errors.append(
                    "Finish entering a valid robot count to select its controllers."
                )
            else:
                controller_keys = tuple(
                    selector.value for selector in self.controller_selectors[:count]
                )
        if self.robot_mode == "manual" and "num_robots" in values:
            count = values["num_robots"]
            if len(self.robot_drafts) != count:
                result.general_errors.append(
                    "Finish entering a valid robot count to create its configuration rows."
                )
            else:
                parsed: list[RobotConfiguration | None] = []
                positions: dict[tuple[int, int], list[int]] = {}
                for index, draft in enumerate(self.robot_drafts):
                    errors: dict[str, str] = {}
                    try:
                        x = int(draft.x.text)
                    except ValueError:
                        x = 0
                        errors["x"] = f"Robot {index + 1} X must be an integer."
                    try:
                        y = int(draft.y.text)
                    except ValueError:
                        y = 0
                        errors["y"] = f"Robot {index + 1} Y must be an integer."
                    try:
                        battery = float(draft.battery.text)
                    except ValueError:
                        battery = 0.0
                        errors["battery"] = (
                            f"Robot {index + 1} battery must be a number."
                        )
                    else:
                        if not math.isfinite(battery) or battery < 0:
                            errors["battery"] = (
                                f"Robot {index + 1} battery must be nonnegative and finite."
                            )

                    if dimensions_ready:
                        if "x" not in errors and not 0 <= x < values["width"]:
                            errors["x"] = (
                                f"Robot {index + 1} X must be between 0 and "
                                f"{values['width'] - 1}."
                            )
                        if "y" not in errors and not 0 <= y < values["height"]:
                            errors["y"] = (
                                f"Robot {index + 1} Y must be between 0 and "
                                f"{values['height'] - 1}."
                            )
                    if not errors:
                        position = (x, y)
                        positions.setdefault(position, []).append(index)
                        parsed.append(
                            RobotConfiguration(
                                position=position,
                                battery_level=battery,
                                controller_key=(
                                    controller_keys[index]
                                    if index < len(controller_keys)
                                    else self._default_controller_key
                                ),
                            )
                        )
                    else:
                        parsed.append(None)
                        result.robot_errors[index] = errors

                for position, indices in positions.items():
                    if len(indices) <= 1:
                        continue
                    for index in indices:
                        result.robot_errors.setdefault(index, {})["position"] = (
                            f"Robots cannot share cell {position}."
                        )
                if not result.robot_errors and all(value is not None for value in parsed):
                    configurations = tuple(
                        value for value in parsed if value is not None
                    )

        if result.field_errors or result.robot_errors or result.general_errors:
            return result
        if any(name not in values for name in self._INTEGER_FIELDS):
            return result
        if any(name not in costs for name in self._COST_FIELDS):
            return result

        result.configuration = SetupConfiguration(
            width=values["width"],
            height=values["height"],
            num_robots=values["num_robots"],
            num_items=values["num_items"],
            seed=values["seed"],
            max_steps=values["max_steps"],
            action_battery_costs=ActionBatteryCosts(
                movement=costs["movement_cost"],
                pickup=costs["pickup_cost"],
                drop=costs["drop_cost"],
                wait=costs["wait_cost"],
            ),
            robot_mode=self.robot_mode,
            controller_keys=controller_keys,
            robot_configurations=configurations,
        )
        return result

    def preview_scenario(self) -> InitialScenario | None:
        validation = self.validate()
        configuration = validation.configuration
        if configuration is None:
            self._preview_scenario = None
            self._preview_error = None
            return None

        key = (
            configuration.width,
            configuration.height,
            configuration.num_robots,
            configuration.num_items,
            configuration.seed,
            configuration.action_battery_costs,
            configuration.robot_mode,
            configuration.controller_keys,
            configuration.robot_configurations,
        )
        if key == self._preview_key:
            return self._preview_scenario

        self._preview_key = key
        try:
            self._preview_scenario = create_initial_scenario(
                width=configuration.width,
                height=configuration.height,
                num_robots=configuration.num_robots,
                num_items=configuration.num_items,
                seed=configuration.seed,
                robot_configurations=configuration.robot_configurations,
                action_battery_costs=configuration.action_battery_costs,
                controller_keys=configuration.controller_keys,
            )
        except ValueError as exc:
            self._preview_scenario = None
            self._preview_error = str(exc)
        else:
            self._preview_error = None
        return self._preview_scenario

    @property
    def preview_error(self) -> str | None:
        return self._preview_error

    def _positive_width_or_one(self) -> int:
        try:
            return max(1, int(self.fields["width"].text))
        except ValueError:
            return 1

    @staticmethod
    def _format_number(value: float) -> str:
        number = float(value)
        return str(int(number)) if number.is_integer() else str(number)

    def _seed_manual_rows_from_random_preview(self) -> None:
        scenario = self.preview_scenario()
        if scenario is None:
            self.sync_robot_drafts()
            return
        self.robot_drafts = [
            RobotDraft.create(configuration.position, configuration.battery_level)
            for configuration in scenario.robot_configurations
        ]
        self.selected_robot = 0


class PygameSimulationApp:
    """Run setup and interactive playback in a single resizable window."""

    _INITIAL_WINDOW_SIZE = (1080, 720)
    _MIN_WINDOW_SIZE = (800, 600)
    _UI_FPS = 60
    _TOOLBAR_HEIGHT = 102
    _ROW_HEIGHT = 42

    def __init__(
        self,
        width: int,
        height: int,
        num_robots: int,
        num_items: int,
        seed: int,
        max_steps: int,
        step_rate: int,
        show_labels: bool = True,
        movement_cost: float = 1,
        pickup_cost: float = 1,
        wait_cost: float = 0,
        controller_registry: ControllerRegistry | None = None,
        drop_cost: float = 1,
    ) -> None:
        if type(step_rate) is not int or not 1 <= step_rate <= 30:
            raise ValueError("step_rate must be an integer between 1 and 30")

        self.controller_registry = (
            create_default_controller_registry()
            if controller_registry is None
            else controller_registry
        )
        controller_choices = tuple(
            (definition.key, definition.display_name)
            for definition in self.controller_registry.definitions
        )
        if not controller_choices:
            raise ValueError("controller_registry must contain at least one controller")

        self.setup = InitializationState(
            width=width,
            height=height,
            num_robots=num_robots,
            num_items=num_items,
            seed=seed,
            max_steps=max_steps,
            movement_cost=movement_cost,
            pickup_cost=pickup_cost,
            drop_cost=drop_cost,
            wait_cost=wait_cost,
            controller_choices=controller_choices,
        )
        self.show_labels = bool(show_labels)
        self.session: SimulationSession | None = None
        self.screen_name: Literal["setup", "simulation"] = "setup"
        self.running = False
        self._last_world: SimulationWorld | None = None
        self._step_rate = step_rate
        self._accumulator = 0.0

        self.surface: pygame.Surface | None = None
        self._clock: pygame.time.Clock | None = None
        self._font: pygame.font.Font | None = None
        self._small_font: pygame.font.Font | None = None
        self._title_font: pygame.font.Font | None = None

        self._setup_field_labels = {
            "width": "Grid width",
            "height": "Grid height",
            "num_robots": "Robots",
            "num_items": "Items",
            "seed": "Seed",
            "max_steps": "Maximum steps",
            "movement_cost": "Move cost",
            "pickup_cost": "Pickup cost",
            "drop_cost": "Drop cost",
            "wait_cost": "Wait cost",
        }
        self._random_button = Button("Random")
        self._manual_button = Button("Manual")
        self._reroll_button = Button("Reroll")
        self._start_button = Button("Start simulation", primary=True)
        self._back_button = Button("Back")
        self._play_button = Button("Play", primary=True)
        self._forward_button = Button("Forward")
        self._new_setup_button = Button("New setup")
        self._rate_slider = IntegerSlider(1, 30, step_rate)

        self._left_panel = pygame.Rect(0, 0, 1, 1)
        self._preview_panel = pygame.Rect(0, 0, 1, 1)
        self._preview_grid = pygame.Rect(0, 0, 1, 1)
        self._manual_view = pygame.Rect(0, 0, 1, 1)
        self._manual_row_rects: list[pygame.Rect] = []
        self._simulation_grid_area = pygame.Rect(0, 0, 1, 1)
        self._status_panel = pygame.Rect(0, 0, 1, 1)
        self._status_view = pygame.Rect(0, 0, 1, 1)
        self._status_scroll_offset = 0
        self._status_row_height = 88

    @property
    def setup_is_valid(self) -> bool:
        return self.setup.validate().valid

    def initialize(self) -> None:
        """Initialize Pygame; separated from ``__init__`` for testability."""

        if self.surface is not None:
            return
        pygame.init()
        self.surface = pygame.display.set_mode(
            self._INITIAL_WINDOW_SIZE,
            pygame.RESIZABLE,
        )
        pygame.display.set_caption("Multi-Agent 2D Simulation - Setup")
        self._clock = pygame.time.Clock()
        self._font = pygame.font.Font(None, 22)
        self._small_font = pygame.font.Font(None, 18)
        self._title_font = pygame.font.Font(None, 34)
        self.running = True
        self._layout_setup()

    def run(self) -> SimulationWorld | None:
        """Run until closed and return the most recently active world, if any."""

        self.initialize()
        assert self._clock is not None
        try:
            while self.running:
                elapsed_seconds = self._clock.tick(self._UI_FPS) / 1000.0
                self._layout_current_screen()
                for event in pygame.event.get():
                    self.process_event(event)
                    if not self.running:
                        break
                self.update(elapsed_seconds)
                self.render()
        finally:
            current_world = self.session.world if self.session is not None else None
            if current_world is not None:
                self._last_world = current_world
            pygame.quit()
            self.surface = None
            self._clock = None
        return self._last_world

    def process_event(self, event: pygame.event.Event) -> bool:
        """Process one Pygame event and return whether the app remains open."""

        if event.type == pygame.QUIT:
            self.running = False
            return False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.running = False
            return False
        if event.type == pygame.VIDEORESIZE:
            size = (
                max(self._MIN_WINDOW_SIZE[0], event.w),
                max(self._MIN_WINDOW_SIZE[1], event.h),
            )
            self.surface = pygame.display.set_mode(size, pygame.RESIZABLE)
            self._layout_current_screen()
            return True

        if self.screen_name == "setup":
            self._process_setup_event(event)
        else:
            self._process_simulation_event(event)
        return self.running

    def update(self, elapsed_seconds: float) -> None:
        """Advance playback using elapsed time while keeping rendering at 60 FPS."""

        if elapsed_seconds < 0:
            raise ValueError("elapsed_seconds cannot be negative")
        session = self.session
        if self.screen_name != "simulation" or session is None or not session.playing:
            self._accumulator = 0.0
            return

        interval = 1.0 / session.step_rate
        self._accumulator += elapsed_seconds
        # A generous cap prevents an unresponsive catch-up spiral after the
        # process is suspended, while leaving any unprocessed time queued.
        steps_this_frame = 0
        while self._accumulator + 1e-12 >= interval and steps_this_frame < 120:
            if not session.can_step_forward:
                session.pause()
                self._accumulator = 0.0
                break
            session.step_forward()
            self._accumulator -= interval
            steps_this_frame += 1
            if not session.playing:
                self._accumulator = 0.0
                break

    def render(self) -> None:
        if self.surface is None:
            return
        self.surface.fill(PALETTE.background)
        if self.screen_name == "setup":
            self._draw_setup()
        else:
            self._draw_simulation()
        pygame.display.flip()

    def close(self) -> None:
        """Request loop termination; useful to embedding code and tests."""

        self.running = False

    def _layout_current_screen(self) -> None:
        if self.screen_name == "setup":
            self._layout_setup()
        else:
            self._layout_simulation()

    def _layout_setup(self) -> None:
        if self.surface is None:
            return
        width, height = self.surface.get_size()
        margin = 22
        content_top = 68
        gap = 16
        left_width = max(430, min(560, int(width * 0.52)))
        self._left_panel = pygame.Rect(
            margin,
            content_top,
            left_width,
            height - content_top - margin,
        )
        self._preview_panel = pygame.Rect(
            self._left_panel.right + gap,
            content_top,
            width - self._left_panel.right - gap - margin,
            height - content_top - margin,
        )

        inner = self._left_panel.inflate(-28, -24)
        field_gap = 12
        row_height = 62
        field_rows = (
            ("width", "height"),
            ("num_robots", "num_items"),
            ("seed", "max_steps"),
            ("movement_cost", "pickup_cost", "drop_cost", "wait_cost"),
        )
        for row, names in enumerate(field_rows):
            y = inner.y + 24 + row * row_height
            field_width = (
                inner.width - field_gap * (len(names) - 1)
            ) // len(names)
            for column, name in enumerate(names):
                x = inner.x + column * (field_width + field_gap)
                self.setup.fields[name].rect = pygame.Rect(x, y + 20, field_width, 32)

        mode_y = inner.y + 24 + len(field_rows) * row_height + 2
        self._random_button.rect = pygame.Rect(inner.x, mode_y + 20, 94, 32)
        self._manual_button.rect = pygame.Rect(inner.x + 100, mode_y + 20, 94, 32)
        self._reroll_button.rect = pygame.Rect(inner.right - 82, mode_y + 20, 82, 32)

        footer_height = 104
        manual_top = mode_y + 64
        self._manual_view = pygame.Rect(
            inner.x,
            manual_top + 24,
            inner.width,
            max(40, inner.bottom - footer_height - manual_top - 24),
        )
        self._clamp_manual_scroll()
        self._start_button.rect = pygame.Rect(
            inner.right - 154,
            inner.bottom - 38,
            154,
            36,
        )

        self._manual_row_rects = []
        for index, draft in enumerate(self.setup.robot_drafts):
            row_rect = pygame.Rect(
                self._manual_view.x,
                self._manual_view.y + index * self._ROW_HEIGHT - self.setup.scroll_offset,
                self._manual_view.width - 10,
                self._ROW_HEIGHT - 4,
            )
            self._manual_row_rects.append(row_rect)
            label_width = 68
            gap = 4
            selector = self.setup.controller_selectors[index]
            if self.setup.robot_mode == "manual":
                remaining = row_rect.width - label_width - gap * 4
                x_width = 42
                y_width = 42
                battery_width = 68
                controller_width = max(
                    80,
                    remaining - x_width - y_width - battery_width,
                )
                x = row_rect.x + label_width
                draft.x.rect = pygame.Rect(
                    x, row_rect.y + 4, x_width, row_rect.height - 8
                )
                x += x_width + gap
                draft.y.rect = pygame.Rect(
                    x, row_rect.y + 4, y_width, row_rect.height - 8
                )
                x += y_width + gap
                draft.battery.rect = pygame.Rect(
                    x, row_rect.y + 4, battery_width, row_rect.height - 8
                )
                x += battery_width + gap
                selector.rect = pygame.Rect(
                    x, row_rect.y + 4, controller_width, row_rect.height - 8
                )
            else:
                selector.rect = pygame.Rect(
                    row_rect.x + label_width,
                    row_rect.y + 4,
                    row_rect.width - label_width - gap,
                    row_rect.height - 8,
                )

        preview_inner = self._preview_panel.inflate(-28, -28)
        preview_area = pygame.Rect(
            preview_inner.x,
            preview_inner.y + 58,
            preview_inner.width,
            max(1, preview_inner.height - 58),
        )
        validation = self.setup.validate()
        configuration = validation.configuration
        grid_width = configuration.width if configuration else self._safe_positive_field("width")
        grid_height = configuration.height if configuration else self._safe_positive_field("height")
        if grid_width is None or grid_height is None:
            self._preview_grid = preview_area.inflate(-24, -24)
        else:
            scale = min(preview_area.width / grid_width, preview_area.height / grid_height)
            pixel_width = max(1, round(grid_width * scale))
            pixel_height = max(1, round(grid_height * scale))
            self._preview_grid = pygame.Rect(0, 0, pixel_width, pixel_height)
            self._preview_grid.center = preview_area.center

    def _layout_simulation(self) -> None:
        if self.surface is None:
            return
        width, height = self.surface.get_size()
        top = 16
        x = 20
        self._back_button.rect = pygame.Rect(x, top, 74, 34)
        x += 82
        self._play_button.rect = pygame.Rect(x, top, 102, 34)
        x += 110
        self._forward_button.rect = pygame.Rect(x, top, 84, 34)
        x += 114
        self._rate_slider.rect = pygame.Rect(x + 94, top + 2, min(220, max(90, width - x - 310)), 30)
        self._new_setup_button.rect = pygame.Rect(width - 126, top, 106, 34)

        content = pygame.Rect(
            20,
            self._TOOLBAR_HEIGHT + 16,
            width - 40,
            height - self._TOOLBAR_HEIGHT - 36,
        )
        gap = 16
        status_width = max(240, min(320, round(content.width * 0.3)))
        self._status_panel = pygame.Rect(
            content.right - status_width,
            content.y,
            status_width,
            content.height,
        )
        self._simulation_grid_area = pygame.Rect(
            content.x,
            content.y,
            max(1, content.width - status_width - gap),
            content.height,
        )
        self._status_view = pygame.Rect(
            self._status_panel.x + 12,
            self._status_panel.y + 48,
            self._status_panel.width - 24,
            max(1, self._status_panel.height - 60),
        )
        self._clamp_status_scroll()

    def _process_setup_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
            self._focus_next_input(reverse=bool(event.mod & pygame.KMOD_SHIFT))
            return

        if self._random_button.handle_event(event):
            self.setup.set_robot_mode("random")
        if self._manual_button.handle_event(event):
            self.setup.set_robot_mode("manual")
        if self._reroll_button.handle_event(event):
            self.setup.reroll()

        validation = self.setup.validate()
        self._start_button.enabled = validation.valid
        if self._start_button.handle_event(event):
            self._start_simulation()
            return

        changed_fields: list[str] = []
        for name, input_field in self.setup.fields.items():
            if input_field.handle_event(event):
                changed_fields.append(name)
        if changed_fields:
            if any(name in ("width", "height", "num_robots") for name in changed_fields):
                self.setup.sync_robot_drafts()
            self.setup.invalidate_preview()

        if event.type == pygame.MOUSEWHEEL and self._manual_view.collidepoint(
            getattr(event, "pos", pygame.mouse.get_pos())
        ):
            self.setup.scroll_offset -= event.y * self._ROW_HEIGHT
            self._clamp_manual_scroll()
            self._layout_setup()
            return

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            for index, rect in enumerate(self._manual_row_rects):
                if rect.collidepoint(event.pos) and self._manual_view.collidepoint(event.pos):
                    self.setup.selected_robot = index
                    break

        for selector in self.setup.controller_selectors[: len(self.setup.robot_drafts)]:
            is_mouse_click = (
                event.type == pygame.MOUSEBUTTONDOWN and event.button in (1, 3)
            )
            if is_mouse_click and not self._manual_view.collidepoint(event.pos):
                selector.focused = False
                continue
            if is_mouse_click and not selector.rect.colliderect(self._manual_view):
                selector.focused = False
                continue
            if selector.handle_event(event):
                self.setup.invalidate_preview()

        if self.setup.robot_mode != "manual":
            return

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._place_selected_robot(event.pos)

        is_mouse_click = (
            event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
        )
        for draft in self.setup.robot_drafts:
            inputs = (draft.x, draft.y, draft.battery)
            if is_mouse_click and not self._manual_view.collidepoint(event.pos):
                for input_field in inputs:
                    input_field.focused = False
                continue

            changed = False
            for input_field in inputs:
                if is_mouse_click and not input_field.rect.colliderect(
                    self._manual_view
                ):
                    input_field.focused = False
                    continue
                changed = input_field.handle_event(event) or changed
            if changed:
                self.setup.invalidate_preview()

    def _process_simulation_event(self, event: pygame.event.Event) -> None:
        session = self.session
        if session is None:
            return

        if event.type == pygame.MOUSEWHEEL and self._status_view.collidepoint(
            getattr(event, "pos", pygame.mouse.get_pos())
        ):
            self._status_scroll_offset -= event.y * self._status_row_height
            self._clamp_status_scroll()
            return

        self._back_button.enabled = session.can_step_back
        self._forward_button.enabled = session.can_step_forward
        self._play_button.enabled = session.can_step_forward
        controls_have_focus = self._rate_slider.focused or any(
            button.enabled and button.focused
            for button in (
                self._back_button,
                self._play_button,
                self._forward_button,
                self._new_setup_button,
            )
        )
        if event.type == pygame.KEYDOWN and not controls_have_focus:
            if event.key == pygame.K_SPACE and session.can_step_forward:
                session.toggle_playing()
                self._accumulator = 0.0
            elif event.key == pygame.K_LEFT and session.can_step_back:
                session.pause()
                session.step_back()
                self._accumulator = 0.0
            elif event.key == pygame.K_RIGHT and session.can_step_forward:
                session.pause()
                session.step_forward()
                self._accumulator = 0.0

        if self._back_button.handle_event(event):
            session.pause()
            session.step_back()
            self._accumulator = 0.0
        if self._play_button.handle_event(event):
            session.toggle_playing()
            self._accumulator = 0.0
        if self._forward_button.handle_event(event):
            session.pause()
            session.step_forward()
            self._accumulator = 0.0
        if self._new_setup_button.handle_event(event):
            self._return_to_setup()
            return
        if self._rate_slider.handle_event(event):
            session.step_rate = self._rate_slider.value
            self._step_rate = self._rate_slider.value

    def _focus_next_input(self, *, reverse: bool) -> None:
        inputs = list(self.setup.fields.values())
        if self.setup.robot_mode == "manual":
            for draft in self.setup.robot_drafts:
                inputs.extend((draft.x, draft.y, draft.battery))
        if not inputs:
            return
        focused = next((index for index, value in enumerate(inputs) if value.focused), -1)
        offset = -1 if reverse else 1
        if focused < 0:
            next_index = len(inputs) - 1 if reverse else 0
        else:
            next_index = (focused + offset) % len(inputs)
        for index, input_field in enumerate(inputs):
            input_field.focused = index == next_index
        if next_index >= len(self.setup.fields):
            robot_index = (next_index - len(self.setup.fields)) // 3
            self.setup.selected_robot = robot_index
            self._scroll_robot_into_view(robot_index)

    def _place_selected_robot(self, position: tuple[int, int]) -> None:
        if not self._preview_grid.collidepoint(position) or not self.setup.robot_drafts:
            return
        width = self._safe_positive_field("width")
        height = self._safe_positive_field("height")
        if width is None or height is None:
            return
        relative_x = position[0] - self._preview_grid.x
        relative_y = position[1] - self._preview_grid.y
        grid_x = min(width - 1, int(relative_x * width / self._preview_grid.width))
        grid_y = min(height - 1, int(relative_y * height / self._preview_grid.height))
        draft = self.setup.robot_drafts[self.setup.selected_robot]
        draft.x.set_text(str(grid_x))
        draft.y.set_text(str(grid_y))
        self.setup.invalidate_preview()

    def _start_simulation(self) -> None:
        validation = self.setup.validate()
        configuration = validation.configuration
        if configuration is None:
            return
        scenario = self.setup.preview_scenario()
        if scenario is None:
            return
        self.session = SimulationSession(
            scenario,
            max_steps=configuration.max_steps,
            step_rate=self._step_rate,
            controller_registry=self.controller_registry,
        )
        self._rate_slider.value = self._step_rate
        self._last_world = self.session.world
        self.screen_name = "simulation"
        self._accumulator = 0.0
        self._status_scroll_offset = 0
        pygame.display.set_caption("Multi-Agent 2D Simulation")
        self._layout_simulation()

    def _return_to_setup(self) -> None:
        if self.session is not None:
            self.session.pause()
            self._step_rate = self.session.step_rate
        self.session = None
        self._last_world = None
        self.screen_name = "setup"
        self._accumulator = 0.0
        self._status_scroll_offset = 0
        pygame.display.set_caption("Multi-Agent 2D Simulation - Setup")
        self._layout_setup()

    def _draw_setup(self) -> None:
        assert self.surface is not None
        assert self._font is not None
        assert self._small_font is not None
        assert self._title_font is not None

        self._layout_setup()
        mouse = pygame.mouse.get_pos()
        draw_text(
            self.surface,
            self._title_font,
            "Configure simulation",
            (22, 22),
        )
        draw_text(
            self.surface,
            self._small_font,
            "Set the world, inspect the preview, then start paused at step 0.",
            (290, 31),
            color=PALETTE.text_muted,
        )
        self._draw_panel(self._left_panel)
        self._draw_panel(self._preview_panel)

        validation = self.setup.validate()
        self._start_button.enabled = validation.valid and self.setup.preview_scenario() is not None

        for name, label in self._setup_field_labels.items():
            input_field = self.setup.fields[name]
            draw_text(
                self.surface,
                self._small_font,
                label,
                (input_field.rect.x, input_field.rect.y - 18),
                color=PALETTE.text_muted,
            )
            input_field.draw(
                self.surface,
                self._font,
                invalid=name in validation.field_errors,
            )

        mode_y = self._random_button.rect.y - 18
        draw_text(
            self.surface,
            self._small_font,
            "Robot configuration",
            (self._random_button.rect.x, mode_y),
            color=PALETTE.text_muted,
        )
        self._random_button.primary = self.setup.robot_mode == "random"
        self._manual_button.primary = self.setup.robot_mode == "manual"
        self._random_button.draw(self.surface, self._small_font, mouse)
        self._manual_button.draw(self.surface, self._small_font, mouse)
        self._reroll_button.draw(self.surface, self._small_font, mouse)

        self._draw_manual_rows(validation)

        messages = list(validation.messages)
        if self.setup.preview_error:
            messages.append(self.setup.preview_error)
        error_y = self._left_panel.bottom - 84
        if messages:
            visible = messages[:2]
            if len(messages) > 2:
                visible[-1] = f"{visible[-1]} (+{len(messages) - 2} more)"
            draw_wrapped_text(
                self.surface,
                self._small_font,
                " • ".join(visible),
                pygame.Rect(
                    self._left_panel.x + 14,
                    error_y,
                    self._start_button.rect.x - self._left_panel.x - 24,
                    44,
                ),
                color=PALETTE.error,
                max_lines=2,
            )
        else:
            draw_text(
                self.surface,
                self._small_font,
                "Configuration ready",
                (self._left_panel.x + 14, error_y + 8),
                color=(41, 126, 83),
            )
        self._start_button.draw(self.surface, self._small_font, mouse)
        self._draw_preview()

    def _draw_manual_rows(self, validation: SetupValidation) -> None:
        assert self.surface is not None
        assert self._small_font is not None
        instruction = (
            "Select a robot and cell; fields are X / Y / Battery / Controller."
            if self.setup.robot_mode == "manual"
            else "Positions and battery 100 are seeded; choose each controller."
        )
        draw_text(
            self.surface,
            self._small_font,
            instruction,
            (self._manual_view.x, self._manual_view.y - 20),
            color=PALETTE.text_muted,
        )
        if not self.setup.robot_drafts:
            draw_text(
                self.surface,
                self._small_font,
                "No robots to configure.",
                (self._manual_view.x + 8, self._manual_view.y + 10),
                color=PALETTE.text_muted,
            )
            return

        old_clip = self.surface.get_clip()
        self.surface.set_clip(self._manual_view)
        mouse = pygame.mouse.get_pos()
        for index, (draft, row_rect) in enumerate(
            zip(self.setup.robot_drafts, self._manual_row_rects, strict=True)
        ):
            if not row_rect.colliderect(self._manual_view):
                continue
            if index == self.setup.selected_robot:
                pygame.draw.rect(
                    self.surface,
                    PALETTE.primary_soft,
                    row_rect,
                    border_radius=5,
                )
            draw_text(
                self.surface,
                self._small_font,
                f"robot_{index + 1}",
                (row_rect.x + 6, row_rect.centery - self._small_font.get_height() // 2),
                color=PALETTE.text,
            )
            errors = validation.robot_errors.get(index, {})
            if self.setup.robot_mode == "manual":
                draft.x.draw(
                    self.surface,
                    self._small_font,
                    invalid="x" in errors or "position" in errors,
                )
                draft.y.draw(
                    self.surface,
                    self._small_font,
                    invalid="y" in errors or "position" in errors,
                )
                draft.battery.draw(
                    self.surface,
                    self._small_font,
                    invalid="battery" in errors,
                )
            self.setup.controller_selectors[index].draw(
                self.surface,
                self._small_font,
                mouse,
            )
        self.surface.set_clip(old_clip)

        content_height = len(self.setup.robot_drafts) * self._ROW_HEIGHT
        if content_height > self._manual_view.height:
            track = pygame.Rect(
                self._manual_view.right - 4,
                self._manual_view.y,
                4,
                self._manual_view.height,
            )
            pygame.draw.rect(self.surface, PALETTE.border, track, border_radius=2)
            thumb_height = max(
                20,
                round(track.height * self._manual_view.height / content_height),
            )
            maximum = content_height - self._manual_view.height
            thumb_y = track.y + round(
                (track.height - thumb_height) * self.setup.scroll_offset / maximum
            )
            pygame.draw.rect(
                self.surface,
                PALETTE.border_strong,
                (track.x, thumb_y, track.width, thumb_height),
                border_radius=2,
            )

    def _draw_preview(self) -> None:
        assert self.surface is not None
        assert self._font is not None
        assert self._small_font is not None
        draw_text(
            self.surface,
            self._font,
            "Initial layout preview",
            (self._preview_panel.x + 16, self._preview_panel.y + 14),
        )
        instruction = (
            "Click a cell to place the selected robot."
            if self.setup.robot_mode == "manual" and self.setup.robot_drafts
            else "Orange diamonds are items; blue circles are robots."
        )
        draw_text(
            self.surface,
            self._small_font,
            instruction,
            (self._preview_panel.x + 16, self._preview_panel.y + 38),
            color=PALETTE.text_muted,
        )

        scenario = self.setup.preview_scenario()
        if scenario is None:
            pygame.draw.rect(
                self.surface,
                PALETTE.panel_alt,
                self._preview_grid,
                border_radius=5,
            )
            draw_text(
                self.surface,
                self._small_font,
                "Fix the highlighted fields to show a preview.",
                (
                    self._preview_grid.centerx - 135,
                    self._preview_grid.centery - 8,
                ),
                color=PALETTE.text_muted,
            )
            return
        self._draw_scenario_grid(scenario)

    def _draw_scenario_grid(self, scenario: InitialScenario) -> None:
        assert self.surface is not None
        self._draw_grid_background(scenario.width, scenario.height, self._preview_grid)
        cell_width = self._preview_grid.width / scenario.width
        cell_height = self._preview_grid.height / scenario.height

        for position in scenario.item_positions:
            center = self._position_center(position, self._preview_grid, cell_width, cell_height)
            radius = max(2, round(min(cell_width, cell_height) * 0.34))
            points = [
                (center[0], center[1] - radius),
                (center[0] + radius, center[1]),
                (center[0], center[1] + radius),
                (center[0] - radius, center[1]),
            ]
            pygame.draw.polygon(self.surface, (238, 139, 46), points)
            pygame.draw.polygon(self.surface, (145, 77, 14), points, width=1)

        for index, configuration in enumerate(scenario.robot_configurations):
            center = self._position_center(
                configuration.position,
                self._preview_grid,
                cell_width,
                cell_height,
            )
            radius = max(2, round(min(cell_width, cell_height) * 0.27))
            if self.setup.robot_mode == "manual" and index == self.setup.selected_robot:
                pygame.draw.circle(self.surface, PALETTE.primary_soft, center, radius + 5)
            pygame.draw.circle(self.surface, (52, 101, 212), center, radius)
            pygame.draw.circle(self.surface, (29, 62, 133), center, radius, width=1)

    def _draw_simulation(self) -> None:
        assert self.surface is not None
        assert self._font is not None
        assert self._small_font is not None
        session = self.session
        if session is None:
            return
        self._layout_simulation()
        mouse = pygame.mouse.get_pos()

        pygame.draw.rect(
            self.surface,
            PALETTE.panel,
            (0, 0, self.surface.get_width(), self._TOOLBAR_HEIGHT),
        )
        pygame.draw.line(
            self.surface,
            PALETTE.border,
            (0, self._TOOLBAR_HEIGHT - 1),
            (self.surface.get_width(), self._TOOLBAR_HEIGHT - 1),
        )
        self._back_button.enabled = session.can_step_back
        self._forward_button.enabled = session.can_step_forward
        self._play_button.enabled = session.can_step_forward
        self._play_button.label = "Pause" if session.playing else "Play"
        self._back_button.draw(self.surface, self._small_font, mouse)
        self._play_button.draw(self.surface, self._small_font, mouse)
        self._forward_button.draw(self.surface, self._small_font, mouse)
        self._new_setup_button.draw(self.surface, self._small_font, mouse)

        draw_text(
            self.surface,
            self._small_font,
            f"Rate: {session.step_rate} steps/s",
            (self._rate_slider.rect.x - 94, self._rate_slider.rect.y + 7),
            color=PALETTE.text_muted,
        )
        self._rate_slider.draw(self.surface)

        world = session.world
        robot_count = len(world.get_entities(Robot))
        item_count = len(world.get_entities(Item))
        status = "Running" if session.playing else "Paused"
        status_color = (36, 130, 82) if session.playing else PALETTE.text_muted
        draw_text(
            self.surface,
            self._small_font,
            status,
            (20, 67),
            color=status_color,
        )
        draw_text(
            self.surface,
            self._small_font,
            f"Step {session.cursor}  |  Recorded through {session.history_length}  |  "
            f"Maximum {session.max_steps}",
            (96, 67),
            color=PALETTE.text,
        )
        counts = f"Robots: {robot_count}   Items: {item_count}"
        counts_width = self._small_font.size(counts)[0]
        draw_text(
            self.surface,
            self._small_font,
            counts,
            (self.surface.get_width() - counts_width - 20, 67),
            color=PALETTE.text_muted,
        )

        available = self._simulation_grid_area
        scale = min(available.width / world.width, available.height / world.height)
        grid = pygame.Rect(
            0,
            0,
            max(1, round(world.width * scale)),
            max(1, round(world.height * scale)),
        )
        grid.center = available.center
        self._draw_world(world, grid)
        self._draw_robot_status_panel(session)

    def _draw_robot_status_panel(self, session: SimulationSession) -> None:
        assert self.surface is not None
        assert self._font is not None
        assert self._small_font is not None

        self._draw_panel(self._status_panel)
        draw_text(
            self.surface,
            self._font,
            "Robot status",
            (self._status_panel.x + 14, self._status_panel.y + 14),
        )

        robots = session.world.get_entities(Robot)
        if not robots:
            draw_text(
                self.surface,
                self._small_font,
                "No robots in this simulation.",
                (self._status_view.x + 4, self._status_view.y + 8),
                color=PALETTE.text_muted,
            )
            return

        old_clip = self.surface.get_clip()
        self.surface.set_clip(self._status_view)
        for index, robot in enumerate(robots):
            row = pygame.Rect(
                self._status_view.x,
                self._status_view.y
                + index * self._status_row_height
                - self._status_scroll_offset,
                self._status_view.width - 8,
                self._status_row_height - 6,
            )
            if not row.colliderect(self._status_view):
                continue
            pygame.draw.rect(
                self.surface,
                PALETTE.panel_alt if index % 2 == 0 else PALETTE.panel,
                row,
                border_radius=5,
            )
            battery = self._format_status_number(robot.battery_level)
            draw_text(
                self.surface,
                self._small_font,
                f"{robot.robot_id}   Battery: {battery}",
                (row.x + 8, row.y + 7),
            )

            controller_name = session.controller_display_name_for(robot.robot_id)
            carried_item_id = getattr(robot, "carried_item_id", None)
            carried = carried_item_id if carried_item_id is not None else "empty"
            draw_text(
                self.surface,
                self._small_font,
                f"{controller_name}   Carrying: {carried}",
                (row.x + 8, row.y + 28),
                color=PALETTE.text_muted,
            )

            warning = session.controller_errors.get(robot.robot_id)
            if warning:
                draw_wrapped_text(
                    self.surface,
                    self._small_font,
                    f"Warning: {warning}",
                    pygame.Rect(row.x + 8, row.y + 49, row.width - 16, 30),
                    color=PALETTE.error,
                    line_gap=1,
                    max_lines=2,
                )
        self.surface.set_clip(old_clip)

        content_height = len(robots) * self._status_row_height
        if content_height > self._status_view.height:
            track = pygame.Rect(
                self._status_view.right - 4,
                self._status_view.y,
                4,
                self._status_view.height,
            )
            pygame.draw.rect(self.surface, PALETTE.border, track, border_radius=2)
            thumb_height = max(
                20,
                round(track.height * self._status_view.height / content_height),
            )
            maximum = content_height - self._status_view.height
            thumb_y = track.y + round(
                (track.height - thumb_height)
                * self._status_scroll_offset
                / maximum
            )
            pygame.draw.rect(
                self.surface,
                PALETTE.border_strong,
                (track.x, thumb_y, track.width, thumb_height),
                border_radius=2,
            )

    def _draw_world(self, world: SimulationWorld, rect: pygame.Rect) -> None:
        assert self.surface is not None
        assert self._small_font is not None
        self._draw_grid_background(world.width, world.height, rect)
        cell_width = rect.width / world.width
        cell_height = rect.height / world.height
        minimum_cell = min(cell_width, cell_height)

        for item in world.get_entities(Item):
            center = self._position_center(item.position, rect, cell_width, cell_height)
            radius = max(2, round(minimum_cell * 0.36))
            points = [
                (center[0], center[1] - radius),
                (center[0] + radius, center[1]),
                (center[0], center[1] + radius),
                (center[0] - radius, center[1]),
            ]
            pygame.draw.polygon(self.surface, (238, 139, 46), points)
            pygame.draw.polygon(self.surface, (145, 77, 14), points, width=1)

        for robot in world.get_entities(Robot):
            center = self._position_center(robot.position, rect, cell_width, cell_height)
            radius = max(2, round(minimum_cell * 0.27))
            pygame.draw.circle(self.surface, (52, 101, 212), center, radius)
            pygame.draw.circle(self.surface, (29, 62, 133), center, radius, width=1)
            if self.show_labels and minimum_cell >= 24:
                label = self._small_font.render(robot.robot_id, True, PALETTE.text)
                self.surface.blit(
                    label,
                    (center[0] - label.get_width() // 2, center[1] + radius + 1),
                )

    def _draw_grid_background(self, width: int, height: int, rect: pygame.Rect) -> None:
        assert self.surface is not None
        pygame.draw.rect(self.surface, PALETTE.panel, rect)
        line_color = (199, 205, 214)
        # Dense grids become a grey block if every sub-pixel line is drawn.
        x_stride = max(1, math.ceil(width / max(1, rect.width // 3)))
        y_stride = max(1, math.ceil(height / max(1, rect.height // 3)))
        for x in range(0, width + 1, x_stride):
            pixel_x = rect.x + round(rect.width * x / width)
            pygame.draw.line(
                self.surface,
                line_color,
                (pixel_x, rect.y),
                (pixel_x, rect.bottom),
            )
        for y in range(0, height + 1, y_stride):
            pixel_y = rect.y + round(rect.height * y / height)
            pygame.draw.line(
                self.surface,
                line_color,
                (rect.x, pixel_y),
                (rect.right, pixel_y),
            )
        pygame.draw.rect(self.surface, PALETTE.border_strong, rect, width=1)

    @staticmethod
    def _position_center(
        position: tuple[int, int],
        rect: pygame.Rect,
        cell_width: float,
        cell_height: float,
    ) -> tuple[int, int]:
        return (
            round(rect.x + (position[0] + 0.5) * cell_width),
            round(rect.y + (position[1] + 0.5) * cell_height),
        )

    def _draw_panel(self, rect: pygame.Rect) -> None:
        assert self.surface is not None
        shadow = rect.move(0, 2)
        pygame.draw.rect(self.surface, (224, 229, 237), shadow, border_radius=9)
        pygame.draw.rect(self.surface, PALETTE.panel, rect, border_radius=9)
        pygame.draw.rect(self.surface, PALETTE.border, rect, width=1, border_radius=9)

    def _safe_positive_field(self, name: str) -> int | None:
        try:
            value = int(self.setup.fields[name].text)
        except ValueError:
            return None
        return value if value > 0 else None

    def _clamp_manual_scroll(self) -> None:
        content_height = len(self.setup.robot_drafts) * self._ROW_HEIGHT
        maximum = max(0, content_height - self._manual_view.height)
        self.setup.scroll_offset = max(0, min(maximum, self.setup.scroll_offset))

    def _clamp_status_scroll(self) -> None:
        robot_count = (
            len(self.session.world.get_entities(Robot))
            if self.session is not None
            else 0
        )
        content_height = robot_count * self._status_row_height
        maximum = max(0, content_height - self._status_view.height)
        self._status_scroll_offset = max(
            0,
            min(maximum, self._status_scroll_offset),
        )

    @staticmethod
    def _format_status_number(value: float) -> str:
        return f"{float(value):.6g}"

    def _scroll_robot_into_view(self, index: int) -> None:
        top = index * self._ROW_HEIGHT
        bottom = top + self._ROW_HEIGHT
        if top < self.setup.scroll_offset:
            self.setup.scroll_offset = top
        elif bottom > self.setup.scroll_offset + self._manual_view.height:
            self.setup.scroll_offset = bottom - self._manual_view.height
        self._clamp_manual_scroll()
        self._layout_setup()


def run_pygame_application(
    width: int,
    height: int,
    num_robots: int,
    num_items: int,
    seed: int,
    max_steps: int,
    step_rate: int,
    show_labels: bool = True,
    movement_cost: float = 1,
    pickup_cost: float = 1,
    wait_cost: float = 0,
    controller_registry: ControllerRegistry | None = None,
    drop_cost: float = 1,
) -> SimulationWorld | None:
    """Convenience wrapper around :class:`PygameSimulationApp`."""

    return PygameSimulationApp(
        width=width,
        height=height,
        num_robots=num_robots,
        num_items=num_items,
        seed=seed,
        max_steps=max_steps,
        step_rate=step_rate,
        show_labels=show_labels,
        movement_cost=movement_cost,
        pickup_cost=pickup_cost,
        drop_cost=drop_cost,
        wait_cost=wait_cost,
        controller_registry=controller_registry,
    ).run()


__all__ = [
    "InitializationState",
    "PygameSimulationApp",
    "RobotDraft",
    "SetupConfiguration",
    "SetupValidation",
    "run_pygame_application",
]
