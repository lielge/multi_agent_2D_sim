"""Small dependency-free Pygame widgets used by the simulation application.

The project intentionally avoids a second GUI dependency.  These controls are
deliberately modest, but include the keyboard and disabled-state behaviour the
setup form and playback toolbar need.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import pygame


Color = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class Palette:
    background: Color = (244, 247, 251)
    panel: Color = (255, 255, 255)
    panel_alt: Color = (236, 241, 248)
    border: Color = (195, 204, 217)
    border_strong: Color = (139, 151, 170)
    text: Color = (28, 35, 48)
    text_muted: Color = (93, 105, 123)
    primary: Color = (48, 100, 210)
    primary_hover: Color = (39, 85, 184)
    primary_soft: Color = (221, 232, 255)
    disabled: Color = (224, 229, 236)
    disabled_text: Color = (145, 154, 168)
    error: Color = (184, 48, 48)
    focus: Color = (48, 100, 210)


PALETTE = Palette()


class Button:
    """A rectangular push button with hover, focus, and disabled states."""

    def __init__(
        self,
        label: str,
        rect: pygame.Rect | tuple[int, int, int, int] = (0, 0, 1, 1),
        *,
        primary: bool = False,
    ) -> None:
        self.label = label
        self.rect = pygame.Rect(rect)
        self.primary = primary
        self.focused = False
        self._enabled = True

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, enabled: bool) -> None:
        self._enabled = bool(enabled)
        if not self._enabled:
            self.focused = False

    def handle_event(self, event: pygame.event.Event) -> bool:
        if not self.enabled:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.focused = self.rect.collidepoint(event.pos)
            return self.focused
        if (
            event.type == pygame.KEYDOWN
            and self.focused
            and event.key in (pygame.K_RETURN, pygame.K_SPACE)
        ):
            return True
        return False

    def draw(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        mouse_position: tuple[int, int],
    ) -> None:
        hovered = self.enabled and self.rect.collidepoint(mouse_position)
        if not self.enabled:
            background = PALETTE.disabled
            foreground = PALETTE.disabled_text
            border = PALETTE.border
        elif self.primary:
            background = PALETTE.primary_hover if hovered else PALETTE.primary
            foreground = (255, 255, 255)
            border = background
        else:
            background = PALETTE.panel_alt if hovered else PALETTE.panel
            foreground = PALETTE.text
            border = PALETTE.focus if self.focused else PALETTE.border_strong

        pygame.draw.rect(surface, background, self.rect, border_radius=6)
        pygame.draw.rect(surface, border, self.rect, width=1, border_radius=6)
        text = font.render(self.label, True, foreground)
        surface.blit(text, text.get_rect(center=self.rect.center))


class TextInput:
    """A single-line text input with editing and selection shortcuts."""

    def __init__(
        self,
        text: str = "",
        rect: pygame.Rect | tuple[int, int, int, int] = (0, 0, 1, 1),
        *,
        max_length: int = 24,
        character_filter: Callable[[str], bool] | None = None,
    ) -> None:
        self.text = text
        self.rect = pygame.Rect(rect)
        self.max_length = max_length
        self.character_filter = character_filter
        self.focused = False
        self.cursor = len(text)
        self._select_all = False

    def set_text(self, text: str) -> None:
        self.text = str(text)[: self.max_length]
        self.cursor = len(self.text)
        self._select_all = False

    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handle one event and return whether the text changed."""

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.focused = self.rect.collidepoint(event.pos)
            if self.focused:
                # Exact glyph hit-testing adds little value for these short
                # numeric fields.  Clicking places the caret at the end.
                self.cursor = len(self.text)
                self._select_all = False
            return False

        if event.type != pygame.KEYDOWN or not self.focused:
            return False

        modifiers = pygame.key.get_mods()
        if event.key == pygame.K_a and modifiers & pygame.KMOD_CTRL:
            self._select_all = True
            self.cursor = len(self.text)
            return False
        if event.key == pygame.K_HOME:
            self.cursor = 0
            self._select_all = False
            return False
        if event.key == pygame.K_END:
            self.cursor = len(self.text)
            self._select_all = False
            return False
        if event.key == pygame.K_LEFT:
            self.cursor = max(0, self.cursor - 1)
            self._select_all = False
            return False
        if event.key == pygame.K_RIGHT:
            self.cursor = min(len(self.text), self.cursor + 1)
            self._select_all = False
            return False
        if event.key == pygame.K_BACKSPACE:
            if self._select_all:
                changed = bool(self.text)
                self.text = ""
                self.cursor = 0
                self._select_all = False
                return changed
            if self.cursor > 0:
                self.text = self.text[: self.cursor - 1] + self.text[self.cursor :]
                self.cursor -= 1
                return True
            return False
        if event.key == pygame.K_DELETE:
            if self._select_all:
                changed = bool(self.text)
                self.text = ""
                self.cursor = 0
                self._select_all = False
                return changed
            if self.cursor < len(self.text):
                self.text = self.text[: self.cursor] + self.text[self.cursor + 1 :]
                return True
            return False

        value = getattr(event, "unicode", "")
        if (
            not value
            or not value.isprintable()
            or (self.character_filter is not None and not self.character_filter(value))
        ):
            return False

        if self._select_all:
            candidate = value
            cursor = len(value)
        else:
            candidate = self.text[: self.cursor] + value + self.text[self.cursor :]
            cursor = self.cursor + len(value)
        if len(candidate) > self.max_length:
            return False
        self.text = candidate
        self.cursor = cursor
        self._select_all = False
        return True

    def draw(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        *,
        invalid: bool = False,
    ) -> None:
        pygame.draw.rect(surface, PALETTE.panel, self.rect, border_radius=5)
        border = PALETTE.error if invalid else (
            PALETTE.focus if self.focused else PALETTE.border
        )
        pygame.draw.rect(surface, border, self.rect, width=2 if self.focused else 1, border_radius=5)

        inner = self.rect.inflate(-12, -6)
        old_clip = surface.get_clip()
        surface.set_clip(inner)
        rendered = font.render(self.text, True, PALETTE.text)
        text_x = inner.x
        if rendered.get_width() > inner.width:
            text_x = inner.right - rendered.get_width()
        surface.blit(rendered, (text_x, inner.centery - rendered.get_height() // 2))

        if self.focused and pygame.time.get_ticks() % 1000 < 540:
            before = font.render(self.text[: self.cursor], True, PALETTE.text)
            cursor_x = min(inner.right, text_x + before.get_width())
            pygame.draw.line(
                surface,
                PALETTE.text,
                (cursor_x, inner.y + 2),
                (cursor_x, inner.bottom - 2),
            )
        surface.set_clip(old_clip)


class IntegerSlider:
    """A mouse/keyboard integer slider."""

    def __init__(
        self,
        minimum: int,
        maximum: int,
        value: int,
        rect: pygame.Rect | tuple[int, int, int, int] = (0, 0, 1, 1),
    ) -> None:
        if minimum >= maximum:
            raise ValueError("minimum must be less than maximum")
        self.minimum = minimum
        self.maximum = maximum
        self.rect = pygame.Rect(rect)
        self.focused = False
        self.dragging = False
        self._value = minimum
        self.value = value

    @property
    def value(self) -> int:
        return self._value

    @value.setter
    def value(self, value: int) -> None:
        self._value = max(self.minimum, min(self.maximum, int(value)))

    def handle_event(self, event: pygame.event.Event) -> bool:
        old_value = self.value
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.focused = self.rect.inflate(0, 14).collidepoint(event.pos)
            self.dragging = self.focused
            if self.dragging:
                self._set_from_x(event.pos[0])
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._set_from_x(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.dragging:
                self._set_from_x(event.pos[0])
            self.dragging = False
        elif event.type == pygame.KEYDOWN and self.focused:
            if event.key in (pygame.K_LEFT, pygame.K_DOWN):
                self.value -= 1
            elif event.key in (pygame.K_RIGHT, pygame.K_UP):
                self.value += 1
            elif event.key == pygame.K_HOME:
                self.value = self.minimum
            elif event.key == pygame.K_END:
                self.value = self.maximum
        return self.value != old_value

    def draw(self, surface: pygame.Surface) -> None:
        track = pygame.Rect(self.rect.x, self.rect.centery - 2, self.rect.width, 4)
        pygame.draw.rect(surface, PALETTE.border, track, border_radius=2)
        fraction = (self.value - self.minimum) / (self.maximum - self.minimum)
        filled = pygame.Rect(track.x, track.y, round(track.width * fraction), track.height)
        if filled.width:
            pygame.draw.rect(surface, PALETTE.primary, filled, border_radius=2)
        knob_x = round(track.x + track.width * fraction)
        pygame.draw.circle(surface, PALETTE.panel, (knob_x, track.centery), 9)
        pygame.draw.circle(
            surface,
            PALETTE.focus if self.focused else PALETTE.primary,
            (knob_x, track.centery),
            9,
            width=2,
        )

    def _set_from_x(self, mouse_x: int) -> None:
        if self.rect.width <= 0:
            return
        fraction = (mouse_x - self.rect.x) / self.rect.width
        fraction = max(0.0, min(1.0, fraction))
        span = self.maximum - self.minimum
        self.value = self.minimum + round(span * fraction)


class ChoiceSelector:
    """A compact selector that cycles through stable key/display-name choices.

    A left click or Right/Down advances to the next choice.  A right click or
    Left/Up selects the previous choice.  This deliberately small interaction
    model works well inside clipped, scrollable robot rows without needing a
    pop-over menu that can escape the clipping region.
    """

    def __init__(
        self,
        choices: Sequence[tuple[str, str]],
        value: str | None = None,
        rect: pygame.Rect | tuple[int, int, int, int] = (0, 0, 1, 1),
    ) -> None:
        normalized = tuple((str(key), str(label)) for key, label in choices)
        if not normalized:
            raise ValueError("choices cannot be empty")
        keys = tuple(key for key, _ in normalized)
        if any(not key.strip() for key in keys):
            raise ValueError("choice keys must be non-empty strings")
        if len(set(keys)) != len(keys):
            raise ValueError("choice keys must be unique")
        if any(not label.strip() for _, label in normalized):
            raise ValueError("choice labels must be non-empty strings")

        self.choices = normalized
        self.rect = pygame.Rect(rect)
        self.focused = False
        self._value = keys[0]
        self.value = self._value if value is None else value

    @property
    def value(self) -> str:
        return self._value

    @value.setter
    def value(self, value: str) -> None:
        if value not in self.keys:
            raise ValueError(f"unknown choice key: {value!r}")
        self._value = value

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(key for key, _ in self.choices)

    @property
    def display_name(self) -> str:
        return next(label for key, label in self.choices if key == self.value)

    def handle_event(self, event: pygame.event.Event) -> bool:
        old_value = self.value
        if event.type == pygame.MOUSEBUTTONDOWN and event.button in (1, 3):
            self.focused = self.rect.collidepoint(event.pos)
            if self.focused:
                self._cycle(-1 if event.button == 3 else 1)
        elif event.type == pygame.KEYDOWN and self.focused:
            if event.key in (pygame.K_RIGHT, pygame.K_DOWN, pygame.K_SPACE):
                self._cycle(1)
            elif event.key in (pygame.K_LEFT, pygame.K_UP):
                self._cycle(-1)
            elif event.key == pygame.K_HOME:
                self._value = self.keys[0]
            elif event.key == pygame.K_END:
                self._value = self.keys[-1]
        return self.value != old_value

    def draw(
        self,
        surface: pygame.Surface,
        font: pygame.font.Font,
        mouse_position: tuple[int, int],
    ) -> None:
        hovered = self.rect.collidepoint(mouse_position)
        background = PALETTE.panel_alt if hovered else PALETTE.panel
        border = PALETTE.focus if self.focused else PALETTE.border
        pygame.draw.rect(surface, background, self.rect, border_radius=5)
        pygame.draw.rect(
            surface,
            border,
            self.rect,
            width=2 if self.focused else 1,
            border_radius=5,
        )

        arrow_width = 22
        text_rect = pygame.Rect(
            self.rect.x + 7,
            self.rect.y + 2,
            max(1, self.rect.width - arrow_width - 9),
            max(1, self.rect.height - 4),
        )
        old_clip = surface.get_clip()
        surface.set_clip(text_rect)
        rendered = font.render(self.display_name, True, PALETTE.text)
        surface.blit(
            rendered,
            (text_rect.x, text_rect.centery - rendered.get_height() // 2),
        )
        surface.set_clip(old_clip)

        arrow_x = self.rect.right - 12
        arrow_y = self.rect.centery
        pygame.draw.polygon(
            surface,
            PALETTE.text_muted,
            ((arrow_x - 4, arrow_y - 2), (arrow_x + 4, arrow_y - 2), (arrow_x, arrow_y + 3)),
        )

    def _cycle(self, offset: int) -> None:
        keys = self.keys
        self._value = keys[(keys.index(self.value) + offset) % len(keys)]


def draw_text(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    position: tuple[int, int],
    *,
    color: Color = PALETTE.text,
) -> pygame.Rect:
    rendered = font.render(text, True, color)
    return surface.blit(rendered, position)


def draw_wrapped_text(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    rect: pygame.Rect,
    *,
    color: Color = PALETTE.text,
    line_gap: int = 3,
    max_lines: int | None = None,
) -> int:
    """Draw wrapped text and return the y-coordinate after the last line."""

    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and font.size(candidate)[0] > rect.width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    if max_lines is not None:
        lines = lines[:max_lines]

    y = rect.y
    for line in lines:
        draw_text(surface, font, line, (rect.x, y), color=color)
        y += font.get_linesize() + line_gap
    return y
