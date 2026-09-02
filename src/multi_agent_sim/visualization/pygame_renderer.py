"""Interactive Pygame renderer for a :class:`SimulationWorld`."""

from __future__ import annotations

try:
    import pygame
except ImportError as exc:  # pragma: no cover - depends on optional installation
    raise ImportError(
        "Pygame visualization is optional. Install it with "
        "`pip install -e \".[visualization]\"`."
    ) from exc

from multi_agent_sim.entities import DeliveryDestination, Item, Robot
from multi_agent_sim.world import SimulationWorld


class PygameRenderer:
    """Draw a world without modifying or advancing its simulation state."""

    _BACKGROUND = (245, 247, 250)
    _GRID = (199, 205, 214)
    _ROBOT = (52, 101, 212)
    _ROBOT_EDGE = (29, 62, 133)
    _ITEM = (238, 139, 46)
    _ITEM_EDGE = (145, 77, 14)
    _DESTINATION_EMPTY = (191, 174, 226)
    _DESTINATION_WRONG = (241, 190, 92)
    _DESTINATION_CORRECT = (104, 190, 133)
    _DESTINATION_EDGE = (91, 70, 133)
    _TEXT = (24, 30, 40)

    def __init__(
        self,
        cell_size: int = 30,
        fps: int = 5,
        show_labels: bool = True,
        max_window_size: int = 900,
    ) -> None:
        if type(cell_size) is not int or cell_size < 4:
            raise ValueError("cell_size must be an integer of at least 4")
        if type(fps) is not int or fps <= 0:
            raise ValueError("fps must be a positive integer")
        if type(max_window_size) is not int or max_window_size < 100:
            raise ValueError("max_window_size must be an integer of at least 100")

        pygame.init()
        self._requested_cell_size = cell_size
        self._fps = fps
        self._show_labels = show_labels
        self._max_window_size = max_window_size
        self._clock = pygame.time.Clock()
        self._surface: pygame.Surface | None = None
        self._font: pygame.font.Font | None = None
        self._world_size: tuple[int, int] | None = None
        self._cell_size = cell_size
        self._header_height = 32

    def process_events(self) -> bool:
        """Return ``False`` when the user closes the window or presses Escape."""

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def render(self, world: SimulationWorld) -> None:
        """Render the current state without advancing ``world``."""

        self._ensure_display(world)
        assert self._surface is not None

        self._surface.fill(self._BACKGROUND)
        self._draw_header(world)
        self._draw_grid(world)

        # Destinations are floor markers. Items are diamonds and robots are
        # smaller circles, so later entities remain visible on the marker.
        for destination in world.get_entities(DeliveryDestination):
            self._draw_destination(world, destination)
        for item in world.get_entities(Item):
            self._draw_item(item)
        for robot in world.get_entities(Robot):
            self._draw_robot(robot)

        pygame.display.flip()

    def tick(self) -> None:
        """Limit the caller's loop to the configured frames per second."""

        self._clock.tick(self._fps)

    def close(self) -> None:
        pygame.quit()
        self._surface = None
        self._font = None

    def __enter__(self) -> PygameRenderer:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _ensure_display(self, world: SimulationWorld) -> None:
        world_size = (world.width, world.height)
        if self._surface is not None and world_size == self._world_size:
            return

        horizontal_limit = max(4, self._max_window_size // world.width)
        vertical_limit = max(
            4,
            (self._max_window_size - self._header_height) // world.height,
        )
        self._cell_size = min(
            self._requested_cell_size,
            horizontal_limit,
            vertical_limit,
        )
        window_size = (
            world.width * self._cell_size,
            world.height * self._cell_size + self._header_height,
        )
        self._surface = pygame.display.set_mode(window_size)
        self._world_size = world_size
        font_size = max(10, min(16, self._cell_size // 2))
        self._font = pygame.font.Font(None, font_size)
        pygame.display.set_caption("Multi-Agent 2D Simulation")

    def _draw_header(self, world: SimulationWorld) -> None:
        assert self._surface is not None
        assert self._font is not None
        destinations = world.get_entities(DeliveryDestination)
        delivered_count = sum(
            any(
                isinstance(occupant, Item)
                and occupant.item_id == destination.target_item_id
                for occupant in world.get_entities_at(destination.position)
            )
            for destination in destinations
        )
        text = self._font.render(
            f"Step {world.timestep}   Robots: {len(world.get_entities(Robot))}   "
            f"Items: {len(world.get_entities(Item))}   "
            f"Delivered: {delivered_count}/{len(destinations)}",
            True,
            self._TEXT,
        )
        self._surface.blit(text, (8, max(4, (self._header_height - text.get_height()) // 2)))

    def _draw_grid(self, world: SimulationWorld) -> None:
        assert self._surface is not None
        top = self._header_height
        width = world.width * self._cell_size
        height = world.height * self._cell_size
        for x in range(world.width + 1):
            pixel_x = x * self._cell_size
            pygame.draw.line(
                self._surface,
                self._GRID,
                (pixel_x, top),
                (pixel_x, top + height),
            )
        for y in range(world.height + 1):
            pixel_y = top + y * self._cell_size
            pygame.draw.line(
                self._surface,
                self._GRID,
                (0, pixel_y),
                (width, pixel_y),
            )

    def _draw_item(self, item: Item) -> None:
        assert self._surface is not None
        center_x, center_y = self._cell_center(item.position)
        radius = max(2, int(self._cell_size * 0.38))
        points = [
            (center_x, center_y - radius),
            (center_x + radius, center_y),
            (center_x, center_y + radius),
            (center_x - radius, center_y),
        ]
        pygame.draw.polygon(self._surface, self._ITEM, points)
        pygame.draw.polygon(self._surface, self._ITEM_EDGE, points, width=1)

    def _draw_destination(
        self,
        world: SimulationWorld,
        destination: DeliveryDestination,
    ) -> None:
        assert self._surface is not None
        items = tuple(
            occupant
            for occupant in world.get_entities_at(destination.position)
            if isinstance(occupant, Item)
        )
        if not items:
            color = self._DESTINATION_EMPTY
        elif any(item.item_id == destination.target_item_id for item in items):
            color = self._DESTINATION_CORRECT
        else:
            color = self._DESTINATION_WRONG

        x, y = destination.position
        inset = max(1, round(self._cell_size * 0.1))
        marker = pygame.Rect(
            x * self._cell_size + inset,
            self._header_height + y * self._cell_size + inset,
            max(1, self._cell_size - 2 * inset),
            max(1, self._cell_size - 2 * inset),
        )
        pygame.draw.rect(self._surface, color, marker, border_radius=3)
        pygame.draw.rect(
            self._surface,
            self._DESTINATION_EDGE,
            marker,
            width=1,
            border_radius=3,
        )

    def _draw_robot(self, robot: Robot) -> None:
        assert self._surface is not None
        center = self._cell_center(robot.position)
        radius = max(2, int(self._cell_size * 0.28))
        pygame.draw.circle(self._surface, self._ROBOT, center, radius)
        pygame.draw.circle(self._surface, self._ROBOT_EDGE, center, radius, width=1)

        if self._show_labels and self._cell_size >= 18:
            assert self._font is not None
            label = self._font.render(robot.robot_id, True, self._TEXT)
            label_x = center[0] - label.get_width() // 2
            label_y = center[1] + radius
            self._surface.blit(label, (label_x, label_y))

    def _cell_center(self, position: tuple[int, int]) -> tuple[int, int]:
        x, y = position
        return (
            x * self._cell_size + self._cell_size // 2,
            self._header_height + y * self._cell_size + self._cell_size // 2,
        )
