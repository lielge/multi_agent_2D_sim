"""Optional visualization adapters.

Importing :mod:`multi_agent_sim` never imports Pygame. Import the renderer from
this subpackage only when visualization support is installed.
"""

from .pygame_renderer import PygameRenderer
from .pygame_app import PygameSimulationApp, run_pygame_application

__all__ = [
    "PygameRenderer",
    "PygameSimulationApp",
    "run_pygame_application",
]
