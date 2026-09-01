# Multi-Agent 2D Simulation

A small, modular foundation for multi-agent experiments on a discrete 2D grid.
It provides deterministic world transitions, battery-aware movement, item
pickup and drop, immutable controller observations, pluggable robot policies,
and an optional interactive Pygame application.

## Installation

Python 3.11 or newer is required. From this directory, create and activate a
virtual environment, then choose the components you need:

```bash
python -m venv .venv
```

PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -e ".[visualization,dev]"
```

macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install -e ".[visualization,dev]"
```

The simulation core has no runtime dependencies. For a headless-only install,
use `python -m pip install -e .`. The `visualization` extra adds Pygame and the
`dev` extra adds pytest.

## Quick start

Run the interactive 20x20 demonstration with three robots and five items:

```bash
python examples/basic_simulation.py
```

The window opens on a setup screen. Choose the grid dimensions, robot and item
counts, seed, maximum number of steps, and the nonnegative **Move**, **Pickup**,
**Drop**, and **Wait** battery costs. You can generate robot positions or
configure each robot's position and initial battery manually. In either
placement mode, each robot has its own controller selector. The built-ins are
**Random** and the deterministic **Nearest Item** test controller; custom
runnable registry entries appear in the same selectors automatically.

In manual mode, select a robot row and click a cell in the preview to place it.
Items are generated from the seed and never initially overlap a robot. Invalid
or duplicate positions, batteries, costs, and other settings are shown inline
and must be fixed before Start is enabled.

The simulation starts paused. Its toolbar provides:

- **Back** and **Forward** for exact one-step timeline navigation.
- **Play/Pause** and a 1-30 steps-per-second rate slider.
- The current step, recorded-history position, and robot/item counts.
- A scrollable status panel with each robot's battery, selected controller,
  carried item, and latest controller warning.
- **New Setup** to discard the current run and return to the setup screen while
  retaining the latest settings.

Moving Back or Forward pauses playback first. Previously recorded steps replay
the same actions and controller warnings without invoking controllers again.
Once playback reaches the end of history, the configured controllers choose new
actions from a single immutable world snapshot. Playback pauses automatically
at the configured maximum step. Press Escape or close the window to exit.

A deterministic headless run is also available and does not require Pygame:

```bash
python examples/basic_simulation.py --headless --steps 25 --seed 42
```

Useful options include `--width`, `--height`, `--robots`, `--items`, `--steps`,
`--seed`, `--fps`, `--move-cost`, `--pickup-cost`, `--drop-cost`,
`--wait-cost`, and `--no-labels`. The four cost flags accept finite,
nonnegative numbers and default to `1`, `1`, `1`, and `0`, respectively. In
visual mode these values prefill the setup screen: `--steps` is the initial
maximum step and `--fps` is the initial step rate, which must be between 1 and
30. In headless mode, `--steps` is the number of steps to execute and `--fps`
is ignored; scenario creation, controllers, action costs, and world transitions
otherwise use the same path as the visual application.

Create and step a world programmatically:

```python
from multi_agent_sim import (
    Action,
    ActionBatteryCosts,
    Item,
    Robot,
    SimulationWorld,
)

world = SimulationWorld(
    width=10,
    height=10,
    action_battery_costs=ActionBatteryCosts(
        movement=1,
        pickup=2,
        drop=1,
        wait=0,
    ),
)
world.add_entity(Robot(robot_id="robot_1", position=(2, 3)))
world.add_entity(Item(item_id="item_1", position=(3, 3)))

results = world.step({"robot_1": Action.MOVE_RIGHT})
print(world.get_entity("robot_1").position)  # (3, 3)
print(results["robot_1"].success)            # True

results = world.step({"robot_1": Action.PICK_UP})
robot = world.get_entity("robot_1")
print(robot.carried_item_id)                 # item_1
print(results["robot_1"].battery_spent)      # 2.0

results = world.step({"robot_1": Action.DROP})
print(robot.carried_item_id)                  # None
print(world.get_entity("item_1").position)   # (3, 3)
print(results["robot_1"].dropped_item_id)    # item_1
```

Or generate a reproducible scenario:

```python
from multi_agent_sim import generate_random_world

world = generate_random_world(
    width=30,
    height=30,
    num_robots=5,
    num_items=10,
    seed=42,
)
```

## Architecture

- `entities.py` defines the shared `Entity` abstraction and the initial
  `Robot` and `Item` types. Entity positions and robot inventory are read-only
  to callers; transitions go through the owning world so its indexes and
  invariants remain correct.
- `world.py` owns dimensions, entity registration, positional indexing,
  occupancy policy, battery charging, pickup/drop, simulation time, and batched
  state transitions.
- `actions.py` contains the action vocabulary, immutable battery-cost
  configuration, and structured per-robot action results.
- `controllers.py` defines immutable world/robot/item observation objects, the
  per-robot controller protocol, the controller registry, the two built-in
  policies, and the multi-agent adapter.
- `session.py` constructs controllers, collects one action per robot, records
  controller warnings, and provides exact replay and rewind by rebuilding the
  initial world and replaying recorded action batches.
- `generation.py` creates scenarios with a private seeded random generator and
  globally unique initial positions.
- `visualization/` contains the optional Pygame renderer, interactive setup UI,
  controller/status controls, and timeline application. The core package never
  imports Pygame.

The package root exports the simulation-facing API. Visualization must be
imported explicitly with:

```python
from multi_agent_sim.visualization import PygameRenderer, PygameSimulationApp
```

## World semantics

Coordinates are zero-based. `(0, 0)` is the top-left cell, `x` grows to the
right, and `y` grows downward. Valid coordinates satisfy:

```text
0 <= x < width
0 <= y < height
```

The default occupancy policy allows a robot and an item to share a cell. Two
robots cannot share a cell, nor can two items. Random generation is stricter:
all entities start in different cells.

`world.step(actions)` resolves movement from a snapshot of the step's starting
state. Missing robot actions become `WAIT`. Out-of-bounds moves fail; all robots
contesting the same destination fail; and a cell occupied by another robot at
the start remains blocked even if that robot is moving away. Independent valid
moves still succeed. Each call advances `world.timestep` exactly once and
returns an `ActionResult` for every robot.

Battery charging is owned by `SimulationWorld`, so GUI, headless, and direct
API use have identical semantics. An affordable action consumes its configured
cost even if it fails because of a boundary, collision, missing item, full
inventory, empty inventory, or blocked drop position. If the robot cannot
afford the action, it fails with
`INSUFFICIENT_BATTERY`, consumes nothing, and has no other effect. Spending the
last available battery is valid; a battery can reach zero but never become
negative. `WAIT` uses its configured cost, including when it is substituted for
a missing action or a controller failure. `ActionResult` reports battery before
and after the attempt, the amount spent, and any picked-up or dropped item ID.

`PICK_UP` acts on the robot's current cell. A successful pickup removes the
item from the world and stores its ID in the robot's read-only
`carried_item_id`. A robot can carry one item. Pickup fails with `NO_ITEM` when
the cell has no item and with `INVENTORY_FULL` when the robot already carries
one.

`DROP` places the carried item on the robot's current cell, restores the same
stable item ID to the world, and clears the robot's inventory. It fails with
`NO_CARRIED_ITEM` when the robot is empty and with `OCCUPIED` when the cell
already contains an item or the occupancy policy rejects placement. Carried
item IDs remain reserved while off-grid, so another entity cannot reuse the ID.
There is no item transfer action or direct manual Drop button; controllers issue
`DROP` through the normal action interface.

Administrative setup or environment code may use `world.move_entity(...)`.
Robot controllers should use the batched `step()` interface so multi-agent
updates do not depend on iteration order.

## Controller integration

The world owns truth and transitions but never decides actions. A
`SimulationSession` gives every controller the same immutable
`WorldObservation` for a timestep, calls controllers synchronously, and then
submits the completed action batch to the world. Controllers never receive a
mutable `SimulationWorld`, `Robot`, or `Item`.

The primary interface is deliberately small:

```python
from multi_agent_sim import Action, WorldObservation


class MyController:
    def choose_action(
        self,
        observation: WorldObservation,
        robot_id: str,
    ) -> Action:
        robot = observation.get_robot(robot_id)
        if robot.carried_item_id is not None:
            return Action.DROP
        return Action.WAIT if robot.battery_level == 0 else Action.MOVE_RIGHT
```

`RandomController` samples `DROP` along with every other action, even when its
inventory is empty. `NearestItemController` still waits after collecting an
item because this version has no delivery destination. Custom controllers can
choose `DROP` whenever their own task logic requires it.

This same protocol can be implemented directly by a rule-based controller or
planner, or by a thin adapter around an RL policy or neural network. Register a
factory under a stable key to make it selectable by scenarios and the setup UI:

```python
from multi_agent_sim import (
    SimulationSession,
    create_default_controller_registry,
    create_initial_scenario,
)

registry = create_default_controller_registry()
registry.register(
    "my_planner",
    "My Planner",
    lambda context: MyController(),
)

scenario = create_initial_scenario(
    width=10,
    height=10,
    num_robots=2,
    num_items=4,
    seed=42,
    controller_keys=("my_planner", "nearest_item"),
)
session = SimulationSession(scenario, controller_registry=registry)
```

Each registry factory receives a `ControllerFactoryContext` containing the
scenario seed and robot ID, and each robot normally receives its own controller
instance. If a controller raises an exception or returns a non-`Action`, the
session substitutes `WAIT`, records a warning for that robot, and continues the
step. Recorded actions and warnings are replayed exactly without calling the
controller again.

A joint team policy can implement `MultiAgentController.choose_actions` and be
wrapped once. Share that same adapter instance through `controller_overrides`
so its joint policy is evaluated once per timestep and cached for every team
member:

```python
from multi_agent_sim import Action, MultiAgentControllerAdapter, SimulationSession


class TeamPolicy:
    def choose_actions(self, observation, robot_ids):
        return {robot_id: Action.WAIT for robot_id in robot_ids}


team = ("robot_1", "robot_2")
adapter = MultiAgentControllerAdapter(TeamPolicy(), team)
session = SimulationSession(
    scenario,
    controller_registry=registry,
    controller_overrides={robot_id: adapter for robot_id in team},
)
```

The package does not add an ML framework, training loop, HTTP or subprocess
transport, or asynchronous inference. Those concerns stay inside custom
controller adapters. Future extensions can add partial observations, rewards,
transfer or delivery actions, or local belief state without coupling
decision-making to rendering or world mutation.

## Tests

With the development extra installed:

```bash
python -m pytest
```

The tests use the standard-library unittest API and can also run without pytest:

```bash
python -m unittest discover -s tests -v
```
