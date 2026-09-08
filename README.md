# Multi-Agent 2D Simulation

A small, modular foundation for multi-agent experiments on a discrete 2D grid.
It provides deterministic world transitions, battery-aware movement, item
pickup and drop, reversible delivery destinations, immutable controller
observations, pluggable robot policies, and an optional interactive Pygame
application. Trained shared-MLP policies can be published into a local
controller catalog and selected like the built-in policies.

## Installation

Python 3.11 or newer is required. From this directory, create and activate a
virtual environment, then choose the components you need:

```bash
python -m venv .venv
```

PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -e ".[visualization,training,dev]"
```

macOS or Linux:

```bash
source .venv/bin/activate
python -m pip install -e ".[visualization,training,dev]"
```

The simulation core, episode scorer, and fixed-slot environment have no runtime
dependencies. For a headless-only install, use `python -m pip install -e .`.
The `visualization` extra adds Pygame, `training` adds NumPy and PyTorch, and
`dev` adds pytest.

## Quick start

Run the interactive 20x20 demonstration with three robots and five items:

```bash
python examples/basic_simulation.py
```

The window opens on a setup screen. Choose the grid dimensions, robot and item
counts, seed, maximum number of steps, and the nonnegative **Move**, **Pickup**,
**Drop**, and **Wait** battery costs. One empty delivery destination is generated
for every item. You can generate robot positions or
configure each robot's position and initial battery manually. In either
placement mode, each robot has its own controller selector. The built-ins are
**Random** and the deterministic **Nearest Item** test controller; valid saved
neural controllers from `controllers/` appear after them. Click **Reload
controllers** to rescan that directory without losing selections that are
still valid.

In manual mode, select a robot row and click a cell in the preview to place it.
Items and destinations are generated from the seed; robots, items, and
destinations all start in distinct cells. A scenario therefore needs at least
`robot count + 2 * item count` cells. Invalid or duplicate positions, batteries,
costs, and other settings are shown inline and must be fixed before Start is
enabled.

The simulation starts paused. Its toolbar provides:

- **Back** and **Forward** for exact one-step timeline navigation.
- **Play/Pause** and a 1-30 steps-per-second rate slider.
- The current step, recorded-history position, robot/item counts, and delivered
  destination count.
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

List the saved neural controllers and their stable IDs:

```powershell
.\.venv\Scripts\python.exe examples\train_mlp.py controllers list
```

Open the setup screen with the catalog available for independent per-robot
selection:

```powershell
.\.venv\Scripts\python.exe examples\basic_simulation.py `
  --controllers-dir controllers
```

To initialize every robot with one catalog entry, copy its `controller_id`
from the list command:

```powershell
.\.venv\Scripts\python.exe examples\basic_simulation.py `
  --controllers-dir controllers `
  --controller <controller-id>
```

Add `--headless` for a nonvisual run. Saved policies use masked, greedy CPU
inference by default; `--controller-device auto` or
`--controller-device cuda` changes the device. They support at most four
robots and eight items, matching the training encoding.

Create and step a world programmatically:

```python
from multi_agent_sim import (
    Action,
    ActionBatteryCosts,
    DeliveryDestination,
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
world.add_entity(
    DeliveryDestination(
        destination_id="destination_1",
        position=(6, 7),
        target_item_id="item_1",
    )
)

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

- `entities.py` defines the shared `Entity` abstraction and the `Robot`, `Item`,
  and `DeliveryDestination` types. Entity positions and robot inventory are
  read-only to callers; transitions go through the owning world so its indexes
  and invariants remain correct.
- `world.py` owns dimensions, entity registration, positional indexing,
  occupancy policy, battery charging, pickup/drop, simulation time, and batched
  state transitions.
- `actions.py` contains the action vocabulary, immutable battery-cost
  configuration, and structured per-robot action results.
- `controllers.py` defines immutable world, robot, item, and destination
  observation objects, the per-robot controller protocol, the controller
  registry, the two built-in policies, and the multi-agent adapter.
- `session.py` constructs controllers, collects one action per robot, records
  controller warnings, and provides exact replay and rewind by rebuilding the
  initial world and replaying recorded action batches.
- `generation.py` creates scenarios with a private seeded random generator and
  globally unique initial positions.
- `episode.py` scores immutable before/after observations and action results,
  and provides a headless episode runner without changing world semantics.
- `training_env.py` contains the dependency-free 4-robot/8-item encoder,
  one-hot command boundary, and training environment.
- `learning.py` is an optional PyTorch-only layer containing the shared MLP,
  per-robot learned controllers, REINFORCE trainer, and curriculum helpers.
- `controller_catalog.py` discovers and validates dependency-free immutable
  controller manifests, while `saved_mlp_controllers.py` provides the optional
  PyTorch session adapter.
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

The default occupancy policy allows robots and items to occupy destination
cells. Two robots, two items, or two destinations cannot share a cell. Random
generation is stricter: all robots, items, and destinations start in different
cells. Each generated `destination_n` targets `item_n`.

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

A destination is empty when its cell has no item, correctly fulfilled when its
target item occupies the cell, and incorrectly occupied when another item is
there. Correct and incorrect drops use the normal `DROP` behavior. Any item on
the cell blocks a second drop, and picking the item up makes the destination
empty again. The visualizers show empty destinations in purple, incorrect
occupancy in amber, and correct delivery in green.

Administrative setup or environment code may use `world.move_entity(...)`.
Robot controllers should use the batched `step()` interface so multi-agent
updates do not depend on iteration order.

## Controller integration

The world owns truth and transitions but never decides actions. A
`SimulationSession` gives every controller the same immutable
`WorldObservation` for a timestep, calls controllers synchronously, and then
submits the completed action batch to the world. Controllers never receive a
mutable `SimulationWorld` or live entity.

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
inventory is empty. `NearestItemController` is intentionally unchanged and
still waits after collecting an item. Custom controllers can inspect
`observation.destinations`, use `get_destination(...)`, and choose `DROP`
whenever their task logic requires it.

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

These rule-based controller interfaces are unchanged. Learned controllers use
the separate fixed-slot environment described below and return length-seven
one-hot commands; the environment validates and decodes those commands before
the episode runner calls `SimulationWorld.step()`.

## External episode cost and reward

Episode scoring is deliberately outside `SimulationWorld` and
`SimulationSession`. `TeamCostEvaluator` consumes immutable observations plus
the `ActionResult` mapping produced by a transition. `EpisodeRunner` is a small
headless composition helper; it owns a fresh scenario world but delegates all
scoring to the evaluator.

The default accrued/final episode cost is:

```text
C = 10 F + 2 U + 0.5 (T / T_max) + 0.5 (E / E_max)
    + 0.1 (R_final - R_initial)
    + 1.0 (Q_initial - Q_final)
```

`F` is one for an unsuccessful terminal episode, `U` is the final undelivered
fraction, `T` counts simultaneous team steps once, and `E` is the sum of actual
`ActionResult.battery_spent` values. `Q` is the fraction of items currently at
their own destinations. `R` is a route potential, normalized by the grid's
maximum Manhattan distance and averaged across items: a delivered item is
zero, a carried item uses carrier-to-destination distance, and an uncarried
item uses nearest-robot-to-item plus item-to-destination distance. All six
coefficients are configurable through `EpisodeCostWeights`; `item_progress`
defaults to `0.1` and `delivery` defaults to `1.0`.

Each transition receives the shared reward:

```text
r_t = -time_weight / T_max
      -energy_weight * delta_E_t / E_max
      +item_progress_weight * (R_previous - R_next)
      +delivery_weight * (Q_next - Q_previous)
```

The failure and undelivered components are also subtracted on a terminal
failure. If initial team battery is zero, the normalized energy term is zero.
Success is checked before timeout, so delivery on the final allowed step
succeeds. With no items, an episode is immediately successful with zero cost.
An item counts as delivered only while it is physically located at its own
destination. Placing it there earns delivery credit; picking it up removes the
same credit. Wrong drops receive no delivery credit. Approaching an
uncarried item and carrying it toward its destination both improve the route
potential. All route and delivery terms are signed potentials, so reversible
movement and delivery/removal cycles have zero net reward.

Run a small successful scored episode and inspect every metric:

```bash
python examples/scored_episode.py
```

Programmatic use keeps action selection outside the runner:

```python
from multi_agent_sim import Action, EpisodeRunner, create_initial_scenario

scenario = create_initial_scenario(6, 6, 1, 1, seed=42)
episode = EpisodeRunner(scenario, max_steps=64)
while not episode.terminated:
    transition = episode.step({"robot_1": Action.WAIT})
    print(transition.reward, transition.metrics.total_cost)
```

The sum of undiscounted rewards equals `-metrics.total_cost`. A discounted
training return (`gamma < 1`) does not retain that equality.

## Fixed-slot MLP training

`FixedSlotTeamEnv` wraps `EpisodeRunner` without NumPy or Torch. It rejects
scenarios above four robots or eight items and encodes four robot-specific rows
of 155 values. Stable logical item slots include destination coordinates,
on-grid/delivered/carried state, and carrier identity; padded robot and item
slots include presence flags and are zero-filled. Each active row appends the
controlled robot's four-way identity.

The checkpoint-stable command order is:

```text
0 up, 1 down, 2 left, 3 right, 4 pick up, 5 wait, 6 drop
```

The optional shared model is exactly `155 -> 128 -> 128 -> 7`, with ReLU after
each hidden layer. One `MLPRobotController` is created per robot and all share
the same model. A controller samples during training, uses argmax during greedy
evaluation, and returns only its own immutable one-hot command. Advisory legal
action masks are enabled by default and can be disabled. The learned-policy
mask protects a correctly delivered item from `Pick Up`; the underlying
simulator action remains legal for unmasked and rule-based controllers.

Train with the eight-stage curriculum and the default one-hour budget:

```bash
python examples/train_mlp.py train
```

For a short pipeline check:

```bash
python examples/train_mlp.py train --minutes 0.1 --max-stages 1 \
  --evaluation-interval 10 --evaluation-episodes 5
```

The curriculum is:

| Stage | Grid | Robots | Items | Initially delivered | `T_max` |
|---|---:|---:|---:|---:|---:|
| 1 | 4x4 | 1 | 1 | 0 | 32 |
| 2 | 4x4 | 1 | 2 | 1 | 64 |
| 3 | 4x4 | 1 | 2 | 0 | 64 |
| 4 | 5x5 | 1 | 2 | 0 | 64 |
| 5 | 6x6 | 2 | 2 | 0 | 64 |
| 6 | 8x8 | 2 | 4 | 0 | 128 |
| 7 | 10x10 | 4 | 4 | 0 | 160 |
| 8 | 12x12 | 4 | 8 | 0 | 256 |

Stage 2 explicitly teaches the policy to leave one completed delivery alone
and pursue the remaining item. Stage 3 then presents the original two-item
problem with neither item initially delivered.

Normal training selects a completed earlier stage for 40% of episodes. If a
held-out evaluation finds an earlier stage below 80% success, training switches
exclusively to the earliest regressed stage until all preceding stages recover.
Override these defaults with `--previous-stage-probability` and
`--retention-threshold`.

Two distinct checkpoints are written:

- `artifacts/shared_mlp_best.pt` is the best model-only inference checkpoint;
  its JSON metadata contains the held-out results.
- `artifacts/shared_mlp_latest.pt` is atomically replaced after every completed
  optimizer update and contains the model, Adam state, random-generator state,
  curriculum position, counters, and recovery state needed for continuation.

At the end of every training invocation, including a graceful Ctrl+C, the best
model is also published as an immutable model/manifest pair under
`controllers/`. This is separate from both mutable files above. Use
`--no-publish-controller` to disable publication, `--controller-name NAME` to
add a readable label, or `--controllers-dir PATH` to use another catalog.
Identical model bytes are deduplicated.

Warm-start the new curriculum from an existing model-only checkpoint:

```powershell
.\.venv\Scripts\python.exe examples\train_mlp.py train `
  --initialize-from artifacts\shared_mlp_best.pt
```

Continue a later run exactly from its saved optimizer boundary:

```powershell
.\.venv\Scripts\python.exe examples\train_mlp.py train `
  --resume-from artifacts\shared_mlp_latest.pt
```

Resume requires the same architecture, encoding/action versions, curriculum,
objective weights, and training-critical options. With the same Torch version
and device type, continuation from the boundary is numerically exact; changing
device type can introduce floating-point differences. Passing an old
model-only file to `--resume-from` still warm-starts with a warning, but
`--initialize-from` is clearer.

The route-and-delivery objective and protected-pickup mask have compatibility
schemas. Existing catalog models remain usable for inference, but an older
full training bundle cannot be resumed exactly under the changed reward,
mask, or curriculum. Start the first protected-policy five-hour run from
random weights with distinct artifact names:

```powershell
.\.venv\Scripts\python.exe examples\train_mlp.py train `
  --minutes 300 `
  --checkpoint artifacts\shared_mlp_protected_v3_best.pt `
  --latest-checkpoint artifacts\shared_mlp_protected_v3_latest.pt `
  --controller-name protected_route_v3
```

Do not add `--resume-from` or `--initialize-from` to that first revised run.
Later, resume this new run with
`--resume-from artifacts\shared_mlp_protected_v3_latest.pt` and the same
training-critical options.

Publish an existing model-only best checkpoint without retraining:

```powershell
.\.venv\Scripts\python.exe examples\train_mlp.py controllers publish `
  --checkpoint artifacts\shared_mlp_best.pt `
  --name my_controller
```

Each committed catalog entry has a uniquely named `.pt` file and matching
`.controller.json` manifest containing its persistent run ID, versions,
curriculum, objective configuration, validation result, counters, and SHA-256.
Qualified entries show their qualified stage; other usable entries are clearly
marked **Unqualified**. The manifest is written last, so incomplete packages
are not discovered. Corrupt, incompatible, missing, path-escaping, or
checksum-mismatched entries are skipped with a concise warning.

Evaluate the best checkpoint greedily and write full per-episode metrics with:

```bash
python examples/train_mlp.py evaluate \
  --checkpoint artifacts/shared_mlp_best.pt \
  --report artifacts/evaluation.json
```

Run the saved controller greedily and visualize the same fixed-slot episode
through the read-only Pygame renderer:

```bash
python examples/visualize_mlp_controller.py \
  --checkpoint artifacts/shared_mlp_protected_v3_best.pt \
  --stage 1 \
  --inspect-robot robot_1 \
  --trace-report artifacts/protected_v3_trace.json
```

The default seed is the first held-out seed for the selected stage. Close the
window or press Escape when finished. The side panel and optional JSON trace
show the selected action, detached masked probabilities, entropy, route and
delivery state, and every step-reward component. Add `--stochastic` to sample
commands, or `--headless` to exercise the identical controller loop without
Pygame.

The trainer uses parameter-sharing episodic REINFORCE, one team reward per
world step, and the sum of the active robots' log probabilities for the joint
command. Its default `gamma` is one. Route progress is dense and delivery
credit is sparse; both are signed, cycle-neutral potentials. There is no
standalone pickup/drop bonus. Ordinary visual and headless simulation
composition also stops issuing new forward steps once every item is
simultaneously delivered, while replay and Back remain available in Pygame.

## Tests

With the development extra installed:

```bash
python -m pytest
```

The tests use the standard-library unittest API and can also run without pytest:

```bash
python -m unittest discover -s tests -v
```
