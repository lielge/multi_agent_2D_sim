# Copilot instructions

This repository is a deterministic 2D multi-agent simulator. Keep the simulation core isolated from visualization and preserve the world-owned state model.

## Core architecture

- `src/multi_agent_sim/world.py`: authoritative simulation state and batched transitions.
- `src/multi_agent_sim/controllers.py`: immutable observations and controller interfaces.
- `src/multi_agent_sim/session.py`: scenario setup, controller execution, and replay.
- `src/multi_agent_sim/entities.py`: robot/item entities and validation.
- `src/multi_agent_sim/actions.py`: action vocabulary and energy costs.
- `src/multi_agent_sim/visualization/`: optional Pygame UI only.

## Expected behavior

- Controllers must receive immutable observations, not live world objects.
- The world should remain the single source of truth for occupancy, battery, and item state.
- Simulation outcomes should be deterministic and stable across equivalent inputs.
- Changes to world rules or controller contracts should be accompanied by tests.

## Validation

Run the project tests with:

```bash
python -m pytest -q
```

For targeted checks during development:

```bash
python -m pytest tests/test_world.py tests/test_controllers.py -q
```

## Style

- Keep changes minimal and domain-focused.
- Prefer explicit validation, small dataclasses, and clear public APIs.
- Do not add UI imports into the core package.
- Maintain backwards compatibility with the package exports in `src/multi_agent_sim/__init__.py`.
