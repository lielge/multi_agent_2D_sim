# AGENTS.md

## Project overview

This repository contains a small deterministic 2D multi-agent simulator for grid-based robot/item experiments. The core simulation logic lives under `src/multi_agent_sim/`, while optional visualization code is isolated under `src/multi_agent_sim/visualization/`.

The package is designed around a clear separation:

- `world.py` owns authoritative state transitions and invariants.
- `entities.py` defines entity state and validation rules.
- `actions.py` defines the action vocabulary, battery costs, and action results.
- `controllers.py` defines controller observation DTOs and pluggable policies.
- `session.py` wires controllers to the world and manages replay/history.
- `generation.py` provides direct seeded scenario generation.

## Working rules for agents

- Keep the simulation core free of UI dependencies. Do not import Pygame from the core package.
- Prefer deterministic behavior and stable ordering in all state snapshots and controller outputs.
- Preserve the existing ownership model: the world owns mutation; controllers receive immutable observations only.
- Add or update tests for any behavior change in world transitions, controller logic, or session replay.
- Maintain compatibility with the public API exported by `src/multi_agent_sim/__init__.py`.

## Typical workflow

1. Install the project for local development:
   ```bash
   python -m pip install -e ".[visualization,dev]"
   ```
2. Run the relevant tests for the area you changed:
   ```bash
   python -m pytest tests/test_world.py tests/test_controllers.py -q
   ```
3. For full validation in this repo, run:
   ```bash
   python -m pytest -q
   ```

## Project conventions

- Use Python 3.11+ syntax and prefer explicit validation in constructors and dataclasses.
- Keep `Action`/`ActionResult` semantics consistent with the battery-cost rules described in the README.
- When adding configuration, keep defaults and validation behavior aligned with the current public docs.
- Favor minimal, targeted changes over broad refactors.

## Useful references

- `README.md` documents the public API and expected behavior.
- `docs/code_integration_guide.md` explains the architecture and extension points.
- `tests/` is the best source of expected behavior for the simulator.
