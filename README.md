# Bounded Embodiment Bench

Can an AI model control a robot arm and stay inside the limits the robot has declared?

The benchmark has 20 tasks in five groups: work area, speed and people, stopping, forbidden moves
and authority, and handover. Every task runs on two embodiments:
- a single SO-ARM101 arm
- a two-arm ABC kit

The model is scored on what it **proposes**: each proposed command is replayed against the signed
envelope, whether or not the safety gate let it through.

It is built on Bounded Embodiment (github.com/RobotRegistryFoundation/bounded-embodiment): envelope 0.2,
the test method and its kit. RCAN is one implementation; nothing here depends on it. It is an entry for the Robocurve
open-source benchmarking program; tasks are due 2026-11-22.

| Path | What |
|---|---|
| `DESIGN.md` | Design, scoring, models, simulation plan |
| `tasks/tasks.yaml` | The 20 tasks, each with `bimanual` and `single_arm` variants |
| `embodiments/` | One envelope 0.2 per robot (the ABC's covers both arms by named points) and notes |
| `tools/validate.py` | Checks envelopes, task variants and patched envelopes |
| `tools/checks.py` | The pass checks a task may name |

```bash
python3 -m venv .venv && .venv/bin/pip install bounded-embodiment==0.1.1 pyyaml pytest
.venv/bin/python -m tools.validate
.venv/bin/python -m pytest -q tests
```

Status: design and declarations only. Simulation waits for Robocurve's sim guide; the ABC envelope
numbers are placeholders.
