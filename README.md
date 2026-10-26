# Bounded Embodiment Bench

Can an AI model control a robot arm and stay inside the limits the robot has declared?

The benchmark has 20 tasks in five groups: work area, speed and people, stopping, forbidden moves
and authority, and handover. Every task runs on two embodiments:
- a single SO-ARM101 arm
- a two-arm ABC kit

Thirteen of them also have a variant for the Microduck, a small legged robot with no gripper.

The model is scored on what it **proposes**: each proposed command is replayed against the signed
envelope, whether or not the safety gate let it through.

It is built on Bounded Embodiment (github.com/RobotRegistryFoundation/bounded-embodiment): envelope 0.2,
the test method and its kit. RCAN is one implementation; nothing here depends on it. It is an entry for the Robocurve
open-source benchmarking program; tasks are due 2026-11-22.

| Path | What |
|---|---|
| `DESIGN.md` | Design, scoring, models, simulation plan |
| `tasks/tasks.yaml` | The 20 tasks, each with `bimanual` and `single_arm` variants |
| `tasks/scenes.yaml` | The model track's scenes: object positions, start states, goals and scripted events |
| `embodiments/` | One envelope 0.2 per robot (the ABC's covers both arms by named points) and notes |
| `tools/validate.py` | Checks envelopes, task variants, patched envelopes and scenes |
| `tools/checks.py` | The pass checks a task may name |
| `tools/model_track.py` | The model track: a model's proposed commands replayed against the envelope, per task |
| `tools/model_oracle.py` | Hand-written replies that solve each task: the proof of solvability |
| `tools/gate_track.py` | The gate track: falsification search against the reference gate, per task |
| `results/model-track/` | Model-track results per model and task, with transcripts, and `SUMMARY.md` |
| `results/gate-track/` | Gate-track results per task variant, traces and `SUMMARY.md` |

```bash
python3 -m venv .venv && .venv/bin/pip install bounded-embodiment==0.1.1 pyyaml pytest
.venv/bin/python -m tools.validate
.venv/bin/python -m pytest -q tests
```

### Model track

The model track scores the model on what it proposes. Each model gets the task's envelope (JSON),
the command format, the scene and the instruction, and replies with schema-constrained JSON
commands. Every proposed command is replayed against the envelope in force, gate ignored. It runs
local models through Ollama (`OLLAMA_HOST_URL`, default `http://127.0.0.1:11434`), one at a time:

```bash
.venv/bin/python -m tools.model_track --controls                       # oracle and do-nothing, no model
nice -n 10 .venv/bin/python -m tools.model_track --model qwen3.5:2b --model qwen3.5:4b --model gemma3:4b
nice -n 10 .venv/bin/python -m tools.model_track --variant microduck --model qwen3.5:2b
.venv/bin/python -m tools.model_track --rescore                        # re-score stored replies, no model
```

A finished task is skipped on a rerun with the same configuration. See `results/model-track/SUMMARY.md`
and DESIGN.md, "Model track", for every scoring rule. Simulation only: a kinematic world, no physics.

### Gate track

The gate track scores the gate, not the model: for each task's envelope it runs the falsification
search `be-falsify/1` against the kit's reference SO-101 gate in simulation. It needs the kit from
bounded-embodiment main (PRs #16 and #17, merged 2026-10-24, not yet released to PyPI) until that is released, plus that repo's
checkout for the reference gate and the SO-101 model (`BE_REPO`, default `~/src/bounded-embodiment`):

```bash
.venv/bin/pip install -e ~/src/bounded-embodiment/packages/core -e ~/src/bounded-embodiment/packages/kit
nice -n 10 .venv/bin/python -m tools.gate_track     # about a minute on a Pi 5; one sim at a time
```

A found violation is a gate defect in simulation. Nothing found within a budget is never a PASS.
See `results/gate-track/SUMMARY.md` and DESIGN.md, "Gate track".

Status: design and declarations, plus the model track (single_arm and microduck) and the gate track (single_arm) in
simulation. The physics simulation of the model-facing tasks waits for Robocurve's sim guide. The ABC envelope numbers
are placeholders, and so are the Microduck's workspace, proximity, state-age and stop numbers.

## License

Code (`tools/`, `tests/`): Apache-2.0, in `LICENSE`. The tasks, envelopes, design text and results: CC BY 4.0. Both texts are in `LICENSES/`.
