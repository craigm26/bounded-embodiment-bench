# A Bounded Embodiment gate for the simulated YAM

Robocurve's final-stage tasks run Dream-1 on two YAM arms in Isaac Lab-Arena. Each of my tasks puts one declared limit
in front of the model (a keep-out area, a keep-in area, a height band, a speed limit) and scores whether the model kept
to it. This package is the other half: the [Bounded Embodiment](https://github.com/RobotRegistryFoundation/bounded-embodiment)
gate that enforces the same limit on every command, so a task can be run twice, with the same model and scene:
Dream-1 straight to the arms, and Dream-1 through the gate.

- `be_yam_gate/gate.py`: the gate. For each arm and each command row it caps joint speed, samples the joint-space line
  from the last command it sent, runs forward kinematics on the arm's declared points (tool, wrist, elbow, fingertips),
  and keeps the longest prefix whose clearance (`Envelope.clearance` from the method's own `bounded_embodiment` core)
  stays at or above the margin; it holds the arm if none does, and caps the tool point's speed. Grippers pass through.
- `be_yam_gate/policy.py`: Inspect Robots policy `dream_be_gated` (Dream-1 with the gate in front of the arms). The
  gate's interventions go into the trial transcript.
- `envelopes/<task>.envelope.json`: one envelope 0.2 document per task, linted with `be envelope lint`.
- `tests/test_gate.py`: offline tests (no Isaac, no Dream): the first draft of the e-stop task's demo enters the
  keep-out ungated and never does gated; the resubmitted demo passes untouched; invalid rows hold; speed is capped.

Run, on an instance set up by Robocurve's walkthrough, after `pip install -e path/to/bounded-embodiment/packages/core -e .`:

    inspect-robots run --embodiment yam_arena --policy dream_be_gated -P envelope=envelopes/keep-estop-zone-clear.envelope.json \
      -E task=<task file> -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
      --instruction "<the task's instruction>" --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/gated

What this is not: it runs in simulation only, it has not been measured against a real arm, and its runs are not EV-03
or EV-10 results. It is an example of the method's gate on this robot, not a product.
