# Pack with a hand mid-task, ported to Isaac Lab-Arena

The MuJoCo task in `sim/yam_pack` (B3, bimanual YAM) as an Isaac Lab-Arena external environment, so
the sim track does not depend on which simulator Robocurve's guide names. Same spec, same seeds, same
four reference policies, same oracle on the same trace format. Pinned to IsaacLab-Arena `e87874d`
(v0.3.0, 2026-10-30), which pins Isaac Lab `e3f7d9ff`.

Everything outside `isaac/` runs on a CPU with MuJoCo and usd-core and is tested here. Everything in
`isaac/` needs Isaac Sim and runs on a GPU box; `BREV.md` says how, and what is still unverified.

```bash
.venv/bin/pip install usd-core==26.11
.venv/bin/python -m sim.yam_pack_arena.usd_build      # writes assets/*.usda from the MJCF
.venv/bin/python -m sim.yam_pack_arena.usd_fk         # USD forward kinematics vs MuJoCo
.venv/bin/python -m sim.yam_pack_arena.carry_check    # carry vs weld, 4 policies x 20 seeds, ~8 min
.venv/bin/python -m pytest -q tests/test_sim_yam_arena.py
```

## What is in it

| File | What |
|---|---|
| `usd_build.py` | YAM arm, box, block, hand marker and table as USD, from the same MJCF MuJoCo runs (compiled by MuJoCo, so kinematics, limits, inertials and capsules are MuJoCo's numbers). Optional meshed copy from Menagerie for rendered clips. |
| `usd_fk.py` | Forward kinematics read straight from the USD joints, and the comparison with MuJoCo |
| `core.py` | The episode driver with the simulator behind a small `Backend` protocol: ramped joint targets, the hand, the sensor, grasping, the trace. `MujocoBackend` runs it in MuJoCo. |
| `carry_check.py` | Runs every committed policy and seed with the weld and with the carry, and compares the verdicts |
| `assets/` | The generated USD (committed so Isaac needs no MuJoCo to load it); `assets/build/` is the optional meshed copy, not committed |
| `isaac/backend.py` | `Backend` over Isaac Lab assets, the action term that runs the driver inside Isaac Lab's step, the observation term, the collision filter |
| `isaac/embodiment.py` | `yam_bimanual`: two fixed-base YAM articulations with MuJoCo's gains, armature and joint friction |
| `isaac/environment.py` | The scene, task and physics settings (2 ms, 50 substeps per 10 Hz step) as an Arena external environment |
| `isaac/policy.py` | The four reference policies as an Arena policy (`--policy_type sim.yam_pack_arena.isaac.policy.ReferencePolicy`) |
| `isaac/run.py` | Runs policies and seeds in Isaac, scores each trace with the oracle, compares with the committed MuJoCo verdict; `--check-fk` checks Isaac's reading of the USD |

## What changed from MuJoCo, and what that does to verdicts

- **Grasp.** MuJoCo welds a grasped object to `link_6`. Isaac Lab has no runtime weld, so a grasp
  records the object's pose relative to `link_6` and, before every physics step, applies a force and
  torque at the object's centre of mass that pull it to that pose (a soft weld, with MuJoCo's default
  weld stiffness and damping, plus the object's weight; Isaac Lab's external wrench). The object stays
  dynamic, so blocks in the box ride along by friction. Run in MuJoCo on all 80 committed episodes,
  the carry gives the same verdict, the same result on every check and the same violation kinds as
  the weld; named points stay within 1.3 mm of the weld run and objects come to rest within 17.4 mm
  (`results/sim-track/yam-pack-arena/CARRY.md`). The kinematic carry it replaced wrote the pose
  straight into the simulator, which slid the box out from under its contents.
- **Table.** A static box with its top at z = 0 instead of MuJoCo's ground plane, because Isaac's
  ground plane loads from Nucleus.
- **Friction.** MuJoCo's default sliding friction (1.0) is authored as a physics material on the
  table, box and blocks; PhysX would otherwise use 0.5.
- **Arm contacts.** The trace's `arm_contacts` list (left arm touching right arm) is empty in Isaac.
  The envelope has no rule on it and the oracle only counts it in a metric.
- **Solver.** PhysX's TGS and implicit drives are not MuJoCo's implicitfast and position actuators. The
  same targets will not give bitwise the same motion; the question `isaac/run.py` answers is whether
  they give the same verdicts.

## The YAM model

The USD comes from Robocurve's vendored collision model (inspect-robots-yam `9b5ebb5`, a reduction of
Menagerie `i2rt_yam` at `71f066a`), so the port and the MuJoCo results describe the same arm. i2rt's
own release (v1.4.0, `fe78bae`, 2026-10-28) describes the same joints differently in body frames
only: the joint zeros, axis directions and names are the same, so the 14-D wire format's angles mean the
same pose on both. What differs is the shoulder (everything from joint 2 up sits 9.5 mm higher in i2rt's
model), the gripper bodies, the masses, and three limits (joint1 up to 3.14 rad against 3.05 rad,
joint3 up to 3.14 rad against 3.67 rad, joint4 down to -1.69 rad against -1.57 rad). An earlier version
of this note said the joint zeros differ; they do not (`sim/yam_pack/models/fit.py --check i2rt_yam_v1`).
The joint-mapping layer (`sim/yam_pack/joint_map.py`, `sim/yam_pack/README.md`, "Other YAM models")
runs the MuJoCo scene and this driver on i2rt's model with the same verdicts; `MujocoBackend` and
`core.episode` take `model=`. The Isaac backend still loads the canonical USD only. Which description
ships with the ABC kit is still unconfirmed (`embodiments/abc-bimanual/embodiment.yaml`).
