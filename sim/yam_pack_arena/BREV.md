# Running the Isaac half on a GPU box (Brev)

Nothing here has been run on Isaac yet. These are the steps, in order, with what each one proves and
what to do if it fails. Budget: one RTX-class GPU instance (Isaac Sim needs RT cores; an L40S, L4 or
A10G works, an A100 or H100 does not render). PhysX itself runs on the CPU device for verdict runs.

## 1. Build Arena at the pinned commit

```bash
git clone https://github.com/isaac-sim/IsaacLab-Arena.git && cd IsaacLab-Arena
git checkout e87874def4aa689f675d4dfc1a58e1f2c6b9236b
git submodule update --init submodules/IsaacLab        # pins Isaac Lab e3f7d9ff1f5d7f8280de3ece9000b7117f66bcf4
./docker/run_docker.sh                                  # builds isaaclab_arena:latest (Isaac Sim 6.2.0 base) and opens a shell
```

Inside the container, smoke-test Arena itself before adding anything:

```bash
python isaaclab_arena/evaluation/policy_runner.py --headless --policy_type zero_action --num_steps 20 \
    --external_environment_class_path isaaclab_arena_examples.external_environments.basic:ExternalFrankaTableEnvironment \
    franka_table
```

## 2. Add this repo

```bash
cd /tmp && git clone https://github.com/craigm26/bounded-embodiment-bench.git && cd bounded-embodiment-bench
python -m pip install mujoco==3.3.7 pyyaml bounded-embodiment==0.1.1    # the policies' IK and the oracle
export PYTHONPATH=$PWD:$PYTHONPATH
```

The USD in `sim/yam_pack_arena/assets/` is committed, so no build step is needed here.

## 3. Check Isaac reads the arm the way MuJoCo does

```bash
python -m sim.yam_pack_arena.isaac.run --headless --check-fk
```

Writes random joint values into both arms, reads the named points back from PhysX, and compares them
with MuJoCo's forward kinematics. Pass: `"ok": true`, worst gap under 0.1 mm. This is the first place
the joint frames, axis signs and the fixed base meet PhysX.

If it fails: a gap of a whole link length means a joint sign or frame (look at `usd_build.frame_with_x`
and the `localRot` attributes); the whole arm offset means the root is not fixed or the base pose is
applied twice (`embodiment.py`, `fix_root_link` and `init_state.pos`).

## 4. One episode per policy

```bash
python -m sim.yam_pack_arena.isaac.run --headless --seeds 1
```

Pass: each policy prints `Isaac <verdict>, MuJoCo <verdict>` with no `differs`, and the run exits 0.
Expected verdicts: comply PASS, ignore FAIL, right_only FAIL, do_nothing FAIL. Each episode is
about 25 to 45 simulated seconds at 2 ms; the per-substep Python in the driver is the slow part, so
expect minutes per episode, not seconds.

If comply fails `task_success`: the blocks are not ending in the box. Check the carry (blocks should
ride with the right gripper) and the collision filter (the arms must not hit the box or blocks; look
for `/World/collision_groups` in the stage). If a policy fails a timing check MuJoCo passes, the arms
are slower or springier than MuJoCo's; compare `joint_pos` against the targets for one step.

## 5. All committed seeds

```bash
python -m sim.yam_pack_arena.isaac.run --headless --seeds 20
```

Writes `results/sim-track/yam-pack-isaac/<policy>.json` and `SUMMARY.md` with Isaac's verdict and the
committed MuJoCo verdict side by side. Bring those files back and commit them.

## Notes

- `--device cpu` is the runner's default. GPU PhysX is not bitwise repeatable, and the oracle's
  determinism check (same seed, same trace) is part of the result. Pass `--device cuda:0` to try it.
- One environment per process. The driver, the hand script and the trace are per episode.
- If the pinned Isaac Lab renamed an API, `backend.py` tries the known names (`_call`); anything else
  shows up as an `AttributeError` with the method name in it.
- The Isaac Lab conversion tools (`scripts/tools/convert_mjcf.py`, `convert_urdf.py`) are an
  alternative to `usd_build.py` if Robocurve's guide ships its own YAM USD; rerun step 3 on whatever
  asset is used.
