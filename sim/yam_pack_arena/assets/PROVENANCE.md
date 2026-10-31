# Generated USD

Every file here is written by `python -m sim.yam_pack_arena.usd_build` and checked against a fresh
build by `tests/test_sim_yam_arena.py`; edit the builder, not these files.

- `yam.usda`: one YAM arm, from `sim/yam_pack/assets/yam_collision.xml` (Robocurve's collision
  model, inspect-robots-yam `9b5ebb5`, a reduction of MuJoCo Menagerie `i2rt_yam` at `71f066a`;
  MIT, i2rt robotics, license text in `sim/yam_pack/assets/MENAGERIE_LICENSE`), compiled by MuJoCo 3.3.7.
- `tray.usda`, `block.usda`, `hand.usda`, `table.usda`: from the numbers in `sim/yam_pack/spec.yaml`.

`build/yam_visual.usdc` (not committed) adds Menagerie's visual meshes when the builder is given
`--visual /path/to/mujoco_menagerie/i2rt_yam/yam.xml`; its kinematics are checked against the
collision model's before it is written.
