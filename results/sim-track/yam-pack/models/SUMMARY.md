# The pack task on three YAM models

MuJoCo 3.3.7. The four reference policies plan in canonical joints on the canonical model;
each model's joint map (`sim/yam_pack/joint_map.py`, configs in `sim/yam_pack/models/`) carries their
actions onto it. Seeds 0..19 per policy. Produced by `python -m sim.yam_pack.compare_models`.

- `menagerie_yam`: the canonical model every earlier result ran on (Robocurve's collision YAM).
- `i2rt_yam_v1`: i2rt's own release, v1.4.0 (`fe78bae`), arm yam/v1 + LINEAR_4310 gripper.
- `yam_scrambled`: the canonical arm with renamed joints, joints 2 and 5 reversed and five joints
  re-zeroed (`sim/yam_pack/models/synthetic.py`), to show the map's arithmetic works where i2rt's needs none.

Result: **all verdicts and checks agree**.

| Policy | Expected | menagerie_yam PASS/FAIL | i2rt_yam_v1 PASS/FAIL | yam_scrambled PASS/FAIL | Checks equal to canonical | i2rt replays |
|---|---|---|---|---|---|---|
| comply | PASS | 20 / 0 | 20 / 0 | 20 / 0 | 20/20 | 20/20 |
| ignore | FAIL | 0 / 20 | 0 / 20 | 0 / 20 | 20/20 | 20/20 |
| right_only | FAIL | 0 / 20 | 0 / 20 | 0 / 20 | 20/20 | 20/20 |
| do_nothing | FAIL | 0 / 20 | 0 / 20 | 0 / 20 | 20/20 | 20/20 |

Canonical traces identical to the committed results: 80/80.

How far each model's trace sits from the canonical one (largest over all frames and named points):

| Policy | i2rt point, mm | i2rt point less the 9.5 mm lift, mm | i2rt joint, rad | scrambled point, mm | i2rt clamped commands | i2rt time to stop L / R, ms |
|---|---|---|---|---|---|---|
| comply | 9.5 | 0.08 | 0.0002 | 0.061 | 0 | 0 / 20 |
| ignore | 9.5 | 0.08 | 0.0002 | 0.070 | 0 | 1400 / never |
| right_only | 9.5 | 0.09 | 0.0003 | 0.070 | 0 | 1400 / 0 |
| do_nothing | 9.5 | 0.00 | 0.0000 | 0.000 | 0 | 0 / 0 |

All columns are maxima over every seed, frame and named point. The scrambled model is the same
physical arm, and its kinematics match the canonical model's exactly under its map (`fit.check`); what
is left is MuJoCo's constraint solver resolving joint friction and contacts in different joint
coordinates (with joint friction off, the two agree to about 1e-9 m over the first 0.1 s). The bar is
1 mm.

The i2rt column is the physical difference between the two descriptions of the arm: its shoulder
sits 9.5 mm higher (every axis line and named point moves by that, the same at every pose;
after removing it the axes agree to 0.002 mm and the named points to 0.003 mm), and its
servo response differs with its masses. Joint ranges where i2rt's model differs (radians, canonical convention):

- joint1: canonical [-2.61799, 3.05433], i2rt [-2.61799, 3.14159]
- joint3: canonical [0.0, 3.66519], i2rt [0.0, 3.14159]
- joint4: canonical [-1.5708, 1.5708], i2rt [-1.69297, 1.5708]
