# Carry vs weld in MuJoCo

Isaac Lab has no runtime weld, so the port carries a grasped object kinematically
(sim/yam_pack_arena/core.py). This runs both grasps in MuJoCo on every committed seed.

| Policy | Episodes | Same verdict | Same checks | Same violation kinds | Max named-point gap (mm) | Max final object gap (mm) |
|---|---|---|---|---|---|---|
| comply | 20 | 20 | 20 | 20 | 1.3 | 71.8 |
| ignore | 20 | 20 | 20 | 20 | 1.3 | 72.3 |
| right_only | 20 | 20 | 20 | 20 | 1.3 | 73.2 |
| do_nothing | 20 | 20 | 20 | 20 | 0.0 | 0.0 |

Final object gap is where the blocks come to rest. With the carry they settle up to 7 cm from the weld run,
still inside the box (14 x 16 cm), so no check changes.
