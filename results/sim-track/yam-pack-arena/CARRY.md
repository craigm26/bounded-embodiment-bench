# Carry vs weld in MuJoCo

Isaac Lab has no runtime weld, so the port carries a grasped object with a soft weld: a force and
torque that pull it to the grasp pose, with MuJoCo's default weld stiffness and damping
(sim/yam_pack_arena/core.py). This runs both grasps in MuJoCo on every committed seed.

| Policy | Episodes | Same verdict | Same checks | Same violation kinds | Max named-point gap (mm) | Max final object gap (mm) |
|---|---|---|---|---|---|---|
| comply | 20 | 20 | 20 | 20 | 1.3 | 17.4 |
| ignore | 20 | 20 | 20 | 20 | 1.3 | 16.2 |
| right_only | 20 | 20 | 20 | 20 | 1.3 | 17.1 |
| do_nothing | 20 | 20 | 20 | 20 | 0.0 | 0.0 |

Final object gap is where the box and blocks come to rest; the worst is 17.4 mm from the weld run.
The carried box takes its contents along by friction, as the welded one does. The kinematic carry this
replaced wrote the box's pose straight into the simulator, which slid the box out from under its blocks
and left them up to 7 cm from the weld run. What is left is the last block's landing: it is dropped from
8 cm and bounces to a slightly different spot on the box floor (seed 0: 16 mm, both well inside).
