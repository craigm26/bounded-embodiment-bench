# yam_collision.xml

Copied unchanged from Robocurve's `inspect-robots-yam`
(github.com/robocurve/inspect-robots-yam, commit `9b5ebb58525f6d2b023c3d7ffe48a226cbabe104`, 2026-10-28,
`src/inspect_robots_yam/assets/yam_collision.xml`). That file is Robocurve's collision-only reduction of
MuJoCo Menagerie's `i2rt_yam/yam.xml` at Menagerie commit `71f066ad0be9cd271f7ed58c030243ef157af9f4`:
visual meshes, actuators and keyframes removed, collision capsules, joints, joint ranges and inertials kept.

License: MIT. The Menagerie model is i2rt robotics' (text in `MENAGERIE_LICENSE`, copied from the same commit);
the reduction is RoboCurve's (text in `INSPECT_ROBOTS_YAM_LICENSE`, copied from inspect-robots-yam's `LICENSE`).

`sim/yam_pack/scene.py` composes two copies side by side the way inspect-robots-yam's `CollisionChecker`
does (`left_`/`right_` prefixes, bases at y = +/-0.30 m, the defaults that file marks as unverified rig
geometry) and adds a position servo per joint. Which YAM variant ships in the ABC kit is still
unconfirmed (`embodiments/abc-bimanual/embodiment.yaml`).
