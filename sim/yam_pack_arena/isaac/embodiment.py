"""Two YAM arms as one Isaac Lab-Arena embodiment, registered as ``yam_bimanual``.

Each arm is its own articulation (``left_robot``, ``right_robot``) spawned from ``assets/yam.usda`` at
the MuJoCo scene's base positions, root fixed to the world. Drive gains, armature and joint friction
are MuJoCo's, so the same joint targets give the same motion as closely as two solvers allow:

    joints 1..3 (DM4340)  kp 1500  kv 60  armature 0.032   frictionloss 0.1
    joints 4..6 (DM4310)  kp  200  kv  6  armature 0.0018  frictionloss 0.1
    fingers               kp   50  kv  2  armature 0.1     frictionloss 0.1, held open

(``sim/yam_pack/spec.yaml`` robot.servo, ``assets/yam_collision.xml`` defaults). MuJoCo's position
actuators have no force limit; the effort limits here are high enough never to bind.
"""
from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase

from sim.yam_pack.spec import load_spec
from sim.yam_pack_arena.isaac.backend import PackActionsCfg, pack_state

ASSETS = Path(__file__).resolve().parents[1] / "assets"
_SPEC = load_spec()
_SV = _SPEC["robot"]["servo"]
EFFORT_LIMIT = 1.0e4       # N·m / N; never binds, like MuJoCo's unlimited actuators
VELOCITY_LIMIT = 100.0     # rad/s; never binds


def _arm_cfg(side: str) -> ArticulationCfg:
    q0 = _SPEC["robot"]["start_q"][side]
    joint_pos = {f"joint{j + 1}": float(v) for j, v in enumerate(q0)}
    joint_pos.update({"left_finger": 0.037524, "right_finger": -0.037524})   # open (MJCF ranges)
    return ArticulationCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{side}_yam",
        spawn=sim_utils.UsdFileCfg(usd_path=str(ASSETS / "yam.usda"), fix_root_link=True),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(float(v) for v in _SPEC["robot"]["bases"][side]),
            rot=(0.0, 0.0, 0.0, 1.0),          # x, y, z, w
            joint_pos=joint_pos,
        ),
        actuators={
            "dm4340": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-3]"], stiffness=float(_SV["kp_big"]), damping=float(_SV["kv_big"]),
                armature=0.032, friction=0.1, joint_effort_limit=EFFORT_LIMIT, joint_velocity_limit=VELOCITY_LIMIT),
            "dm4310": ImplicitActuatorCfg(
                joint_names_expr=["joint[4-6]"], stiffness=float(_SV["kp_small"]), damping=float(_SV["kv_small"]),
                armature=0.0018, friction=0.1, joint_effort_limit=EFFORT_LIMIT, joint_velocity_limit=VELOCITY_LIMIT),
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=["left_finger", "right_finger"], stiffness=50.0, damping=2.0,
                armature=0.1, friction=0.1, joint_effort_limit=EFFORT_LIMIT, joint_velocity_limit=VELOCITY_LIMIT),
        },
    )


@configclass
class YamBimanualSceneCfg:
    left_robot: ArticulationCfg = _arm_cfg("left")
    right_robot: ArticulationCfg = _arm_cfg("right")


@configclass
class YamBimanualObservationsCfg:
    @configclass
    class PolicyCfg(ObservationGroupCfg):
        state = ObservationTermCfg(func=pack_state)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@register_asset
class YamBimanualEmbodiment(EmbodimentBase):
    """Two fixed-base YAM arms 0.6 m apart, driven by ``PackDriverAction``."""

    registry_name = "yam_bimanual"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scene_config = YamBimanualSceneCfg()
        self.action_config = PackActionsCfg()
        self.observation_config = YamBimanualObservationsCfg()

    def get_scene_key(self) -> str:
        return "left_robot"
