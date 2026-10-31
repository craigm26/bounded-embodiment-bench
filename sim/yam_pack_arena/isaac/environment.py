"""The pack-with-hand task as an Isaac Lab-Arena external environment.

    python isaaclab_arena/evaluation/policy_runner.py --device cpu --num_envs 1 \\
        --policy_type sim.yam_pack_arena.isaac.policy.ReferencePolicy --mode comply --num_episodes 1 \\
        --external_environment_class_path sim.yam_pack_arena.isaac.environment:YamPackEnvironment yam_pack

(``run.py`` wraps this and scores each episode with the oracle; ``BREV.md`` has the setup.)

The scene is the MuJoCo scene's: a table top at z = 0, the open box and three blocks from
``assets/`` at the seed's positions, the hand marker, and the two arms of ``yam_bimanual``. Physics
runs at the spec's 2 ms with 50 substeps per 10 Hz control step; Arena's defaults (1/120 s, 8
substeps) are replaced in ``env_cfg_callback``. The task ends only on time (the spec's 45 s plus a
margin); success and safety are the oracle's call on the trace, not a termination term.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg
from isaaclab.utils import configclass

from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

from sim.yam_pack.spec import load_spec
from sim.yam_pack_arena.isaac.backend import filter_arm_object_collisions

ASSETS = Path(__file__).resolve().parents[1] / "assets"
_SPEC = load_spec()


def _rigid(name: str, usd: str, pos) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        spawn=sim_utils.UsdFileCfg(usd_path=str(ASSETS / usd)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(float(v) for v in pos), rot=(0.0, 0.0, 0.0, 1.0)),
    )


@configclass
class YamPackSceneCfg:
    table = AssetBaseCfg(prim_path="{ENV_REGEX_NS}/table", spawn=sim_utils.UsdFileCfg(usd_path=str(ASSETS / "table.usda")))
    tray = _rigid("tray", "tray.usda", (*_SPEC["objects"]["box"]["centre"][:2], 0.0))
    block0 = _rigid("block0", "block.usda", _SPEC["objects"]["blocks"]["at"][0])
    block1 = _rigid("block1", "block.usda", _SPEC["objects"]["blocks"]["at"][1])
    block2 = _rigid("block2", "block.usda", _SPEC["objects"]["blocks"]["at"][2])
    hand = _rigid("hand", "hand.usda", _SPEC["hand"]["start"])
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))


@configclass
class YamPackEventsCfg:
    no_arm_object_contact = EventTermCfg(
        func=filter_arm_object_collisions,
        mode="prestartup",
        params={"arm_paths": ["{ENV_REGEX_NS}/left_yam", "{ENV_REGEX_NS}/right_yam"],
                "object_paths": ["{ENV_REGEX_NS}/tray", "{ENV_REGEX_NS}/block0", "{ENV_REGEX_NS}/block1",
                                 "{ENV_REGEX_NS}/block2"]},
    )


class YamPackTask(TaskBase):
    def __init__(self):
        super().__init__(episode_length_s=float(_SPEC["episode"]["max_time"]) + 5.0,
                         task_description=_SPEC.get("instruction", "Pack the blocks into the box."))

    def get_scene_cfg(self):
        return YamPackSceneCfg()

    def get_termination_cfg(self) -> TaskTerminationCfg:
        return TaskTerminationCfg(timeout_s=self.episode_length_s)

    def get_events_cfg(self):
        return YamPackEventsCfg()

    def get_mimic_env_cfg(self, arm_mode):
        raise NotImplementedError

    def get_metrics(self):
        return []


def physics_like_mujoco(env_cfg):
    """2 ms physics, 50 substeps per 10 Hz control step, gravity -9.81 z (MuJoCo's default)."""
    env_cfg.sim.dt = float(_SPEC["robot"]["physics_dt"])
    env_cfg.decimation = int(round(1.0 / _SPEC["robot"]["control_hz"] / env_cfg.sim.dt))
    env_cfg.sim.render_interval = env_cfg.decimation
    env_cfg.sim.gravity = (0.0, 0.0, -9.81)
    env_cfg.scene.num_envs = 1
    return env_cfg


class YamPackEnvironment(ExampleEnvironmentBase):
    name: str = "yam_pack"

    def get_env(self, args_cli: argparse.Namespace):
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene

        import sim.yam_pack_arena.isaac.embodiment  # noqa: F401  (registers yam_bimanual)

        embodiment = self.asset_registry.get_asset_by_name("yam_bimanual")()
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=Scene(assets=[]),
            task=YamPackTask(),
            env_cfg_callback=physics_like_mujoco,
        )

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        pass
