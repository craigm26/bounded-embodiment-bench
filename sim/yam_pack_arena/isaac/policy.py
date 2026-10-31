"""The four reference policies (``sim/yam_pack/policies.py``) as an Arena policy.

``--policy_type sim.yam_pack_arena.isaac.policy.ReferencePolicy --mode comply|ignore|right_only|do_nothing``

The policy is unchanged: it plans with IK on the same compiled MJCF (never stepped) and reads the
episode through ``core.PolicyView``, so its joint targets are the ones it sends in MuJoCo for the
same seed, up to the small differences in where the arms actually are.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import torch

from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg

from sim.yam_pack import scene
from sim.yam_pack.policies import Packer
from sim.yam_pack_arena.core import PolicyView


@dataclass
class ReferencePolicyCfg(PolicyCfg):
    mode: str = "comply"


@register_policy
class ReferencePolicy(PolicyBase[ReferencePolicyCfg]):
    name = "yam_pack_reference"

    def __init__(self, config: ReferencePolicyCfg):
        super().__init__(config)
        self.packer: Packer | None = None
        self._ik_model: mujoco.MjModel | None = None

    def _model(self, driver) -> mujoco.MjModel:
        if self._ik_model is None:
            self._ik_model = scene.build_model(driver.spec, driver.block_xy)
        return self._ik_model

    def get_action(self, env, observation) -> torch.Tensor:
        term = env.unwrapped.action_manager.get_term("driver")
        term.sync()
        drv = term.ensure_driver()
        if self.packer is None:
            self._ik_model = None       # block positions change per seed; the IK model carries them
            self.packer = Packer(PolicyView(drv, self._model(drv)), self.config.mode)
        a = self.packer.act(drv.observation())
        return torch.tensor(a, dtype=torch.float32, device=env.unwrapped.device).reshape(1, -1)

    def done(self) -> bool:
        return self.packer is not None and self.packer.done()

    def reset(self, env_ids=None) -> None:
        self.packer = None
