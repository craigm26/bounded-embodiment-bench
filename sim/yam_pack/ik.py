"""Damped least-squares IK for one YAM arm in the composed scene, used by the scripted policies.

Primary task: the tool point (grasp_site) at a position, with the gripper's approach axis pointing
straight down. Secondary: keep the elbow (joint 3 axis) at least ``ELBOW_MIN_X`` from the table edge,
because the ABC envelope bounds the elbow to x >= 0.10. Runs on a scratch MjData, never the episode's.
"""
from __future__ import annotations

import mujoco
import numpy as np

ELBOW_MIN_X = 0.12
DOWN = np.array([0.0, 0.0, -1.0])


class ArmIK:
    def __init__(self, model: mujoco.MjModel, side: str):
        self.m, self.side = model, side
        self.d = mujoco.MjData(model)
        self.site = model.site(f"{side}_grasp_site").id
        self.elbow = model.body(f"{side}_link_3").id
        joints = [model.joint(f"{side}_joint{j}").id for j in range(1, 7)]
        self.qadr = np.array([model.jnt_qposadr[j] for j in joints])
        self.dadr = np.array([model.jnt_dofadr[j] for j in joints])
        self.lo = model.jnt_range[joints, 0] + 1e-3
        self.hi = model.jnt_range[joints, 1] - 1e-3

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        R = self.d.site_xmat[self.site].reshape(3, 3)
        return self.d.site_xpos[self.site].copy(), R[:, 2].copy(), self.d.xpos[self.elbow].copy()

    def solve(self, target: np.ndarray, q0: np.ndarray, iters: int = 200, tol: float = 1e-4) -> tuple[np.ndarray, float]:
        q = np.clip(np.array(q0, float), self.lo, self.hi)
        jp = np.zeros((3, self.m.nv))
        jr = np.zeros((3, self.m.nv))
        je = np.zeros((3, self.m.nv))
        err = np.inf
        for _ in range(iters):
            p, z, e = self.fk(q)
            ep = target - p
            er = np.cross(z, DOWN)
            err = float(np.linalg.norm(ep))
            if err < tol and np.linalg.norm(er) < 1e-3 and e[0] >= ELBOW_MIN_X - 1e-3:
                break
            mujoco.mj_jacSite(self.m, self.d, jp, jr, self.site)
            mujoco.mj_jacBody(self.m, self.d, je, None, self.elbow)
            rows = [jp[:, self.dadr], 0.3 * jr[:, self.dadr]]
            errs = [ep, 0.3 * er]
            if e[0] < ELBOW_MIN_X:
                rows.append(2.0 * je[0:1, self.dadr])
                errs.append(np.array([2.0 * (ELBOW_MIN_X - e[0])]))
            J = np.vstack(rows)
            r = np.concatenate(errs)
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(J.shape[0]), r)
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        return q, err
