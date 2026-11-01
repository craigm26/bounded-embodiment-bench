"""The Isaac Lab-Arena port, checked without Isaac: the USD assets against MuJoCo, and the
simulator-agnostic driver (with the carry that replaces MuJoCo's weld) against the MuJoCo results."""
import json

import numpy as np
import pytest

pxr = pytest.importorskip("pxr", reason="usd-core not installed (pip install usd-core)")

from sim.yam_pack.oracle import score  # noqa: E402
from sim.yam_pack.run import ENVELOPE, OUT, episode as weld_episode  # noqa: E402
from sim.yam_pack.spec import load_spec  # noqa: E402
from sim.yam_pack_arena import core, usd_build, usd_fk  # noqa: E402

SPEC = load_spec()
ENV = json.loads(ENVELOPE.read_text())


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("usd")
    col = usd_build.Compiled(usd_build.COLLISION_MJCF)
    usd_build.write_robot(col, out / "yam.usda")
    usd_build.write_tray(SPEC, out / "tray.usda")
    usd_build.write_block(SPEC, out / "block.usda")
    usd_build.write_hand(SPEC, out / "hand.usda")
    usd_build.write_table(out / "table.usda")
    return out


def test_committed_assets_are_current(built):
    """The committed USD is what the builder writes now (rerun ``python -m sim.yam_pack_arena.usd_build``)."""
    for name in ("yam.usda", "tray.usda", "block.usda", "hand.usda", "table.usda"):
        assert (built / name).read_text() == (usd_build.ASSETS / name).read_text(), name


def test_usd_kinematics_match_mujoco(built):
    r = usd_fk.compare_with_mujoco(usd_build.COLLISION_MJCF, built / "yam.usda", n=100)
    assert r["body_pos_m"] < 1e-6 and r["point_m"] < 1e-6
    assert r["body_angle_rad"] < 1e-5
    assert r["limit_gap"] < 1e-6
    assert r["rest_pos_m"] < 1e-6 and r["rest_angle_rad"] < 1e-5      # authored poses agree with the joints
    assert abs(r["mass_usd_kg"] - r["mass_mujoco_kg"]) < 1e-5


def test_usd_structure(built):
    arm = usd_fk.UsdArm(built / "yam.usda")
    assert arm.base == "arm"
    assert [j.name for j in arm.joints] == [f"joint{i}" for i in range(1, 7)] + ["left_finger", "right_finger"]
    assert {j.kind for j in arm.joints[:6]} == {"revolute"} and {j.kind for j in arm.joints[6:]} == {"prismatic"}
    stage = pxr.Usd.Stage.Open(str(built / "yam.usda"))
    bodies = [p for p in stage.Traverse() if p.HasAPI(pxr.UsdPhysics.RigidBodyAPI)]
    assert all(p.GetAttribute("physxRigidBody:disableGravity").Get() for p in bodies)    # MuJoCo gravcomp
    assert sum(1 for p in stage.Traverse() if p.HasAPI(pxr.UsdPhysics.CollisionAPI)) > 20


def test_tray_matches_mujoco_box(built):
    """Same mass, centre of mass and inertia as MuJoCo's box body (uniform density walls)."""
    import mujoco

    from sim.yam_pack.scene import build_model

    m = build_model(SPEC, [b[:2] for b in SPEC["objects"]["blocks"]["at"]])
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    b = m.body("box").id
    stage = pxr.Usd.Stage.Open(str(built / "tray.usda"))     # keep the stage alive while reading
    prim = stage.GetPrimAtPath("/tray")
    mapi = pxr.UsdPhysics.MassAPI(prim)
    assert abs(mapi.GetMassAttr().Get() - m.body_mass[b]) < 1e-6
    assert np.allclose(np.array(mapi.GetCenterOfMassAttr().Get()), m.body_ipos[b], atol=1e-6)
    assert np.allclose(sorted(mapi.GetDiagonalInertiaAttr().Get()), sorted(m.body_inertia[b]), rtol=1e-4)


def _isaac_order(drv: core.PackDriver, be: core.MujocoBackend, action) -> None:
    """One control period called the way Isaac Lab's env step calls the action term: process once,
    then per substep apply (which first finishes the previous substep) and a physics step; the last
    substep is finished by the observation term's sync."""
    stepped = False
    drv.begin(action)
    for _ in range(drv.n_sub):
        if stepped:
            drv.post_physics()
        drv.pre_physics()
        be.physics_step()
        stepped = True
    drv.post_physics()


def test_isaac_call_order_gives_the_same_trace():
    from sim.yam_pack.episode import draw
    from sim.yam_pack.oracle import sha256_json
    from sim.yam_pack.policies import Packer

    traces = []
    for isaac in (False, True):
        dr = draw(SPEC, 0)
        be = core.MujocoBackend(SPEC, dr.block_xy)
        drv = core.PackDriver(be, SPEC, 0, draws=dr)
        pol = Packer(core.PolicyView(drv, be.m), "comply")
        while drv.t < 6.0:
            a = pol.act(drv.observation())
            _isaac_order(drv, be, a) if isaac else drv.step(a, be.physics_step)
        traces.append(sha256_json(drv.trace))
    assert traces[0] == traces[1]


@pytest.mark.parametrize("policy", ["comply", "ignore", "right_only", "do_nothing"])
def test_carry_gives_the_weld_verdicts(policy):
    """The soft-weld carry Isaac uses gets the verdicts and per-check results the MuJoCo weld gets,
    and the ones committed in results/sim-track/yam-pack (seed 0 here; ``python -m sim.yam_pack_arena.carry_check``
    runs all 20)."""
    tr, drv = core.episode(policy, 0, SPEC)
    v = score(tr, ENV, SPEC)
    committed = json.loads((OUT / f"{policy}.json").read_text())[0]
    assert v["verdict"] == committed["verdict"]
    assert {k: c["result"] for k, c in v["checks"].items()} == committed["checks"]
    weld, _ = weld_episode(policy, 0, SPEC)
    n = min(len(tr), len(weld))
    gap = max(np.linalg.norm(np.array(a["points"][p]) - np.array(b["points"][p]))
              for a, b in zip(tr[:n], weld[:n]) for p in a["points"])
    assert gap < 0.01       # the arms follow the same path; the carry only changes how objects ride


def test_carried_box_takes_its_contents_along():
    """The box is slid twice after block0 lands in it (6 cm in all). With the soft-weld carry block0
    rides with the box and ends where the weld leaves it; the old kinematic carry left it ~2.4 cm off."""
    tr, _ = core.episode("comply", 0, SPEC)
    weld, _ = weld_episode("comply", 0, SPEC)

    def local(f, name):
        c = np.array(f["objects"]["box"])
        return np.array(f["objects"][name]) - c       # the box stays upright: tilt is checked by task_success
    assert np.linalg.norm(local(tr[-1], "block0") - local(weld[-1], "block0")) < 0.005


def test_driver_is_deterministic():
    from sim.yam_pack.oracle import sha256_json

    a, _ = core.episode("ignore", 1, SPEC)
    b, _ = core.episode("ignore", 1, SPEC)
    assert sha256_json(a) == sha256_json(b)
