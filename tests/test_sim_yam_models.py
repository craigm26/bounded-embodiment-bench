"""The YAM joint-mapping layer: model configs, the map's arithmetic, the fit that derives a config from a
model file, and the pack task's verdicts on every model."""
import json

import numpy as np
import pytest
import yaml

from sim.yam_pack.joint_map import CANONICAL, JointMap, load_config
from sim.yam_pack.spec import load_spec
from tools.validate import ROOT

ENV = json.loads((ROOT / "embodiments/abc-bimanual/envelope.json").read_text())
SPEC = load_spec()
I2RT = "i2rt_yam_v1"
OUT = ROOT / "results/sim-track/yam-pack"


# ----- configs and the map (no simulator) ---------------------------------------------------------
def test_shipped_configs_load():
    canon, i2rt = load_config(), load_config(I2RT)
    assert canon.name == CANONICAL and canon.is_identity and canon.model.exists()
    assert i2rt.name == I2RT and i2rt.model.exists()
    # i2rt's release shares the canonical joint zeros, axes and names (see models/i2rt_yam_v1.yaml).
    assert i2rt.is_identity
    assert i2rt.weld_body == "gripper" and [f["joint"] for f in i2rt.fingers] == ["joint7", "joint8"]
    assert (i2rt.model.parent / "I2RT_LICENSE").exists()


@pytest.mark.parametrize("change,message", [
    (lambda c: c["joints"].reverse(), "in order"),
    (lambda c: c["joints"][1].update(sign=2), "must be \\+1 or -1"),
    (lambda c: c["points"].pop("elbow"), "tool, wrist and elbow"),
])
def test_bad_configs_are_refused(tmp_path, change, message):
    raw = yaml.safe_load(load_config().path.read_text())
    raw["model"] = str(load_config().model)
    change(raw)
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match=message):
        load_config(p)


def _scrambled_config(tmp_path):
    from sim.yam_pack.models import synthetic
    return load_config(synthetic.write(tmp_path)), synthetic


def test_map_round_trips_and_counts_clamps(tmp_path):
    raw = yaml.safe_load(load_config().path.read_text())
    raw["model"] = str(load_config().model)
    signs, offsets = [1, -1, 1, -1, 1, -1], [0.1, -0.2, 0.0, 1.0, -1.5, 3.0]
    for j, s, o in zip(raw["joints"], signs, offsets):
        j.update(sign=s, offset=o)
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(raw))
    cfg = load_config(p)
    assert not cfg.is_identity
    q = np.random.default_rng(0).uniform(-1, 1, 6)
    jm = JointMap(cfg)
    n = jm.to_native(q)
    np.testing.assert_allclose(n, np.array(signs) * q + offsets)
    np.testing.assert_allclose(jm.to_canonical(n), q, atol=1e-15)
    ranges = np.stack([n - 0.5, n + 0.5], axis=1)
    jm = JointMap(cfg, ranges)
    np.testing.assert_allclose(jm.to_native(q), n)
    assert jm.clips == 0
    out = jm.to_native(q + np.array([0, 1.0, 0, 0, -1.0, 0]))     # two joints past their limits
    assert jm.clips == 2
    assert out[1] == pytest.approx(ranges[1, 0]) and out[4] == pytest.approx(ranges[4, 0])   # sign -1 / +1
    cr = jm.canonical_ranges()
    assert np.all(cr[:, 0] <= q) and np.all(q <= cr[:, 1])


# ----- the fit and check, in MuJoCo --------------------------------------------------------------
def test_i2rt_model_matches_canonical_up_to_the_shoulder_lift():
    pytest.importorskip("mujoco")
    from sim.yam_pack.models import fit
    c = fit.check(load_config(I2RT), n=200)
    assert c["lift_mm"][:2] == [0.0, 0.0] and c["lift_mm"][2] == pytest.approx(9.5, abs=0.05)
    assert c["axis_angle_deg"] < 1e-3
    assert max(c["axis_line_after_lift_mm"]) < 0.01
    assert max(c["points_after_lift_mm"].values()) < 0.01


def test_fit_recovers_the_scrambled_map(tmp_path):
    pytest.importorskip("mujoco")
    from sim.yam_pack.models import fit
    cfg, synthetic = _scrambled_config(tmp_path)
    got = fit.fit(cfg.model, list(synthetic.NAMES), {"tool": "slink_6", "wrist": "slink_5", "elbow": "slink_3"},
                  "left_finger")
    assert [j["sign"] for j in got["joints"]] == list(synthetic.SIGNS)
    np.testing.assert_allclose([j["offset"] for j in got["joints"]], synthetic.OFFSETS, atol=1e-5)
    c = fit.check(cfg, n=200)
    assert c["axis_angle_deg"] < 1e-4 and max(c["axis_line_mm"]) < 1e-4      # acos rounding near 1
    assert max(c["points_mm"].values()) < 1e-4


def test_fit_recovers_i2rt_as_identity():
    pytest.importorskip("mujoco")
    from sim.yam_pack.models import fit
    cfg = load_config(I2RT)
    got = fit.fit(cfg.model, [j.native for j in cfg.joints], {"tool": "gripper", "wrist": "link5", "elbow": "link3"},
                  "joint7")
    assert [j["sign"] for j in got["joints"]] == [1] * 6
    np.testing.assert_allclose([j["offset"] for j in got["joints"]], 0.0, atol=1e-5)
    assert got["lift_m"][2] == pytest.approx(0.0095, abs=5e-5)
    np.testing.assert_allclose(got["points"]["tool"]["pos"], cfg.points["tool"]["pos"], atol=1e-5)


# ----- the pack task on each model ---------------------------------------------------------------
def _run(policy, model, seed=0):
    from sim.yam_pack.compare_models import episode
    from sim.yam_pack.oracle import score
    tr, clips = episode(policy, seed, SPEC, model)
    v = score(tr, ENV, SPEC)
    return tr, clips, v["verdict"], {k: c["result"] for k, c in v["checks"].items()}


def test_canonical_model_by_name_reproduces_the_committed_trace():
    pytest.importorskip("mujoco")
    from sim.yam_pack.oracle import sha256_json
    tr, clips, verdict, _ = _run("comply", CANONICAL)
    committed = json.loads((OUT / "comply.json").read_text())[0]
    assert sha256_json(tr) == committed["trace_sha256"] and clips == 0 and verdict == "PASS"


@pytest.mark.parametrize("policy", ["comply", "ignore", "right_only", "do_nothing"])
def test_i2rt_model_gives_the_canonical_verdicts(policy):
    pytest.importorskip("mujoco")
    _, clips, verdict, checks = _run(policy, I2RT)
    committed = json.loads((OUT / f"{policy}.json").read_text())[0]
    assert verdict == committed["verdict"] and checks == committed["checks"]
    assert clips == 0


def test_scrambled_model_reproduces_the_canonical_episode(tmp_path):
    """Same physical arm under a non-trivial map: same verdict, points within solver noise, and joints
    reported in canonical values. A wrong sign or offset would put the arm centimetres away."""
    pytest.importorskip("mujoco")
    cfg, _ = _scrambled_config(tmp_path)
    a, _, va, ca = _run("comply", CANONICAL)
    b, clips, vb, cb = _run("comply", cfg.path)
    assert (vb, cb) == (va, ca) and clips == 0
    assert set(b[0]["joints"]) == set(a[0]["joints"])      # joints reported under canonical names
    n = min(len(a), len(b))
    gap = max(np.linalg.norm(np.subtract(x["points"][p], y["points"][p])) for x, y in zip(a[:n], b[:n]) for p in x["points"])
    jgap = max(abs(x["joints"][j] - y["joints"][j]) for x, y in zip(a[:n], b[:n]) for j in x["joints"])
    assert gap < 1e-3 and jgap < 1e-2


def test_arena_driver_runs_a_mapped_model():
    pytest.importorskip("mujoco")
    from sim.yam_pack.oracle import score
    from sim.yam_pack_arena import core
    tr, _ = core.episode("comply", 0, SPEC, model=I2RT)
    assert score(tr, ENV, SPEC)["verdict"] == "PASS"
