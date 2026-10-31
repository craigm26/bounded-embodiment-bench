"""Load spec.yaml. Kept apart from scene.py so the oracle and its tests run without MuJoCo."""
from __future__ import annotations

from pathlib import Path

import yaml

SPEC = Path(__file__).resolve().parent / "spec.yaml"


def load_spec(path: Path = SPEC) -> dict:
    return yaml.safe_load(path.read_text())
