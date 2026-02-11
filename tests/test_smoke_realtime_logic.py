from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_smoke_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "smoke_realtime.py"
    spec = importlib.util.spec_from_file_location("smoke_realtime_module", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load smoke_realtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_status_unverified_when_samples_insufficient():
    m = _load_smoke_module()
    status = m.resolve_status(
        sample_count=2,
        min_samples=3,
        fad_gate_pass=True,
        e2e_gate_pass=True,
    )
    assert status == "UNVERIFIED"


def test_resolve_status_pass_and_fail():
    m = _load_smoke_module()

    status_ok = m.resolve_status(
        sample_count=3,
        min_samples=3,
        fad_gate_pass=True,
        e2e_gate_pass=True,
    )
    assert status_ok == "PASS"

    status_fail = m.resolve_status(
        sample_count=3,
        min_samples=3,
        fad_gate_pass=False,
        e2e_gate_pass=True,
    )
    assert status_fail == "FAIL"
