from __future__ import annotations

from types import SimpleNamespace

import pytest

import app.backend.main as backend_main


def test_gpu_gate_skips_in_fake_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        backend_main,
        "config",
        SimpleNamespace(
            fake_mode=True,
            require_gpu=True,
            model_device="cuda:0",
            asr_device="cuda",
        ),
    )
    monkeypatch.setattr(
        backend_main,
        "_gpu_status",
        backend_main._GpuRuntimeStatus(available=False, device_count=0, device_name=None),
    )

    backend_main._enforce_runtime_constraints()


def test_gpu_gate_rejects_missing_cuda(monkeypatch) -> None:
    monkeypatch.setattr(
        backend_main,
        "config",
        SimpleNamespace(
            fake_mode=False,
            require_gpu=True,
            model_device="cuda:0",
            asr_device="cuda",
        ),
    )
    monkeypatch.setattr(
        backend_main,
        "_gpu_status",
        backend_main._GpuRuntimeStatus(
            available=False,
            device_count=0,
            device_name=None,
            error="cuda unavailable",
        ),
    )

    with pytest.raises(RuntimeError, match="GPU is required in real mode"):
        backend_main._enforce_runtime_constraints()


def test_gpu_gate_rejects_cpu_device_config(monkeypatch) -> None:
    monkeypatch.setattr(
        backend_main,
        "config",
        SimpleNamespace(
            fake_mode=False,
            require_gpu=True,
            model_device="cpu",
            asr_device="cuda",
        ),
    )
    monkeypatch.setattr(
        backend_main,
        "_gpu_status",
        backend_main._GpuRuntimeStatus(available=True, device_count=1, device_name="RTX"),
    )

    with pytest.raises(RuntimeError, match="RVC_MODEL_DEVICE"):
        backend_main._enforce_runtime_constraints()
