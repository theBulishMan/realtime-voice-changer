from __future__ import annotations

import importlib
import queue
import time

import pytest


def _load_main(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.setenv("RVC_FAKE_MODE", "1")
    monkeypatch.setenv("RVC_DATA_DIR", str(tmp_path / "data"))
    module = importlib.import_module("app.backend.main")
    return importlib.reload(module)


def _create_voice(client):
    resp = client.post(
        "/api/v1/voices/design",
        json={
            "name": "inject-voice",
            "voice_prompt": "stable and clear",
            "preview_text": "hello inject",
            "language": "Auto",
            "save": True,
        },
    )
    assert resp.status_code == 200
    return resp.json()["voice"]["id"]


def test_realtime_inject_end_to_end(monkeypatch, tmp_path):
    testclient_mod = pytest.importorskip("fastapi.testclient")
    TestClient = testclient_mod.TestClient
    module = _load_main(monkeypatch, tmp_path)

    with TestClient(module.app) as client:
        voice_id = _create_voice(client)
        preview = client.post(
            f"/api/v1/voices/{voice_id}/preview",
            json={"text": "this is a loopback sample", "language": "Auto"},
        )
        assert preview.status_code == 200
        audio_b64 = preview.json()["audio_b64"]

        start = client.post("/api/v1/realtime/start", json={"voice_id": voice_id, "language": "Auto"})
        assert start.status_code == 200

        inject = client.post(
            "/api/v1/realtime/inject",
            json={
                "audio_b64": audio_b64,
                "inject_mode": "segment",
                "realtime_pacing": False,
                "repeat": 2,
                "pause_ms": 0,
                "append_silence_ms": 300,
            },
        )
        assert inject.status_code == 200
        payload = inject.json()
        assert payload["ok"] is True
        assert payload["inject_mode"] == "segment"
        assert payload["injected_segments"] == 2

        sample_count = 0
        deadline = time.time() + 2.0
        while time.time() < deadline:
            metrics = client.get("/api/v1/metrics/current")
            assert metrics.status_code == 200
            sample_count = int(metrics.json()["sample_count"])
            if sample_count > 0:
                break
            time.sleep(0.05)

        stop = client.post("/api/v1/realtime/stop")
        assert stop.status_code == 200
        assert sample_count > 0


def test_realtime_inject_queue_full_returns_409(monkeypatch, tmp_path):
    testclient_mod = pytest.importorskip("fastapi.testclient")
    TestClient = testclient_mod.TestClient
    module = _load_main(monkeypatch, tmp_path)

    class _AlwaysFullQueue:
        def put(self, *args, **kwargs):
            raise queue.Full

        def put_nowait(self, *args, **kwargs):
            raise queue.Full

        def get(self, *args, **kwargs):
            raise queue.Empty

    with TestClient(module.app) as client:
        voice_id = _create_voice(client)
        preview = client.post(
            f"/api/v1/voices/{voice_id}/preview",
            json={"text": "queue full test sample", "language": "Auto"},
        )
        assert preview.status_code == 200
        audio_b64 = preview.json()["audio_b64"]

        start = client.post("/api/v1/realtime/start", json={"voice_id": voice_id, "language": "Auto"})
        assert start.status_code == 200
        module._engine._queue = _AlwaysFullQueue()

        inject = client.post(
            "/api/v1/realtime/inject",
            json={
                "audio_b64": audio_b64,
                "inject_mode": "queue",
                "realtime_pacing": False,
                "repeat": 1,
                "pause_ms": 0,
                "append_silence_ms": 0,
            },
        )
        assert inject.status_code == 409
        assert "inject queue is full" in inject.json()["detail"]

        stop = client.post("/api/v1/realtime/stop")
        assert stop.status_code == 200


def test_realtime_stop_drops_late_inflight_metrics(monkeypatch, tmp_path):
    testclient_mod = pytest.importorskip("fastapi.testclient")
    TestClient = testclient_mod.TestClient
    module = _load_main(monkeypatch, tmp_path)

    with TestClient(module.app) as client:
        voice_id = _create_voice(client)
        preview = client.post(
            f"/api/v1/voices/{voice_id}/preview",
            json={"text": "inflight late sample", "language": "Auto"},
        )
        assert preview.status_code == 200
        audio_b64 = preview.json()["audio_b64"]

        start = client.post("/api/v1/realtime/start", json={"voice_id": voice_id, "language": "Auto"})
        assert start.status_code == 200

        def _slow_partial(audio, language):
            time.sleep(0.6)
            return "[partial]"

        module._engine.asr_stage1.transcribe_partial = _slow_partial
        module._engine.asr_stage2.transcribe_final = lambda audio, language: "late final"

        inject = client.post(
            "/api/v1/realtime/inject",
            json={
                "audio_b64": audio_b64,
                "inject_mode": "queue",
                "realtime_pacing": False,
                "repeat": 1,
                "pause_ms": 0,
                "append_silence_ms": 0,
            },
        )
        assert inject.status_code == 200

        time.sleep(0.1)
        stop = client.post("/api/v1/realtime/stop")
        assert stop.status_code == 200

        time.sleep(0.7)
        metrics = client.get("/api/v1/metrics/current")
        assert metrics.status_code == 200
        assert int(metrics.json()["sample_count"]) == 0
