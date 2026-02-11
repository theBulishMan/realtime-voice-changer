from __future__ import annotations

from pathlib import Path

from app.backend.realtime.metrics import SlidingRealtimeMetrics


def test_metrics_snapshot(tmp_path: Path):
    m = SlidingRealtimeMetrics(buffer_size=32, log_path=tmp_path / "metrics.ndjson")
    for i in range(1, 8):
        m.record(asr_ms=i * 10, tts_ms=i * 8, fad_ms=i * 20, e2e_ms=i * 50)
    snap = m.snapshot()
    assert snap.sample_count == 7
    assert snap.p95_fad_ms >= snap.p50_fad_ms
    assert snap.p95_e2e_ms >= snap.p50_e2e_ms


def test_metrics_reset(tmp_path: Path):
    log_path = tmp_path / "metrics.ndjson"
    m = SlidingRealtimeMetrics(buffer_size=8, log_path=log_path)
    m.record(asr_ms=10, tts_ms=20, fad_ms=30, e2e_ms=40)
    assert m.snapshot().sample_count == 1
    assert log_path.exists()
    assert log_path.read_text(encoding="utf-8").strip()

    m.reset(clear_log=True)
    snap = m.snapshot()
    assert snap.sample_count == 0
    assert snap.drop_rate == 0.0
    assert log_path.read_text(encoding="utf-8") == ""
