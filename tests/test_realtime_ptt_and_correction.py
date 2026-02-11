from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from app.backend.config import AppConfig
from app.backend.realtime.realtime_engine import RealtimeEngine
from app.backend.types import RealtimeSettings


class _DummyDB:
    def upsert_setting(self, key, value):
        return None

    def record_metric(self, ts, payload):
        return None

    def clear_metrics(self):
        return 0


class _DummyVoiceService:
    def get_voice(self, voice_id):
        return None


class _DummyASR:
    ready = True

    def transcribe_partial(self, segment, language):
        return ""

    def transcribe_final(self, segment, language):
        return ""


class _DummyTTS:
    pass


def _make_engine(tmp_path, *, ptt_enabled: bool = True):
    config = AppConfig.from_env()
    config.data_dir = tmp_path / "data"
    config.frontend_dir = tmp_path / "frontend"
    config.ensure_paths()
    events = []
    engine = RealtimeEngine(
        config=config,
        db=_DummyDB(),
        voice_service=_DummyVoiceService(),
        tts_manager=_DummyTTS(),
        asr_stage1=_DummyASR(),
        asr_stage2=_DummyASR(),
        settings=RealtimeSettings(ptt_enabled=ptt_enabled),
        event_sink=events.append,
    )
    return engine, events


def test_audio_callback_blocks_frames_when_ptt_enabled_and_not_pressed(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RVC_FAKE_MODE", "1")
    monkeypatch.setenv("RVC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RVC_FRONTEND_DIR", str(tmp_path / "frontend"))
    engine, _events = _make_engine(tmp_path, ptt_enabled=True)

    frame = np.ones((320, 1), dtype=np.float32) * 0.02
    engine._ptt_active = False
    engine._audio_callback(frame, 320, None, None)
    assert engine._queue.empty()

    engine._ptt_active = True
    engine._audio_callback(frame, 320, None, None)
    assert not engine._queue.empty()


def test_maybe_correct_final_text_emits_event_when_text_changes(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RVC_FAKE_MODE", "1")
    monkeypatch.setenv("RVC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RVC_FRONTEND_DIR", str(tmp_path / "frontend"))
    engine, events = _make_engine(tmp_path, ptt_enabled=False)

    class _StubCorrector:
        def correct_text(self, text: str, language_hint: str):
            return SimpleNamespace(
                text=f"{text}!",
                changed=True,
                used_model=True,
                warning=None,
            )

    engine.text_corrector = _StubCorrector()
    corrected = engine._maybe_correct_final_text("hello", "English")
    assert corrected == "hello!"
    assert any(evt.type == "asr_corrected" for evt in events)


def test_submit_or_buffer_ptt_segment_buffers_while_pressed(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RVC_FAKE_MODE", "1")
    monkeypatch.setenv("RVC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RVC_FRONTEND_DIR", str(tmp_path / "frontend"))
    engine, _events = _make_engine(tmp_path, ptt_enabled=True)

    engine._running = True
    engine._session_id = 3
    engine._ptt_active = True

    submitted = []

    def _capture(seg, *, session_id: int):
        submitted.append((seg, session_id))

    engine._submit_segment = _capture  # type: ignore[method-assign]
    engine._submit_or_buffer_ptt_segment(np.ones(160, dtype=np.float32), session_id=3)
    assert submitted == []
    assert len(engine._ptt_pending_segments) == 1

    engine._ptt_active = False
    engine._submit_or_buffer_ptt_segment(np.ones(160, dtype=np.float32), session_id=3)
    assert len(submitted) == 1
    assert submitted[0][1] == 3


def test_set_ptt_active_release_merges_pending_segments(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RVC_FAKE_MODE", "1")
    monkeypatch.setenv("RVC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RVC_FRONTEND_DIR", str(tmp_path / "frontend"))
    engine, _events = _make_engine(tmp_path, ptt_enabled=True)

    engine._running = True
    engine._session_id = 7
    engine._ptt_active = True
    engine._ptt_pending_segments = [
        np.ones(120, dtype=np.float32),
        np.ones(80, dtype=np.float32),
    ]
    engine._ptt_pending_samples = 200

    class _Seg:
        def flush(self):
            return [np.ones(40, dtype=np.float32)]

    engine._segmenter = _Seg()

    submitted = []

    def _capture(seg, *, session_id: int):
        submitted.append((seg, session_id))

    engine._submit_segment = _capture  # type: ignore[method-assign]

    state = engine.set_ptt_active(False)
    assert state.ptt_active is False
    assert len(submitted) == 1
    merged, sid = submitted[0]
    assert sid == 7
    assert merged.shape[0] == 240
