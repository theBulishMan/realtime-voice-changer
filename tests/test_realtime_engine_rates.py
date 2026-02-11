from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from app.backend.config import AppConfig
from app.backend.realtime import realtime_engine as realtime_engine_module
from app.backend.realtime.realtime_engine import (
    RealtimeEngine,
    _continuous_ambience_chunk,
    _apply_edge_fade,
    _apply_low_level_gain,
    _apply_gain_db,
    _mix_ambience_noise,
    _apply_soft_limiter,
    _normalized_level,
    _ordered_capture_rates,
    _ordered_output_rates,
    _sanitize_output_wav,
    _speech_activity_ratio,
    _soft_noise_suppress,
    _trim_silence,
)
from app.backend.types import RealtimeSettings


def test_ordered_capture_rates_prefers_configured_rate() -> None:
    assert _ordered_capture_rates(16000) == [16000, 48000, 32000, 8000]


def test_ordered_capture_rates_falls_back_to_supported_defaults() -> None:
    assert _ordered_capture_rates(44100) == [48000, 32000, 16000, 8000]


def test_ordered_output_rates_prioritizes_preferred_and_defaults() -> None:
    assert _ordered_output_rates(24000, [48000, 44100])[:4] == [24000, 48000, 44100, 32000]


def test_normalized_level_zero_for_silence() -> None:
    silence = np.zeros(256, dtype=np.float32)
    assert _normalized_level(silence) == 0.0


def test_normalized_level_one_for_full_scale_signal() -> None:
    full = np.ones(256, dtype=np.float32)
    assert _normalized_level(full) == 1.0


def test_normalized_level_mid_range_for_quiet_signal() -> None:
    quiet = np.full(256, 0.1, dtype=np.float32)
    level = _normalized_level(quiet)
    assert 0.7 <= level <= 0.8


def test_sanitize_output_wav_converts_stereo_to_mono() -> None:
    stereo = np.array([[1.5, -1.5], [0.5, -0.5]], dtype=np.float32)
    mono = _sanitize_output_wav(stereo)
    assert mono.ndim == 1
    assert mono.shape[0] == 2
    assert np.max(np.abs(mono)) <= 1.0


def test_sanitize_output_wav_replaces_nan_and_inf() -> None:
    raw = np.array([0.2, np.nan, np.inf, -np.inf], dtype=np.float32)
    clean = _sanitize_output_wav(raw)
    assert np.isfinite(clean).all()


def test_apply_low_level_gain_boosts_quiet_audio() -> None:
    quiet = np.full(2048, 0.001, dtype=np.float32)
    boosted, peak, gain = _apply_low_level_gain(quiet)
    assert gain > 1.0
    assert boosted.shape == quiet.shape
    assert peak > 0.003


def test_apply_low_level_gain_keeps_normal_audio() -> None:
    normal = np.full(2048, 0.1, dtype=np.float32)
    boosted, peak, gain = _apply_low_level_gain(normal)
    assert gain == 1.0
    assert np.allclose(boosted, normal)
    assert np.isclose(peak, 0.1)


def test_apply_gain_db_boosts_audio() -> None:
    wav = np.full(1024, 0.1, dtype=np.float32)
    gained = _apply_gain_db(wav, 6.0)
    assert np.max(np.abs(gained)) > np.max(np.abs(wav))


def test_apply_gain_db_attenuates_audio() -> None:
    wav = np.full(1024, 0.1, dtype=np.float32)
    gained = _apply_gain_db(wav, -6.0)
    assert np.max(np.abs(gained)) < np.max(np.abs(wav))


def test_soft_noise_suppress_returns_finite_audio() -> None:
    noise = np.random.default_rng(0).normal(0.0, 0.01, size=16000).astype(np.float32)
    denoised = _soft_noise_suppress(noise, sample_rate=16000, strength=0.5)
    assert denoised.shape == noise.shape
    assert np.isfinite(denoised).all()


def test_speech_activity_ratio_distinguishes_silence_from_speech() -> None:
    quiet = np.zeros(1600, dtype=np.float32)
    speech = np.concatenate(
        [np.zeros(800, dtype=np.float32), np.full(800, 0.08, dtype=np.float32)]
    )
    assert _speech_activity_ratio(quiet) == 0.0
    assert _speech_activity_ratio(speech) > 0.4


def test_trim_silence_removes_edges() -> None:
    sr = 16000
    audio = np.concatenate(
        [
            np.zeros(int(sr * 0.12), dtype=np.float32),
            np.full(int(sr * 0.2), 0.04, dtype=np.float32),
            np.zeros(int(sr * 0.1), dtype=np.float32),
        ]
    )
    trimmed = _trim_silence(audio, sr)
    assert trimmed.size < audio.size
    assert trimmed.size > int(sr * 0.16)


def test_mix_ambience_noise_adds_subtle_texture() -> None:
    wav = np.zeros(24000, dtype=np.float32)
    out = _mix_ambience_noise(
        wav,
        sample_rate=24000,
        level=0.008,
        rng=np.random.default_rng(7),
    )
    assert out.shape == wav.shape
    assert np.isfinite(out).all()
    assert float(np.max(np.abs(out))) > 0.0


def test_continuous_ambience_chunk_outputs_noise_and_keeps_state() -> None:
    rng = np.random.default_rng(42)
    chunk1, state1 = _continuous_ambience_chunk(512, level=0.01, rng=rng, prev_state=0.0)
    chunk2, state2 = _continuous_ambience_chunk(512, level=0.01, rng=rng, prev_state=state1)

    assert chunk1.shape == (512,)
    assert chunk2.shape == (512,)
    assert np.isfinite(chunk1).all()
    assert np.isfinite(chunk2).all()
    assert float(np.max(np.abs(chunk1))) > 0.0
    assert float(np.max(np.abs(chunk2))) > 0.0
    assert np.isfinite(state1)
    assert np.isfinite(state2)


def test_continuous_ambience_chunk_zero_level_is_silent() -> None:
    rng = np.random.default_rng(7)
    chunk, state = _continuous_ambience_chunk(256, level=0.0, rng=rng, prev_state=0.123)
    assert chunk.shape == (256,)
    assert np.max(np.abs(chunk)) == 0.0
    assert state == 0.123


def test_sanitize_output_wav_scales_int16_pcm() -> None:
    src = np.array([0, 8192, -8192, 16384], dtype=np.int16)
    out = _sanitize_output_wav(src)
    assert out.dtype == np.float32
    assert np.max(np.abs(out)) <= 0.51


def test_apply_edge_fade_reduces_boundaries() -> None:
    wav = np.ones(24000, dtype=np.float32) * 0.4
    faded = _apply_edge_fade(wav, sample_rate=24000, fade_ms=10.0)
    assert faded[0] == 0.0
    assert faded[-1] == 0.0
    assert faded[12000] > 0.35


def test_apply_soft_limiter_reduces_hot_peaks() -> None:
    wav = np.ones(2048, dtype=np.float32) * 1.5
    limited = _apply_soft_limiter(wav, ceiling=0.96)
    assert np.max(np.abs(limited)) <= 0.9601


def test_sanitize_runtime_settings_falls_back_to_available_devices(monkeypatch, tmp_path) -> None:
    class FakeSoundDevice:
        default = SimpleNamespace(device=(0, 1))

        @staticmethod
        def query_devices(device=None):
            devices = [
                {
                    "name": "Mic",
                    "max_input_channels": 1,
                    "max_output_channels": 0,
                },
                {
                    "name": "Speaker",
                    "max_input_channels": 0,
                    "max_output_channels": 2,
                },
                {
                    "name": "Virtual",
                    "max_input_channels": 0,
                    "max_output_channels": 2,
                },
            ]
            if device is None:
                return devices
            idx = int(device)
            if idx < 0 or idx >= len(devices):
                raise ValueError("invalid device id")
            return devices[idx]

    monkeypatch.setitem(sys.modules, "sounddevice", FakeSoundDevice)
    monkeypatch.setattr(
        realtime_engine_module,
        "list_audio_devices",
        lambda _: SimpleNamespace(virtual_mic_output_id=2),
    )
    monkeypatch.setenv("RVC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RVC_FRONTEND_DIR", str(tmp_path / "frontend"))

    config = AppConfig.from_env()
    config.ensure_paths()

    class DummyDB:
        def __init__(self):
            self.saved = None

        def upsert_setting(self, key, value):
            self.saved = (key, value)

        def record_metric(self, ts, payload):
            return None

        def clear_metrics(self):
            return 0

    class DummyVoiceService:
        def get_voice(self, voice_id):
            return None

    class DummyASR:
        ready = True

        def transcribe_partial(self, segment, language):
            return ""

        def transcribe_final(self, segment, language):
            return ""

    class DummyTTS:
        pass

    events = []
    db = DummyDB()
    settings = RealtimeSettings(
        input_device_id=999,
        monitor_device_id=998,
        virtual_mic_device_id=997,
        monitor_enabled=False,
        vad_silence_ms=240,
        max_segment_ms=12000,
    )
    engine = RealtimeEngine(
        config=config,
        db=db,
        voice_service=DummyVoiceService(),
        tts_manager=DummyTTS(),
        asr_stage1=DummyASR(),
        asr_stage2=DummyASR(),
        settings=settings,
        event_sink=events.append,
    )

    engine._sanitize_runtime_settings()

    assert engine.settings.input_device_id == 0
    assert engine.settings.monitor_device_id == 1
    assert engine.settings.virtual_mic_device_id == 2
    assert db.saved is not None
    assert db.saved[0] == "realtime_settings"
    assert events
    assert events[-1].type == "error"


def test_sanitize_runtime_settings_upgrades_virtual_mic_to_preferred(monkeypatch, tmp_path) -> None:
    class FakeSoundDevice:
        default = SimpleNamespace(device=(0, 1))

        @staticmethod
        def query_devices(device=None):
            devices = [
                {
                    "name": "Mic",
                    "max_input_channels": 1,
                    "max_output_channels": 0,
                },
                {
                    "name": "Speaker",
                    "max_input_channels": 0,
                    "max_output_channels": 2,
                },
                {
                    "name": "CABLE Input (MME)",
                    "max_input_channels": 0,
                    "max_output_channels": 2,
                },
                {
                    "name": "CABLE Input (WASAPI)",
                    "max_input_channels": 0,
                    "max_output_channels": 2,
                },
            ]
            if device is None:
                return devices
            idx = int(device)
            if idx < 0 or idx >= len(devices):
                raise ValueError("invalid device id")
            return devices[idx]

    monkeypatch.setitem(sys.modules, "sounddevice", FakeSoundDevice)
    monkeypatch.setattr(
        realtime_engine_module,
        "list_audio_devices",
        lambda _: SimpleNamespace(virtual_mic_output_id=3),
    )
    monkeypatch.setenv("RVC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RVC_FRONTEND_DIR", str(tmp_path / "frontend"))

    config = AppConfig.from_env()
    config.ensure_paths()

    class DummyDB:
        def __init__(self):
            self.saved = None

        def upsert_setting(self, key, value):
            self.saved = (key, value)

        def record_metric(self, ts, payload):
            return None

        def clear_metrics(self):
            return 0

    class DummyVoiceService:
        def get_voice(self, voice_id):
            return None

    class DummyASR:
        ready = True

        def transcribe_partial(self, segment, language):
            return ""

        def transcribe_final(self, segment, language):
            return ""

    class DummyTTS:
        pass

    db = DummyDB()
    settings = RealtimeSettings(
        input_device_id=0,
        monitor_device_id=1,
        virtual_mic_device_id=2,  # valid but not preferred
        monitor_enabled=False,
        vad_silence_ms=240,
        max_segment_ms=12000,
    )
    events = []
    engine = RealtimeEngine(
        config=config,
        db=db,
        voice_service=DummyVoiceService(),
        tts_manager=DummyTTS(),
        asr_stage1=DummyASR(),
        asr_stage2=DummyASR(),
        settings=settings,
        event_sink=events.append,
    )

    engine._sanitize_runtime_settings()

    assert engine.settings.virtual_mic_device_id == 3
    assert db.saved is not None
    assert db.saved[0] == "realtime_settings"
    assert events
    assert events[-1].type == "notice"


def test_sanitize_runtime_settings_keeps_monitor_enabled_on_feedback_risk(monkeypatch, tmp_path) -> None:
    class FakeSoundDevice:
        default = SimpleNamespace(device=(0, 1))

        @staticmethod
        def query_devices(device=None):
            devices = [
                {
                    "name": "Headset (VXE V1)",
                    "max_input_channels": 1,
                    "max_output_channels": 2,
                },
                {
                    "name": "Speakers (VXE V1)",
                    "max_input_channels": 0,
                    "max_output_channels": 2,
                },
                {
                    "name": "CABLE Input (VB-Audio Virtual Cable)",
                    "max_input_channels": 0,
                    "max_output_channels": 2,
                },
            ]
            if device is None:
                return devices
            idx = int(device)
            if idx < 0 or idx >= len(devices):
                raise ValueError("invalid device id")
            return devices[idx]

    monkeypatch.setitem(sys.modules, "sounddevice", FakeSoundDevice)
    monkeypatch.setattr(
        realtime_engine_module,
        "list_audio_devices",
        lambda _: SimpleNamespace(virtual_mic_output_id=2),
    )
    monkeypatch.setenv("RVC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RVC_FRONTEND_DIR", str(tmp_path / "frontend"))

    config = AppConfig.from_env()
    config.ensure_paths()

    class DummyDB:
        def __init__(self):
            self.saved = None

        def upsert_setting(self, key, value):
            self.saved = (key, value)

        def record_metric(self, ts, payload):
            return None

        def clear_metrics(self):
            return 0

    class DummyVoiceService:
        def get_voice(self, voice_id):
            return None

    class DummyASR:
        ready = True

        def transcribe_partial(self, segment, language):
            return ""

        def transcribe_final(self, segment, language):
            return ""

    class DummyTTS:
        pass

    db = DummyDB()
    settings = RealtimeSettings(
        input_device_id=0,
        monitor_device_id=0,
        virtual_mic_device_id=2,
        monitor_enabled=True,
        vad_silence_ms=240,
        max_segment_ms=12000,
    )
    events = []
    engine = RealtimeEngine(
        config=config,
        db=db,
        voice_service=DummyVoiceService(),
        tts_manager=DummyTTS(),
        asr_stage1=DummyASR(),
        asr_stage2=DummyASR(),
        settings=settings,
        event_sink=events.append,
    )

    engine._sanitize_runtime_settings()

    assert engine.settings.monitor_enabled is True
    assert db.saved is None
    assert events
    assert events[-1].type == "notice"


def test_resolve_output_sample_rate_falls_back_from_24000_to_48000(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RVC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RVC_FRONTEND_DIR", str(tmp_path / "frontend"))

    config = AppConfig.from_env()
    config.ensure_paths()

    class DummyDB:
        def upsert_setting(self, key, value):
            return None

        def record_metric(self, ts, payload):
            return None

        def clear_metrics(self):
            return 0

    class DummyVoiceService:
        def get_voice(self, voice_id):
            return None

    class DummyASR:
        ready = True

        def transcribe_partial(self, segment, language):
            return ""

        def transcribe_final(self, segment, language):
            return ""

    class DummyTTS:
        pass

    engine = RealtimeEngine(
        config=config,
        db=DummyDB(),
        voice_service=DummyVoiceService(),
        tts_manager=DummyTTS(),
        asr_stage1=DummyASR(),
        asr_stage2=DummyASR(),
        settings=RealtimeSettings(),
        event_sink=lambda _evt: None,
    )

    class FakeSD:
        @staticmethod
        def query_devices(device):
            return {"default_samplerate": 48000.0}

        @staticmethod
        def check_output_settings(device, channels, samplerate, dtype):
            if samplerate != 48000:
                raise ValueError("invalid sample rate")
            return None

    rate, tried = engine._resolve_output_sample_rate(
        FakeSD,
        device_ids=[1, 2],
        channels=1,
        preferred_rate=24000,
    )
    assert rate == 48000
    assert 24000 in tried


def test_prepare_asr_audio_resamples_to_16k(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RVC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RVC_FRONTEND_DIR", str(tmp_path / "frontend"))

    config = AppConfig.from_env()
    config.ensure_paths()

    class DummyDB:
        def upsert_setting(self, key, value):
            return None

        def record_metric(self, ts, payload):
            return None

        def clear_metrics(self):
            return 0

    class DummyVoiceService:
        def get_voice(self, voice_id):
            return None

    class DummyASR:
        ready = True

        def transcribe_partial(self, segment, language):
            return ""

        def transcribe_final(self, segment, language):
            return ""

    class DummyTTS:
        pass

    engine = RealtimeEngine(
        config=config,
        db=DummyDB(),
        voice_service=DummyVoiceService(),
        tts_manager=DummyTTS(),
        asr_stage1=DummyASR(),
        asr_stage2=DummyASR(),
        settings=RealtimeSettings(),
        event_sink=lambda _evt: None,
    )
    engine._capture_sample_rate = 48000

    audio = np.zeros(48000, dtype=np.float32)
    asr_audio = engine._prepare_asr_audio(audio)
    assert 15900 <= asr_audio.size <= 16100


def test_start_resolves_auto_language_from_voice_hint(monkeypatch, tmp_path) -> None:
    class FakeSoundDevice:
        default = SimpleNamespace(device=(0, 1))

        @staticmethod
        def query_devices(device=None):
            devices = [
                {"name": "Mic", "max_input_channels": 1, "max_output_channels": 0},
                {"name": "Speaker", "max_input_channels": 0, "max_output_channels": 2},
                {"name": "CABLE Input", "max_input_channels": 0, "max_output_channels": 2},
            ]
            if device is None:
                return devices
            idx = int(device)
            if idx < 0 or idx >= len(devices):
                raise ValueError("invalid device id")
            return devices[idx]

    monkeypatch.setitem(sys.modules, "sounddevice", FakeSoundDevice)
    monkeypatch.setattr(
        realtime_engine_module,
        "list_audio_devices",
        lambda _hint: SimpleNamespace(virtual_mic_output_id=2),
    )
    monkeypatch.setenv("RVC_FAKE_MODE", "1")
    monkeypatch.setenv("RVC_FAKE_CAPTURE_INPUT", "0")
    monkeypatch.setenv("RVC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RVC_FRONTEND_DIR", str(tmp_path / "frontend"))

    config = AppConfig.from_env()
    config.ensure_paths()

    class DummyDB:
        def upsert_setting(self, key, value):
            return None

        def record_metric(self, ts, payload):
            return None

        def clear_metrics(self):
            return 0

    class DummyVoiceService:
        def get_voice(self, voice_id):
            return SimpleNamespace(id=voice_id, language_hint="Chinese")

    class DummyASR:
        ready = True

        def transcribe_partial(self, segment, language):
            return ""

        def transcribe_final(self, segment, language):
            return ""

    class DummyTTS:
        pass

    events = []
    settings = RealtimeSettings(
        input_device_id=0,
        monitor_device_id=1,
        virtual_mic_device_id=2,
        monitor_enabled=False,
        vad_silence_ms=180,
        max_segment_ms=4000,
    )
    engine = RealtimeEngine(
        config=config,
        db=DummyDB(),
        voice_service=DummyVoiceService(),
        tts_manager=DummyTTS(),
        asr_stage1=DummyASR(),
        asr_stage2=DummyASR(),
        settings=settings,
        event_sink=events.append,
    )

    engine.start(voice_id="v1", language="Auto")
    state = engine.state()
    engine.stop()

    assert state.language == "Chinese"


def test_process_segment_suppresses_fast_repeated_final_text(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RVC_FAKE_MODE", "0")
    monkeypatch.setenv("RVC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RVC_FRONTEND_DIR", str(tmp_path / "frontend"))
    config = AppConfig.from_env()
    config.ensure_paths()

    class DummyDB:
        def upsert_setting(self, key, value):
            return None

        def record_metric(self, ts, payload):
            return None

        def clear_metrics(self):
            return 0

    class DummyVoiceService:
        def get_voice(self, voice_id):
            return SimpleNamespace(
                id=voice_id,
                prompt_path=str(tmp_path / "dummy.pt"),
                ref_text="hello",
                language_hint="Auto",
            )

    class DummyASR:
        ready = True

        def transcribe_partial(self, segment, language):
            return "partial"

        def transcribe_final(self, segment, language):
            return "thank you"

    class DummyTTS:
        def __init__(self):
            self.calls = 0

        def synthesize_with_prompt_path(self, **kwargs):
            self.calls += 1
            return np.zeros(2400, dtype=np.float32), 24000

    events = []
    tts = DummyTTS()
    engine = RealtimeEngine(
        config=config,
        db=DummyDB(),
        voice_service=DummyVoiceService(),
        tts_manager=tts,
        asr_stage1=DummyASR(),
        asr_stage2=DummyASR(),
        settings=RealtimeSettings(),
        event_sink=events.append,
    )

    engine._voice_id = "v1"
    engine._language = "Auto"
    engine._capture_sample_rate = 16000
    engine._play_audio = lambda *args, **kwargs: None
    segment = np.ones(3200, dtype=np.float32) * 0.01

    engine._process_segment(segment)
    engine._process_segment(segment)

    assert tts.calls == 1
    assert any(evt.type == "notice" for evt in events)


def test_process_segment_suppresses_low_info_text(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RVC_FAKE_MODE", "0")
    monkeypatch.setenv("RVC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RVC_FRONTEND_DIR", str(tmp_path / "frontend"))
    config = AppConfig.from_env()
    config.ensure_paths()

    class DummyDB:
        def upsert_setting(self, key, value):
            return None

        def record_metric(self, ts, payload):
            return None

        def clear_metrics(self):
            return 0

    class DummyVoiceService:
        def get_voice(self, voice_id):
            return SimpleNamespace(
                id=voice_id,
                prompt_path=str(tmp_path / "dummy.pt"),
                ref_text="hello",
                language_hint="Auto",
            )

    class DummyASR:
        ready = True

        def transcribe_partial(self, segment, language):
            return ""

        def transcribe_final(self, segment, language):
            return "嗯"

    class DummyTTS:
        def __init__(self):
            self.calls = 0

        def synthesize_with_prompt_path(self, **kwargs):
            self.calls += 1
            return np.ones(2400, dtype=np.float32) * 0.03, 24000

    events = []
    tts = DummyTTS()
    engine = RealtimeEngine(
        config=config,
        db=DummyDB(),
        voice_service=DummyVoiceService(),
        tts_manager=tts,
        asr_stage1=DummyASR(),
        asr_stage2=DummyASR(),
        settings=RealtimeSettings(),
        event_sink=events.append,
    )
    engine._voice_id = "v1"
    engine._language = "Auto"
    engine._capture_sample_rate = 16000
    engine._play_audio = lambda *args, **kwargs: None

    # short segment should be suppressed as low-info filler.
    seg = np.ones(16000, dtype=np.float32) * 0.03
    engine._process_segment(seg)

    assert tts.calls == 0
    assert any(
        evt.type == "notice" and "low-info text suppressed" in str(evt.data.get("message", ""))
        for evt in events
    )
