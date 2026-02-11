from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

from app.backend.config import AppConfig
from app.backend.db import AppDatabase
from app.backend.errors import ClientInputError
from app.backend.services.tts_manager import TTSManager
from app.backend.services.voice_service import VoiceService


def _build_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> VoiceService:
    monkeypatch.setenv("RVC_FAKE_MODE", "1")
    monkeypatch.setenv("RVC_DATA_DIR", str(tmp_path / "data"))
    cfg = AppConfig.from_env()
    cfg.ensure_paths()
    db = AppDatabase(cfg.db_path)
    db.init_schema()
    return VoiceService(cfg, db, TTSManager(cfg))


def _wav_bytes(seconds: float, sr: int = 24000) -> bytes:
    sf = pytest.importorskip("soundfile")
    frames = max(1, int(seconds * sr))
    wav = np.zeros(frames, dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV")
    return buf.getvalue()


def test_clone_rejects_unsupported_extension(tmp_path: Path, monkeypatch):
    svc = _build_service(tmp_path, monkeypatch)
    with pytest.raises(ClientInputError, match="Unsupported audio format"):
        svc.create_clone_voice(
            name="v1",
            language="Auto",
            ref_text="hello",
            ref_audio_bytes=b"abc",
            filename="ref.txt",
        )


def test_clone_rejects_oversized_file(tmp_path: Path, monkeypatch):
    svc = _build_service(tmp_path, monkeypatch)
    huge = b"0" * (svc.MAX_CLONE_FILE_SIZE_BYTES + 1)
    with pytest.raises(ClientInputError, match="Audio file too large"):
        svc.create_clone_voice(
            name="v1",
            language="Auto",
            ref_text="hello",
            ref_audio_bytes=huge,
            filename="ref.wav",
        )


def test_clone_rejects_too_short_audio(tmp_path: Path, monkeypatch):
    svc = _build_service(tmp_path, monkeypatch)
    with pytest.raises(ClientInputError, match="too short"):
        svc.create_clone_voice(
            name="v1",
            language="Auto",
            ref_text="hello",
            ref_audio_bytes=_wav_bytes(1.0, sr=24000),
            filename="ref.wav",
        )


def test_clone_rejects_too_long_audio(tmp_path: Path, monkeypatch):
    svc = _build_service(tmp_path, monkeypatch)
    with pytest.raises(ClientInputError, match="too long"):
        svc.create_clone_voice(
            name="v1",
            language="Auto",
            ref_text="hello",
            ref_audio_bytes=_wav_bytes(65.0, sr=24000),
            filename="ref.wav",
        )


def test_clone_rejects_low_sample_rate(tmp_path: Path, monkeypatch):
    svc = _build_service(tmp_path, monkeypatch)
    with pytest.raises(ClientInputError, match="sample rate too low"):
        svc.create_clone_voice(
            name="v1",
            language="Auto",
            ref_text="hello",
            ref_audio_bytes=_wav_bytes(4.0, sr=16000),
            filename="ref.wav",
        )


def test_clone_rejects_duplicate_name(tmp_path: Path, monkeypatch):
    svc = _build_service(tmp_path, monkeypatch)
    audio = _wav_bytes(4.0, sr=24000)
    svc.create_clone_voice(
        name="same-name",
        language="Auto",
        ref_text="hello",
        ref_audio_bytes=audio,
        filename="ref.wav",
    )

    with pytest.raises(ClientInputError, match="Voice name already exists"):
        svc.create_clone_voice(
            name=" same-name ",
            language="Auto",
            ref_text="hello",
            ref_audio_bytes=audio,
            filename="ref.wav",
        )
