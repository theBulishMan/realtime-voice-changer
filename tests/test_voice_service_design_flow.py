from __future__ import annotations

from pathlib import Path

import pytest

from app.backend.config import AppConfig
from app.backend.db import AppDatabase
from app.backend.services.tts_manager import TTSManager
from app.backend.services.voice_service import VoiceService
from app.backend.types import DesignVoiceRequest


def _build_service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> VoiceService:
    monkeypatch.setenv("RVC_FAKE_MODE", "1")
    monkeypatch.setenv("RVC_DATA_DIR", str(tmp_path / "data"))
    cfg = AppConfig.from_env()
    cfg.ensure_paths()
    db = AppDatabase(cfg.db_path)
    db.init_schema()
    return VoiceService(cfg, db, TTSManager(cfg))


def test_design_preview_not_saved(tmp_path: Path, monkeypatch):
    svc = _build_service(tmp_path, monkeypatch)
    req = DesignVoiceRequest(
        name="preview-only",
        voice_prompt="warm and clear",
        preview_text="hello world",
        language="Auto",
        save=False,
    )
    result = svc.create_design_voice(req)

    assert result.voice.mode == "design"
    assert result.preview_audio_b64
    assert svc.list_voices() == []
    assert not (svc.config.voices_dir / result.voice.id).exists()


def test_design_saved_persists_voice(tmp_path: Path, monkeypatch):
    svc = _build_service(tmp_path, monkeypatch)
    req = DesignVoiceRequest(
        name="saved-design",
        voice_prompt="warm and clear",
        preview_text="hello world",
        language="Auto",
        save=True,
    )
    result = svc.create_design_voice(req)

    voices = svc.list_voices()
    assert len(voices) == 1
    assert voices[0].id == result.voice.id
    assert (svc.config.voices_dir / result.voice.id / "meta.json").exists()


def test_design_preview_uses_runtime_clone_path(tmp_path: Path, monkeypatch):
    svc = _build_service(tmp_path, monkeypatch)
    called = {"runtime_preview": False}

    original = svc.tts.synthesize_with_prompt

    def _wrapped(text, language, prompt_items, **kwargs):
        called["runtime_preview"] = True
        return original(text, language, prompt_items, **kwargs)

    svc.tts.synthesize_with_prompt = _wrapped  # type: ignore[assignment]

    req = DesignVoiceRequest(
        name="runtime-preview-design",
        voice_prompt="warm and clear",
        preview_text="hello world",
        language="Auto",
        save=True,
    )
    result = svc.create_design_voice(req)

    assert result.preview_audio_b64
    assert called["runtime_preview"] is True


def test_design_save_reuses_preview_cache(tmp_path: Path, monkeypatch):
    svc = _build_service(tmp_path, monkeypatch)
    preview_req = DesignVoiceRequest(
        name="design-cache-preview",
        voice_prompt="warm and clear",
        preview_text="hello world",
        language="Auto",
        save=False,
    )
    preview_result = svc.create_design_voice(preview_req)
    assert preview_result.preview_cache_key

    def _should_not_run(*_args, **_kwargs):
        raise RuntimeError("design preview should be reused from cache")

    svc.tts.generate_voice_design_preview = _should_not_run  # type: ignore[assignment]

    save_req = DesignVoiceRequest(
        name="design-cache-saved",
        voice_prompt="warm and clear",
        preview_text="hello world",
        language="Auto",
        save=True,
        preview_cache_key=preview_result.preview_cache_key,
    )
    save_result = svc.create_design_voice(save_req)
    assert save_result.preview_reused is True
    assert save_result.preview_cache_key is None
    voices = svc.list_voices()
    assert len(voices) == 1
