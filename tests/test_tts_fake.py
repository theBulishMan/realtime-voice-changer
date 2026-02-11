from __future__ import annotations

import builtins
import importlib
from pathlib import Path

import pytest

from app.backend.config import AppConfig
from app.backend.services.tts_manager import TTSManager


def test_fake_tts_synthesis(tmp_path: Path, monkeypatch):
    pytest.importorskip("soundfile")
    monkeypatch.setenv("RVC_FAKE_MODE", "1")
    monkeypatch.setenv("RVC_DATA_DIR", str(tmp_path / "data"))
    cfg = AppConfig.from_env()
    tts = TTSManager(cfg)
    wav, sr = tts.synthesize_with_prompt("hello world", "Auto", prompt_items=[])
    assert sr == cfg.tts_sample_rate
    assert wav.ndim == 1
    assert len(wav) > 0

    out = tmp_path / "out.wav"
    tts.save_wav(wav, sr, out)
    assert Path(out).exists()


def test_fake_prompt_path_does_not_import_qwen(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RVC_FAKE_MODE", "1")
    monkeypatch.setenv("RVC_DATA_DIR", str(tmp_path / "data"))
    cfg = AppConfig.from_env()
    tts = TTSManager(cfg)

    real_import = builtins.__import__

    def guard_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "qwen_tts":
            raise AssertionError("qwen_tts should not be imported in fake mode")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guard_import)

    wav, sr = tts.synthesize_with_prompt_path(
        text="hello fake mode",
        language="Auto",
        prompt_path=tmp_path / "missing_prompt.pt",
    )
    assert sr == cfg.tts_sample_rate
    assert wav.ndim == 1
    assert len(wav) > 0


def test_estimate_max_new_tokens_respects_bounds(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RVC_FAKE_MODE", "1")
    monkeypatch.setenv("RVC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RVC_TTS_MAX_NEW_TOKENS_MIN", "300")
    monkeypatch.setenv("RVC_TTS_MAX_NEW_TOKENS_CAP", "900")
    monkeypatch.setenv("RVC_TTS_TOKENS_PER_CHAR", "10")
    cfg = AppConfig.from_env()
    tts = TTSManager(cfg)

    short = tts._estimate_max_new_tokens("你好")
    long = tts._estimate_max_new_tokens("这是一个用于测试实时变声系统长文本合成预算的句子。")

    assert short >= 300
    assert short <= 900
    assert long >= short
    assert long <= 900


def test_prompt_items_cache_avoids_reloading(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("RVC_FAKE_MODE", "1")
    monkeypatch.setenv("RVC_DATA_DIR", str(tmp_path / "data"))
    cfg = AppConfig.from_env()
    tts = TTSManager(cfg)

    prompt_path = tmp_path / "prompt.pt"
    prompt_path.write_bytes(b"placeholder")

    prompt_store = importlib.import_module("app.backend.services.prompt_store")
    calls = {"count": 0}

    def _fake_load(path: Path, *, ref_text_override: str | None = None):
        calls["count"] += 1
        return [{"ref_text": ref_text_override or "none"}]

    monkeypatch.setattr(prompt_store, "load_prompt_items", _fake_load)

    first = tts._load_prompt_items_cached(prompt_path, ref_text_override="ref-a")
    second = tts._load_prompt_items_cached(prompt_path, ref_text_override="ref-a")

    assert calls["count"] == 1
    assert first[0]["ref_text"] == "ref-a"
    first[0]["ref_text"] = "mutated"
    assert second[0]["ref_text"] == "ref-a"


def test_fake_custom_voice_preview_and_catalog(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RVC_FAKE_MODE", "1")
    monkeypatch.setenv("RVC_DATA_DIR", str(tmp_path / "data"))
    cfg = AppConfig.from_env()
    tts = TTSManager(cfg)

    speakers, languages = tts.get_custom_voice_catalog()
    assert any(item["id"] == "Vivian" for item in speakers)
    assert "Auto" in languages

    wav, sr = tts.generate_custom_voice_preview(
        text="hello custom voice",
        language="English",
        speaker="Ryan",
        instruct="Very happy.",
    )
    assert sr == cfg.tts_sample_rate
    assert wav.ndim == 1
    assert len(wav) > 0


def test_parallel_tts_env_config_applies(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("RVC_FAKE_MODE", "1")
    monkeypatch.setenv("RVC_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RVC_TTS_INFER_WORKERS", "3")
    monkeypatch.setenv("RVC_TTS_MODEL_REPLICAS", "4")
    cfg = AppConfig.from_env()
    tts = TTSManager(cfg)
    assert cfg.tts_infer_workers == 3
    assert cfg.tts_model_replicas == 4
    assert tts._base_model_target_replicas() == 3
