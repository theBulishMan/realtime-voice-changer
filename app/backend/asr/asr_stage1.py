from __future__ import annotations

import gc
import threading
from typing import Any

import numpy as np

from app.backend.config import AppConfig


def _map_language(language_hint: str) -> str | None:
    hint = (language_hint or "auto").strip().lower()
    if hint in {"auto", ""}:
        return None
    if hint in {"chinese", "zh-cn", "zh"}:
        return "Chinese"
    if hint in {"english", "en-us", "en"}:
        return "English"
    return None


def _map_language_whisper(language_hint: str) -> str | None:
    mapped = _map_language(language_hint)
    if mapped == "Chinese":
        return "zh"
    if mapped == "English":
        return "en"
    return None


class Stage1ASR:
    """Low-latency partial ASR."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._model: Any | None = None
        self._lock = threading.RLock()

    @property
    def ready(self) -> bool:
        if self.config.fake_mode:
            return True
        if self.config.asr_backend.strip().lower() == "qwen3":
            return True
        return self._model is not None

    def _load(self) -> None:
        if self._model is not None or self.config.fake_mode:
            return
        if self.config.asr_backend.strip().lower() == "qwen3":
            # Stage2 handles full Qwen3-ASR decoding; keep stage1 lightweight.
            self._model = "qwen3-stage1-skip"
            return
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self.config.asr_small_model,
            device=self.config.asr_device,
            compute_type=self.config.asr_compute_type,
        )

    def transcribe_partial(self, audio: np.ndarray, language_hint: str) -> str:
        if self.config.fake_mode:
            energy = float(np.mean(np.abs(audio))) if audio.size else 0.0
            return "" if energy < 0.01 else "[partial speech]"
        if self.config.asr_backend.strip().lower() == "qwen3":
            # Avoid double ASR decoding cost in realtime path.
            return ""
        with self._lock:
            self._load()
            assert self._model is not None
            segments, _ = self._model.transcribe(
                audio.astype(np.float32),
                language=_map_language_whisper(language_hint),
                beam_size=1,
                best_of=1,
                vad_filter=False,
                condition_on_previous_text=False,
                temperature=0.0,
            )
        text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
        return text.strip()

    def warmup(self) -> None:
        if self.config.fake_mode:
            return
        with self._lock:
            self._load()

    def loaded_models(self) -> list[dict[str, Any]]:
        if self.config.fake_mode:
            return []
        backend = self.config.asr_backend.strip().lower()
        if backend == "qwen3":
            # qwen3 stage1 is a lightweight skip path and does not hold GPU model weights.
            return []
        acquired = self._lock.acquire(timeout=0.02)
        try:
            if self._model is None:
                return []
            return [
                {
                    "key": "asr_stage1",
                    "label": "ASR Stage1",
                    "replicas": 1,
                    "device": str(self.config.asr_device),
                    "unloadable": True,
                }
            ]
        finally:
            if acquired:
                self._lock.release()

    def unload_model(self) -> bool:
        if self.config.fake_mode:
            return False
        with self._lock:
            changed = self._model is not None
            self._model = None
        if changed:
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            except Exception:
                pass
        return changed
