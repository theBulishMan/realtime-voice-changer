from __future__ import annotations

import gc
import re
import threading
from typing import Any
from pathlib import Path

import numpy as np

from app.backend.config import AppConfig
from app.backend.asr.asr_stage1 import _map_language

_AUTO_LANG_MIN_PROB = 0.60
_AUTO_HALLUCINATION_TEXTS = {
    "thankyou",
    "thankyouforwatching",
    "thanksforwatching",
    "thanks",
}


def _normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _map_language_whisper(language_hint: str) -> str | None:
    mapped = _map_language(language_hint)
    if mapped == "Chinese":
        return "zh"
    if mapped == "English":
        return "en"
    return None


class Stage2ASR:
    """Higher-quality final ASR."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._model: Any | None = None
        self._lock = threading.RLock()

    @property
    def ready(self) -> bool:
        if self.config.fake_mode:
            return True
        return self._model is not None

    def _load(self) -> None:
        if self._model is not None or self.config.fake_mode:
            return

        backend = self.config.asr_backend.strip().lower()
        if backend == "qwen3":
            try:
                import torch
                from qwen_asr import Qwen3ASRModel
            except Exception as exc:
                raise RuntimeError(
                    "Qwen3-ASR backend requested but `qwen-asr` is not installed. "
                    "Install with: pip install -U qwen-asr"
                ) from exc

            model_path = self.config.qwen_asr_local_dir.strip() or self.config.qwen_asr_model_id
            local_dir = self.config.qwen_asr_local_dir.strip()
            if local_dir:
                local_path = Path(local_dir)
                if not local_path.exists():
                    raise RuntimeError(
                        "Qwen3-ASR local model directory is missing: "
                        f"{local_path}. Run: python scripts/predownload_models.py "
                        "--all --provider modelscope --models-dir .cache/models"
                    )
            dtype_name = (self.config.model_dtype or "bfloat16").strip().lower()
            if dtype_name in {"bf16", "bfloat16"}:
                dtype = torch.bfloat16
            elif dtype_name in {"fp16", "float16", "half"}:
                dtype = torch.float16
            else:
                dtype = torch.float32

            self._model = Qwen3ASRModel.from_pretrained(
                model_path,
                dtype=dtype,
                device_map=self.config.model_device,
                max_new_tokens=max(32, int(self.config.qwen_asr_max_new_tokens)),
                max_inference_batch_size=max(1, int(self.config.qwen_asr_max_inference_batch_size)),
            )
            return

        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self.config.asr_large_model,
            device=self.config.asr_device,
            compute_type=self.config.asr_compute_type,
        )

    def _transcribe_once_whisper(self, audio: np.ndarray, language: str | None) -> tuple[str, Any]:
        assert self._model is not None
        segments, info = self._model.transcribe(
            audio.astype(np.float32),
            language=language,
            beam_size=5,
            best_of=5,
            vad_filter=False,
            condition_on_previous_text=False,
        )
        text = " ".join(seg.text.strip() for seg in segments if seg.text.strip()).strip()
        return text, info

    def _transcribe_qwen(self, audio: np.ndarray, language_hint: str) -> str:
        assert self._model is not None
        forced_lang = _map_language(language_hint)
        results = self._model.transcribe(
            audio=(audio.astype(np.float32), 16000),
            language=forced_lang,
            return_time_stamps=False,
        )
        if not results:
            return ""
        text = str(getattr(results[0], "text", "") or "").strip()
        if not text:
            return ""

        # Auto mode robustness: suppress frequent short English hallucinations on Chinese speech.
        if forced_lang is None:
            normalized = _normalize_text(text)
            predicted_lang = str(getattr(results[0], "language", "") or "").strip().lower()
            if normalized in _AUTO_HALLUCINATION_TEXTS and predicted_lang.startswith("english"):
                retry_results = self._model.transcribe(
                    audio=(audio.astype(np.float32), 16000),
                    language="Chinese",
                    return_time_stamps=False,
                )
                retry_text = (
                    str(getattr(retry_results[0], "text", "") or "").strip() if retry_results else ""
                )
                if retry_text and _normalize_text(retry_text) not in _AUTO_HALLUCINATION_TEXTS:
                    return retry_text
                return ""

        return text

    def transcribe_final(self, audio: np.ndarray, language_hint: str) -> str:
        if self.config.fake_mode:
            energy = float(np.mean(np.abs(audio))) if audio.size else 0.0
            return "" if energy < 0.01 else "simulated final transcription"

        with self._lock:
            self._load()
            assert self._model is not None

            backend = self.config.asr_backend.strip().lower()
            if backend == "qwen3":
                return self._transcribe_qwen(audio, language_hint)

            mapped = _map_language_whisper(language_hint)
            text, info = self._transcribe_once_whisper(audio, mapped)
            if not text:
                return ""

            if mapped is None:
                normalized = _normalize_text(text)
                lang_prob = float(getattr(info, "language_probability", 0.0) or 0.0)
                if normalized in _AUTO_HALLUCINATION_TEXTS and lang_prob < _AUTO_LANG_MIN_PROB:
                    retry_text, _retry_info = self._transcribe_once_whisper(audio, "zh")
                    retry_normalized = _normalize_text(retry_text)
                    if retry_text and retry_normalized not in _AUTO_HALLUCINATION_TEXTS:
                        return retry_text
                    return ""

        return text

    def warmup(self) -> None:
        if self.config.fake_mode:
            return
        with self._lock:
            self._load()

    def loaded_models(self) -> list[dict[str, Any]]:
        if self.config.fake_mode:
            return []
        acquired = self._lock.acquire(timeout=0.02)
        try:
            if self._model is None:
                return []
            backend = self.config.asr_backend.strip().lower()
            if backend == "qwen3":
                label = "Qwen3-ASR"
                device = str(self.config.model_device)
            else:
                label = f"Whisper {self.config.asr_large_model}"
                device = str(self.config.asr_device)
            return [
                {
                    "key": "asr_stage2",
                    "label": label,
                    "replicas": 1,
                    "device": device,
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
