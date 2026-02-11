from __future__ import annotations

import gc
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.backend.config import AppConfig
from app.backend.services.siliconflow_client import SiliconFlowClient

logger = logging.getLogger("rvc.text_corrector")

_FILLER_PATTERNS = (
    r"(?:^|[\s,.;!?])(?:uh+|um+|erm+|hmm+)(?=[\s,.;!?]|$)",
    r"(?:^|[\s,.;!?])(?:mm+|ah+|eh+)(?=[\s,.;!?]|$)",
    r"(?:^|[\s,.;!?])(?:\u55ef+|\u5443+|\u554a+|\u989d+)(?=[\s,.;!?]|$)",
)
_REPEATED_PUNCT = re.compile(r"([,.;!?])\1+")
_MULTI_SPACES = re.compile(r"\s+")
_QUOTED_PREFIX = re.compile(r"^(?:corrected(?: text)?\s*[:：]\s*)", re.IGNORECASE)


@dataclass(slots=True)
class TextCorrectionResult:
    text: str
    changed: bool
    used_model: bool
    warning: str | None = None


def _normalize_text(text: str) -> str:
    return _MULTI_SPACES.sub(" ", str(text or "").strip())


def _strip_fillers(text: str) -> str:
    out = f" {_normalize_text(text)} "
    for pattern in _FILLER_PATTERNS:
        out = re.sub(pattern, " ", out, flags=re.IGNORECASE)
    out = _MULTI_SPACES.sub(" ", out).strip()
    out = _REPEATED_PUNCT.sub(r"\1", out)
    out = re.sub(r"^[,.;!?]+\s*", "", out)
    out = re.sub(r"\s*[,.;!?]+$", "", out)
    return out.strip()


def _safe_model_path(model_id: str) -> Path | None:
    model_id = str(model_id or "").strip()
    if not model_id:
        return None
    path = Path(model_id)
    if path.exists() and path.is_dir():
        return path
    return None


class TextCorrector:
    """Optional LLM pass for ASR post-correction before TTS."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._siliconflow = SiliconFlowClient(config)
        self._lock = threading.RLock()
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._load_failed_reason: str | None = None
        self._warned_unavailable = False

    def _provider(self) -> str:
        return str(self.config.text_correction_provider or "local").strip().lower()

    @property
    def ready(self) -> bool:
        if self.config.fake_mode:
            return True
        if self._provider() == "siliconflow":
            return self._siliconflow.available()
        with self._lock:
            return self._model is not None and self._tokenizer is not None

    def _dtype(self):
        import torch

        value = str(self.config.model_dtype or "bfloat16").strip().lower()
        if value in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if value in {"fp16", "float16", "half"}:
            return torch.float16
        return torch.float32

    def _load(self) -> None:
        if self.config.fake_mode or self._model is not None:
            return
        model_id = str(self.config.text_correction_model_id or "").strip()
        local_path = _safe_model_path(model_id)
        if local_path is None:
            raise RuntimeError(
                "text correction model is missing locally: "
                f"{model_id}. Download to models dir first."
            )

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:
            raise RuntimeError(
                "transformers is required for text correction model. "
                "Install with: pip install -U transformers"
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(str(local_path), trust_remote_code=True)
        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        kwargs: dict[str, Any] = {
            "torch_dtype": self._dtype(),
            "trust_remote_code": True,
        }
        device = str(self.config.model_device or "cuda:0").strip()
        if not device:
            device = "cuda:0"
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                str(local_path),
                device_map={"": device},
                **kwargs,
            )
        except Exception:
            self._model = AutoModelForCausalLM.from_pretrained(str(local_path), **kwargs)
            self._model.to(device)
        self._model.eval()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def warmup(self) -> None:
        if self.config.fake_mode:
            return
        if self._provider() == "siliconflow":
            # Remote provider has no local weights to preload.
            return
        with self._lock:
            self._load()
            if self._model is None or self._tokenizer is None:
                return
        _ = self.correct_text("warmup test", language_hint="Auto")

    def _build_prompt(self, text: str, language_hint: str) -> str:
        hint = str(language_hint or "Auto").strip() or "Auto"
        return (
            "You are an ASR text correction model for low-latency voice conversion.\n"
            "Task: fix obvious recognition errors, punctuation, and repeated fillers.\n"
            "Rules: keep original meaning, do not paraphrase, do not add new content.\n"
            "Output only the corrected final text.\n"
            f"Language hint: {hint}\n"
            f"Input: {text}\n"
            "Corrected:"
        )

    def _decode_new_tokens(self, outputs, input_len: int) -> str:
        if self._tokenizer is None:
            return ""
        ids = outputs[0][input_len:]
        return str(self._tokenizer.decode(ids, skip_special_tokens=True) or "").strip()

    def _extract_text(self, raw: str) -> str:
        candidate = str(raw or "").strip()
        if not candidate:
            return ""
        lines = [ln.strip() for ln in candidate.splitlines() if ln.strip()]
        if lines:
            candidate = lines[0]
        candidate = candidate.strip().strip('"').strip("'")
        candidate = _QUOTED_PREFIX.sub("", candidate).strip()
        candidate = _MULTI_SPACES.sub(" ", candidate).strip()
        return candidate

    def _run_model(self, text: str, language_hint: str) -> str:
        assert self._model is not None
        assert self._tokenizer is not None
        import torch

        prompt = self._build_prompt(text, language_hint)
        encoded = self._tokenizer(prompt, return_tensors="pt")
        device = getattr(self._model, "device", None)
        if device is not None:
            try:
                encoded = {k: v.to(device) for k, v in encoded.items()}
            except Exception:
                pass

        input_len = int(encoded["input_ids"].shape[-1])
        with torch.inference_mode():
            outputs = self._model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max(16, int(self.config.text_correction_max_new_tokens)),
                pad_token_id=self._tokenizer.eos_token_id,
            )
        decoded = self._decode_new_tokens(outputs, input_len)
        return self._extract_text(decoded)

    def _run_siliconflow_model(self, text: str, language_hint: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "你是语音识别后处理器。请只做最小纠错：修正错字、补充自然标点、"
                    "去除重复口头禅（如嗯、呃、uh、um），不要改写语义，不要扩写。"
                    "只输出最终纠错文本。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"语言提示: {language_hint}\n"
                    f"原文本: {text}\n"
                    "请输出纠错后的文本："
                ),
            },
        ]
        raw = self._siliconflow.chat(
            messages=messages,
            max_tokens=max(32, int(self.config.text_correction_max_new_tokens)),
            temperature=0.1,
            top_p=0.7,
        )
        return self._extract_text(raw)

    def correct_text(self, text: str, language_hint: str) -> TextCorrectionResult:
        src = _normalize_text(text)
        if not src:
            return TextCorrectionResult(text="", changed=False, used_model=False)

        fallback = _strip_fillers(src) or src
        if self.config.fake_mode:
            return TextCorrectionResult(
                text=fallback,
                changed=(fallback != src),
                used_model=False,
            )

        provider = self._provider()
        if provider == "siliconflow":
            if not self._siliconflow.available():
                warning = None
                if not self._warned_unavailable:
                    warning = (
                        "SiliconFlow key missing. Fallback correction is in use."
                    )
                    self._warned_unavailable = True
                    logger.warning("%s", warning)
                return TextCorrectionResult(
                    text=fallback,
                    changed=(fallback != src),
                    used_model=False,
                    warning=warning,
                )
            try:
                corrected = self._run_siliconflow_model(src, language_hint)
                if not corrected:
                    corrected = fallback
                return TextCorrectionResult(
                    text=corrected,
                    changed=(corrected != src),
                    used_model=True,
                )
            except Exception as exc:
                logger.warning("siliconflow correction failed, fallback to rule pass: %s", exc)
                return TextCorrectionResult(
                    text=fallback,
                    changed=(fallback != src),
                    used_model=False,
                    warning="SiliconFlow correction failed once; fallback correction used.",
                )

        with self._lock:
            if self._model is None and self._load_failed_reason is None:
                try:
                    self._load()
                except Exception as exc:
                    self._load_failed_reason = str(exc)

            if self._model is None or self._tokenizer is None:
                warning = None
                if not self._warned_unavailable:
                    warning = (
                        "LLM text correction model unavailable, fallback correction is in use."
                    )
                    self._warned_unavailable = True
                    logger.warning("%s", self._load_failed_reason or warning)
                return TextCorrectionResult(
                    text=fallback,
                    changed=(fallback != src),
                    used_model=False,
                    warning=warning,
                )

            try:
                corrected = self._run_model(src, language_hint)
                if not corrected:
                    corrected = fallback
                return TextCorrectionResult(
                    text=corrected,
                    changed=(corrected != src),
                    used_model=True,
                )
            except Exception as exc:
                logger.warning("text correction inference failed, fallback to rule pass: %s", exc)
                return TextCorrectionResult(
                    text=fallback,
                    changed=(fallback != src),
                    used_model=False,
                    warning="LLM text correction failed once; fallback correction used.",
                )

    def loaded_models(self) -> list[dict[str, Any]]:
        if self.config.fake_mode:
            return []
        if self._provider() == "siliconflow":
            if not self._siliconflow.available():
                return []
            display = str(self.config.siliconflow_model_display_name or "").strip()
            model_name = str(self.config.siliconflow_model or "").strip()
            return [
                {
                    "key": "text_corrector",
                    "label": display or f"OpenAI 兼容 ({model_name})",
                    "replicas": 1,
                    "device": "remote",
                    "unloadable": False,
                }
            ]
        acquired = self._lock.acquire(timeout=0.02)
        try:
            if self._model is None:
                return []
            return [
                {
                    "key": "text_corrector",
                    "label": "ASR Text Corrector",
                    "replicas": 1,
                    "device": str(self.config.model_device),
                    "unloadable": True,
                }
            ]
        finally:
            if acquired:
                self._lock.release()

    def unload_model(self) -> bool:
        if self.config.fake_mode:
            return False
        if self._provider() == "siliconflow":
            return False
        with self._lock:
            changed = self._model is not None or self._tokenizer is not None
            self._model = None
            self._tokenizer = None
            self._load_failed_reason = None
            self._warned_unavailable = False
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
