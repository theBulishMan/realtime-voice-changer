from __future__ import annotations

import contextlib
import copy
import gc
import io
import math
import logging
import os
import re
import shutil
import threading
import time
from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from app.backend.config import AppConfig

if TYPE_CHECKING:
    import torch

logger = logging.getLogger("rvc.tts")


@dataclass
class _FakePromptItem:
    ref_code: Any | None
    ref_spk_embedding: Any | None
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: str | None = None


DEFAULT_CUSTOM_VOICE_SPEAKERS: list[dict[str, str]] = [
    {
        "id": "Vivian",
        "label": "薇薇安 (Vivian)",
        "description": "明亮、略带棱角的年轻女声。",
        "native_language": "Chinese",
    },
    {
        "id": "Serena",
        "label": "塞雷娜 (Serena)",
        "description": "温暖、柔和的年轻女性声音。",
        "native_language": "Chinese",
    },
    {
        "id": "Uncle_Fu",
        "label": "福叔 (Uncle_Fu)",
        "description": "经验丰富的男声，音色低沉醇厚。",
        "native_language": "Chinese",
    },
    {
        "id": "Dylan",
        "label": "迪伦 (Dylan)",
        "description": "年轻的北京男声，音色清晰自然。",
        "native_language": "Chinese (Beijing)",
    },
    {
        "id": "Eric",
        "label": "埃里克 (Eric)",
        "description": "活泼的成都男声，略带沙哑的明亮音色。",
        "native_language": "Chinese (Sichuan)",
    },
    {
        "id": "Ryan",
        "label": "瑞安 (Ryan)",
        "description": "富有活力、节奏感强的男声。",
        "native_language": "English",
    },
    {
        "id": "Aiden",
        "label": "艾登 (Aiden)",
        "description": "阳光的美式男声，中音清晰。",
        "native_language": "English",
    },
    {
        "id": "Ono_Anna",
        "label": "小野安娜 (Ono_Anna)",
        "description": "活泼俏皮的日语女声，音色轻快灵动。",
        "native_language": "Japanese",
    },
    {
        "id": "Sohee",
        "label": "昭熙 (Sohee)",
        "description": "温暖且富感染力的韩语女声。",
        "native_language": "Korean",
    },
]

DEFAULT_CUSTOM_VOICE_LANGUAGES = ["Auto", "Chinese", "English", "Japanese", "Korean"]


class TTSManager:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._base_model: Any | None = None
        self._base_model_pool: list[Any] = []
        self._base_model_pool_locks: list[threading.RLock] = []
        self._base_model_rr_index = 0
        self._design_model: Any | None = None
        self._custom_model: Any | None = None
        self._lock = threading.RLock()
        self._prompt_cache: dict[tuple[str, str], tuple[int, list[Any]]] = {}
        self._prompt_cache_order: list[tuple[str, str]] = []
        self._prompt_cache_max_items = 64
        self._prepare_runtime_env()

    def _prepare_runtime_env(self) -> None:
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "1200")
        os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
        self._ensure_sox_path()
        self._configure_torch_runtime()
        logging.getLogger("transformers.generation.configuration_utils").setLevel(logging.ERROR)
        logging.getLogger("qwen_tts").setLevel(logging.WARNING)

    @contextlib.contextmanager
    def _suppress_third_party_output(self):
        # qwen_tts emits non-critical startup text via print(); keep launcher output clean.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            yield

    def _configure_torch_runtime(self) -> None:
        if not self.config.tts_gpu_turbo:
            return
        if "cuda" not in (self.config.model_device or "").lower():
            return
        try:
            import torch
        except Exception:
            return
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    def _ensure_sox_path(self) -> None:
        if shutil.which("sox"):
            return

        candidates: list[Path] = []
        local_app_data = os.getenv("LOCALAPPDATA", "").strip()
        if local_app_data:
            winget_root = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
            candidates.append(
                winget_root
                / "ChrisBagwell.SoX_Microsoft.Winget.Source_8wekyb3d8bbwe"
                / "sox-14.4.2"
            )
            if winget_root.exists():
                for pkg_dir in winget_root.glob("ChrisBagwell.SoX*"):
                    direct = pkg_dir / "sox-14.4.2"
                    candidates.append(direct)
                    candidates.append(pkg_dir)
        candidates.append(Path.cwd() / "tools" / "sox" / "sox-14.4.2")
        candidates.append(Path.cwd() / "tools" / "sox")

        for candidate in candidates:
            exe_path = candidate / "sox.exe"
            if not exe_path.exists():
                continue
            os.environ["PATH"] = f"{candidate}{os.pathsep}{os.environ.get('PATH', '')}"
            if shutil.which("sox"):
                return

    def _wrap_model_load_error(self, model_id: str, exc: Exception) -> RuntimeError:
        hint = (
            "Model load failed. "
            "Run `python scripts/predownload_models.py --all --provider modelscope --models-dir .cache/models` and retry."
        )
        if model_id == self.config.design_model_id:
            hint = (
                "Model load failed. "
                "Run `python scripts/predownload_models.py --all --provider modelscope --models-dir .cache/models` and retry."
            )

        message = str(exc).lower()
        network_markers = (
            "timed out",
            "connectionpool",
            "cas-bridge.xethub.hf.co",
            "failed to establish a new connection",
            "name resolution",
        )
        if any(marker in message for marker in network_markers):
            return RuntimeError(
                f"Failed to download model `{model_id}` due to network timeout/connection issue. {hint}"
            )
        return RuntimeError(f"Failed to load model `{model_id}`. {hint} Root cause: {exc}")

    def _ensure_sox_available(self) -> None:
        self._ensure_sox_path()
        if shutil.which("sox"):
            return
        raise RuntimeError(
            "SoX executable not found. Voice clone prompt extraction requires SoX. "
            "Run `powershell -ExecutionPolicy Bypass -File scripts/install_sox.ps1` and retry."
        )

    @property
    def model_ready(self) -> bool:
        if self.config.fake_mode:
            return True
        return self._base_model is not None or bool(self._base_model_pool)

    @property
    def custom_voice_model_ready(self) -> bool:
        if self.config.fake_mode:
            return True
        local_path = Path(self.config.custom_voice_model_id)
        return local_path.exists() and local_path.is_dir()

    def _dtype(self) -> "torch.dtype":
        import torch

        value = self.config.model_dtype.lower()
        if value in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if value in {"fp16", "float16", "half"}:
            return torch.float16
        return torch.float32

    def _preferred_attn_implementation(self) -> str | None:
        device = (self.config.model_device or "").lower()
        if "cuda" not in device:
            return None
        if find_spec("flash_attn") is None:
            if self.config.tts_force_sdpa:
                return "sdpa"
            return None
        return "flash_attention_2"

    def _load_model_with_fallback(self, model_id: str) -> Any:
        logging.getLogger("sox").setLevel(logging.ERROR)
        with self._suppress_third_party_output():
                from qwen_tts import Qwen3TTSModel

        kwargs: dict[str, Any] = {
            "device_map": self.config.model_device,
            "dtype": self._dtype(),
        }
        preferred = self._preferred_attn_implementation()
        if preferred:
            kwargs["attn_implementation"] = preferred

        try:
            with self._suppress_third_party_output():
                return Qwen3TTSModel.from_pretrained(model_id, **kwargs)
        except Exception as exc:
            if "attn_implementation" in kwargs:
                failed_impl = str(kwargs.get("attn_implementation"))
                fallback_kwargs = dict(kwargs)
                fallback_kwargs.pop("attn_implementation", None)
                logger.warning(
                    "Attention implementation '%s' unavailable for %s, "
                    "falling back to default attention: %s",
                    failed_impl,
                    model_id,
                    exc,
                )
                try:
                    with self._suppress_third_party_output():
                        return Qwen3TTSModel.from_pretrained(model_id, **fallback_kwargs)
                except Exception as fallback_exc:
                    raise self._wrap_model_load_error(model_id, fallback_exc) from fallback_exc
            raise self._wrap_model_load_error(model_id, exc) from exc

    def _base_model_target_replicas(self) -> int:
        workers = max(1, int(self.config.tts_infer_workers))
        requested = max(1, int(self.config.tts_model_replicas))
        return max(1, min(workers, requested))

    def _is_probable_oom(self, exc: Exception) -> bool:
        text = str(exc).lower()
        return (
            "out of memory" in text
            or "cuda error: out of memory" in text
            or "cublas_status_alloc_failed" in text
            or "cuda" in text
            and "memory" in text
        )

    def _load_base_model(self) -> None:
        if self.config.fake_mode:
            return
        target_replicas = self._base_model_target_replicas()
        if self._base_model is not None and len(self._base_model_pool) >= target_replicas:
            return
        local_path = Path(self.config.base_model_id)
        if not local_path.exists():
            raise RuntimeError(
                "Base TTS local model directory is missing: "
                f"{local_path}. Run: python scripts/predownload_models.py "
                "--all --provider modelscope --models-dir .cache/models"
            )
        if self._base_model is None:
            self._base_model = self._load_model_with_fallback(self.config.base_model_id)
        if not self._base_model_pool:
            self._base_model_pool.append(self._base_model)
            self._base_model_pool_locks.append(threading.RLock())
            logger.info(
                "TTS base model pool initialized (replicas=%d, workers=%d)",
                1,
                int(self.config.tts_infer_workers),
            )
        while len(self._base_model_pool) < target_replicas:
            try:
                replica = self._load_model_with_fallback(self.config.base_model_id)
            except Exception as exc:
                if self._is_probable_oom(exc):
                    logger.warning(
                        "TTS replica scaling stopped at %d/%d due to memory pressure: %s",
                        len(self._base_model_pool),
                        target_replicas,
                        exc,
                    )
                    break
                raise
            self._base_model_pool.append(replica)
            self._base_model_pool_locks.append(threading.RLock())
            logger.info(
                "Loaded TTS base model replica %d/%d",
                len(self._base_model_pool),
                target_replicas,
            )

    def _primary_base_model_slot(self) -> tuple[Any, threading.RLock]:
        with self._lock:
            self._load_base_model()
            if not self._base_model_pool:
                assert self._base_model is not None
                self._base_model_pool = [self._base_model]
                self._base_model_pool_locks = [threading.RLock()]
            return self._base_model_pool[0], self._base_model_pool_locks[0]

    def _pick_base_model_slot(self) -> tuple[Any, threading.RLock]:
        with self._lock:
            self._load_base_model()
            if not self._base_model_pool:
                assert self._base_model is not None
                self._base_model_pool = [self._base_model]
                self._base_model_pool_locks = [threading.RLock()]
            pool_size = len(self._base_model_pool)
            idx = self._base_model_rr_index % pool_size
            self._base_model_rr_index += 1
            return self._base_model_pool[idx], self._base_model_pool_locks[idx]

    def _load_design_model(self) -> None:
        if self._design_model is not None or self.config.fake_mode:
            return
        local_path = Path(self.config.design_model_id)
        if not local_path.exists():
            raise RuntimeError(
                "VoiceDesign local model directory is missing: "
                f"{local_path}. Run: python scripts/predownload_models.py "
                "--all --provider modelscope --models-dir .cache/models"
            )
        self._design_model = self._load_model_with_fallback(self.config.design_model_id)

    def _load_custom_model(self) -> None:
        if self._custom_model is not None or self.config.fake_mode:
            return
        if not self.config.custom_voice_enabled:
            raise RuntimeError("CustomVoice is disabled by configuration (RVC_CUSTOM_VOICE_ENABLED=0).")
        local_path = Path(self.config.custom_voice_model_id)
        if not local_path.exists():
            raise RuntimeError(
                "CustomVoice local model directory is missing: "
                f"{local_path}. Run: python scripts/predownload_models.py "
                "--custom --provider modelscope --models-dir .cache/models"
            )
        self._custom_model = self._load_model_with_fallback(self.config.custom_voice_model_id)

    def warmup(self) -> None:
        if self.config.fake_mode:
            return
        with self._lock:
            self._load_base_model()

    def _fake_wav(self, text: str, sample_rate: int | None = None) -> np.ndarray:
        sr = sample_rate or self.config.tts_sample_rate
        duration = max(0.35, min(3.0, 0.03 * max(len(text), 1)))
        t = np.linspace(0, duration, int(sr * duration), endpoint=False, dtype=np.float32)
        f = 180.0 + (len(text) % 80)
        return (0.18 * np.sin(2.0 * math.pi * f * t)).astype(np.float32)

    def _apply_prompt_ref_text(self, prompt_items: list[Any], ref_text: str | None) -> None:
        if ref_text is None:
            return
        normalized = str(ref_text).strip()
        if not normalized:
            return
        for item in prompt_items:
            if isinstance(item, dict):
                item["ref_text"] = normalized
                continue
            try:
                setattr(item, "ref_text", normalized)
            except Exception:
                continue

    def _validate_prompt_items(self, prompt_items: list[Any]) -> None:
        if not prompt_items:
            raise RuntimeError("Voice prompt is empty. Recreate this voice profile.")
        for idx, item in enumerate(prompt_items):
            if isinstance(item, dict):
                ref_code = item.get("ref_code")
                ref_spk_embedding = item.get("ref_spk_embedding")
            else:
                ref_code = getattr(item, "ref_code", None)
                ref_spk_embedding = getattr(item, "ref_spk_embedding", None)
            if ref_code is None and ref_spk_embedding is None:
                raise RuntimeError(
                    "Voice prompt is invalid for real mode (missing prompt embedding). "
                    f"Prompt item #{idx} is empty. Recreate this voice in real mode."
                )

    def generate_voice_design_preview(
        self, voice_prompt: str, preview_text: str, language: str
    ) -> tuple[np.ndarray, int]:
        if self.config.fake_mode:
            return self._fake_wav(preview_text), self.config.tts_sample_rate
        with self._lock:
            self._load_design_model()
            assert self._design_model is not None
            with self._suppress_third_party_output():
                wavs, sr = self._design_model.generate_voice_design(
                    text=preview_text,
                    instruct=voice_prompt,
                    language=language,
                    non_streaming_mode=True,
                )
        return wavs[0].astype(np.float32), int(sr)

    def get_custom_voice_catalog(self) -> tuple[list[dict[str, str]], list[str]]:
        speakers = [dict(item) for item in DEFAULT_CUSTOM_VOICE_SPEAKERS]
        languages = list(DEFAULT_CUSTOM_VOICE_LANGUAGES)
        return speakers, languages

    def custom_voice_speaker_ids(self) -> set[str]:
        return {item["id"] for item in DEFAULT_CUSTOM_VOICE_SPEAKERS}

    def generate_custom_voice_preview(
        self,
        *,
        text: str,
        language: str,
        speaker: str,
        instruct: str = "",
    ) -> tuple[np.ndarray, int]:
        normalized_text = str(text or "").strip()
        normalized_lang = str(language or "Auto").strip() or "Auto"
        normalized_speaker = str(speaker or "").strip()
        normalized_instruct = str(instruct or "").strip()

        if self.config.fake_mode:
            fake_seed = f"{normalized_speaker}:{normalized_text}:{normalized_instruct}"
            return self._fake_wav(fake_seed), self.config.tts_sample_rate

        if not normalized_speaker:
            raise RuntimeError("CustomVoice speaker is required.")
        if not normalized_text:
            raise RuntimeError("CustomVoice preview text is required.")

        kwargs: dict[str, Any] = {
            "text": normalized_text,
            "language": normalized_lang,
            "speaker": normalized_speaker,
        }
        if normalized_instruct:
            kwargs["instruct"] = normalized_instruct

        with self._lock:
            self._load_custom_model()
            assert self._custom_model is not None
            with self._suppress_third_party_output():
                wavs, sr = self._custom_model.generate_custom_voice(**kwargs)
        return wavs[0].astype(np.float32), int(sr)

    def create_clone_prompt_from_audio(
        self, ref_audio_path: Path, ref_text: str
    ) -> list[Any]:
        if self.config.fake_mode:
            return [
                _FakePromptItem(
                    ref_code=None,
                    ref_spk_embedding=None,
                    x_vector_only_mode=False,
                    icl_mode=True,
                    ref_text=ref_text,
                )
            ]
        self._ensure_sox_available()
        model, model_lock = self._primary_base_model_slot()
        with model_lock:
            with self._suppress_third_party_output():
                prompt_items = model.create_voice_clone_prompt(
                    ref_audio=str(ref_audio_path),
                    ref_text=ref_text,
                    x_vector_only_mode=False,
                )
        self._apply_prompt_ref_text(prompt_items, ref_text)
        return prompt_items

    def create_clone_prompt_from_wave(
        self, wav: np.ndarray, sr: int, ref_text: str
    ) -> list[Any]:
        if self.config.fake_mode:
            return [
                _FakePromptItem(
                    ref_code=None,
                    ref_spk_embedding=None,
                    x_vector_only_mode=False,
                    icl_mode=True,
                    ref_text=ref_text,
                )
            ]
        self._ensure_sox_available()
        model, model_lock = self._primary_base_model_slot()
        with model_lock:
            with self._suppress_third_party_output():
                prompt_items = model.create_voice_clone_prompt(
                    ref_audio=(wav, sr),
                    ref_text=ref_text,
                    x_vector_only_mode=False,
                )
        self._apply_prompt_ref_text(prompt_items, ref_text)
        return prompt_items

    def synthesize_with_prompt(
        self,
        text: str,
        language: str,
        prompt_items: list[Any],
        *,
        ref_text_override: str | None = None,
    ) -> tuple[np.ndarray, int]:
        if self.config.fake_mode:
            return self._fake_wav(text), self.config.tts_sample_rate
        self._apply_prompt_ref_text(prompt_items, ref_text_override)
        self._validate_prompt_items(prompt_items)
        max_new_tokens = self._estimate_max_new_tokens(text)
        model, model_lock = self._pick_base_model_slot()
        with model_lock:
            with self._suppress_third_party_output():
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language=language,
                    voice_clone_prompt=prompt_items,
                    non_streaming_mode=self.config.tts_non_streaming_mode,
                    max_new_tokens=max_new_tokens,
                )
        return wavs[0].astype(np.float32), int(sr)

    def synthesize_with_prompt_path(
        self,
        text: str,
        language: str,
        prompt_path: Path,
        *,
        ref_text_override: str | None = None,
    ) -> tuple[np.ndarray, int]:
        if self.config.fake_mode:
            return self._fake_wav(text), self.config.tts_sample_rate

        prompt_items = self._load_prompt_items_cached(
            prompt_path, ref_text_override=ref_text_override
        )
        return self.synthesize_with_prompt(
            text, language, prompt_items, ref_text_override=ref_text_override
        )

    def preload_prompt_path(self, prompt_path: Path, *, ref_text_override: str | None = None) -> None:
        if self.config.fake_mode:
            return
        self._load_prompt_items_cached(prompt_path, ref_text_override=ref_text_override)

    def save_wav(self, wav: np.ndarray, sr: int, path: Path) -> None:
        import soundfile as sf

        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, wav, sr)

    def create_design_reference_file(
        self, voice_prompt: str, preview_text: str, language: str, output_path: Path
    ) -> tuple[np.ndarray, int]:
        wav, sr = self.generate_voice_design_preview(voice_prompt, preview_text, language)
        self.save_wav(wav, sr, output_path)
        return wav, sr

    def timestamp_token(self) -> str:
        return str(int(time.time() * 1000))

    def _estimate_max_new_tokens(self, text: str) -> int:
        min_tokens = max(64, int(self.config.tts_max_new_tokens_min))
        cap_tokens = max(min_tokens, int(self.config.tts_max_new_tokens_cap))
        per_char = max(1.0, float(self.config.tts_tokens_per_char))

        normalized = re.sub(r"\s+", "", str(text or ""))
        char_count = len(normalized)
        punctuation_bonus = 64 if any(ch in "。！？!?;；,，" for ch in str(text)) else 0
        estimate = int(min_tokens + (char_count * per_char) + punctuation_bonus)
        return int(max(min_tokens, min(cap_tokens, estimate)))

    def _clone_prompt_items(self, prompt_items: list[Any]) -> list[Any]:
        try:
            cloned = copy.deepcopy(prompt_items)
            if isinstance(cloned, list):
                return cloned
        except Exception:
            pass
        return list(prompt_items)

    def _evict_prompt_cache_if_needed(self) -> None:
        while len(self._prompt_cache_order) > self._prompt_cache_max_items:
            oldest = self._prompt_cache_order.pop(0)
            self._prompt_cache.pop(oldest, None)

    def _load_prompt_items_cached(
        self, prompt_path: Path, *, ref_text_override: str | None = None
    ) -> list[Any]:
        from app.backend.services.prompt_store import load_prompt_items

        normalized_path = str(prompt_path.resolve())
        normalized_ref = (ref_text_override or "").strip()
        cache_key = (normalized_path, normalized_ref)
        mtime_ns = int(prompt_path.stat().st_mtime_ns)

        with self._lock:
            cached = self._prompt_cache.get(cache_key)
            if cached is not None and cached[0] == mtime_ns:
                return self._clone_prompt_items(cached[1])

        prompt_items = load_prompt_items(prompt_path, ref_text_override=ref_text_override)
        with self._lock:
            self._prompt_cache[cache_key] = (mtime_ns, self._clone_prompt_items(prompt_items))
            if cache_key in self._prompt_cache_order:
                self._prompt_cache_order.remove(cache_key)
            self._prompt_cache_order.append(cache_key)
            self._evict_prompt_cache_if_needed()
        return self._clone_prompt_items(prompt_items)

    def _cleanup_cuda_cache(self) -> None:
        if "cuda" not in (self.config.model_device or "").lower():
            return
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            return

    def loaded_models(self) -> list[dict[str, Any]]:
        if self.config.fake_mode:
            return []
        acquired = self._lock.acquire(timeout=0.03)
        try:
            return self._loaded_models_snapshot_nolock()
        finally:
            if acquired:
                self._lock.release()

    def _loaded_models_snapshot_nolock(self) -> list[dict[str, Any]]:
        models: list[dict[str, Any]] = []
        if self._base_model is not None or self._base_model_pool:
            replicas = max(
                1,
                len(self._base_model_pool) if self._base_model_pool else 1,
            )
            models.append(
                {
                    "key": "tts_base",
                    "label": "Qwen3-TTS Base",
                    "replicas": int(replicas),
                    "device": str(self.config.model_device),
                    "unloadable": True,
                }
            )
        if self._design_model is not None:
            models.append(
                {
                    "key": "tts_design",
                    "label": "Qwen3-TTS VoiceDesign",
                    "replicas": 1,
                    "device": str(self.config.model_device),
                    "unloadable": True,
                }
            )
        if self._custom_model is not None:
            models.append(
                {
                    "key": "tts_custom",
                    "label": "Qwen3-TTS CustomVoice",
                    "replicas": 1,
                    "device": str(self.config.model_device),
                    "unloadable": True,
                }
            )
        return models

    def unload_model(self, model_key: str) -> bool:
        key = str(model_key or "").strip().lower()
        changed = False
        with self._lock:
            if key == "tts_base":
                changed = (
                    self._base_model is not None
                    or bool(self._base_model_pool)
                    or bool(self._base_model_pool_locks)
                )
                self._base_model = None
                self._base_model_pool = []
                self._base_model_pool_locks = []
                self._base_model_rr_index = 0
            elif key == "tts_design":
                changed = self._design_model is not None
                self._design_model = None
            elif key == "tts_custom":
                changed = self._custom_model is not None
                self._custom_model = None
            else:
                raise ValueError(f"Unsupported model key: {model_key}")
        if changed:
            gc.collect()
            self._cleanup_cuda_cache()
        return changed
