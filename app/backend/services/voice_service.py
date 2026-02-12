from __future__ import annotations

import base64
import json
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from app.backend.config import AppConfig
from app.backend.db import AppDatabase
from app.backend.errors import ClientInputError
from app.backend.types import (
    CustomVoiceAssistRequest,
    CustomVoiceAssistResponse,
    CustomVoiceCatalogResponse,
    CustomVoiceRequest,
    CustomVoiceSpeaker,
    DesignVoiceRequest,
    VoiceDesignAssistRequest,
    VoiceDesignAssistResponse,
    VoiceCreateResponse,
    VoicePreviewResponse,
    VoiceProfileMeta,
    utc_now_iso,
)
from app.backend.services.prompt_store import save_prompt_items
from app.backend.services.siliconflow_client import SiliconFlowClient
from app.backend.services.tts_manager import TTSManager


class VoiceService:
    MAX_CLONE_FILE_SIZE_BYTES = 10 * 1024 * 1024
    MIN_CLONE_DURATION_SEC = 3.0
    MAX_CLONE_DURATION_SEC = 60.0
    MIN_CLONE_SAMPLE_RATE = 24000
    ALLOWED_CLONE_SUFFIX = {".wav", ".mp3", ".m4a"}
    PREVIEW_CACHE_TTL_SEC = 20 * 60
    PREVIEW_CACHE_MAX_ITEMS = 24
    _DEFAULT_PREVIEW_TEXT = (
        "你好，这是实时变声器的统一试听文本。请保持自然语速，吐字清晰，"
        "语气稳定，避免过度夸张。接下来补一段英文：Hello everyone, this is a latency and timbre consistency check."
    )

    def __init__(
        self,
        config: AppConfig,
        db: AppDatabase,
        tts: TTSManager,
        siliconflow_client: SiliconFlowClient | None = None,
    ) -> None:
        self.config = config
        self.db = db
        self.tts = tts
        self.siliconflow = siliconflow_client or SiliconFlowClient(config)
        self._preview_cache_lock = threading.Lock()
        self._preview_cache: dict[str, dict[str, Any]] = {}

    def list_voices(self) -> list[VoiceProfileMeta]:
        rows = self.db.list_voices()
        return [VoiceProfileMeta(**row) for row in rows]

    def get_voice(self, voice_id: str) -> VoiceProfileMeta | None:
        row = self.db.get_voice(voice_id)
        if row is None:
            return None
        return VoiceProfileMeta(**row)

    def get_custom_voice_catalog(self) -> CustomVoiceCatalogResponse:
        speakers, languages = self.tts.get_custom_voice_catalog()
        speaker_rows = [CustomVoiceSpeaker(**item) for item in speakers]
        return CustomVoiceCatalogResponse(speakers=speaker_rows, languages=languages)

    def _assist_prompt_fallback(
        self, req: VoiceDesignAssistRequest
    ) -> VoiceDesignAssistResponse:
        brief = " ".join(str(req.brief or "").split()).strip()
        voice_prompt = (
            "自然、贴近真人的声音；吐字清楚，语速中等，句尾收音自然。"
            f"风格重点：{brief}。避免机械感、金属电音和过度情绪化。"
        )
        return VoiceDesignAssistResponse(
            voice_prompt=voice_prompt[:1800],
            preview_text=self._DEFAULT_PREVIEW_TEXT,
            model="",
            source="fallback",
        )

    def assist_design_prompt(
        self, req: VoiceDesignAssistRequest
    ) -> VoiceDesignAssistResponse:
        if not self.siliconflow.available():
            return self._assist_prompt_fallback(req)
        brief = " ".join(str(req.brief or "").split()).strip()
        language = str(req.language or "Auto").strip() or "Auto"
        messages = [
            {
                "role": "system",
                "content": (
                    "你是 TTS 音色编排助手。根据用户给的简短需求，生成两个字段："
                    "voice_prompt 和 preview_text。"
                    "输出必须是 JSON，不要额外解释。"
                    "voice_prompt 要适合 Qwen3-TTS VoiceDesign，80-220字。"
                    "preview_text 要是可试听的长文本，80-180字，便于测试稳定性与自然度。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"语言提示: {language}\n"
                    f"用户期望: {brief}\n"
                    '请输出 JSON: {"voice_prompt":"...","preview_text":"..."}'
                ),
            },
        ]
        try:
            raw = self.siliconflow.chat(
                messages=messages,
                max_tokens=320,
                temperature=0.4,
                top_p=0.8,
            )
            cleaned = str(raw or "").strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`").strip()
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:].strip()
            parsed = json.loads(cleaned)
            voice_prompt = " ".join(str(parsed.get("voice_prompt", "")).split()).strip()
            preview_text = " ".join(str(parsed.get("preview_text", "")).split()).strip()
            if not voice_prompt or not preview_text:
                return self._assist_prompt_fallback(req)
            return VoiceDesignAssistResponse(
                voice_prompt=voice_prompt[:1800],
                preview_text=preview_text[:1000],
                model=str(self.config.siliconflow_model or ""),
                source="siliconflow",
            )
        except Exception:
            return self._assist_prompt_fallback(req)

    def _assist_custom_voice_fallback(
        self, req: CustomVoiceAssistRequest
    ) -> CustomVoiceAssistResponse:
        brief = " ".join(str(req.brief or "").split()).strip()
        speaker = " ".join(str(req.speaker or "").split()).strip()
        speaker_hint = f"面向说话人 {speaker}，" if speaker else ""
        instruct = (
            f"{speaker_hint}语速中等，咬字清晰，情绪自然稳定，"
            f"重点风格：{brief}。避免机械感、过度夸张和电音感。"
        )
        return CustomVoiceAssistResponse(
            instruct=instruct[:1000],
            preview_text=self._DEFAULT_PREVIEW_TEXT,
            model="",
            source="fallback",
        )

    def assist_custom_voice_prompt(
        self, req: CustomVoiceAssistRequest
    ) -> CustomVoiceAssistResponse:
        if not self.siliconflow.available():
            return self._assist_custom_voice_fallback(req)
        brief = " ".join(str(req.brief or "").split()).strip()
        language = str(req.language or "Auto").strip() or "Auto"
        speaker = " ".join(str(req.speaker or "").split()).strip()
        messages = [
            {
                "role": "system",
                "content": (
                    "你是 TTS 官方预置音色编排助手。"
                    "请根据用户需求输出两个字段：instruct 和 preview_text。"
                    "输出必须是 JSON，不要额外解释。"
                    "instruct 用于模型语气指令，长度 10-120 字；"
                    "preview_text 用于试听，长度 80-180 字。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"语言提示: {language}\n"
                    f"官方说话人: {speaker or 'Auto'}\n"
                    f"用户期望: {brief}\n"
                    '请输出 JSON: {"instruct":"...","preview_text":"..."}'
                ),
            },
        ]
        try:
            raw = self.siliconflow.chat(
                messages=messages,
                max_tokens=260,
                temperature=0.35,
                top_p=0.8,
            )
            cleaned = str(raw or "").strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`").strip()
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:].strip()
            parsed = json.loads(cleaned)
            instruct = " ".join(str(parsed.get("instruct", "")).split()).strip()
            preview_text = " ".join(str(parsed.get("preview_text", "")).split()).strip()
            if not instruct or not preview_text:
                return self._assist_custom_voice_fallback(req)
            return CustomVoiceAssistResponse(
                instruct=instruct[:1000],
                preview_text=preview_text[:1000],
                model=str(self.config.siliconflow_model or ""),
                source="siliconflow",
            )
        except Exception:
            return self._assist_custom_voice_fallback(req)

    # Re-define assist templates below to avoid terse/ambiguous prompts and
    # ensure detailed controllable outputs for voice design/custom voice flows.
    _DEFAULT_PREVIEW_TEXT = (
        "你好，这是一段标准试听文本，用于检查音色一致性、自然度和清晰度。"
        "请保持中速表达、吐字清楚、句尾自然收音，并观察是否存在电音感、齿音过重或机械断句。"
        "Now we switch to a short English segment for robustness check: "
        "hello everyone, this is a latency and timbre consistency check."
    )

    @staticmethod
    def _normalize_text(value: str | None) -> str:
        return " ".join(str(value or "").split()).strip()

    @staticmethod
    def _language_label(language: str) -> str:
        normalized = str(language or "Auto").strip()
        if normalized == "Chinese":
            return "中文"
        if normalized == "English":
            return "英文"
        if normalized == "Japanese":
            return "日文"
        if normalized == "Korean":
            return "韩文"
        return "自动适配"

    def _build_detailed_voice_prompt(self, *, brief: str, language: str) -> str:
        target = brief or "自然、清晰、贴近真人、适合长句稳定输出"
        lang = self._language_label(language)
        return (
            f"目标语言：{lang}。角色设定：{target}。"
            "请采用真实人声质感：中低频有厚度但不过闷，齿音适中，鼻音轻微，气声比例低到中等。"
            "语速中等偏稳，短句干净利落，长句保持连贯，不要一个字一个字往外蹦。"
            "句内停顿要自然，逗号轻停、句号收尾平稳，句尾下落但不拖尾。"
            "情绪曲线为“平静起步-轻微增强-自然回落”，整体克制、可信、可长期聆听。"
            "重点避免：机械感、电音感、金属失真、过度夸张、音高剧烈波动、忽大忽小的音量跳变。"
        )

    def _build_detailed_custom_instruct(
        self, *, brief: str, language: str, speaker: str
    ) -> str:
        target = brief or "自然真实、稳定耐听"
        lang = self._language_label(language)
        speaker_hint = f"说话人基底参考 {speaker}；" if speaker else ""
        return (
            f"{speaker_hint}目标语言 {lang}；风格目标 {target}；"
            "语速中等偏稳，连读顺滑；音高中区间，波动小；"
            "咬字清晰但不生硬，辅音不过冲，气声轻微；"
            "情绪克制、亲和、可信，避免朗诵腔；"
            "逗号轻停、句号自然收尾，长句保持呼吸感和连续性；"
            "禁止机械电音、金属感、夸张情绪、过强齿音和忽快忽慢。"
        )

    def _assist_prompt_fallback(
        self, req: VoiceDesignAssistRequest
    ) -> VoiceDesignAssistResponse:
        brief = self._normalize_text(req.brief)
        language = self._normalize_text(req.language or "Auto") or "Auto"
        voice_prompt = self._build_detailed_voice_prompt(brief=brief, language=language)
        return VoiceDesignAssistResponse(
            voice_prompt=voice_prompt[:1800],
            preview_text=self._DEFAULT_PREVIEW_TEXT,
            model="",
            source="fallback",
        )

    def assist_design_prompt(
        self, req: VoiceDesignAssistRequest
    ) -> VoiceDesignAssistResponse:
        brief = self._normalize_text(req.brief)
        language = self._normalize_text(req.language or "Auto") or "Auto"
        if not self.siliconflow.available():
            return self._assist_prompt_fallback(req)
        messages = [
            {
                "role": "system",
                "content": (
                    "你是资深配音导演和 TTS 提示词工程师。"
                    "请根据用户需求输出 JSON，字段为 voice_prompt 与 preview_text。"
                    "voice_prompt 必须是高细节、可执行的声音规范（120-320字），"
                    "至少覆盖：音色基底、音高范围、语速节奏、停连方式、咬字清晰度、情绪曲线、句尾处理、禁止项。"
                    "preview_text 必须是 120-220 字的试听长文本，包含短句与长句，能测试自然度与稳定性。"
                    "不要输出 markdown，不要输出任何解释，只能输出 JSON。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"目标语言：{language}\n"
                    f"用户需求：{brief}\n"
                    '请输出 JSON: {"voice_prompt":"...","preview_text":"..."}'
                ),
            },
        ]
        try:
            raw = self.siliconflow.chat(
                messages=messages,
                max_tokens=420,
                temperature=0.25,
                top_p=0.85,
            )
            cleaned = str(raw or "").strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`").strip()
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:].strip()
            parsed = json.loads(cleaned)
            voice_prompt = self._normalize_text(parsed.get("voice_prompt"))
            preview_text = self._normalize_text(parsed.get("preview_text"))
            if len(voice_prompt) < 80:
                voice_prompt = self._build_detailed_voice_prompt(
                    brief=brief, language=language
                )
            if len(preview_text) < 30:
                preview_text = self._DEFAULT_PREVIEW_TEXT
            return VoiceDesignAssistResponse(
                voice_prompt=voice_prompt[:1800],
                preview_text=preview_text[:1000],
                model=str(self.config.siliconflow_model or ""),
                source="siliconflow",
            )
        except Exception:
            return self._assist_prompt_fallback(req)

    def _assist_custom_voice_fallback(
        self, req: CustomVoiceAssistRequest
    ) -> CustomVoiceAssistResponse:
        brief = self._normalize_text(req.brief)
        language = self._normalize_text(req.language or "Auto") or "Auto"
        speaker = self._normalize_text(req.speaker)
        instruct = self._build_detailed_custom_instruct(
            brief=brief, language=language, speaker=speaker
        )
        return CustomVoiceAssistResponse(
            instruct=instruct[:1000],
            preview_text=self._DEFAULT_PREVIEW_TEXT,
            model="",
            source="fallback",
        )

    def assist_custom_voice_prompt(
        self, req: CustomVoiceAssistRequest
    ) -> CustomVoiceAssistResponse:
        brief = self._normalize_text(req.brief)
        language = self._normalize_text(req.language or "Auto") or "Auto"
        speaker = self._normalize_text(req.speaker)
        if not self.siliconflow.available():
            return self._assist_custom_voice_fallback(req)
        messages = [
            {
                "role": "system",
                "content": (
                    "你是官方预置音色的语气编排助理。"
                    "根据用户需求输出 JSON，字段为 instruct 与 preview_text。"
                    "instruct 必须详细、可执行（40-220字），"
                    "至少覆盖：语速、音高、力度、停顿、咬字、情绪强度、句尾处理、禁止项。"
                    "preview_text 需要 120-220 字，用于试听稳定性和人声真实感。"
                    "不要输出 markdown，不要输出任何解释，只能输出 JSON。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"目标语言：{language}\n"
                    f"官方说话人：{speaker or 'Auto'}\n"
                    f"用户需求：{brief}\n"
                    '请输出 JSON: {"instruct":"...","preview_text":"..."}'
                ),
            },
        ]
        try:
            raw = self.siliconflow.chat(
                messages=messages,
                max_tokens=360,
                temperature=0.2,
                top_p=0.85,
            )
            cleaned = str(raw or "").strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`").strip()
                if cleaned.lower().startswith("json"):
                    cleaned = cleaned[4:].strip()
            parsed = json.loads(cleaned)
            instruct = self._normalize_text(parsed.get("instruct"))
            preview_text = self._normalize_text(parsed.get("preview_text"))
            if len(instruct) < 36:
                instruct = self._build_detailed_custom_instruct(
                    brief=brief, language=language, speaker=speaker
                )
            if len(preview_text) < 30:
                preview_text = self._DEFAULT_PREVIEW_TEXT
            return CustomVoiceAssistResponse(
                instruct=instruct[:1000],
                preview_text=preview_text[:1000],
                model=str(self.config.siliconflow_model or ""),
                source="siliconflow",
            )
        except Exception:
            return self._assist_custom_voice_fallback(req)

    def _voice_dir(self, voice_id: str) -> Path:
        return self.config.voices_dir / voice_id

    def _write_meta_file(self, voice: VoiceProfileMeta) -> None:
        voice_dir = self._voice_dir(voice.id)
        voice_dir.mkdir(parents=True, exist_ok=True)
        (voice_dir / "meta.json").write_text(
            json.dumps(voice.model_dump(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def _audio_to_b64(self, wav: np.ndarray, sample_rate: int) -> str:
        import io
        import soundfile as sf

        buffer = io.BytesIO()
        sf.write(buffer, wav, sample_rate, format="WAV")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _ensure_unique_name(self, name: str) -> None:
        normalized = name.strip().lower()
        if not normalized:
            raise ClientInputError("Voice name is required.")
        for voice in self.list_voices():
            if voice.name.strip().lower() == normalized:
                raise ClientInputError(f"Voice name already exists: {name}")

    def _inspect_audio(self, path: Path) -> tuple[np.ndarray, int, int, float]:
        """
        Returns: mono_wav, sample_rate, channels, duration_seconds
        """
        wav: np.ndarray
        sr: int
        channels: int

        try:
            import soundfile as sf

            wav_raw, sr = sf.read(str(path), dtype="float32", always_2d=True)
            # soundfile returns shape [frames, channels]
            channels = int(wav_raw.shape[1])
            wav = wav_raw.mean(axis=1).astype(np.float32)
        except Exception:
            try:
                import librosa

                wav_raw, sr_raw = librosa.load(str(path), sr=None, mono=False)
                sr = int(sr_raw)
                if isinstance(wav_raw, np.ndarray) and wav_raw.ndim == 2:
                    channels = int(wav_raw.shape[0])
                    wav = wav_raw.mean(axis=0).astype(np.float32)
                else:
                    channels = 1
                    wav = np.asarray(wav_raw, dtype=np.float32)
            except Exception as exc:
                raise ClientInputError(
                    "Failed to decode reference audio. Please upload a valid WAV/MP3/M4A file."
                ) from exc

        if sr <= 0:
            raise ClientInputError("Invalid sample rate in reference audio.")
        duration = float(len(wav)) / float(sr)
        return wav, int(sr), channels, duration

    def _validate_and_prepare_clone_audio(
        self, voice_dir: Path, filename: str, ref_audio_bytes: bytes
    ) -> tuple[Path, np.ndarray, int]:
        suffix = Path(filename).suffix.lower() or ".wav"
        if suffix not in self.ALLOWED_CLONE_SUFFIX:
            allowed = ", ".join(sorted(self.ALLOWED_CLONE_SUFFIX))
            raise ClientInputError(f"Unsupported audio format: {suffix}. Allowed: {allowed}")
        if len(ref_audio_bytes) > self.MAX_CLONE_FILE_SIZE_BYTES:
            raise ClientInputError(
                f"Audio file too large: {len(ref_audio_bytes)} bytes. Max: {self.MAX_CLONE_FILE_SIZE_BYTES}."
            )
        if len(ref_audio_bytes) == 0:
            raise ClientInputError("Uploaded audio file is empty.")

        raw_path = voice_dir / f"ref{suffix}"
        raw_path.write_bytes(ref_audio_bytes)

        wav, sr, channels, duration = self._inspect_audio(raw_path)
        if duration < self.MIN_CLONE_DURATION_SEC:
            raise ClientInputError(
                f"Reference audio is too short ({duration:.2f}s). Min: {self.MIN_CLONE_DURATION_SEC:.1f}s."
            )
        if duration > self.MAX_CLONE_DURATION_SEC:
            raise ClientInputError(
                f"Reference audio is too long ({duration:.2f}s). Max: {self.MAX_CLONE_DURATION_SEC:.1f}s."
            )
        if sr < self.MIN_CLONE_SAMPLE_RATE:
            raise ClientInputError(
                f"Reference audio sample rate too low ({sr} Hz). Min: {self.MIN_CLONE_SAMPLE_RATE} Hz."
            )
        if channels <= 0:
            raise ClientInputError("Invalid channel count in reference audio.")

        ref_wav_path = voice_dir / "ref.wav"
        self.tts.save_wav(wav, sr, ref_wav_path)
        return ref_wav_path, wav, sr

    def _is_sox_missing_error(self, exc: Exception) -> bool:
        message = str(exc).lower()
        if "sox" in message and ("not found" in message or "could not be found" in message):
            return True
        if "winerror 2" in message and shutil.which("sox") is None:
            return True
        return False

    def _preview_fingerprint_design(self, req: DesignVoiceRequest) -> str:
        prompt = self._normalize_text(req.voice_prompt)
        text = self._normalize_text(req.preview_text)
        lang = self._normalize_text(req.language or "Auto")
        return f"design|{prompt}|{text}|{lang}"

    def _preview_fingerprint_custom(self, req: CustomVoiceRequest) -> str:
        speaker = self._normalize_text(req.speaker)
        instruct = self._normalize_text(req.instruct)
        text = self._normalize_text(req.preview_text)
        lang = self._normalize_text(req.language or "Auto")
        return f"custom|{speaker}|{instruct}|{text}|{lang}"

    def _prune_preview_cache_locked(self, now_ts: float) -> None:
        ttl = float(self.PREVIEW_CACHE_TTL_SEC)
        expired = [
            key
            for key, item in self._preview_cache.items()
            if (now_ts - float(item.get("created_at", 0.0))) > ttl
        ]
        for key in expired:
            self._preview_cache.pop(key, None)

        if len(self._preview_cache) <= self.PREVIEW_CACHE_MAX_ITEMS:
            return
        ordered = sorted(
            self._preview_cache.items(),
            key=lambda kv: float(kv[1].get("created_at", 0.0)),
        )
        overflow = len(self._preview_cache) - int(self.PREVIEW_CACHE_MAX_ITEMS)
        for key, _ in ordered[: max(0, overflow)]:
            self._preview_cache.pop(key, None)

    def _store_preview_cache(
        self,
        *,
        kind: str,
        fingerprint: str,
        ref_wav: np.ndarray,
        ref_sr: int,
        prompt_items: list[Any],
        runtime_preview_wav: np.ndarray,
        runtime_preview_sr: int,
    ) -> str:
        cache_key = uuid.uuid4().hex
        now_ts = time.time()
        entry = {
            "kind": kind,
            "fingerprint": fingerprint,
            "created_at": now_ts,
            "ref_wav": np.asarray(ref_wav, dtype=np.float32).copy(),
            "ref_sr": int(ref_sr),
            "prompt_items": prompt_items,
            "runtime_preview_wav": np.asarray(runtime_preview_wav, dtype=np.float32).copy(),
            "runtime_preview_sr": int(runtime_preview_sr),
        }
        with self._preview_cache_lock:
            self._prune_preview_cache_locked(now_ts)
            self._preview_cache[cache_key] = entry
        return cache_key

    def _take_preview_cache(
        self,
        *,
        cache_key: str | None,
        kind: str,
        fingerprint: str,
    ) -> dict[str, Any] | None:
        key = self._normalize_text(cache_key)
        if not key:
            return None
        now_ts = time.time()
        with self._preview_cache_lock:
            self._prune_preview_cache_locked(now_ts)
            entry = self._preview_cache.get(key)
            if not entry:
                return None
            if entry.get("kind") != kind or entry.get("fingerprint") != fingerprint:
                return None
            self._preview_cache.pop(key, None)
            return entry

    def _build_design_assets(
        self, req: DesignVoiceRequest
    ) -> tuple[np.ndarray, int, list[Any], np.ndarray, int]:
        design_wav, design_sr = self.tts.generate_voice_design_preview(
            req.voice_prompt, req.preview_text, req.language
        )
        prompt_items = self.tts.create_clone_prompt_from_wave(
            design_wav, design_sr, req.preview_text
        )
        runtime_preview_wav, runtime_preview_sr = self.tts.synthesize_with_prompt(
            text=req.preview_text,
            language=req.language,
            prompt_items=prompt_items,
            ref_text_override=req.preview_text,
        )
        return design_wav, int(design_sr), prompt_items, runtime_preview_wav, int(
            runtime_preview_sr
        )

    def _build_custom_assets(
        self, req: CustomVoiceRequest
    ) -> tuple[np.ndarray, int, list[Any], np.ndarray, int]:
        custom_wav, custom_sr = self.tts.generate_custom_voice_preview(
            text=req.preview_text,
            language=req.language,
            speaker=req.speaker,
            instruct=req.instruct,
        )
        prompt_items = self.tts.create_clone_prompt_from_wave(
            custom_wav, custom_sr, req.preview_text
        )
        runtime_preview_wav, runtime_preview_sr = self.tts.synthesize_with_prompt(
            text=req.preview_text,
            language=req.language,
            prompt_items=prompt_items,
            ref_text_override=req.preview_text,
        )
        return custom_wav, int(custom_sr), prompt_items, runtime_preview_wav, int(
            runtime_preview_sr
        )

    def create_design_voice(self, req: DesignVoiceRequest) -> VoiceCreateResponse:
        self._ensure_unique_name(req.name)
        voice_id = str(uuid.uuid4())
        created_at = utc_now_iso()
        voice_dir = self._voice_dir(voice_id)
        ref_path = voice_dir / "ref.wav"
        prompt_path = voice_dir / "prompt.pt"
        fingerprint = self._preview_fingerprint_design(req)
        preview_cache_key: str | None = None
        preview_reused = False

        cached = None
        if req.save:
            cached = self._take_preview_cache(
                cache_key=req.preview_cache_key,
                kind="design",
                fingerprint=fingerprint,
            )

        if cached is not None:
            design_wav = np.asarray(cached["ref_wav"], dtype=np.float32)
            design_sr = int(cached["ref_sr"])
            prompt_items = cached["prompt_items"]
            runtime_preview_wav = np.asarray(cached["runtime_preview_wav"], dtype=np.float32)
            runtime_preview_sr = int(cached["runtime_preview_sr"])
            preview_reused = True
        else:
            design_wav, design_sr, prompt_items, runtime_preview_wav, runtime_preview_sr = (
                self._build_design_assets(req)
            )
            if not req.save:
                preview_cache_key = self._store_preview_cache(
                    kind="design",
                    fingerprint=fingerprint,
                    ref_wav=design_wav,
                    ref_sr=design_sr,
                    prompt_items=prompt_items,
                    runtime_preview_wav=runtime_preview_wav,
                    runtime_preview_sr=runtime_preview_sr,
                )
        meta = VoiceProfileMeta(
            id=voice_id,
            name=req.name,
            mode="design",
            language_hint=req.language,
            ref_audio_path=str(ref_path),
            ref_text=req.preview_text,
            prompt_path=str(prompt_path),
            created_at=created_at,
            updated_at=created_at,
        )
        if req.save:
            voice_dir.mkdir(parents=True, exist_ok=True)
            self.tts.save_wav(design_wav, design_sr, ref_path)
            save_prompt_items(prompt_path, prompt_items)
            self.db.create_voice(meta.model_dump())
            self._write_meta_file(meta)
        else:
            if voice_dir.exists():
                shutil.rmtree(voice_dir, ignore_errors=True)
        return VoiceCreateResponse(
            voice=meta,
            preview_audio_b64=self._audio_to_b64(runtime_preview_wav, runtime_preview_sr),
            sample_rate=runtime_preview_sr,
            preview_cache_key=preview_cache_key,
            preview_reused=preview_reused,
        )

    def create_custom_voice(self, req: CustomVoiceRequest) -> VoiceCreateResponse:
        self._ensure_unique_name(req.name)
        supported_speakers = self.tts.custom_voice_speaker_ids()
        if req.speaker not in supported_speakers:
            raise ClientInputError(
                f"Unsupported custom speaker: {req.speaker}. "
                f"Supported: {', '.join(sorted(supported_speakers))}"
            )

        voice_id = str(uuid.uuid4())
        created_at = utc_now_iso()
        voice_dir = self._voice_dir(voice_id)
        ref_path = voice_dir / "ref.wav"
        prompt_path = voice_dir / "prompt.pt"
        fingerprint = self._preview_fingerprint_custom(req)
        preview_cache_key: str | None = None
        preview_reused = False

        cached = None
        if req.save:
            cached = self._take_preview_cache(
                cache_key=req.preview_cache_key,
                kind="custom",
                fingerprint=fingerprint,
            )

        try:
            if cached is not None:
                custom_wav = np.asarray(cached["ref_wav"], dtype=np.float32)
                custom_sr = int(cached["ref_sr"])
                prompt_items = cached["prompt_items"]
                runtime_preview_wav = np.asarray(
                    cached["runtime_preview_wav"], dtype=np.float32
                )
                runtime_preview_sr = int(cached["runtime_preview_sr"])
                preview_reused = True
            else:
                custom_wav, custom_sr, prompt_items, runtime_preview_wav, runtime_preview_sr = (
                    self._build_custom_assets(req)
                )
                if not req.save:
                    preview_cache_key = self._store_preview_cache(
                        kind="custom",
                        fingerprint=fingerprint,
                        ref_wav=custom_wav,
                        ref_sr=custom_sr,
                        prompt_items=prompt_items,
                        runtime_preview_wav=runtime_preview_wav,
                        runtime_preview_sr=runtime_preview_sr,
                    )
        except RuntimeError as exc:
            if voice_dir.exists():
                shutil.rmtree(voice_dir, ignore_errors=True)
            raise ClientInputError(str(exc)) from exc

        meta = VoiceProfileMeta(
            id=voice_id,
            name=req.name,
            mode="custom",
            language_hint=req.language,
            ref_audio_path=str(ref_path),
            ref_text=req.preview_text,
            prompt_path=str(prompt_path),
            created_at=created_at,
            updated_at=created_at,
        )
        if req.save:
            voice_dir.mkdir(parents=True, exist_ok=True)
            self.tts.save_wav(custom_wav, custom_sr, ref_path)
            save_prompt_items(prompt_path, prompt_items)
            self.db.create_voice(meta.model_dump())
            self._write_meta_file(meta)
        else:
            if voice_dir.exists():
                shutil.rmtree(voice_dir, ignore_errors=True)

        return VoiceCreateResponse(
            voice=meta,
            preview_audio_b64=self._audio_to_b64(runtime_preview_wav, runtime_preview_sr),
            sample_rate=runtime_preview_sr,
            preview_cache_key=preview_cache_key,
            preview_reused=preview_reused,
        )

    def create_clone_voice(
        self,
        name: str,
        language: str,
        ref_text: str,
        ref_audio_bytes: bytes,
        filename: str,
    ) -> VoiceCreateResponse:
        self._ensure_unique_name(name)
        if not ref_text.strip():
            raise ClientInputError("Reference transcript (ref_text) is required.")
        voice_id = str(uuid.uuid4())
        created_at = utc_now_iso()
        voice_dir = self._voice_dir(voice_id)
        voice_dir.mkdir(parents=True, exist_ok=True)

        prompt_path = voice_dir / "prompt.pt"
        try:
            ref_path, wav, sr = self._validate_and_prepare_clone_audio(
                voice_dir=voice_dir,
                filename=filename,
                ref_audio_bytes=ref_audio_bytes,
            )
            prompt_items = self.tts.create_clone_prompt_from_wave(wav, sr, ref_text)
            save_prompt_items(prompt_path, prompt_items)
        except Exception as exc:
            shutil.rmtree(voice_dir, ignore_errors=True)
            if self._is_sox_missing_error(exc):
                raise ClientInputError(
                    "SoX is not available. Run `powershell -ExecutionPolicy Bypass -File scripts/install_sox.ps1`, "
                    "restart the app, then retry voice clone."
                ) from exc
            raise

        preview_wav, preview_sr = self.tts.synthesize_with_prompt(
            text=ref_text,
            language=language,
            prompt_items=prompt_items,
            ref_text_override=ref_text,
        )
        meta = VoiceProfileMeta(
            id=voice_id,
            name=name,
            mode="clone",
            language_hint=language,
            ref_audio_path=str(ref_path),
            ref_text=ref_text,
            prompt_path=str(prompt_path),
            created_at=created_at,
            updated_at=created_at,
        )
        self.db.create_voice(meta.model_dump())
        self._write_meta_file(meta)
        return VoiceCreateResponse(
            voice=meta,
            preview_audio_b64=self._audio_to_b64(preview_wav, preview_sr),
            sample_rate=preview_sr,
        )

    def preview_voice(self, voice_id: str, text: str, language: str) -> VoicePreviewResponse:
        voice = self.get_voice(voice_id)
        if voice is None:
            raise ValueError(f"Voice not found: {voice_id}")
        prompt_path = Path(voice.prompt_path)
        wav, sr = self.tts.synthesize_with_prompt_path(
            text, language, prompt_path, ref_text_override=voice.ref_text
        )
        return VoicePreviewResponse(
            voice_id=voice_id,
            audio_b64=self._audio_to_b64(wav, sr),
            sample_rate=sr,
        )

    def delete_voice(self, voice_id: str) -> bool:
        deleted = self.db.delete_voice(voice_id)
        voice_dir = self._voice_dir(voice_id)
        if voice_dir.exists():
            shutil.rmtree(voice_dir, ignore_errors=True)
        return deleted > 0
