from __future__ import annotations

import base64
import json
import shutil
import uuid
from pathlib import Path

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

    def create_design_voice(self, req: DesignVoiceRequest) -> VoiceCreateResponse:
        self._ensure_unique_name(req.name)
        voice_id = str(uuid.uuid4())
        created_at = utc_now_iso()
        voice_dir = self._voice_dir(voice_id)
        ref_path = voice_dir / "ref.wav"
        prompt_path = voice_dir / "prompt.pt"

        design_wav, design_sr = self.tts.create_design_reference_file(
            req.voice_prompt, req.preview_text, req.language, ref_path
        )
        prompt_items = self.tts.create_clone_prompt_from_wave(design_wav, design_sr, req.preview_text)
        runtime_preview_wav, runtime_preview_sr = self.tts.synthesize_with_prompt(
            text=req.preview_text,
            language=req.language,
            prompt_items=prompt_items,
            ref_text_override=req.preview_text,
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

        try:
            custom_wav, custom_sr = self.tts.generate_custom_voice_preview(
                text=req.preview_text,
                language=req.language,
                speaker=req.speaker,
                instruct=req.instruct,
            )
            self.tts.save_wav(custom_wav, custom_sr, ref_path)
            prompt_items = self.tts.create_clone_prompt_from_wave(custom_wav, custom_sr, req.preview_text)
        except RuntimeError as exc:
            if voice_dir.exists():
                shutil.rmtree(voice_dir, ignore_errors=True)
            raise ClientInputError(str(exc)) from exc

        runtime_preview_wav, runtime_preview_sr = self.tts.synthesize_with_prompt(
            text=req.preview_text,
            language=req.language,
            prompt_items=prompt_items,
            ref_text_override=req.preview_text,
        )

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
