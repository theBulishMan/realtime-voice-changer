from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class AppConfig:
    data_dir: Path
    db_path: Path
    voices_dir: Path
    logs_dir: Path
    frontend_dir: Path
    models_dir: Path

    fake_mode: bool
    require_gpu: bool
    base_model_id: str
    design_model_id: str
    custom_voice_model_id: str
    custom_voice_enabled: bool
    model_device: str
    model_dtype: str
    tts_gpu_turbo: bool
    tts_force_sdpa: bool
    tts_infer_workers: int
    tts_model_replicas: int
    tts_non_streaming_mode: bool
    tts_max_new_tokens_min: int
    tts_max_new_tokens_cap: int
    tts_tokens_per_char: float
    text_correction_provider: str
    text_correction_model_id: str
    text_correction_enabled_default: bool
    text_correction_max_new_tokens: int
    text_correction_warmup: bool
    siliconflow_api_base: str
    siliconflow_api_key: str
    siliconflow_model: str
    siliconflow_timeout_s: float
    siliconflow_credential_name: str
    siliconflow_model_display_name: str
    siliconflow_context_length: int
    siliconflow_max_tokens: int
    siliconflow_compat_mode: str
    siliconflow_enable_thinking: bool

    asr_small_model: str
    asr_large_model: str
    asr_backend: str
    qwen_asr_model_id: str
    qwen_asr_local_dir: str
    qwen_asr_max_new_tokens: int
    qwen_asr_max_inference_batch_size: int
    asr_device: str
    asr_compute_type: str

    capture_sample_rate: int
    tts_sample_rate: int
    frame_ms: int
    max_segment_ms: int
    vad_silence_ms: int

    monitor_enabled_default: bool
    ptt_enabled_default: bool
    llm_correction_enabled_default: bool
    virtual_mic_name_hint: str
    monitor_device_name_hint: str
    input_device_name_hint: str
    metrics_buffer_size: int
    startup_warmup: bool
    fake_capture_input: bool

    @classmethod
    def from_env(cls) -> "AppConfig":
        root = Path(os.getenv("RVC_DATA_DIR", "data")).resolve()
        frontend = Path(os.getenv("RVC_FRONTEND_DIR", "app/frontend")).resolve()
        models_dir = Path(os.getenv("RVC_MODELS_DIR", ".cache/models")).resolve()
        return cls(
            data_dir=root,
            db_path=root / "app.db",
            voices_dir=root / "voices",
            logs_dir=root / "logs",
            frontend_dir=frontend,
            models_dir=models_dir,
            fake_mode=_as_bool(os.getenv("RVC_FAKE_MODE"), default=False),
            require_gpu=_as_bool(os.getenv("RVC_REQUIRE_GPU"), default=True),
            base_model_id=os.getenv(
                "RVC_BASE_MODEL_ID",
                str((models_dir / "Qwen3-TTS-12Hz-1.7B-Base").resolve()),
            ),
            design_model_id=os.getenv(
                "RVC_DESIGN_MODEL_ID",
                str((models_dir / "Qwen3-TTS-12Hz-1.7B-VoiceDesign").resolve()),
            ),
            custom_voice_model_id=os.getenv(
                "RVC_CUSTOM_VOICE_MODEL_ID",
                str((models_dir / "Qwen3-TTS-12Hz-1.7B-CustomVoice").resolve()),
            ),
            custom_voice_enabled=_as_bool(
                os.getenv("RVC_CUSTOM_VOICE_ENABLED"), default=True
            ),
            model_device=os.getenv("RVC_MODEL_DEVICE", "cuda:0"),
            model_dtype=os.getenv("RVC_MODEL_DTYPE", "bfloat16"),
            tts_gpu_turbo=_as_bool(os.getenv("RVC_TTS_GPU_TURBO"), default=True),
            tts_force_sdpa=_as_bool(os.getenv("RVC_TTS_FORCE_SDPA"), default=True),
            tts_infer_workers=max(1, int(os.getenv("RVC_TTS_INFER_WORKERS", "1"))),
            tts_model_replicas=max(1, int(os.getenv("RVC_TTS_MODEL_REPLICAS", "1"))),
            tts_non_streaming_mode=_as_bool(
                os.getenv("RVC_TTS_NON_STREAMING_MODE"), default=False
            ),
            tts_max_new_tokens_min=int(os.getenv("RVC_TTS_MAX_NEW_TOKENS_MIN", "320")),
            tts_max_new_tokens_cap=int(os.getenv("RVC_TTS_MAX_NEW_TOKENS_CAP", "1024")),
            tts_tokens_per_char=float(os.getenv("RVC_TTS_TOKENS_PER_CHAR", "12")),
            text_correction_provider=os.getenv("RVC_TEXT_CORRECTION_PROVIDER", "local"),
            text_correction_model_id=os.getenv(
                "RVC_TEXT_CORRECTION_MODEL_ID",
                str((models_dir / "Qwen2.5-1.5B-Instruct").resolve()),
            ),
            text_correction_enabled_default=_as_bool(
                os.getenv("RVC_TEXT_CORRECTION_ENABLED_DEFAULT"), default=True
            ),
            text_correction_max_new_tokens=int(
                os.getenv("RVC_TEXT_CORRECTION_MAX_NEW_TOKENS", "80")
            ),
            text_correction_warmup=_as_bool(
                os.getenv("RVC_TEXT_CORRECTION_WARMUP"), default=False
            ),
            siliconflow_api_base=os.getenv(
                "RVC_SILICONFLOW_API_BASE",
                "https://api.siliconflow.cn/v1",
            ),
            siliconflow_api_key=os.getenv(
                "RVC_SILICONFLOW_API_KEY",
                os.getenv("SILICONFLOW_API_KEY", ""),
            ),
            siliconflow_model=os.getenv(
                "RVC_SILICONFLOW_MODEL",
                "zai-org/GLM-4.5-Air",
            ),
            siliconflow_timeout_s=float(
                os.getenv("RVC_SILICONFLOW_TIMEOUT_S", "12")
            ),
            siliconflow_credential_name=os.getenv(
                "RVC_SILICONFLOW_CREDENTIAL_NAME",
                "API KEY 1",
            ),
            siliconflow_model_display_name=os.getenv(
                "RVC_SILICONFLOW_MODEL_DISPLAY_NAME",
                "远端纠错模型",
            ),
            siliconflow_context_length=max(
                2048,
                int(os.getenv("RVC_SILICONFLOW_CONTEXT_LENGTH", "131072")),
            ),
            siliconflow_max_tokens=max(
                64,
                int(os.getenv("RVC_SILICONFLOW_MAX_TOKENS", "4096")),
            ),
            siliconflow_compat_mode=os.getenv(
                "RVC_SILICONFLOW_COMPAT_MODE",
                "strict_openai",
            ),
            siliconflow_enable_thinking=_as_bool(
                os.getenv("RVC_SILICONFLOW_ENABLE_THINKING"),
                default=False,
            ),
            asr_small_model=os.getenv("RVC_ASR_SMALL_MODEL", "small"),
            asr_large_model=os.getenv("RVC_ASR_LARGE_MODEL", "large-v3-turbo"),
            asr_backend=os.getenv("RVC_ASR_BACKEND", "qwen3"),
            qwen_asr_model_id=os.getenv("RVC_QWEN_ASR_MODEL_ID", "Qwen/Qwen3-ASR-0.6B"),
            qwen_asr_local_dir=os.getenv(
                "RVC_QWEN_ASR_LOCAL_DIR",
                str((models_dir / "Qwen3-ASR-0.6B").resolve()),
            ),
            qwen_asr_max_new_tokens=int(os.getenv("RVC_QWEN_ASR_MAX_NEW_TOKENS", "128")),
            qwen_asr_max_inference_batch_size=int(
                os.getenv("RVC_QWEN_ASR_MAX_INFERENCE_BATCH_SIZE", "1")
            ),
            asr_device=os.getenv("RVC_ASR_DEVICE", "cuda"),
            asr_compute_type=os.getenv("RVC_ASR_COMPUTE_TYPE", "float16"),
            capture_sample_rate=int(os.getenv("RVC_CAPTURE_SR", "16000")),
            tts_sample_rate=int(os.getenv("RVC_TTS_SR", "48000")),
            frame_ms=int(os.getenv("RVC_FRAME_MS", "20")),
            max_segment_ms=int(os.getenv("RVC_MAX_SEGMENT_MS", "2800")),
            vad_silence_ms=int(os.getenv("RVC_VAD_SILENCE_MS", "140")),
            monitor_enabled_default=_as_bool(
                os.getenv("RVC_MONITOR_ENABLED_DEFAULT"), default=False
            ),
            ptt_enabled_default=_as_bool(os.getenv("RVC_PTT_ENABLED_DEFAULT"), default=True),
            llm_correction_enabled_default=_as_bool(
                os.getenv("RVC_LLM_CORRECTION_ENABLED_DEFAULT"), default=True
            ),
            virtual_mic_name_hint=os.getenv("RVC_VIRTUAL_MIC_HINT", "CABLE Input"),
            monitor_device_name_hint=os.getenv("RVC_MONITOR_HINT", ""),
            input_device_name_hint=os.getenv("RVC_INPUT_HINT", ""),
            metrics_buffer_size=int(os.getenv("RVC_METRICS_BUFFER_SIZE", "400")),
            startup_warmup=_as_bool(os.getenv("RVC_STARTUP_WARMUP"), default=False),
            fake_capture_input=_as_bool(os.getenv("RVC_FAKE_CAPTURE_INPUT"), default=False),
        )

    def ensure_paths(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
