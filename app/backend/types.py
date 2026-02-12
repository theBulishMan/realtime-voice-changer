from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: Literal["ok", "degraded"]
    fake_mode: bool
    model_ready: bool
    asr_ready: bool
    gpu_required: bool
    gpu_available: bool
    gpu_device: str | None = None
    asr_backend: str
    model_device: str
    asr_device: str
    custom_voice_enabled: bool = False
    custom_voice_model_ready: bool = False
    detail: str


class LoadedModelItem(BaseModel):
    key: str
    label: str
    replicas: int = 1
    device: str = ""
    unloadable: bool = True


class LoadedModelsResponse(BaseModel):
    gpu_available: bool
    gpu_device: str | None = None
    cuda_memory_allocated_mb: float = 0.0
    cuda_memory_reserved_mb: float = 0.0
    models: list[LoadedModelItem]


class AudioDeviceInfo(BaseModel):
    id: int
    name: str
    max_input_channels: int
    max_output_channels: int
    default_samplerate: float
    is_default_input: bool = False
    is_default_output: bool = False


class AudioDevicesResponse(BaseModel):
    devices: list[AudioDeviceInfo]
    default_input_id: int | None
    default_output_id: int | None
    virtual_mic_output_id: int | None
    virtual_mic_hint: str


class VoiceProfileMeta(BaseModel):
    id: str
    name: str
    mode: Literal["design", "clone", "custom"]
    language_hint: str = "Auto"
    ref_audio_path: str
    ref_text: str | None = None
    prompt_path: str
    created_at: str
    updated_at: str


class DesignVoiceRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    voice_prompt: str = Field(min_length=1, max_length=2048)
    preview_text: str = Field(min_length=1, max_length=1024)
    language: str = "Auto"
    save: bool = True


class VoiceDesignAssistRequest(BaseModel):
    brief: str = Field(min_length=1, max_length=512)
    language: str = "Auto"


class VoiceDesignAssistResponse(BaseModel):
    voice_prompt: str
    preview_text: str
    model: str = ""
    source: Literal["siliconflow", "fallback"] = "fallback"


class CustomVoiceAssistRequest(BaseModel):
    brief: str = Field(min_length=1, max_length=512)
    language: str = "Auto"
    speaker: str | None = Field(default=None, max_length=64)


class CustomVoiceAssistResponse(BaseModel):
    instruct: str
    preview_text: str
    model: str = ""
    source: Literal["siliconflow", "fallback"] = "fallback"


class SiliconFlowSettingsResponse(BaseModel):
    text_correction_provider: Literal["local", "siliconflow"] = "local"
    siliconflow_api_base: str = "https://api.siliconflow.cn/v1"
    credential_name: str = "API KEY 1"
    model_display_name: str = "远端纠错模型"
    endpoint_model_name: str = "zai-org/GLM-4.5-Air"
    siliconflow_model: str = "zai-org/GLM-4.5-Air"
    context_length: int = 131072
    max_tokens: int = 4096
    compat_mode: Literal["strict_openai", "siliconflow"] = "strict_openai"
    thinking_enabled: bool = False
    siliconflow_timeout_s: float = 12.0
    api_key_present: bool = False
    api_key_masked: str = ""


class SiliconFlowSettingsUpdateRequest(BaseModel):
    text_correction_provider: Literal["local", "siliconflow"] | None = None
    siliconflow_api_base: str | None = Field(default=None, max_length=256)
    credential_name: str | None = Field(default=None, max_length=128)
    model_display_name: str | None = Field(default=None, max_length=128)
    endpoint_model_name: str | None = Field(default=None, max_length=256)
    siliconflow_model: str | None = Field(default=None, max_length=256)
    context_length: int | None = Field(default=None, ge=2048, le=262144)
    max_tokens: int | None = Field(default=None, ge=64, le=131072)
    compat_mode: Literal["strict_openai", "siliconflow"] | None = None
    thinking_enabled: bool | None = None
    siliconflow_timeout_s: float | None = Field(default=None, ge=2.0, le=120.0)
    siliconflow_api_key: str | None = Field(default=None, max_length=512)
    clear_api_key: bool = False


class CustomVoiceRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    speaker: str = Field(min_length=1, max_length=64)
    preview_text: str = Field(min_length=1, max_length=1024)
    language: str = "Auto"
    instruct: str = Field(default="", max_length=1024)
    save: bool = True


class VoiceCreateResponse(BaseModel):
    voice: VoiceProfileMeta
    preview_audio_b64: str
    sample_rate: int


class CustomVoiceSpeaker(BaseModel):
    id: str
    label: str
    description: str = ""
    native_language: str = ""


class CustomVoiceCatalogResponse(BaseModel):
    speakers: list[CustomVoiceSpeaker]
    languages: list[str]


class VoicePreviewRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)
    language: str = "Auto"


class VoicePreviewResponse(BaseModel):
    voice_id: str
    audio_b64: str
    sample_rate: int


class RealtimeSettings(BaseModel):
    input_device_id: int | None = None
    monitor_device_id: int | None = None
    virtual_mic_device_id: int | None = None
    monitor_enabled: bool = False
    ptt_enabled: bool = True
    llm_correction_enabled: bool = True
    vad_silence_ms: int = Field(default=140, ge=120, le=1000)
    max_segment_ms: int = Field(default=2800, ge=1000, le=30000)
    input_gain_db: float = Field(default=0.0, ge=-24.0, le=24.0)
    output_gain_db: float = Field(default=0.0, ge=-24.0, le=24.0)
    denoise_enabled: bool = False
    denoise_strength: float = Field(default=0.35, ge=0.0, le=1.0)
    ambience_enabled: bool = True
    ambience_level: float = Field(default=0.003, ge=0.0, le=0.06)


class RealtimeStartRequest(BaseModel):
    voice_id: str
    language: str = "Auto"


class RealtimePttRequest(BaseModel):
    active: bool


class RealtimeSimulateRequest(BaseModel):
    count: int = Field(default=1, ge=1, le=200)
    duration_ms: int = Field(default=500, ge=120, le=10000)
    amplitude: float = Field(default=0.08, gt=0.0, le=1.0)
    frequency_hz: float = Field(default=220.0, ge=20.0, le=5000.0)


class RealtimeInjectAudioRequest(BaseModel):
    audio_b64: str = Field(min_length=1)
    inject_mode: Literal["queue", "segment"] = "queue"
    realtime_pacing: bool = True
    repeat: int = Field(default=1, ge=1, le=50)
    pause_ms: int = Field(default=0, ge=0, le=5000)
    append_silence_ms: int = Field(default=260, ge=0, le=5000)


class RealtimeStateResponse(BaseModel):
    running: bool
    voice_id: str | None = None
    language: str = "Auto"
    ptt_active: bool = False
    settings: RealtimeSettings


class RealtimeMetrics(BaseModel):
    p50_fad_ms: float = 0.0
    p95_fad_ms: float = 0.0
    p50_e2e_ms: float = 0.0
    p95_e2e_ms: float = 0.0
    asr_ms: float = 0.0
    tts_ms: float = 0.0
    drop_rate: float = 0.0
    sample_count: int = 0
    updated_at: str = Field(default_factory=utc_now_iso)


class WsEvent(BaseModel):
    type: str
    ts: str = Field(default_factory=utc_now_iso)
    data: dict[str, Any]
