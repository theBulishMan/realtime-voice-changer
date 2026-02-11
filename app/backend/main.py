from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.backend.asr.asr_stage1 import Stage1ASR
from app.backend.asr.asr_stage2 import Stage2ASR
from app.backend.audio.audio_devices import list_audio_devices
from app.backend.config import AppConfig
from app.backend.db import AppDatabase
from app.backend.errors import ClientInputError
from app.backend.realtime.realtime_engine import RealtimeEngine
from app.backend.services.siliconflow_client import SiliconFlowClient
from app.backend.services.text_corrector import TextCorrector
from app.backend.services.tts_manager import TTSManager
from app.backend.services.voice_service import VoiceService
from app.backend.types import (
    CustomVoiceRequest,
    DesignVoiceRequest,
    HealthResponse,
    LoadedModelItem,
    LoadedModelsResponse,
    RealtimeInjectAudioRequest,
    RealtimePttRequest,
    RealtimeSimulateRequest,
    RealtimeSettings,
    RealtimeStartRequest,
    RealtimeStateResponse,
    SiliconFlowSettingsResponse,
    SiliconFlowSettingsUpdateRequest,
    VoiceDesignAssistRequest,
    VoicePreviewRequest,
)
from app.backend.ws_manager import WebSocketManager

logger = logging.getLogger("rvc")
logging.basicConfig(level=logging.INFO)

config = AppConfig.from_env()
config.ensure_paths()
db = AppDatabase(config.db_path)
tts_manager = TTSManager(config)
asr_stage1 = Stage1ASR(config)
asr_stage2 = Stage2ASR(config)
text_corrector = TextCorrector(config)
siliconflow_client = SiliconFlowClient(config)
ws_manager = WebSocketManager()
voice_service = VoiceService(config, db, tts_manager, siliconflow_client=siliconflow_client)

_loop: asyncio.AbstractEventLoop | None = None
_engine: RealtimeEngine | None = None
_SILICONFLOW_SETTINGS_KEY = "siliconflow_settings_v1"


@dataclass(slots=True)
class _GpuRuntimeStatus:
    available: bool
    device_count: int
    device_name: str | None
    error: str | None = None


def _detect_gpu_runtime() -> _GpuRuntimeStatus:
    try:
        import torch
    except Exception as exc:
        return _GpuRuntimeStatus(
            available=False,
            device_count=0,
            device_name=None,
            error=f"torch import failed: {exc}",
        )

    try:
        available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count()) if available else 0
        name = str(torch.cuda.get_device_name(0)) if available and count > 0 else None
        return _GpuRuntimeStatus(available=available, device_count=count, device_name=name)
    except Exception as exc:
        return _GpuRuntimeStatus(
            available=False,
            device_count=0,
            device_name=None,
            error=f"cuda query failed: {exc}",
        )


def _enforce_runtime_constraints() -> None:
    if config.fake_mode or not config.require_gpu:
        return

    if not _gpu_status.available:
        detail = _gpu_status.error or "torch.cuda.is_available() is false"
        raise RuntimeError(f"GPU is required in real mode, but CUDA is unavailable ({detail}).")

    if "cuda" not in config.model_device.lower():
        raise RuntimeError(
            f"GPU is required in real mode, but RVC_MODEL_DEVICE={config.model_device!r} is not CUDA."
        )

    if "cuda" not in config.asr_device.lower():
        raise RuntimeError(
            f"GPU is required in real mode, but RVC_ASR_DEVICE={config.asr_device!r} is not CUDA."
        )


_gpu_status = _detect_gpu_runtime()


def _event_sink(event) -> None:
    if _loop is None:
        return
    _loop.call_soon_threadsafe(asyncio.create_task, ws_manager.broadcast_json(event.model_dump()))


def _default_settings() -> RealtimeSettings:
    try:
        devices = list_audio_devices(config.virtual_mic_name_hint)
        monitor_id = devices.default_output_id
        input_id = devices.default_input_id
        virtual_id = devices.virtual_mic_output_id
    except Exception:
        monitor_id = None
        input_id = None
        virtual_id = None
    return RealtimeSettings(
        input_device_id=input_id,
        monitor_device_id=monitor_id,
        virtual_mic_device_id=virtual_id,
        monitor_enabled=config.monitor_enabled_default,
        ptt_enabled=config.ptt_enabled_default,
        llm_correction_enabled=config.llm_correction_enabled_default,
        vad_silence_ms=config.vad_silence_ms,
        max_segment_ms=config.max_segment_ms,
    )


def _model_dir_ready(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    if not (path / "config.json").exists():
        return False
    return (
        (path / "model.safetensors").exists()
        or (path / "model.safetensors.index.json").exists()
        or (path / "pytorch_model.bin").exists()
    )


def _get_engine() -> RealtimeEngine:
    if _engine is None:
        raise RuntimeError("Realtime engine is not initialized")
    return _engine


def _cuda_device_index() -> int:
    raw = str(config.model_device or "").strip().lower()
    if raw.startswith("cuda:"):
        suffix = raw.split(":", 1)[1].strip()
        try:
            idx = int(suffix)
            if idx >= 0:
                return idx
        except Exception:
            return 0
    return 0


def _cuda_memory_snapshot_mb() -> tuple[float, float]:
    if not _gpu_status.available:
        return 0.0, 0.0
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0, 0.0
        idx = _cuda_device_index()
        allocated = float(torch.cuda.memory_allocated(idx) / (1024.0 * 1024.0))
        reserved = float(torch.cuda.memory_reserved(idx) / (1024.0 * 1024.0))
        return round(allocated, 1), round(reserved, 1)
    except Exception:
        return 0.0, 0.0


def _loaded_models_snapshot() -> LoadedModelsResponse:
    models: list[LoadedModelItem] = []
    try:
        for item in tts_manager.loaded_models():
            models.append(LoadedModelItem(**item))
    except Exception as exc:
        logger.warning("failed to collect TTS loaded model state: %s", exc)
    try:
        for item in asr_stage1.loaded_models():
            models.append(LoadedModelItem(**item))
    except Exception as exc:
        logger.warning("failed to collect ASR stage1 loaded model state: %s", exc)
    try:
        for item in asr_stage2.loaded_models():
            models.append(LoadedModelItem(**item))
    except Exception as exc:
        logger.warning("failed to collect ASR stage2 loaded model state: %s", exc)
    try:
        for item in text_corrector.loaded_models():
            models.append(LoadedModelItem(**item))
    except Exception as exc:
        logger.warning("failed to collect text corrector model state: %s", exc)

    allocated_mb, reserved_mb = _cuda_memory_snapshot_mb()
    return LoadedModelsResponse(
        gpu_available=_gpu_status.available,
        gpu_device=_gpu_status.device_name,
        cuda_memory_allocated_mb=allocated_mb,
        cuda_memory_reserved_mb=reserved_mb,
        models=models,
    )


def _normalize_provider(value: str | None) -> Literal["local", "siliconflow"]:
    return "siliconflow" if str(value or "").strip().lower() == "siliconflow" else "local"


def _normalize_compat_mode(value: str | None) -> Literal["strict_openai", "siliconflow"]:
    return "siliconflow" if str(value or "").strip().lower() == "siliconflow" else "strict_openai"


def _clamp_int(value: int | None, default: int, minimum: int, maximum: int) -> int:
    try:
        raw = int(value) if value is not None else int(default)
    except Exception:
        raw = int(default)
    return max(minimum, min(maximum, raw))


def _mask_secret(raw: str | None) -> str:
    secret = str(raw or "").strip()
    if not secret:
        return ""
    if len(secret) <= 8:
        return "*" * len(secret)
    return f"{secret[:4]}{'*' * (len(secret) - 8)}{secret[-4:]}"


def _build_siliconflow_settings_response() -> SiliconFlowSettingsResponse:
    key = str(config.siliconflow_api_key or "").strip()
    base = str(config.siliconflow_api_base or "").strip() or "https://api.siliconflow.cn/v1"
    model = str(config.siliconflow_model or "").strip() or "zai-org/GLM-4.5-Air"
    credential_name = str(config.siliconflow_credential_name or "").strip() or "API KEY 1"
    model_display_name = str(config.siliconflow_model_display_name or "").strip() or "远端纠错模型"
    context_length = _clamp_int(config.siliconflow_context_length, 131072, 2048, 262144)
    max_tokens = _clamp_int(config.siliconflow_max_tokens, 4096, 64, 131072)
    compat_mode = _normalize_compat_mode(config.siliconflow_compat_mode)
    thinking_enabled = bool(config.siliconflow_enable_thinking)
    timeout_s = max(2.0, float(config.siliconflow_timeout_s or 12.0))
    return SiliconFlowSettingsResponse(
        text_correction_provider=_normalize_provider(config.text_correction_provider),
        siliconflow_api_base=base,
        credential_name=credential_name,
        model_display_name=model_display_name,
        endpoint_model_name=model,
        siliconflow_model=model,
        context_length=context_length,
        max_tokens=max_tokens,
        compat_mode=compat_mode,
        thinking_enabled=thinking_enabled,
        siliconflow_timeout_s=timeout_s,
        api_key_present=bool(key),
        api_key_masked=_mask_secret(key),
    )


def _apply_siliconflow_settings_from_db() -> None:
    raw = db.get_setting(_SILICONFLOW_SETTINGS_KEY)
    if not isinstance(raw, dict):
        return
    config.text_correction_provider = _normalize_provider(raw.get("text_correction_provider"))
    config.siliconflow_compat_mode = _normalize_compat_mode(raw.get("compat_mode"))
    base = str(raw.get("siliconflow_api_base") or "").strip()
    model = str(raw.get("endpoint_model_name") or raw.get("siliconflow_model") or "").strip()
    credential_name = str(raw.get("credential_name") or "").strip()
    model_display_name = str(raw.get("model_display_name") or "").strip()
    context_length = raw.get("context_length")
    max_tokens = raw.get("max_tokens")
    thinking_enabled = raw.get("thinking_enabled")
    timeout = raw.get("siliconflow_timeout_s")
    api_key = raw.get("siliconflow_api_key")

    if base:
        config.siliconflow_api_base = base
    if model:
        config.siliconflow_model = model
    if credential_name:
        config.siliconflow_credential_name = credential_name
    if model_display_name:
        config.siliconflow_model_display_name = model_display_name
    if context_length is not None:
        config.siliconflow_context_length = _clamp_int(context_length, 131072, 2048, 262144)
    if max_tokens is not None:
        config.siliconflow_max_tokens = _clamp_int(max_tokens, 4096, 64, 131072)
    if thinking_enabled is not None:
        config.siliconflow_enable_thinking = bool(thinking_enabled)
    if timeout is not None:
        try:
            config.siliconflow_timeout_s = max(2.0, float(timeout))
        except Exception:
            pass
    if isinstance(api_key, str):
        config.siliconflow_api_key = api_key.strip()


def _persist_siliconflow_settings() -> None:
    db.upsert_setting(
        _SILICONFLOW_SETTINGS_KEY,
        {
            "text_correction_provider": _normalize_provider(config.text_correction_provider),
            "compat_mode": _normalize_compat_mode(config.siliconflow_compat_mode),
            "siliconflow_api_base": str(config.siliconflow_api_base or "").strip()
            or "https://api.siliconflow.cn/v1",
            "credential_name": str(config.siliconflow_credential_name or "").strip() or "API KEY 1",
            "model_display_name": str(config.siliconflow_model_display_name or "").strip()
            or "远端纠错模型",
            "endpoint_model_name": str(config.siliconflow_model or "").strip()
            or "zai-org/GLM-4.5-Air",
            "siliconflow_model": str(config.siliconflow_model or "").strip() or "zai-org/GLM-4.5-Air",
            "context_length": _clamp_int(config.siliconflow_context_length, 131072, 2048, 262144),
            "max_tokens": _clamp_int(config.siliconflow_max_tokens, 4096, 64, 131072),
            "thinking_enabled": bool(config.siliconflow_enable_thinking),
            "siliconflow_timeout_s": max(2.0, float(config.siliconflow_timeout_s or 12.0)),
            "siliconflow_api_key": str(config.siliconflow_api_key or "").strip(),
        },
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _loop
    global _engine
    _enforce_runtime_constraints()
    db.init_schema()
    _apply_siliconflow_settings_from_db()
    _loop = asyncio.get_running_loop()

    raw_settings = db.get_setting("realtime_settings")
    if raw_settings is None:
        settings = _default_settings()
        db.upsert_setting("realtime_settings", settings.model_dump())
    else:
        settings = RealtimeSettings(**raw_settings)
        # Auto-migrate legacy segmentation defaults to responsive defaults.
        migrated: dict[str, int | bool | None] = {}
        if "ptt_enabled" not in raw_settings:
            migrated["ptt_enabled"] = bool(config.ptt_enabled_default)
        if "llm_correction_enabled" not in raw_settings:
            migrated["llm_correction_enabled"] = bool(config.llm_correction_enabled_default)
        if (
            int(settings.max_segment_ms) == 12000
            and int(settings.vad_silence_ms) == 240
        ):
            migrated["max_segment_ms"] = int(config.max_segment_ms)
            migrated["vad_silence_ms"] = int(config.vad_silence_ms)
        # Migrate old responsive defaults to newer faster profile.
        if (
            int(settings.max_segment_ms) == 4000
            and int(settings.vad_silence_ms) == 180
        ):
            migrated["max_segment_ms"] = int(config.max_segment_ms)
            migrated["vad_silence_ms"] = int(config.vad_silence_ms)
        # Lower historical ambience default to avoid overly strong noise texture.
        if float(settings.ambience_level) == 0.008:
            migrated["ambience_level"] = 0.003
        try:
            devices = list_audio_devices(config.virtual_mic_name_hint)
            preferred_virtual = devices.virtual_mic_output_id
            if preferred_virtual is not None and settings.virtual_mic_device_id != preferred_virtual:
                migrated["virtual_mic_device_id"] = int(preferred_virtual)
        except Exception:
            pass
        if migrated:
            settings = settings.model_copy(update=migrated)
            db.upsert_setting("realtime_settings", settings.model_dump())

    _engine = RealtimeEngine(
        config=config,
        db=db,
        voice_service=voice_service,
        tts_manager=tts_manager,
        asr_stage1=asr_stage1,
        asr_stage2=asr_stage2,
        text_corrector=text_corrector,
        settings=settings,
        event_sink=_event_sink,
    )
    if config.startup_warmup:
        warmup_steps = [
            ("tts", tts_manager.warmup),
            ("asr_stage1", asr_stage1.warmup),
            ("asr_stage2", asr_stage2.warmup),
        ]
        if config.text_correction_warmup:
            warmup_steps.append(("text_corrector", text_corrector.warmup))
        for name, fn in warmup_steps:
            try:
                fn()
                logger.info("warmup ok: %s", name)
            except Exception as exc:
                logger.warning("warmup failed: %s (%s)", name, exc)
                if not config.fake_mode:
                    raise
    yield
    try:
        _get_engine().stop()
    except Exception:
        pass
    db.close()


app = FastAPI(title="Realtime Voice Changer", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(config.frontend_dir)), name="static")


@app.get("/", include_in_schema=False)
async def root_index():
    return FileResponse(config.frontend_dir / "index.html")


@app.get("/voices", include_in_schema=False)
async def voices_index():
    return FileResponse(config.frontend_dir / "voices.html")


@app.get("/settings", include_in_schema=False)
async def settings_index():
    return FileResponse(config.frontend_dir / "settings.html")


@app.get("/api/v1/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    model_ready = tts_manager.model_ready
    asr_ready = asr_stage1.ready and asr_stage2.ready
    gpu_required = (not config.fake_mode) and config.require_gpu
    gpu_available = _gpu_status.available
    gpu_device = _gpu_status.device_name
    custom_voice_model_ready = tts_manager.custom_voice_model_ready
    missing_local_models: list[str] = []
    if not config.fake_mode:
        base_model_dir = Path(config.base_model_id)
        if not _model_dir_ready(base_model_dir):
            missing_local_models.append(f"tts_base:{base_model_dir}")
        design_model_dir = Path(config.design_model_id)
        if not _model_dir_ready(design_model_dir):
            missing_local_models.append(f"tts_design:{design_model_dir}")
        if config.custom_voice_enabled:
            custom_model_dir = Path(config.custom_voice_model_id)
            if not _model_dir_ready(custom_model_dir):
                missing_local_models.append(f"tts_custom:{custom_model_dir}")
        if config.asr_backend.strip().lower() == "qwen3":
            asr_local = (config.qwen_asr_local_dir or "").strip()
            if asr_local:
                asr_model_dir = Path(asr_local)
                if not _model_dir_ready(asr_model_dir):
                    missing_local_models.append(f"qwen_asr:{asr_model_dir}")

    status: Literal["ok", "degraded"]
    detail: str
    if gpu_required and not gpu_available:
        status = "degraded"
        detail = "gpu required but not available"
    elif missing_local_models:
        status = "degraded"
        detail = "missing local models: " + "; ".join(missing_local_models)
    elif model_ready and asr_ready:
        status = "ok"
        detail = "all subsystems are ready"
    else:
        status = "ok"
        detail = "runtime is ready; some subsystems are lazy-loaded and will warm up on first use"
    return HealthResponse(
        status=status,
        fake_mode=config.fake_mode,
        model_ready=model_ready,
        asr_ready=asr_ready,
        gpu_required=gpu_required,
        gpu_available=gpu_available,
        gpu_device=gpu_device,
        asr_backend=config.asr_backend,
        model_device=config.model_device,
        asr_device=config.asr_device,
        custom_voice_enabled=config.custom_voice_enabled,
        custom_voice_model_ready=custom_voice_model_ready,
        detail=detail,
    )


@app.get("/api/v1/models/loaded", response_model=LoadedModelsResponse)
async def loaded_models() -> LoadedModelsResponse:
    return _loaded_models_snapshot()


@app.delete("/api/v1/models/loaded/{model_key}", response_model=LoadedModelsResponse)
async def unload_model(model_key: str) -> LoadedModelsResponse:
    engine = _get_engine()
    if engine.state().running:
        raise HTTPException(status_code=409, detail="请先停止实时链路，再卸载模型。")

    key = str(model_key or "").strip().lower()
    if key in {"tts_base", "tts_design", "tts_custom"}:
        try:
            tts_manager.unload_model(key)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    elif key == "asr_stage1":
        asr_stage1.unload_model()
    elif key == "asr_stage2":
        asr_stage2.unload_model()
    elif key == "text_corrector":
        text_corrector.unload_model()
    else:
        raise HTTPException(status_code=404, detail=f"Unsupported model key: {model_key}")

    return _loaded_models_snapshot()


@app.get("/api/v1/audio/devices")
async def audio_devices():
    try:
        return list_audio_devices(config.virtual_mic_name_hint)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to query audio devices: {exc}") from exc


@app.get("/api/v1/settings/siliconflow", response_model=SiliconFlowSettingsResponse)
async def get_siliconflow_settings() -> SiliconFlowSettingsResponse:
    return _build_siliconflow_settings_response()


@app.patch("/api/v1/settings/siliconflow", response_model=SiliconFlowSettingsResponse)
async def update_siliconflow_settings(
    req: SiliconFlowSettingsUpdateRequest,
) -> SiliconFlowSettingsResponse:
    if req.text_correction_provider is not None:
        next_provider = _normalize_provider(req.text_correction_provider)
        prev_provider = _normalize_provider(config.text_correction_provider)
        if prev_provider != next_provider and next_provider == "siliconflow":
            # Release any local correction model before switching to remote provider.
            text_corrector.unload_model()
        config.text_correction_provider = next_provider

    if req.siliconflow_api_base is not None:
        base = str(req.siliconflow_api_base or "").strip()
        config.siliconflow_api_base = base or "https://api.siliconflow.cn/v1"

    if req.siliconflow_model is not None:
        model = str(req.siliconflow_model or "").strip()
        config.siliconflow_model = model or "zai-org/GLM-4.5-Air"
    if req.endpoint_model_name is not None:
        endpoint_model = str(req.endpoint_model_name or "").strip()
        config.siliconflow_model = endpoint_model or "zai-org/GLM-4.5-Air"
    if req.credential_name is not None:
        config.siliconflow_credential_name = str(req.credential_name or "").strip() or "API KEY 1"
    if req.model_display_name is not None:
        config.siliconflow_model_display_name = (
            str(req.model_display_name or "").strip() or "远端纠错模型"
        )
    if req.context_length is not None:
        config.siliconflow_context_length = _clamp_int(req.context_length, 131072, 2048, 262144)
    if req.max_tokens is not None:
        config.siliconflow_max_tokens = _clamp_int(req.max_tokens, 4096, 64, 131072)
    if req.compat_mode is not None:
        config.siliconflow_compat_mode = _normalize_compat_mode(req.compat_mode)
    if req.thinking_enabled is not None:
        config.siliconflow_enable_thinking = bool(req.thinking_enabled)

    if req.siliconflow_timeout_s is not None:
        config.siliconflow_timeout_s = max(2.0, float(req.siliconflow_timeout_s))

    if req.clear_api_key:
        config.siliconflow_api_key = ""
    elif req.siliconflow_api_key is not None:
        key = str(req.siliconflow_api_key or "").strip()
        if key:
            config.siliconflow_api_key = key

    _persist_siliconflow_settings()
    return _build_siliconflow_settings_response()


@app.get("/api/v1/voices")
async def list_voices():
    return voice_service.list_voices()


@app.get("/api/v1/voices/custom/catalog")
async def custom_voice_catalog():
    return voice_service.get_custom_voice_catalog()


@app.post("/api/v1/voices/design")
async def create_design_voice(req: DesignVoiceRequest):
    try:
        return await run_in_threadpool(voice_service.create_design_voice, req)
    except ClientInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to create design voice: {exc}") from exc


@app.post("/api/v1/voices/design/assist")
async def assist_design_prompt(req: VoiceDesignAssistRequest):
    try:
        return await run_in_threadpool(voice_service.assist_design_prompt, req)
    except ClientInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to assist voice design: {exc}") from exc


@app.post("/api/v1/voices/custom")
async def create_custom_voice(req: CustomVoiceRequest):
    try:
        return await run_in_threadpool(voice_service.create_custom_voice, req)
    except ClientInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to create custom voice: {exc}") from exc


@app.post("/api/v1/voices/clone")
async def create_clone_voice(
    name: str = Form(...),
    language: str = Form("Auto"),
    ref_text: str = Form(...),
    audio_file: UploadFile = File(...),
):
    try:
        payload = await audio_file.read()
        return await run_in_threadpool(
            voice_service.create_clone_voice,
            name=name,
            language=language,
            ref_text=ref_text,
            ref_audio_bytes=payload,
            filename=audio_file.filename or "ref.wav",
        )
    except ClientInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to create clone voice: {exc}") from exc


@app.post("/api/v1/voices/{voice_id}/preview")
async def preview_voice(voice_id: str, req: VoicePreviewRequest):
    try:
        return await run_in_threadpool(
            voice_service.preview_voice,
            voice_id=voice_id,
            text=req.text,
            language=req.language,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to preview voice: {exc}") from exc


@app.delete("/api/v1/voices/{voice_id}")
async def delete_voice(voice_id: str):
    if not voice_service.delete_voice(voice_id):
        raise HTTPException(status_code=404, detail=f"Voice not found: {voice_id}")
    return {"ok": True, "voice_id": voice_id}


@app.post("/api/v1/realtime/start", response_model=RealtimeStateResponse)
async def realtime_start(req: RealtimeStartRequest):
    engine = _get_engine()
    try:
        engine.start(voice_id=req.voice_id, language=req.language)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to start realtime: {exc}") from exc
    return engine.state()


@app.post("/api/v1/realtime/stop", response_model=RealtimeStateResponse)
async def realtime_stop():
    engine = _get_engine()
    engine.stop()
    return engine.state()


@app.post("/api/v1/realtime/ptt", response_model=RealtimeStateResponse)
async def realtime_ptt(req: RealtimePttRequest):
    engine = _get_engine()
    try:
        return engine.set_ptt_active(req.active)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to update PTT state: {exc}") from exc


@app.post("/api/v1/realtime/simulate")
async def realtime_simulate(req: RealtimeSimulateRequest):
    engine = _get_engine()
    try:
        return engine.simulate_segments(
            count=req.count,
            duration_ms=req.duration_ms,
            amplitude=req.amplitude,
            frequency_hz=req.frequency_hz,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to simulate realtime segments: {exc}") from exc


@app.post("/api/v1/realtime/inject")
async def realtime_inject(req: RealtimeInjectAudioRequest):
    import binascii
    import base64
    import io

    import numpy as np
    import soundfile as sf

    engine = _get_engine()
    try:
        raw = base64.b64decode(req.audio_b64.encode("ascii"), validate=True)
        wav_raw, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        wav = wav_raw.mean(axis=1).astype(np.float32)
        return engine.inject_audio(
            wav=wav,
            sample_rate=int(sr),
            inject_mode=req.inject_mode,
            realtime_pacing=req.realtime_pacing,
            repeat=req.repeat,
            pause_ms=req.pause_ms,
            append_silence_ms=req.append_silence_ms,
        )
    except (ValueError, binascii.Error) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to inject realtime audio: {exc}") from exc


@app.patch("/api/v1/realtime/settings", response_model=RealtimeSettings)
async def realtime_settings(settings: RealtimeSettings):
    engine = _get_engine()
    updated = engine.update_settings(settings)
    db.upsert_setting("realtime_settings", updated.model_dump())
    return updated


@app.get("/api/v1/realtime/state", response_model=RealtimeStateResponse)
async def realtime_state():
    return _get_engine().state()


@app.get("/api/v1/metrics/current")
async def metrics_current():
    return _get_engine().metrics_snapshot()


@app.post("/api/v1/metrics/reset")
async def metrics_reset(clear_log: bool = False, clear_db: bool = False):
    return _get_engine().reset_metrics(clear_log=clear_log, clear_db=clear_db)


@app.websocket("/ws/realtime")
async def ws_realtime(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        await websocket.send_json({"type": "state_changed", "data": _get_engine().state().model_dump()})
        await websocket.send_json({"type": "latency_tick", "data": _get_engine().metrics_snapshot().model_dump()})
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        await ws_manager.disconnect(websocket)
    except Exception:
        await ws_manager.disconnect(websocket)
