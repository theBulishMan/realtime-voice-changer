from __future__ import annotations

import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.backend.asr.asr_stage1 import Stage1ASR
from app.backend.asr.asr_stage2 import Stage2ASR
from app.backend.audio.audio_devices import list_audio_devices
from app.backend.audio.segmenter import VadSegmenter
from app.backend.config import AppConfig
from app.backend.db import AppDatabase
from app.backend.realtime.metrics import SlidingRealtimeMetrics
from app.backend.services.text_corrector import TextCorrector
from app.backend.services.tts_manager import TTSManager
from app.backend.services.voice_service import VoiceService
from app.backend.types import RealtimeSettings, RealtimeStateResponse, WsEvent

logger = logging.getLogger("rvc.realtime")


_VAD_SUPPORTED_SAMPLE_RATES = (8000, 16000, 32000, 48000)
_OUTPUT_COMMON_SAMPLE_RATES = (48000, 44100, 32000, 24000, 22050, 16000, 8000)
_ASR_TARGET_SAMPLE_RATE = 16000
_MIN_LEVEL_DB = -75.0
_MAX_LEVEL_DB = 0.0
_LOW_OUTPUT_GAIN_TRIGGER = 0.0025
_LOW_OUTPUT_TARGET_PEAK = 0.03
_LOW_OUTPUT_GAIN_MAX = 3.0
_GAIN_DB_LIMIT = 24.0
_MIN_ASR_RMS = 0.0045
_MIN_ASR_PEAK = 0.02
_SHORT_SEGMENT_MS = 900.0
_LOW_INFO_TEXT = {
    "\u55ef",
    "\u55ef\u55ef",
    "\u55ef\u54fc",
    "\u5443",
    "\u989d",
    "\u554a",
    "\u54ce",
    "\u5509",
    "uh",
    "um",
    "hmm",
    "mm",
}
_VAD_MODE_REALTIME = 3
_VAD_ENERGY_THRESHOLD = 0.018
_LOW_INFO_SUPPRESS_WINDOW_S = 4.0
_GENERAL_REPEAT_SUPPRESS_WINDOW_S = 2.8
_MIN_SPEECH_ACTIVITY_RATIO = 0.17
_ASR_TRIM_SILENCE_THRESHOLD = 0.007
_ASR_TRIM_SILENCE_PAD_MS = 70.0
_OUTPUT_FEEDBACK_SUPPRESS_WINDOW_S = 0.55
_MAX_AMBIENCE_LEVEL = 0.06
_PLAYBACK_QUEUE_MAX = 18
_PLAYBACK_IDLE_TIMEOUT_S = 0.04
_IDLE_AMBIENCE_CHUNK_MS = 40.0
_OUTPUT_STREAM_RETRY_INTERVAL_S = 1.5
_TTS_BUFFER_SOFT_WAIT_S = 0.10
_TTS_BUFFER_HARD_WAIT_S = 0.24
_TTS_BUFFER_FORCE_CHARS = 20
_TERMINAL_PUNCTUATION = tuple("。！？!?.,;；，")
_PTT_MAX_BUFFER_SECONDS = 12.0


def _ordered_capture_rates(preferred: int) -> list[int]:
    rates: list[int] = []
    if preferred in _VAD_SUPPORTED_SAMPLE_RATES:
        rates.append(preferred)
    for rate in reversed(_VAD_SUPPORTED_SAMPLE_RATES):
        if rate not in rates:
            rates.append(rate)
    return rates


def _ordered_output_rates(preferred: int, defaults: list[int]) -> list[int]:
    rates: list[int] = []
    for rate in [preferred, *defaults, *_OUTPUT_COMMON_SAMPLE_RATES]:
        if rate <= 0:
            continue
        if rate not in rates:
            rates.append(rate)
    return rates


def _normalized_level(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    mono = np.asarray(samples, dtype=np.float32).reshape(-1)
    rms = float(np.sqrt(np.mean(np.square(mono), dtype=np.float64)))
    peak = float(np.max(np.abs(mono)))
    if rms <= 1e-12 and peak <= 1e-12:
        return 0.0
    rms_db = 20.0 * np.log10(max(rms, 1e-12))
    peak_db = 20.0 * np.log10(max(peak, 1e-12))
    db = max(rms_db, peak_db - 6.0)
    clamped_db = float(np.clip(db, _MIN_LEVEL_DB, _MAX_LEVEL_DB))
    return (clamped_db - _MIN_LEVEL_DB) / (_MAX_LEVEL_DB - _MIN_LEVEL_DB)


def _sanitize_output_wav(wav: np.ndarray) -> np.ndarray:
    src = np.asarray(wav)
    arr = src.astype(np.float32, copy=False)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim > 1:
        if arr.shape[-1] > 1:
            arr = arr.reshape(arr.shape[0], -1).mean(axis=1)
        else:
            arr = arr.reshape(-1)
    else:
        arr = arr.reshape(-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    # Keep consistent loudness when source is PCM integers (e.g., int16).
    if np.issubdtype(src.dtype, np.integer):
        info = np.iinfo(src.dtype)
        scale = float(max(abs(info.min), abs(info.max), 1))
        arr = (arr / scale).astype(np.float32)
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    if peak > 1.0:
        arr = (arr / max(peak, 1e-8)).astype(np.float32)
    return arr


def _apply_low_level_gain(wav: np.ndarray) -> tuple[np.ndarray, float, float]:
    if wav.size == 0:
        return wav, 0.0, 1.0
    peak = float(np.max(np.abs(wav)))
    if peak <= 0.0 or peak >= _LOW_OUTPUT_GAIN_TRIGGER:
        return wav, peak, 1.0
    gain = min(_LOW_OUTPUT_GAIN_MAX, _LOW_OUTPUT_TARGET_PEAK / max(peak, 1e-8))
    if gain <= 1.0:
        return wav, peak, 1.0
    boosted = np.clip(wav * gain, -1.0, 1.0).astype(np.float32)
    boosted_peak = float(np.max(np.abs(boosted))) if boosted.size else 0.0
    return boosted, boosted_peak, gain


def _db_to_linear(gain_db: float) -> float:
    return float(np.power(10.0, float(gain_db) / 20.0))


def _apply_gain_db(wav: np.ndarray, gain_db: float) -> np.ndarray:
    if wav.size == 0:
        return wav
    clipped_db = float(np.clip(gain_db, -_GAIN_DB_LIMIT, _GAIN_DB_LIMIT))
    if abs(clipped_db) < 1e-4:
        return wav.astype(np.float32)
    linear = _db_to_linear(clipped_db)
    return np.clip(wav.astype(np.float32) * linear, -1.0, 1.0).astype(np.float32)


def _apply_edge_fade(wav: np.ndarray, sample_rate: int, fade_ms: float = 6.0) -> np.ndarray:
    if wav.size == 0 or sample_rate <= 0:
        return wav.astype(np.float32)
    fade_samples = int(sample_rate * fade_ms / 1000.0)
    if fade_samples <= 0:
        return wav.astype(np.float32)
    fade_samples = min(fade_samples, wav.size // 2)
    if fade_samples <= 0:
        return wav.astype(np.float32)
    out = wav.astype(np.float32).copy()
    ramp = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    out[:fade_samples] *= ramp
    out[-fade_samples:] *= ramp[::-1]
    return out


def _apply_soft_limiter(wav: np.ndarray, ceiling: float = 0.96) -> np.ndarray:
    if wav.size == 0:
        return wav.astype(np.float32)
    arr = wav.astype(np.float32)
    peak = float(np.max(np.abs(arr)))
    if peak <= 1e-8:
        return arr
    if peak > ceiling:
        arr = (arr * (ceiling / peak)).astype(np.float32)
    hot_ratio = float(np.mean(np.abs(arr) > 0.90))
    if hot_ratio > 0.12:
        arr = (np.tanh(arr * 1.03) / np.tanh(1.03)).astype(np.float32)
        peak2 = float(np.max(np.abs(arr)))
        if peak2 > ceiling:
            arr = (arr * (ceiling / peak2)).astype(np.float32)
    return arr


def _segment_stats(wav: np.ndarray, sample_rate: int) -> tuple[float, float, float]:
    mono = np.asarray(wav, dtype=np.float32).reshape(-1)
    if mono.size == 0:
        return 0.0, 0.0, 0.0
    rms = float(np.sqrt(np.mean(np.square(mono), dtype=np.float64)))
    peak = float(np.max(np.abs(mono)))
    dur_ms = (mono.size / float(max(sample_rate, 1))) * 1000.0
    return rms, peak, dur_ms


def _is_low_energy_segment(wav: np.ndarray, sample_rate: int) -> bool:
    rms, peak, dur_ms = _segment_stats(wav, sample_rate)
    if peak < _MIN_ASR_PEAK and rms < _MIN_ASR_RMS:
        return True
    if dur_ms <= _SHORT_SEGMENT_MS and peak < 0.03 and rms < 0.006:
        return True
    return False


def _speech_activity_ratio(wav: np.ndarray, threshold: float = 0.01) -> float:
    mono = np.asarray(wav, dtype=np.float32).reshape(-1)
    if mono.size == 0:
        return 0.0
    active = np.abs(mono) >= float(max(threshold, 1e-5))
    return float(np.mean(active))


def _trim_silence(wav: np.ndarray, sample_rate: int) -> np.ndarray:
    mono = np.asarray(wav, dtype=np.float32).reshape(-1)
    if mono.size == 0 or sample_rate <= 0:
        return mono
    mask = np.where(np.abs(mono) >= _ASR_TRIM_SILENCE_THRESHOLD)[0]
    if mask.size == 0:
        return mono
    pad = int(sample_rate * (_ASR_TRIM_SILENCE_PAD_MS / 1000.0))
    start = max(0, int(mask[0]) - pad)
    end = min(mono.size, int(mask[-1]) + pad + 1)
    return mono[start:end].astype(np.float32)


def _mix_ambience_noise(
    wav: np.ndarray, *, sample_rate: int, level: float, rng: np.random.Generator
) -> np.ndarray:
    mono = np.asarray(wav, dtype=np.float32).reshape(-1)
    if mono.size == 0:
        return mono
    tuned = float(np.clip(level, 0.0, _MAX_AMBIENCE_LEVEL)) * 0.45
    if tuned <= 1e-4:
        return mono
    noise = rng.normal(0.0, 1.0, size=mono.size).astype(np.float32)
    std = float(np.std(noise))
    if std > 1e-6:
        noise = (noise / std).astype(np.float32)
    mixed = mono + (noise * tuned)
    return np.clip(mixed, -1.0, 1.0).astype(np.float32)


def _continuous_ambience_chunk(
    sample_count: int,
    *,
    level: float,
    rng: np.random.Generator,
    prev_state: float = 0.0,
    alpha: float = 0.93,
) -> tuple[np.ndarray, float]:
    count = max(0, int(sample_count))
    if count <= 0:
        return np.zeros(0, dtype=np.float32), float(prev_state)
    tuned = float(np.clip(level, 0.0, _MAX_AMBIENCE_LEVEL)) * 0.35
    if tuned <= 1e-5:
        return np.zeros(count, dtype=np.float32), float(prev_state)
    out = rng.normal(0.0, 1.0, size=count).astype(np.float32)
    std = float(np.std(out))
    if std > 1e-6:
        out = (out / std).astype(np.float32)
    out = (out * tuned).astype(np.float32)
    return out, float(prev_state)


def _normalize_text_key(text: str) -> str:
    normalized: list[str] = []
    for ch in text.strip().lower():
        code = ord(ch)
        if 0x30 <= code <= 0x39 or 0x61 <= code <= 0x7A:
            normalized.append(ch)
            continue
        if 0x4E00 <= code <= 0x9FFF:
            normalized.append(ch)
    return "".join(normalized)


def _is_low_information_text(text: str) -> bool:
    key = _normalize_text_key(text)
    if not key:
        return True
    if key in _LOW_INFO_TEXT:
        return True
    if key.startswith("\u55ef") and len(key) <= 2:
        return True
    if key.startswith("uh") and len(key) <= 3:
        return True
    return False


def _ends_with_terminal_punctuation(text: str) -> bool:
    normalized = str(text or "").strip()
    if not normalized:
        return False
    return normalized.endswith(_TERMINAL_PUNCTUATION)


def _merge_tts_text(existing: str, incoming: str) -> str:
    left = str(existing or "").strip()
    right = str(incoming or "").strip()
    if not left:
        return right
    if not right:
        return left
    if left.endswith(tuple("。！？!?.,;；，")):
        return f"{left}{right}"
    return f"{left} {right}".strip()


def _soft_noise_suppress(wav: np.ndarray, sample_rate: int, strength: float) -> np.ndarray:
    mono = np.asarray(wav, dtype=np.float32).reshape(-1)
    if mono.size == 0:
        return mono
    tuned_strength = float(np.clip(strength, 0.0, 1.0))
    if tuned_strength <= 0.0:
        return mono
    if sample_rate <= 0:
        sample_rate = _ASR_TARGET_SAMPLE_RATE

    # Use envelope-based soft gating to suppress stationary background noise
    # while retaining speech transients.
    window = max(24, int(sample_rate * 0.012))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    envelope = np.sqrt(
        np.convolve(np.square(mono, dtype=np.float32), kernel, mode="same") + 1e-8
    ).astype(np.float32)
    floor = float(np.percentile(envelope, 22.0))
    floor = max(floor, 1e-6)
    threshold = floor * (1.4 + 2.0 * tuned_strength)
    soft = np.clip((envelope - threshold) / (threshold + 1e-6), 0.0, 1.0)
    soft = np.power(soft, 0.65 + 0.25 * tuned_strength).astype(np.float32)
    min_gain = max(0.05, 0.22 - (0.16 * tuned_strength))
    gain = np.maximum(min_gain, soft).astype(np.float32)
    denoised = (mono * gain).astype(np.float32)

    original_peak = float(np.max(np.abs(mono)))
    denoised_peak = float(np.max(np.abs(denoised)))
    if original_peak > 1e-6 and denoised_peak > 1e-6:
        restore = min(1.6, original_peak / denoised_peak)
        if restore > 1.0:
            denoised = np.clip(denoised * restore, -1.0, 1.0).astype(np.float32)
    return denoised


class RealtimeEngine:
    def __init__(
        self,
        config: AppConfig,
        db: AppDatabase,
        voice_service: VoiceService,
        tts_manager: TTSManager,
        asr_stage1: Stage1ASR,
        asr_stage2: Stage2ASR,
        settings: RealtimeSettings,
        event_sink: Callable[[WsEvent], None],
        text_corrector: TextCorrector | None = None,
    ) -> None:
        self.config = config
        self.db = db
        self.voice_service = voice_service
        self.tts_manager = tts_manager
        self.asr_stage1 = asr_stage1
        self.asr_stage2 = asr_stage2
        self.text_corrector = text_corrector
        self.settings = settings
        self.event_sink = event_sink

        self._lock = threading.RLock()
        self._running = False
        self._voice_id: str | None = None
        self._language: str = "Auto"
        self._session_id = 0

        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=500)
        self._segment_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
        self._playback_queue: queue.Queue[tuple[np.ndarray, int, int]] = queue.Queue(
            maxsize=_PLAYBACK_QUEUE_MAX
        )
        self._segmenter_lock = threading.RLock()
        self._ptt_segment_lock = threading.RLock()
        self._capture_sample_rate = self.config.capture_sample_rate
        self._segmenter = self._new_segmenter()
        self._ptt_active = not bool(self.settings.ptt_enabled)
        self._ptt_pending_segments: list[np.ndarray] = []
        self._ptt_pending_samples = 0
        self._stop_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._infer_threads: list[threading.Thread] = []
        self._playback_thread: threading.Thread | None = None
        self._input_stream: Any | None = None
        # (stream, device_id, sample_rate, channels)
        self._output_streams: list[tuple[Any, int, int, int]] = []
        self._output_sample_rate: int | None = None
        self._last_final_text: str = ""
        self._last_final_ts: float = 0.0
        self._last_low_info_text: str = ""
        self._last_low_info_ts: float = 0.0
        self._dropped_frames = 0
        self._dropped_segments = 0
        self._last_recorded_dropped_frames = 0
        self._last_recorded_dropped_segments = 0
        self._last_input_level_emit = 0.0
        self._last_output_level_emit = 0.0
        self._last_output_activity_ts = 0.0
        self._noise_rng = np.random.default_rng()
        self._ambience_state_by_sr: dict[int, float] = {}
        self._last_output_open_attempt_ts = 0.0
        self._tts_buffer_lock = threading.RLock()
        self._tts_buffer_text = ""
        self._tts_buffer_voice_id: str | None = None
        self._tts_buffer_language = "Auto"
        self._tts_buffer_prompt_path: Path | None = None
        self._tts_buffer_ref_text: str | None = None
        self._tts_buffer_first_t0 = 0.0
        self._tts_buffer_last_asr_t = 0.0
        self._tts_buffer_updated_ts = 0.0
        self._tts_buffer_session_id = 0
        self._tts_flush_thread: threading.Thread | None = None
        self._level_emit_interval_s = 0.08
        self._metrics = SlidingRealtimeMetrics(
            buffer_size=self.config.metrics_buffer_size,
            log_path=self.config.logs_dir / "metrics.ndjson",
        )

    def _is_valid_device_id(self, device_id: int | None, *, output: bool) -> bool:
        if device_id is None:
            return False
        key = "max_output_channels" if output else "max_input_channels"
        try:
            import sounddevice as sd

            info = sd.query_devices(int(device_id))
            return int(info.get(key, 0)) > 0
        except Exception:
            return False

    def _default_device_id(self, *, output: bool) -> int | None:
        try:
            import sounddevice as sd

            default_in, default_out = sd.default.device
            candidate = default_out if output else default_in
            if candidate is None:
                return None
            idx = int(candidate)
            if idx < 0:
                return None
            if self._is_valid_device_id(idx, output=output):
                return idx
        except Exception:
            pass
        return None

    def _first_available_device_id(self, *, output: bool) -> int | None:
        key = "max_output_channels" if output else "max_input_channels"
        try:
            import sounddevice as sd

            for idx, dev in enumerate(sd.query_devices()):
                if int(dev.get(key, 0)) > 0:
                    return int(idx)
        except Exception:
            return None
        return None

    def _resolve_fallback_output_id(self) -> int | None:
        return self._default_device_id(output=True) or self._first_available_device_id(
            output=True
        )

    def _resolve_fallback_input_id(self) -> int | None:
        return self._default_device_id(output=False) or self._first_available_device_id(
            output=False
        )
    def _has_monitor_feedback_risk(
        self, input_device_id: int | None, monitor_device_id: int | None
    ) -> bool:
        if input_device_id is None or monitor_device_id is None:
            return False
        # Only treat exact same device-id as high feedback risk.
        return input_device_id == monitor_device_id

    def _sanitize_runtime_settings(self) -> None:
        updates: dict[str, int | bool | None] = {}
        invalid_device_fallback = False
        feedback_risk_detected = False
        virtual_preferred_aligned = False
        preferred_virtual: int | None = None
        try:
            devices = list_audio_devices(self.config.virtual_mic_name_hint)
            if devices.virtual_mic_output_id is not None:
                preferred_virtual = int(devices.virtual_mic_output_id)
        except Exception:
            preferred_virtual = None

        if not self._is_valid_device_id(self.settings.input_device_id, output=False):
            updates["input_device_id"] = self._resolve_fallback_input_id()
            invalid_device_fallback = True

        if not self._is_valid_device_id(self.settings.monitor_device_id, output=True):
            updates["monitor_device_id"] = self._resolve_fallback_output_id()
            invalid_device_fallback = True

        current_virtual = self.settings.virtual_mic_device_id
        if (
            preferred_virtual is not None
            and self._is_valid_device_id(preferred_virtual, output=True)
            and current_virtual != preferred_virtual
        ):
            updates["virtual_mic_device_id"] = preferred_virtual
            virtual_preferred_aligned = True
        elif not self._is_valid_device_id(current_virtual, output=True):
            if preferred_virtual is not None and self._is_valid_device_id(
                preferred_virtual, output=True
            ):
                updates["virtual_mic_device_id"] = preferred_virtual
                invalid_device_fallback = True
            else:
                updates["virtual_mic_device_id"] = self._resolve_fallback_output_id()
                invalid_device_fallback = True

        monitor_enabled = bool(updates.get("monitor_enabled", self.settings.monitor_enabled))
        input_device_id = updates.get("input_device_id", self.settings.input_device_id)
        monitor_device_id = updates.get("monitor_device_id", self.settings.monitor_device_id)
        if monitor_enabled and self._has_monitor_feedback_risk(
            input_device_id, monitor_device_id
        ):
            feedback_risk_detected = True

        if not updates:
            if feedback_risk_detected:
                message = (
                    "Monitor is enabled on the same input/output device; acoustic feedback may occur."
                )
                logger.warning(
                    "%s input=%s monitor=%s",
                    message,
                    input_device_id,
                    monitor_device_id,
                )
                self._emit(
                    "notice",
                    {
                        "message": message,
                        "input_device_id": input_device_id,
                        "monitor_device_id": monitor_device_id,
                    },
                )
            return

        before = self.settings.model_dump()
        self.settings = self.settings.model_copy(update=updates)
        with self._segmenter_lock:
            self._segmenter = self._new_segmenter()
        self.db.upsert_setting("realtime_settings", self.settings.model_dump())
        after = self.settings.model_dump()

        message = "Audio devices were auto-adjusted to available devices."
        event_type = "error"
        if not invalid_device_fallback:
            event_type = "notice"
            if feedback_risk_detected:
                message = (
                    "Audio settings were auto-aligned. Monitor remains enabled on the same "
                    "input/output device; feedback may occur."
                )
            elif virtual_preferred_aligned:
                message = "Virtual mic device was auto-aligned to the preferred backend."
            else:
                message = "Audio settings were auto-aligned."
        logger.warning("%s before=%s after=%s", message, before, after)
        self._emit(event_type, {"message": message, "before": before, "after": after})

    def _new_segmenter(self) -> VadSegmenter:
        return VadSegmenter(
            sample_rate=self._capture_sample_rate,
            frame_ms=self.config.frame_ms,
            vad_silence_ms=self.settings.vad_silence_ms,
            max_segment_ms=self.settings.max_segment_ms,
            vad_mode=_VAD_MODE_REALTIME,
            energy_threshold=_VAD_ENERGY_THRESHOLD,
        )

    def _session_active(self, session_id: int) -> bool:
        with self._lock:
            return (
                session_id == self._session_id
                and self._running
                and not self._stop_event.is_set()
            )

    def _mark_dropped_frame(self) -> None:
        with self._lock:
            self._dropped_frames += 1

    def _mark_dropped_segment(self) -> None:
        with self._lock:
            self._dropped_segments += 1

    def _capture_drop_state(self) -> tuple[bool, int, int]:
        with self._lock:
            dropped_frames = self._dropped_frames
            dropped_segments = self._dropped_segments
            dropped = (
                dropped_frames > self._last_recorded_dropped_frames
                or dropped_segments > self._last_recorded_dropped_segments
            )
            self._last_recorded_dropped_frames = dropped_frames
            self._last_recorded_dropped_segments = dropped_segments
        return dropped, dropped_frames, dropped_segments

    def _reset_tts_buffer(self) -> None:
        with self._tts_buffer_lock:
            self._tts_buffer_text = ""
            self._tts_buffer_voice_id = None
            self._tts_buffer_language = "Auto"
            self._tts_buffer_prompt_path = None
            self._tts_buffer_ref_text = None
            self._tts_buffer_first_t0 = 0.0
            self._tts_buffer_last_asr_t = 0.0
            self._tts_buffer_updated_ts = 0.0
            self._tts_buffer_session_id = 0

    def _reset_ptt_pending_segments(self) -> None:
        with self._ptt_segment_lock:
            self._ptt_pending_segments = []
            self._ptt_pending_samples = 0

    def _buffer_ptt_segment(self, segment: np.ndarray) -> None:
        mono = np.asarray(segment, dtype=np.float32).reshape(-1)
        if mono.size == 0:
            return
        max_samples = int(max(1.0, _PTT_MAX_BUFFER_SECONDS) * max(1, int(self._capture_sample_rate)))
        with self._ptt_segment_lock:
            self._ptt_pending_segments.append(mono)
            self._ptt_pending_samples += int(mono.size)
            while self._ptt_pending_segments and self._ptt_pending_samples > max_samples:
                dropped = self._ptt_pending_segments.pop(0)
                self._ptt_pending_samples = max(0, self._ptt_pending_samples - int(dropped.size))

    def _consume_ptt_segments(self, extra_segments: list[np.ndarray] | None = None) -> np.ndarray | None:
        pieces: list[np.ndarray] = []
        with self._ptt_segment_lock:
            if self._ptt_pending_segments:
                pieces.extend(self._ptt_pending_segments)
            self._ptt_pending_segments = []
            self._ptt_pending_samples = 0
        if extra_segments:
            pieces.extend(
                np.asarray(seg, dtype=np.float32).reshape(-1)
                for seg in extra_segments
                if np.asarray(seg).size > 0
            )
        if not pieces:
            return None
        if len(pieces) == 1:
            return pieces[0].astype(np.float32)
        return np.concatenate(pieces).astype(np.float32)

    def _submit_or_buffer_ptt_segment(self, segment: np.ndarray, *, session_id: int) -> None:
        with self._lock:
            ptt_enabled = bool(self.settings.ptt_enabled)
            ptt_active = bool(self._ptt_active)
        if ptt_enabled and ptt_active:
            self._buffer_ptt_segment(segment)
            return
        self._submit_segment(segment, session_id=session_id)

    def state(self) -> RealtimeStateResponse:
        with self._lock:
            return RealtimeStateResponse(
                running=self._running,
                voice_id=self._voice_id,
                language=self._language,
                ptt_active=bool(self._ptt_active),
                settings=self.settings,
            )

    def metrics_snapshot(self):
        return self._metrics.snapshot()

    def reset_metrics(self, *, clear_log: bool = False, clear_db: bool = False) -> dict:
        self._metrics.reset(clear_log=clear_log)
        with self._lock:
            self._dropped_frames = 0
            self._dropped_segments = 0
            self._last_final_text = ""
            self._last_final_ts = 0.0
            self._last_low_info_text = ""
            self._last_low_info_ts = 0.0
            self._last_recorded_dropped_frames = 0
            self._last_recorded_dropped_segments = 0
            self._last_input_level_emit = 0.0
            self._last_output_level_emit = 0.0
            self._last_output_activity_ts = 0.0
        removed = 0
        if clear_db:
            removed = self.db.clear_metrics()
        snapshot = self._metrics.snapshot()
        return {
            "ok": True,
            "clear_log": clear_log,
            "clear_db": clear_db,
            "db_deleted_rows": removed,
            "metrics": snapshot.model_dump(),
        }

    def simulate_segments(
        self,
        *,
        count: int = 1,
        duration_ms: int = 500,
        amplitude: float = 0.08,
        frequency_hz: float = 220.0,
    ) -> dict:
        if not self.config.fake_mode:
            raise RuntimeError("simulate is only available when fake mode is enabled")
        if count <= 0:
            raise ValueError("count must be positive")
        if duration_ms <= 0:
            raise ValueError("duration_ms must be positive")

        with self._lock:
            voice_id = self._voice_id
            session_id = self._session_id
            if not self._running or voice_id is None:
                raise RuntimeError("realtime engine is not running")

        sr = self._capture_sample_rate
        samples = int(sr * duration_ms / 1000)
        if samples < int(sr * 0.12):
            samples = int(sr * 0.12)
        t = np.linspace(0.0, samples / float(sr), samples, endpoint=False, dtype=np.float32)

        for i in range(count):
            phase = np.float32(0.1 * i)
            segment = (amplitude * np.sin((2.0 * np.pi * frequency_hz * t) + phase)).astype(np.float32)
            self._process_segment(segment, session_id=session_id)

        return {
            "ok": True,
            "injected_segments": count,
            "duration_ms": duration_ms,
            "amplitude": amplitude,
            "frequency_hz": frequency_hz,
            "metrics": self._metrics.snapshot().model_dump(),
        }

    def inject_audio(
        self,
        wav: np.ndarray,
        sample_rate: int,
        *,
        inject_mode: str = "queue",
        realtime_pacing: bool = True,
        repeat: int = 1,
        pause_ms: int = 0,
        append_silence_ms: int = 260,
    ) -> dict:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if repeat <= 0:
            raise ValueError("repeat must be positive")
        if pause_ms < 0:
            raise ValueError("pause_ms must be >= 0")
        if append_silence_ms < 0:
            raise ValueError("append_silence_ms must be >= 0")
        if inject_mode not in {"queue", "segment"}:
            raise ValueError("inject_mode must be one of: queue, segment")

        with self._lock:
            if not self._running or self._voice_id is None:
                raise RuntimeError("realtime engine is not running")
            frame_ms = self.config.frame_ms
            capture_sr = self._capture_sample_rate
            session_id = self._session_id

        mono = np.asarray(wav, dtype=np.float32)
        if mono.ndim == 2:
            mono = mono.mean(axis=1).astype(np.float32)
        elif mono.ndim > 2:
            mono = mono.reshape(-1).astype(np.float32)
        if mono.size == 0:
            raise ValueError("wav is empty")

        if sample_rate != capture_sr:
            from scipy.signal import resample_poly

            mono = resample_poly(mono, capture_sr, sample_rate).astype(np.float32)

        frame_samples = max(1, int(capture_sr * frame_ms / 1000))
        remainder = int(mono.size % frame_samples)
        if remainder != 0:
            pad = frame_samples - remainder
            mono = np.pad(mono, (0, pad), mode="constant")
        frames = mono.reshape(-1, frame_samples)

        silence_frames = 0
        if append_silence_ms > 0:
            silence_frames = int(np.ceil(float(append_silence_ms) / float(frame_ms)))
        silence = np.zeros(frame_samples, dtype=np.float32)

        pacing_sleep = frame_ms / 1000.0 if realtime_pacing else 0.0
        injected_frames = 0
        injected_segments = 0
        t0 = time.perf_counter()

        if inject_mode == "segment":
            segment_sleep = (len(mono) / float(capture_sr)) if realtime_pacing else 0.0
            for rep in range(repeat):
                if self._stop_event.is_set():
                    raise RuntimeError("realtime engine stopped during audio injection")
                self._process_segment(mono.copy(), session_id=session_id)
                injected_segments += 1
                if segment_sleep > 0:
                    time.sleep(segment_sleep)
                if append_silence_ms > 0:
                    time.sleep(append_silence_ms / 1000.0)
                if rep < repeat - 1 and pause_ms > 0:
                    time.sleep(pause_ms / 1000.0)

            return {
                "ok": True,
                "inject_mode": inject_mode,
                "injected_frames": 0,
                "injected_segments": injected_segments,
                "frame_ms": frame_ms,
                "input_sample_rate": sample_rate,
                "capture_sample_rate": capture_sr,
                "audio_duration_ms": int((len(mono) / float(capture_sr)) * 1000.0 * repeat),
                "repeat": repeat,
                "append_silence_ms": append_silence_ms,
                "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
                "metrics": self._metrics.snapshot().model_dump(),
            }

        for rep in range(repeat):
            for frame in frames:
                if self._stop_event.is_set():
                    raise RuntimeError("realtime engine stopped during audio injection")
                try:
                    self._queue.put(frame.copy(), timeout=2.0)
                except queue.Full as exc:
                    raise RuntimeError(
                        "inject queue is full; wait for pipeline to drain or reduce inject rate"
                    ) from exc
                injected_frames += 1
                if pacing_sleep > 0:
                    time.sleep(pacing_sleep)

            for _ in range(silence_frames):
                if self._stop_event.is_set():
                    raise RuntimeError("realtime engine stopped during audio injection")
                try:
                    self._queue.put(silence.copy(), timeout=2.0)
                except queue.Full as exc:
                    raise RuntimeError(
                        "inject queue is full; wait for pipeline to drain or reduce inject rate"
                    ) from exc
                injected_frames += 1
                if pacing_sleep > 0:
                    time.sleep(pacing_sleep)

            if rep < repeat - 1 and pause_ms > 0:
                time.sleep(pause_ms / 1000.0)

        return {
            "ok": True,
            "inject_mode": inject_mode,
            "injected_frames": injected_frames,
            "injected_segments": 0,
            "frame_ms": frame_ms,
            "input_sample_rate": sample_rate,
            "capture_sample_rate": capture_sr,
            "audio_duration_ms": int((len(frames) * frame_ms) * repeat),
            "repeat": repeat,
            "append_silence_ms": append_silence_ms,
            "elapsed_ms": (time.perf_counter() - t0) * 1000.0,
            "metrics": self._metrics.snapshot().model_dump(),
        }

    def update_settings(self, settings: RealtimeSettings) -> RealtimeSettings:
        flush_segments: list[np.ndarray] = []
        session_id = 0
        with self._lock:
            prev_ptt_enabled = bool(self.settings.ptt_enabled)
            self.settings = settings
            with self._segmenter_lock:
                self._segmenter = self._new_segmenter()
            if not bool(self.settings.ptt_enabled):
                self._ptt_active = True
                if self._running and prev_ptt_enabled:
                    with self._segmenter_lock:
                        flush_segments = list(self._segmenter.flush())
                    session_id = self._session_id
            elif not prev_ptt_enabled and bool(self.settings.ptt_enabled):
                self._ptt_active = False
            if self._running:
                self._close_output_streams()
        merged = self._consume_ptt_segments(flush_segments)
        if merged is not None and session_id:
            self._submit_segment(merged, session_id=session_id)
        return self.settings

    def set_ptt_active(self, active: bool) -> RealtimeStateResponse:
        to_flush: list[np.ndarray] = []
        session_id = 0
        with self._lock:
            if not self._running:
                raise RuntimeError("Realtime engine is not running")
            desired = bool(active)
            if not bool(self.settings.ptt_enabled):
                desired = True
            changed = desired != bool(self._ptt_active)
            self._ptt_active = desired
            session_id = self._session_id
            if changed and desired:
                self._reset_ptt_pending_segments()
            if changed and (not desired):
                with self._segmenter_lock:
                    to_flush = list(self._segmenter.flush())
            payload = {
                "active": bool(self._ptt_active),
                "enabled": bool(self.settings.ptt_enabled),
            }
        merged = self._consume_ptt_segments(to_flush)
        if merged is not None and session_id:
            self._submit_segment(merged, session_id=session_id)
        self._emit("ptt_state", payload)
        return self.state()

    def start(self, voice_id: str, language: str) -> None:
        with self._lock:
            if self._running:
                raise RuntimeError("Realtime engine is already running")
            voice = self.voice_service.get_voice(voice_id)
            if voice is None:
                raise ValueError(f"Voice not found: {voice_id}")
            requested_language = (language or "").strip() or "Auto"
            resolved_language = requested_language
            if requested_language.lower() == "auto":
                hint = str(getattr(voice, "language_hint", "") or "").strip()
                if hint in {"Chinese", "English"}:
                    resolved_language = hint
            self._voice_id = voice_id
            self._language = resolved_language
            self._stop_event.clear()
            self._running = True
            self._dropped_frames = 0
            self._dropped_segments = 0
            self._last_recorded_dropped_frames = 0
            self._last_recorded_dropped_segments = 0
            self._last_input_level_emit = 0.0
            self._last_output_level_emit = 0.0
            self._last_output_activity_ts = 0.0
            self._ambience_state_by_sr = {}
            self._last_output_open_attempt_ts = 0.0
            self._sanitize_runtime_settings()
            self._session_id += 1
            session_id = self._session_id
            self._queue = queue.Queue(maxsize=500)
            self._segment_queue = queue.Queue(maxsize=64)
            self._playback_queue = queue.Queue(maxsize=_PLAYBACK_QUEUE_MAX)
            self._capture_sample_rate = self.config.capture_sample_rate
            self._output_sample_rate = None
            with self._segmenter_lock:
                self._segmenter = self._new_segmenter()
            self._ptt_active = not bool(self.settings.ptt_enabled)
            self._reset_tts_buffer()
            self._reset_ptt_pending_segments()
            if (
                (not self.config.fake_mode)
                and bool(self.settings.ambience_enabled)
                and float(self.settings.ambience_level) > 1e-4
            ):
                self._last_output_open_attempt_ts = time.perf_counter()
                self._ensure_output_streams(max(1, int(self.config.tts_sample_rate)))
            self._worker_thread = threading.Thread(
                target=self._worker_loop,
                args=(session_id,),
                name="realtime-worker",
                daemon=True,
            )
            infer_workers = max(1, int(self.config.tts_infer_workers))
            self._infer_threads = []
            for idx in range(infer_workers):
                worker = threading.Thread(
                    target=self._infer_loop,
                    args=(session_id,),
                    name=f"realtime-infer-{idx + 1}",
                    daemon=True,
                )
                self._infer_threads.append(worker)
            self._playback_thread = threading.Thread(
                target=self._playback_loop,
                args=(session_id,),
                name="realtime-playback",
                daemon=True,
            )
            self._tts_flush_thread = threading.Thread(
                target=self._tts_flush_loop,
                args=(session_id,),
                name="realtime-tts-flush",
                daemon=True,
            )
            self._worker_thread.start()
            for worker in self._infer_threads:
                worker.start()
            self._playback_thread.start()
            self._tts_flush_thread.start()
            should_capture_input = (not self.config.fake_mode) or self.config.fake_capture_input
            try:
                self.tts_manager.preload_prompt_path(
                    Path(voice.prompt_path),
                    ref_text_override=voice.ref_text,
                )
            except Exception as exc:
                logger.warning("prompt preload failed: %s", exc)
            if should_capture_input:
                try:
                    self._start_input_stream()
                except Exception as exc:
                    if self.config.fake_mode:
                        msg = f"fake input capture unavailable: {exc}"
                        logger.warning(msg)
                        self._emit("error", {"message": msg})
                    else:
                        raise
            self._emit("state_changed", self.state().model_dump())
            self._emit(
                "ptt_state",
                {
                    "enabled": bool(self.settings.ptt_enabled),
                    "active": bool(self._ptt_active),
                },
            )
            self._emit("audio_level", {"input": 0.0, "output": 0.0})

    def stop(self) -> None:
        with self._lock:
            if not self._running:
                return
            self._session_id += 1
            self._stop_event.set()
            self._running = False
            self._emit("state_changed", self.state().model_dump())
            self._emit("audio_level", {"input": 0.0, "output": 0.0})
            self._voice_id = None
            self._language = "Auto"
            self._last_final_text = ""
            self._last_final_ts = 0.0
            self._last_low_info_text = ""
            self._last_low_info_ts = 0.0
            self._last_output_activity_ts = 0.0
            self._ambience_state_by_sr = {}
            self._last_output_open_attempt_ts = 0.0
            self._ptt_active = not bool(self.settings.ptt_enabled)
            self._reset_tts_buffer()
            self._reset_ptt_pending_segments()
            self._close_input_stream()
            self._close_output_streams()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=3.0)
            if self._worker_thread.is_alive():
                logger.warning("Realtime worker is still running after stop timeout")
            self._worker_thread = None
        for infer_thread in self._infer_threads:
            infer_thread.join(timeout=3.0)
            if infer_thread.is_alive():
                logger.warning(
                    "Realtime infer worker is still running after stop timeout: %s",
                    infer_thread.name,
                )
        self._infer_threads = []
        if self._playback_thread is not None:
            self._playback_thread.join(timeout=3.0)
            if self._playback_thread.is_alive():
                logger.warning("Realtime playback worker is still running after stop timeout")
            self._playback_thread = None
        if self._tts_flush_thread is not None:
            self._tts_flush_thread.join(timeout=3.0)
            if self._tts_flush_thread.is_alive():
                logger.warning("Realtime TTS flush worker is still running after stop timeout")
            self._tts_flush_thread = None

    def _emit(self, event_type: str, data: dict) -> None:
        self.event_sink(WsEvent(type=event_type, data=data))

    def _emit_audio_level(
        self,
        *,
        input_level: float | None = None,
        output_level: float | None = None,
        force: bool = False,
    ) -> None:
        now = time.perf_counter()
        payload: dict[str, float] = {}

        if input_level is not None:
            if force or (now - self._last_input_level_emit) >= self._level_emit_interval_s:
                payload["input"] = float(np.clip(input_level, 0.0, 1.0))
                self._last_input_level_emit = now

        if output_level is not None:
            if force or (now - self._last_output_level_emit) >= self._level_emit_interval_s:
                payload["output"] = float(np.clip(output_level, 0.0, 1.0))
                self._last_output_level_emit = now

        if payload:
            self._emit("audio_level", payload)

    def _start_input_stream(self) -> None:
        import sounddevice as sd

        device = self.settings.input_device_id
        channels = 1
        capture_sr, fallback_candidates = self._resolve_capture_sample_rate(
            sd,
            device=device,
            channels=channels,
        )
        self._capture_sample_rate = capture_sr
        with self._segmenter_lock:
            self._segmenter = self._new_segmenter()

        if capture_sr != self.config.capture_sample_rate:
            msg = (
                f"input samplerate fallback: requested={self.config.capture_sample_rate}, "
                f"using={capture_sr}, tried={fallback_candidates}"
            )
            logger.warning(msg)
            self._emit("error", {"message": msg})

        channels = 1
        blocksize = int(capture_sr * self.config.frame_ms / 1000)
        self._input_stream = sd.InputStream(
            samplerate=capture_sr,
            blocksize=blocksize,
            channels=channels,
            dtype="float32",
            device=device,
            callback=self._audio_callback,
            latency="low",
        )
        self._input_stream.start()

    def _resolve_capture_sample_rate(
        self,
        sd,
        *,
        device: int | None,
        channels: int,
    ) -> tuple[int, list[int]]:
        candidates = _ordered_capture_rates(self.config.capture_sample_rate)
        errors: list[str] = []
        for rate in candidates:
            try:
                sd.check_input_settings(
                    device=device,
                    channels=channels,
                    samplerate=rate,
                    dtype="float32",
                )
                return rate, candidates
            except Exception as exc:
                errors.append(f"{rate}:{exc}")

        details = "; ".join(errors) if errors else "no candidate rate was accepted"
        raise RuntimeError(
            "failed to open input stream with supported sample rates. "
            f"device={device}, candidates={candidates}, errors={details}"
        )

    def _close_input_stream(self) -> None:
        if self._input_stream is None:
            return
        try:
            self._input_stream.stop()
            self._input_stream.close()
        finally:
            self._input_stream = None

    def _close_output_streams(self) -> None:
        for stream, _dev_id, _sr, _ch in self._output_streams:
            try:
                stream.stop()
                stream.close()
            except Exception:
                pass
        self._output_streams = []
        self._output_sample_rate = None
        self._ambience_state_by_sr = {}

    def _resolve_output_sample_rate(
        self,
        sd,
        *,
        device_ids: list[int],
        channels: int,
        preferred_rate: int,
    ) -> tuple[int, list[int]]:
        defaults: list[int] = []
        for dev_id in device_ids:
            try:
                info = sd.query_devices(dev_id)
                default_rate = int(round(float(info.get("default_samplerate", 0))))
                if default_rate > 0 and default_rate not in defaults:
                    defaults.append(default_rate)
            except Exception:
                continue

        candidates = _ordered_output_rates(preferred_rate, defaults)
        errors: list[str] = []
        for rate in candidates:
            ok = True
            for dev_id in device_ids:
                try:
                    sd.check_output_settings(
                        device=dev_id,
                        channels=channels,
                        samplerate=rate,
                        dtype="float32",
                    )
                except Exception as exc:
                    ok = False
                    errors.append(f"{rate}@{dev_id}:{exc}")
                    break
            if ok:
                return rate, candidates

        details = "; ".join(errors) if errors else "no candidate rate was accepted"
        raise RuntimeError(
            "failed to open output stream with supported sample rates. "
            f"devices={device_ids}, candidates={candidates}, errors={details}"
        )

    def _resolve_single_output_stream_config(
        self,
        sd,
        *,
        device_id: int,
        preferred_rate: int,
    ) -> tuple[int, int, list[int]]:
        try:
            info = sd.query_devices(device_id)
        except Exception as exc:
            raise RuntimeError(f"query device failed: {exc}") from exc
        max_out = int(info.get("max_output_channels", 0))
        if max_out <= 0:
            raise RuntimeError("device has no output channels")
        candidate_channels: list[int] = [2, 1] if max_out >= 2 else [1]
        default_rate = int(round(float(info.get("default_samplerate", 0))))
        defaults = [default_rate] if default_rate > 0 else []
        candidates = _ordered_output_rates(preferred_rate, defaults)
        errors: list[str] = []
        for rate in candidates:
            for channels in candidate_channels:
                try:
                    sd.check_output_settings(
                        device=device_id,
                        channels=channels,
                        samplerate=rate,
                        dtype="float32",
                    )
                    return rate, channels, candidates
                except Exception as exc:
                    errors.append(f"{rate}Hz/{channels}ch:{exc}")
                    continue
        detail = "; ".join(errors) if errors else "no compatible output config"
        raise RuntimeError(f"device={device_id}, errors={detail}")

    def _ensure_output_streams(self, sample_rate: int) -> None:
        if self._output_streams:
            return
        import sounddevice as sd

        device_ids: list[int] = []
        if self.settings.virtual_mic_device_id is not None:
            device_ids.append(self.settings.virtual_mic_device_id)
        if self.settings.monitor_enabled and self.settings.monitor_device_id is not None:
            if self.settings.monitor_device_id not in device_ids:
                device_ids.append(self.settings.monitor_device_id)
        if not device_ids:
            return

        first_rate: int | None = None
        for dev_id in device_ids:
            try:
                out_rate, out_channels, candidates = self._resolve_single_output_stream_config(
                    sd,
                    device_id=dev_id,
                    preferred_rate=sample_rate,
                )
                stream = sd.OutputStream(
                    samplerate=out_rate,
                    channels=out_channels,
                    dtype="float32",
                    device=dev_id,
                    blocksize=512,
                    latency="low",
                )
                stream.start()
                self._output_streams.append((stream, int(dev_id), int(out_rate), int(out_channels)))
                if first_rate is None:
                    first_rate = int(out_rate)

                if out_rate != sample_rate:
                    msg = (
                        f"output samplerate fallback: requested={sample_rate}, "
                        f"device={dev_id}, using={out_rate}, tried={candidates}"
                    )
                    logger.warning(msg)
            except Exception as exc:
                msg = f"output stream unavailable on device={dev_id}: {exc}"
                logger.warning(msg)
                self._emit("notice", {"message": msg})
                continue

        self._output_sample_rate = first_rate
        if not self._output_streams:
            self._emit("error", {"message": "No available output stream for monitor/virtual mic."})

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if status:
            self._emit("error", {"message": str(status)})
        frame = np.asarray(indata[:, 0], dtype=np.float32).copy()
        frame = _apply_gain_db(frame, float(self.settings.input_gain_db))
        self._emit_audio_level(input_level=_normalized_level(frame))
        with self._lock:
            ptt_enabled = bool(self.settings.ptt_enabled)
            ptt_active = bool(self._ptt_active)
        if ptt_enabled and not ptt_active:
            return
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            self._mark_dropped_frame()

    def _submit_segment(self, segment: np.ndarray, *, session_id: int) -> None:
        if not self._session_active(session_id):
            return
        try:
            self._segment_queue.put(segment, timeout=0.2)
        except queue.Full:
            self._mark_dropped_segment()
            self._emit(
                "error",
                {
                    "message": "segment queue is full; dropping segment",
                    "session_id": session_id,
                },
            )

    def _worker_loop(self, session_id: int) -> None:
        while self._session_active(session_id):
            try:
                frame = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if not self._session_active(session_id):
                break
            with self._segmenter_lock:
                segments = self._segmenter.push(frame)
            for seg in segments:
                if not self._session_active(session_id):
                    break
                self._submit_or_buffer_ptt_segment(seg, session_id=session_id)
        if self._session_active(session_id):
            with self._segmenter_lock:
                flushed = list(self._segmenter.flush())
            for seg in flushed:
                if not self._session_active(session_id):
                    break
                self._submit_or_buffer_ptt_segment(seg, session_id=session_id)

    def _infer_loop(self, session_id: int) -> None:
        while self._session_active(session_id):
            try:
                segment = self._segment_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if not self._session_active(session_id):
                break
            try:
                self._process_segment(segment, session_id=session_id)
            except Exception as exc:
                logger.exception("realtime infer error: %s", exc)
                self._emit(
                    "error",
                    {
                        "message": f"infer pipeline error: {exc}",
                        "session_id": session_id,
                    },
                )

    def _playback_loop(self, session_id: int) -> None:
        while self._session_active(session_id):
            try:
                wav, sr, req_session_id = self._playback_queue.get(timeout=_PLAYBACK_IDLE_TIMEOUT_S)
            except queue.Empty:
                try:
                    self._emit_idle_ambience(session_id)
                except Exception as exc:
                    logger.exception("idle ambience error: %s", exc)
                continue
            if req_session_id != session_id:
                continue
            if not self._session_active(session_id):
                break
            try:
                self._play_audio(wav, sr, session_id=session_id)
            except Exception as exc:
                logger.exception("realtime playback error: %s", exc)
                self._emit(
                    "error",
                    {
                        "message": f"playback pipeline error: {exc}",
                        "session_id": session_id,
                    },
                )

    def _should_emit_idle_ambience(self) -> bool:
        if self.config.fake_mode:
            return False
        with self._lock:
            if not bool(self.settings.ambience_enabled):
                return False
            if float(self.settings.ambience_level) <= 1e-4:
                return False
            return bool(
                self.settings.virtual_mic_device_id is not None
                or (
                    bool(self.settings.monitor_enabled)
                    and self.settings.monitor_device_id is not None
                )
            )

    def _render_idle_ambience(self, *, sample_rate: int, frames: int) -> np.ndarray:
        level = float(np.clip(self.settings.ambience_level, 0.0, _MAX_AMBIENCE_LEVEL))
        prev_state = float(self._ambience_state_by_sr.get(int(sample_rate), 0.0))
        wav, next_state = _continuous_ambience_chunk(
            frames,
            level=level,
            rng=self._noise_rng,
            prev_state=prev_state,
        )
        self._ambience_state_by_sr[int(sample_rate)] = float(next_state)
        wav = _apply_gain_db(wav, float(self.settings.output_gain_db))
        wav = _apply_soft_limiter(wav)
        return wav.astype(np.float32)

    def _emit_idle_ambience(self, session_id: int) -> None:
        if not self._session_active(session_id):
            return
        if not self._should_emit_idle_ambience():
            return

        preferred_out_sr = self._output_sample_rate or self.config.tts_sample_rate
        if not self._output_streams:
            now = time.perf_counter()
            if (now - self._last_output_open_attempt_ts) < _OUTPUT_STREAM_RETRY_INTERVAL_S:
                return
            self._last_output_open_attempt_ts = now
            self._ensure_output_streams(max(1, int(preferred_out_sr)))
            if not self._output_streams:
                return

        for stream, _dev_id, dev_sr, dev_ch in self._output_streams:
            frames = max(1, int((dev_sr * _IDLE_AMBIENCE_CHUNK_MS) / 1000.0))
            mono = self._render_idle_ambience(sample_rate=dev_sr, frames=frames)
            if mono.size == 0:
                continue
            self._emit_audio_level(output_level=_normalized_level(mono))
            if dev_ch >= 2:
                block = np.repeat(mono.reshape(-1, 1), dev_ch, axis=1).astype(np.float32)
            else:
                block = mono.reshape(-1, 1).astype(np.float32)
            stream.write(block)

    def _tts_flush_loop(self, session_id: int) -> None:
        while self._session_active(session_id):
            try:
                self._flush_tts_buffer_if_due(session_id=session_id, force=False)
            except Exception as exc:
                logger.exception("tts flush loop error: %s", exc)
            time.sleep(0.03)

    def _flush_tts_buffer_if_due(self, *, session_id: int, force: bool) -> None:
        text = ""
        voice_id: str | None = None
        language = "Auto"
        prompt_path: Path | None = None
        ref_text: str | None = None
        t0 = 0.0
        t_asr = 0.0

        with self._tts_buffer_lock:
            if (
                not self._tts_buffer_text
                or self._tts_buffer_session_id != session_id
                or self._tts_buffer_prompt_path is None
            ):
                return
            age_s = time.perf_counter() - self._tts_buffer_updated_ts
            should_flush = force
            if not should_flush:
                if age_s >= _TTS_BUFFER_HARD_WAIT_S:
                    should_flush = True
                elif age_s >= _TTS_BUFFER_SOFT_WAIT_S and (
                    len(self._tts_buffer_text) >= 10
                    or _ends_with_terminal_punctuation(self._tts_buffer_text)
                ):
                    should_flush = True
            if not should_flush:
                return

            text = self._tts_buffer_text
            voice_id = self._tts_buffer_voice_id
            language = self._tts_buffer_language
            prompt_path = self._tts_buffer_prompt_path
            ref_text = self._tts_buffer_ref_text
            t0 = self._tts_buffer_first_t0
            t_asr = self._tts_buffer_last_asr_t

            self._tts_buffer_text = ""
            self._tts_buffer_voice_id = None
            self._tts_buffer_language = "Auto"
            self._tts_buffer_prompt_path = None
            self._tts_buffer_ref_text = None
            self._tts_buffer_first_t0 = 0.0
            self._tts_buffer_last_asr_t = 0.0
            self._tts_buffer_updated_ts = 0.0
            self._tts_buffer_session_id = 0

        if not text or voice_id is None or prompt_path is None:
            return
        self._synthesize_buffered_text(
            text=text,
            voice_id=voice_id,
            language=language,
            prompt_path=prompt_path,
            ref_text=ref_text,
            session_id=session_id,
            t0=t0,
            t_asr=t_asr,
        )

    def _append_tts_buffer(
        self,
        *,
        text: str,
        voice_id: str,
        language: str,
        prompt_path: Path,
        ref_text: str | None,
        session_id: int,
        t0: float,
        t_asr: float,
    ) -> None:
        normalized_text = " ".join(str(text).split()).strip()
        if not normalized_text:
            return
        context_switched = False
        flush_now = False
        with self._tts_buffer_lock:
            if (
                self._tts_buffer_text
                and (
                    self._tts_buffer_session_id != session_id
                    or self._tts_buffer_voice_id != voice_id
                    or self._tts_buffer_prompt_path != prompt_path
                )
            ):
                context_switched = True
            else:
                if not self._tts_buffer_text:
                    self._tts_buffer_text = normalized_text
                    self._tts_buffer_voice_id = voice_id
                    self._tts_buffer_language = language
                    self._tts_buffer_prompt_path = Path(prompt_path)
                    self._tts_buffer_ref_text = ref_text
                    self._tts_buffer_first_t0 = t0
                    self._tts_buffer_last_asr_t = t_asr
                    self._tts_buffer_updated_ts = time.perf_counter()
                    self._tts_buffer_session_id = session_id
                else:
                    self._tts_buffer_text = _merge_tts_text(
                        self._tts_buffer_text, normalized_text
                    )
                    self._tts_buffer_last_asr_t = max(self._tts_buffer_last_asr_t, t_asr)
                    self._tts_buffer_updated_ts = time.perf_counter()
                if _ends_with_terminal_punctuation(normalized_text):
                    flush_now = True
                elif len(self._tts_buffer_text) >= _TTS_BUFFER_FORCE_CHARS:
                    flush_now = True

        if context_switched:
            self._flush_tts_buffer_if_due(session_id=session_id, force=True)
            with self._tts_buffer_lock:
                if not self._tts_buffer_text:
                    self._tts_buffer_text = normalized_text
                    self._tts_buffer_voice_id = voice_id
                    self._tts_buffer_language = language
                    self._tts_buffer_prompt_path = Path(prompt_path)
                    self._tts_buffer_ref_text = ref_text
                    self._tts_buffer_first_t0 = t0
                    self._tts_buffer_last_asr_t = t_asr
                    self._tts_buffer_updated_ts = time.perf_counter()
                    self._tts_buffer_session_id = session_id
                else:
                    self._tts_buffer_text = _merge_tts_text(
                        self._tts_buffer_text, normalized_text
                    )
                    self._tts_buffer_last_asr_t = max(self._tts_buffer_last_asr_t, t_asr)
                    self._tts_buffer_updated_ts = time.perf_counter()
                if _ends_with_terminal_punctuation(normalized_text):
                    flush_now = True
                elif len(self._tts_buffer_text) >= _TTS_BUFFER_FORCE_CHARS:
                    flush_now = True

        if flush_now:
            self._flush_tts_buffer_if_due(session_id=session_id, force=True)

    def _synthesize_buffered_text(
        self,
        *,
        text: str,
        voice_id: str,
        language: str,
        prompt_path: Path,
        ref_text: str | None,
        session_id: int,
        t0: float,
        t_asr: float,
    ) -> None:
        if not self._session_active(session_id):
            return
        self._emit("tts_started", {"text": text})

        wav, sr = self.tts_manager.synthesize_with_prompt_path(
            text=text,
            language=language,
            prompt_path=Path(prompt_path),
            ref_text_override=ref_text,
        )
        if not self._session_active(session_id):
            return
        t_tts = time.perf_counter()
        self._enqueue_playback(wav.astype(np.float32), sr, session_id=session_id)
        if not self._session_active(session_id):
            return
        t_end = time.perf_counter()
        dropped, dropped_frames, dropped_segments = self._capture_drop_state()
        playback_queue_size = int(self._playback_queue.qsize())
        effective_t0 = t0 if t0 > 0 else t_asr
        effective_asr = t_asr if t_asr >= effective_t0 else effective_t0
        metrics = self._metrics.record(
            asr_ms=(effective_asr - effective_t0) * 1000.0,
            tts_ms=(t_tts - effective_asr) * 1000.0,
            fad_ms=(t_tts - effective_t0) * 1000.0,
            e2e_ms=(t_end - effective_t0) * 1000.0,
            dropped=dropped,
            extra={
                "dropped_frames": dropped_frames,
                "dropped_segments": dropped_segments,
                "playback_queue_size": playback_queue_size,
                "voice_id": voice_id,
            },
        )
        self.db.record_metric(metrics.updated_at, metrics.model_dump())
        self._emit("tts_first_packet", {"text": text, "fad_ms": metrics.p50_fad_ms})
        self._emit("latency_tick", metrics.model_dump())

    def _synthesize_without_session(
        self,
        *,
        text: str,
        voice_id: str,
        language: str,
        prompt_path: Path,
        ref_text: str | None,
        t0: float,
        t_asr: float,
    ) -> None:
        self._emit("tts_started", {"text": text})
        wav, sr = self.tts_manager.synthesize_with_prompt_path(
            text=text,
            language=language,
            prompt_path=Path(prompt_path),
            ref_text_override=ref_text,
        )
        t_tts = time.perf_counter()
        self._play_audio(wav.astype(np.float32), sr, session_id=None)
        t_end = time.perf_counter()
        dropped, dropped_frames, dropped_segments = self._capture_drop_state()
        metrics = self._metrics.record(
            asr_ms=(t_asr - t0) * 1000.0,
            tts_ms=(t_tts - t_asr) * 1000.0,
            fad_ms=(t_tts - t0) * 1000.0,
            e2e_ms=(t_end - t0) * 1000.0,
            dropped=dropped,
            extra={
                "dropped_frames": dropped_frames,
                "dropped_segments": dropped_segments,
                "playback_queue_size": 0,
                "voice_id": voice_id,
            },
        )
        self.db.record_metric(metrics.updated_at, metrics.model_dump())
        self._emit("tts_first_packet", {"text": text, "fad_ms": metrics.p50_fad_ms})
        self._emit("latency_tick", metrics.model_dump())

    def _enqueue_playback(
        self, wav: np.ndarray, sr: int, *, session_id: int | None = None
    ) -> None:
        if session_id is None:
            self._play_audio(wav, sr, session_id=None)
            return
        if not self._session_active(session_id):
            return
        try:
            self._playback_queue.put((wav.astype(np.float32), int(sr), int(session_id)), timeout=0.2)
        except queue.Full:
            self._emit(
                "error",
                {
                    "message": "playback queue is full; dropping synthesized audio",
                    "session_id": session_id,
                },
            )

    def _process_segment(self, segment: np.ndarray, *, session_id: int | None = None) -> None:
        if segment.size < int(self._capture_sample_rate * 0.12):
            return
        if session_id is not None and not self._session_active(session_id):
            return
        seg_rms, seg_peak, seg_dur_ms = _segment_stats(segment, self._capture_sample_rate)
        if _is_low_energy_segment(segment, self._capture_sample_rate):
            return
        with self._lock:
            monitor_enabled = bool(self.settings.monitor_enabled)
            input_dev = self.settings.input_device_id
            monitor_dev = self.settings.monitor_device_id
        if (
            monitor_enabled
            and self._has_monitor_feedback_risk(input_dev, monitor_dev)
            and (time.perf_counter() - self._last_output_activity_ts) <= _OUTPUT_FEEDBACK_SUPPRESS_WINDOW_S
            and seg_peak < 0.14
            and seg_rms < 0.02
        ):
            return
        self._emit_audio_level(input_level=_normalized_level(segment))
        asr_audio = self._prepare_asr_audio(segment)
        if asr_audio.size < int(_ASR_TARGET_SAMPLE_RATE * 0.12):
            return
        speech_ratio = _speech_activity_ratio(asr_audio)
        if speech_ratio < _MIN_SPEECH_ACTIVITY_RATIO and seg_peak < 0.09:
            return
        with self._lock:
            voice_id = self._voice_id
            language = self._language
            ptt_enabled = bool(self.settings.ptt_enabled)
        if voice_id is None:
            return
        voice = self.voice_service.get_voice(voice_id)
        if voice is None:
            self._emit("error", {"message": f"Voice disappeared during runtime: {voice_id}"})
            return

        t0 = time.perf_counter()
        partial_text = self.asr_stage1.transcribe_partial(asr_audio, language)
        if session_id is not None and not self._session_active(session_id):
            return
        if partial_text:
            self._emit("asr_partial", {"text": partial_text})

        final_text = self.asr_stage2.transcribe_final(asr_audio, language)
        if session_id is not None and not self._session_active(session_id):
            return
        if not final_text:
            return
        normalized_text = " ".join(str(final_text).split()).strip()
        if not normalized_text:
            return
        rms, peak, dur_ms = seg_rms, seg_peak, seg_dur_ms
        text_key = _normalize_text_key(normalized_text)
        if _is_low_information_text(normalized_text):
            suppress_low_info = False
            if dur_ms <= 1400.0:
                suppress_low_info = True
            if rms < 0.012 and peak < 0.08:
                suppress_low_info = True
            if (
                text_key
                and text_key == self._last_low_info_text
                and (time.perf_counter() - self._last_low_info_ts) <= _LOW_INFO_SUPPRESS_WINDOW_S
            ):
                suppress_low_info = True
            self._last_low_info_text = text_key
            self._last_low_info_ts = time.perf_counter()
            if suppress_low_info:
                logger.info("ASR low-info text suppressed: %s", normalized_text)
                return
        now = time.perf_counter()
        if (
            (not self.config.fake_mode)
            and normalized_text
            and normalized_text == self._last_final_text
            and (now - self._last_final_ts) <= _GENERAL_REPEAT_SUPPRESS_WINDOW_S
        ):
            self._emit(
                "notice",
                {"message": f"ASR repeated text suppressed: {normalized_text}"},
            )
            return
        self._last_final_text = normalized_text
        self._last_final_ts = now
        self._emit("asr_final", {"text": normalized_text})

        corrected_text = normalized_text
        if bool(self.settings.llm_correction_enabled):
            corrected_text = self._maybe_correct_final_text(normalized_text, language)
        if not corrected_text:
            corrected_text = normalized_text
        t_asr = time.perf_counter()
        if session_id is None:
            self._synthesize_without_session(
                text=corrected_text,
                voice_id=voice.id,
                language=language,
                prompt_path=Path(voice.prompt_path),
                ref_text=voice.ref_text,
                t0=t0,
                t_asr=t_asr,
            )
            return

        if self.config.fake_mode:
            self._synthesize_buffered_text(
                text=corrected_text,
                voice_id=voice.id,
                language=language,
                prompt_path=Path(voice.prompt_path),
                ref_text=voice.ref_text,
                session_id=session_id,
                t0=t0,
                t_asr=t_asr,
            )
            return

        if ptt_enabled:
            self._synthesize_buffered_text(
                text=corrected_text,
                voice_id=voice.id,
                language=language,
                prompt_path=Path(voice.prompt_path),
                ref_text=voice.ref_text,
                session_id=session_id,
                t0=t0,
                t_asr=t_asr,
            )
            return

        self._append_tts_buffer(
            text=corrected_text,
            voice_id=voice.id,
            language=language,
            prompt_path=Path(voice.prompt_path),
            ref_text=voice.ref_text,
            session_id=session_id,
            t0=t0,
            t_asr=t_asr,
        )

    def _maybe_correct_final_text(self, text: str, language: str) -> str:
        if self.text_corrector is None:
            return text
        source = " ".join(str(text or "").split()).strip()
        if not source:
            return ""
        # Keep short utterances on the fast path.
        if len(source) <= 2:
            return source
        try:
            result = self.text_corrector.correct_text(source, language_hint=language)
        except Exception as exc:
            logger.warning("text correction failed; keep original text: %s", exc)
            self._emit("notice", {"message": f"LLM text correction failed: {exc}"})
            return source

        if result.warning:
            self._emit("notice", {"message": result.warning})
        corrected = " ".join(str(result.text or "").split()).strip() or source
        if corrected != source:
            self._emit(
                "asr_corrected",
                {
                    "before": source,
                    "text": corrected,
                    "used_model": bool(result.used_model),
                },
            )
        return corrected

    def _prepare_asr_audio(self, segment: np.ndarray) -> np.ndarray:
        mono = np.asarray(segment, dtype=np.float32).reshape(-1)
        if mono.size == 0:
            return mono
        if self._capture_sample_rate == _ASR_TARGET_SAMPLE_RATE:
            asr_audio = mono
        else:
            from scipy.signal import resample_poly

            asr_audio = resample_poly(
                mono, _ASR_TARGET_SAMPLE_RATE, self._capture_sample_rate
            ).astype(np.float32)
        if self.settings.denoise_enabled:
            asr_audio = _soft_noise_suppress(
                asr_audio,
                sample_rate=_ASR_TARGET_SAMPLE_RATE,
                strength=float(self.settings.denoise_strength),
            )
        asr_audio = _trim_silence(asr_audio, _ASR_TARGET_SAMPLE_RATE)
        return asr_audio.astype(np.float32)

    def _play_audio(
        self, wav: np.ndarray, sample_rate: int, *, session_id: int | None = None
    ) -> None:
        wav = _sanitize_output_wav(wav)
        wav, peak, gain = _apply_low_level_gain(wav)
        wav = _apply_gain_db(wav, float(self.settings.output_gain_db))
        wav = _apply_soft_limiter(wav)
        if self.settings.ambience_enabled:
            wav = _mix_ambience_noise(
                wav,
                sample_rate=max(sample_rate, 1),
                level=float(self.settings.ambience_level),
                rng=self._noise_rng,
            )
        if wav.size == 0 or peak <= 1e-8:
            msg = (
                "TTS produced silent audio. "
                "Current voice prompt is likely invalid; recreate this voice in real mode."
            )
            logger.warning("%s peak=%s", msg, peak)
            self._emit("error", {"message": msg, "peak": peak})
            self._emit_audio_level(output_level=0.0, force=True)
            return
        if gain > 1.0:
            logger.info("low-level TTS output auto-gain applied: gain=%.2f peak=%.6f", gain, peak)

        output_level = _normalized_level(wav)
        self._emit_audio_level(output_level=output_level)
        if output_level > 0.02:
            self._last_output_activity_ts = time.perf_counter()
        if self.config.fake_mode:
            return
        if session_id is not None and not self._session_active(session_id):
            return
        configured_sr = max(1, int(self.config.tts_sample_rate))
        preferred_out_sr = self._output_sample_rate or configured_sr
        if preferred_out_sr <= 0:
            preferred_out_sr = sample_rate if sample_rate > 0 else 48000
        self._ensure_output_streams(preferred_out_sr)
        if not self._output_streams:
            return
        prepared: list[tuple[Any, np.ndarray]] = []
        for stream, dev_id, dev_sr, dev_ch in self._output_streams:
            dev_wav = wav
            if sample_rate != dev_sr:
                from scipy.signal import resample_poly

                dev_wav = resample_poly(dev_wav, dev_sr, sample_rate).astype(np.float32)
            dev_wav = _apply_edge_fade(dev_wav, dev_sr)
            if dev_ch >= 2:
                block_arr = np.repeat(dev_wav.reshape(-1, 1), dev_ch, axis=1).astype(np.float32)
            else:
                block_arr = dev_wav.reshape(-1, 1).astype(np.float32)
            prepared.append((stream, block_arr))

        chunk = 512
        for stream, block_arr in prepared:
            for start in range(0, len(block_arr), chunk):
                if session_id is not None and not self._session_active(session_id):
                    return
                block = block_arr[start : start + chunk]
                self._emit_audio_level(output_level=_normalized_level(block[:, 0]))
                stream.write(block)

