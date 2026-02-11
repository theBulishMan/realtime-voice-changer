from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import webrtcvad


@dataclass
class VadSegmenter:
    sample_rate: int
    frame_ms: int = 20
    vad_silence_ms: int = 240
    max_segment_ms: int = 12000
    vad_mode: int = 2
    energy_threshold: float = 0.012
    _vad: webrtcvad.Vad = field(init=False)
    _speech_active: bool = field(default=False, init=False)
    _silence_ms: int = field(default=0, init=False)
    _buffer: list[np.ndarray] = field(default_factory=list, init=False)
    _segment_ms: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.sample_rate not in {8000, 16000, 32000, 48000}:
            raise ValueError("WebRTC VAD requires sample_rate in {8000, 16000, 32000, 48000}")
        if self.frame_ms not in {10, 20, 30}:
            raise ValueError("WebRTC VAD requires frame_ms in {10, 20, 30}")
        if not (0.0 <= self.energy_threshold <= 1.0):
            raise ValueError("energy_threshold must be in [0.0, 1.0]")
        self._vad = webrtcvad.Vad(self.vad_mode)

    def _as_pcm_bytes(self, frame: np.ndarray) -> bytes:
        pcm = np.clip(frame, -1.0, 1.0)
        pcm16 = (pcm * 32767.0).astype(np.int16)
        return pcm16.tobytes()

    @staticmethod
    def _rms(frame: np.ndarray) -> float:
        mono = np.asarray(frame, dtype=np.float32).reshape(-1)
        if mono.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(mono), dtype=np.float64)))

    def push(self, frame: np.ndarray) -> list[np.ndarray]:
        frame = np.asarray(frame, dtype=np.float32).reshape(-1)
        pcm_bytes = self._as_pcm_bytes(frame)
        vad_speech = self._vad.is_speech(pcm_bytes, self.sample_rate)
        # VAD can miss soft speech on some consumer microphones; use RMS as fallback.
        is_speech = vad_speech or (self._rms(frame) >= self.energy_threshold)
        segments: list[np.ndarray] = []

        if is_speech and not self._speech_active:
            self._speech_active = True
            self._silence_ms = 0

        if self._speech_active:
            self._buffer.append(frame.copy())
            self._segment_ms += self.frame_ms
            if is_speech:
                self._silence_ms = 0
            else:
                self._silence_ms += self.frame_ms

            if self._silence_ms >= self.vad_silence_ms or self._segment_ms >= self.max_segment_ms:
                segments.append(np.concatenate(self._buffer).astype(np.float32))
                self._buffer.clear()
                self._speech_active = False
                self._silence_ms = 0
                self._segment_ms = 0

        return segments

    def flush(self) -> list[np.ndarray]:
        if not self._buffer:
            return []
        segment = np.concatenate(self._buffer).astype(np.float32)
        self._buffer.clear()
        self._speech_active = False
        self._silence_ms = 0
        self._segment_ms = 0
        return [segment]
