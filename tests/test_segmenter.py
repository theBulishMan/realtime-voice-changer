from __future__ import annotations

import numpy as np
import pytest

from app.backend.audio.segmenter import VadSegmenter


def test_segmenter_uses_energy_fallback_when_vad_misses_speech() -> None:
    seg = VadSegmenter(
        sample_rate=16000,
        frame_ms=20,
        vad_silence_ms=100,
        max_segment_ms=4000,
        energy_threshold=0.01,
    )
    seg._vad.is_speech = lambda *_args, **_kwargs: False  # type: ignore[method-assign]

    frame_len = int(16000 * 0.02)
    speech = np.full(frame_len, 0.05, dtype=np.float32)
    silence = np.zeros(frame_len, dtype=np.float32)

    segments: list[np.ndarray] = []
    for _ in range(5):
        segments.extend(seg.push(speech))
    for _ in range(5):
        segments.extend(seg.push(silence))

    assert segments
    assert segments[0].size >= frame_len


def test_segmenter_does_not_trigger_on_very_low_energy_noise() -> None:
    seg = VadSegmenter(
        sample_rate=16000,
        frame_ms=20,
        vad_silence_ms=100,
        max_segment_ms=4000,
        energy_threshold=0.01,
    )
    seg._vad.is_speech = lambda *_args, **_kwargs: False  # type: ignore[method-assign]

    frame_len = int(16000 * 0.02)
    noise = np.full(frame_len, 0.0015, dtype=np.float32)

    segments: list[np.ndarray] = []
    for _ in range(20):
        segments.extend(seg.push(noise))

    assert segments == []
    assert seg.flush() == []


def test_segmenter_rejects_invalid_energy_threshold() -> None:
    with pytest.raises(ValueError, match="energy_threshold"):
        VadSegmenter(sample_rate=16000, energy_threshold=1.2)
