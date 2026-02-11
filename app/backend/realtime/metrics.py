from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import json

import numpy as np

from app.backend.types import RealtimeMetrics


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


@dataclass
class SlidingRealtimeMetrics:
    buffer_size: int
    log_path: Path
    fad_values: deque[float] = field(init=False)
    e2e_values: deque[float] = field(init=False)
    asr_values: deque[float] = field(init=False)
    tts_values: deque[float] = field(init=False)
    dropped: int = 0
    total: int = 0

    def __post_init__(self) -> None:
        self.fad_values = deque(maxlen=self.buffer_size)
        self.e2e_values = deque(maxlen=self.buffer_size)
        self.asr_values = deque(maxlen=self.buffer_size)
        self.tts_values = deque(maxlen=self.buffer_size)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def _percentile(self, values: deque[float], p: float) -> float:
        if not values:
            return 0.0
        return float(np.percentile(np.array(values, dtype=np.float32), p))

    def record(
        self,
        *,
        asr_ms: float,
        tts_ms: float,
        fad_ms: float,
        e2e_ms: float,
        dropped: bool = False,
        extra: dict | None = None,
    ) -> RealtimeMetrics:
        self.total += 1
        if dropped:
            self.dropped += 1
        self.asr_values.append(float(asr_ms))
        self.tts_values.append(float(tts_ms))
        self.fad_values.append(float(fad_ms))
        self.e2e_values.append(float(e2e_ms))

        payload = {
            "ts": _utc_now(),
            "asr_ms": asr_ms,
            "tts_ms": tts_ms,
            "fad_ms": fad_ms,
            "e2e_ms": e2e_ms,
            "dropped": dropped,
            "extra": extra or {},
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
        return self.snapshot()

    def snapshot(self) -> RealtimeMetrics:
        drop_rate = (self.dropped / self.total) if self.total > 0 else 0.0
        return RealtimeMetrics(
            p50_fad_ms=self._percentile(self.fad_values, 50),
            p95_fad_ms=self._percentile(self.fad_values, 95),
            p50_e2e_ms=self._percentile(self.e2e_values, 50),
            p95_e2e_ms=self._percentile(self.e2e_values, 95),
            asr_ms=self._percentile(self.asr_values, 50),
            tts_ms=self._percentile(self.tts_values, 50),
            drop_rate=drop_rate,
            sample_count=self.total,
            updated_at=_utc_now(),
        )

    def reset(self, clear_log: bool = False) -> None:
        self.fad_values.clear()
        self.e2e_values.clear()
        self.asr_values.clear()
        self.tts_values.clear()
        self.dropped = 0
        self.total = 0
        if clear_log:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text("", encoding="utf-8")
