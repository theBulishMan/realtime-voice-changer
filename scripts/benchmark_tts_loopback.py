from __future__ import annotations

import argparse
import io
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from urllib import error, request

import numpy as np


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def http_json(method: str, url: str, payload: dict | None = None, timeout: float = 30.0) -> dict:
    body = None
    headers: dict[str, str] = {}
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = request.Request(url=url, data=body, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            if not raw.strip():
                return {}
            return json.loads(raw)
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} -> {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"{method} {url} -> network error: {exc}") from exc


def _multipart_body(fields: dict[str, str], files: dict[str, tuple[str, str, bytes]]) -> tuple[bytes, str]:
    boundary = f"----rvc-{uuid.uuid4().hex}"
    chunks: list[bytes] = []

    for key, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            f'Content-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'.encode("utf-8")
        )

    for key, (filename, content_type, content) in files.items():
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            (
                f'Content-Disposition: form-data; name="{key}"; filename="{filename}"\r\n'
                f"Content-Type: {content_type}\r\n\r\n"
            ).encode("utf-8")
        )
        chunks.append(content)
        chunks.append(b"\r\n")

    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))
    body = b"".join(chunks)
    return body, f"multipart/form-data; boundary={boundary}"


def http_multipart(
    url: str,
    *,
    fields: dict[str, str],
    files: dict[str, tuple[str, str, bytes]],
    timeout: float = 60.0,
) -> dict:
    body, content_type = _multipart_body(fields=fields, files=files)
    req = request.Request(
        url=url,
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            if not raw.strip():
                return {}
            return json.loads(raw)
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} -> {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"POST {url} -> network error: {exc}") from exc


def make_bootstrap_wav_bytes(duration_sec: float = 4.0, sample_rate: int = 24000) -> bytes:
    import soundfile as sf

    frames = max(1, int(duration_sec * sample_rate))
    t = np.linspace(0.0, duration_sec, frames, endpoint=False, dtype=np.float32)
    # Mixed sine for stable voiced-like energy.
    wav = (0.08 * np.sin(2.0 * np.pi * 180.0 * t) + 0.04 * np.sin(2.0 * np.pi * 320.0 * t)).astype(
        np.float32
    )
    buf = io.BytesIO()
    sf.write(buf, wav, sample_rate, format="WAV")
    return buf.getvalue()


def choose_voice(voices: list[dict], name_hint: str) -> dict | None:
    normalized = name_hint.strip().lower()
    for voice in voices:
        if str(voice.get("name", "")).strip().lower() == normalized:
            return voice
    return None


def resolve_status(sample_count: int, min_samples: int, p95_fad: float, p95_e2e: float, target_fad: float, target_e2e: float) -> str:
    if sample_count < min_samples:
        return "UNVERIFIED"
    if p95_fad <= target_fad and p95_e2e <= target_e2e:
        return "PASS"
    return "FAIL"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TTS audio and feed into realtime pipeline for latency benchmark.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8787")
    parser.add_argument("--voice-name", default="loopback_voice")
    parser.add_argument("--language", default="Auto")
    parser.add_argument(
        "--input-text",
        default="这是一次实时链路延迟测试。This is a realtime latency loopback benchmark.",
    )
    parser.add_argument("--create-voice-if-missing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--inject-mode", choices=["auto", "queue", "segment"], default="auto")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--inject-batches", type=int, default=8)
    parser.add_argument("--batch-pause-ms", type=int, default=150)
    parser.add_argument("--append-silence-ms", type=int, default=300)
    parser.add_argument("--realtime-pacing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vad-silence-ms", type=int, default=240)
    parser.add_argument("--max-segment-ms", type=int, default=12000)
    parser.add_argument("--min-samples", type=int, default=6)
    parser.add_argument("--max-wait-seconds", type=float, default=45.0)
    parser.add_argument("--poll-interval", type=float, default=0.5)
    parser.add_argument("--target-fad-ms", type=float, default=250.0)
    parser.add_argument("--target-e2e-ms", type=float, default=800.0)
    parser.add_argument(
        "--voice-create-timeout-seconds",
        type=float,
        default=0.0,
        help="Override timeout for creating design/clone voice. 0 means auto by mode.",
    )
    parser.add_argument(
        "--preview-timeout-seconds",
        type=float,
        default=0.0,
        help="Override timeout for /voices/{id}/preview. 0 means auto by mode.",
    )
    parser.add_argument(
        "--inject-timeout-seconds",
        type=float,
        default=0.0,
        help="Override timeout for /realtime/inject requests. 0 means auto from max-wait.",
    )
    parser.add_argument("--strict", action="store_true", default=False)
    parser.add_argument("--report-json", default="reports/tts_loopback_latency.json")
    parser.add_argument("--reset-metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--monitor-enabled", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    created_voice_id: str | None = None
    started = False

    def event(message: str) -> None:
        print(f"[{_now()}] {message}")

    try:
        health = http_json("GET", f"{base_url}/api/v1/health")
        fake_mode = bool(health.get("fake_mode"))
        event(f"health: status={health.get('status')} fake_mode={fake_mode}")

        create_timeout = (
            args.voice_create_timeout_seconds
            if args.voice_create_timeout_seconds > 0
            else (180.0 if fake_mode else 480.0)
        )
        preview_timeout = (
            args.preview_timeout_seconds
            if args.preview_timeout_seconds > 0
            else (180.0 if fake_mode else 600.0)
        )
        inject_timeout = (
            args.inject_timeout_seconds
            if args.inject_timeout_seconds > 0
            else max(20.0, args.max_wait_seconds + 10.0)
        )

        if args.reset_metrics:
            reset = http_json("POST", f"{base_url}/api/v1/metrics/reset")
            event(f"metrics reset: sample_count={reset.get('metrics', {}).get('sample_count', 'n/a')}")

        devices = http_json("GET", f"{base_url}/api/v1/audio/devices")
        settings_payload = {
            "input_device_id": devices.get("default_input_id"),
            "monitor_device_id": devices.get("default_output_id"),
            "virtual_mic_device_id": devices.get("virtual_mic_output_id"),
            "monitor_enabled": bool(args.monitor_enabled),
            "vad_silence_ms": args.vad_silence_ms,
            "max_segment_ms": args.max_segment_ms,
        }
        http_json("PATCH", f"{base_url}/api/v1/realtime/settings", settings_payload)
        event("realtime settings applied")

        voices = http_json("GET", f"{base_url}/api/v1/voices")
        selected = choose_voice(voices, args.voice_name)
        if selected is None:
            if not args.create_voice_if_missing:
                raise RuntimeError("no voice found and --no-create-voice-if-missing was set")
            unique_name = f"{args.voice_name}_{int(time.time())}"
            if fake_mode:
                event(f"creating design voice: {unique_name}")
                created = http_json(
                    "POST",
                    f"{base_url}/api/v1/voices/design",
                    payload={
                        "name": unique_name,
                        "voice_prompt": "Clear, stable, neutral, natural speaking voice.",
                        "preview_text": "Loopback voice initialization.",
                        "language": args.language,
                        "save": True,
                    },
                    timeout=create_timeout,
                )
            else:
                event(f"creating clone voice: {unique_name}")
                bootstrap_audio = make_bootstrap_wav_bytes(duration_sec=4.0, sample_rate=24000)
                created = http_multipart(
                    f"{base_url}/api/v1/voices/clone",
                    fields={
                        "name": unique_name,
                        "language": args.language,
                        "ref_text": "This is a bootstrap clone reference sample.",
                    },
                    files={
                        "audio_file": ("bootstrap.wav", "audio/wav", bootstrap_audio),
                    },
                    timeout=create_timeout,
                )
            selected = created["voice"]
            created_voice_id = str(selected["id"])
        event(f"selected voice: {selected['id']} ({selected.get('name')})")

        preview = http_json(
            "POST",
            f"{base_url}/api/v1/voices/{selected['id']}/preview",
            payload={"text": args.input_text, "language": args.language},
            timeout=preview_timeout,
        )
        audio_b64 = str(preview["audio_b64"])
        event(f"tts source generated: sample_rate={preview.get('sample_rate')}")

        start = http_json(
            "POST",
            f"{base_url}/api/v1/realtime/start",
            payload={"voice_id": selected["id"], "language": args.language},
        )
        started = True
        event(f"realtime started: running={start.get('running')}")

        inject_mode = args.inject_mode
        if inject_mode == "auto":
            inject_mode = "segment" if fake_mode else "queue"
        event(f"inject mode: {inject_mode}")

        timeout_per_batch = inject_timeout
        latest_metrics = {}
        for idx in range(max(1, args.inject_batches)):
            inject_deadline = time.time() + timeout_per_batch
            while True:
                try:
                    injected = http_json(
                        "POST",
                        f"{base_url}/api/v1/realtime/inject",
                        payload={
                            "audio_b64": audio_b64,
                            "inject_mode": inject_mode,
                            "realtime_pacing": bool(args.realtime_pacing),
                            "repeat": args.repeat,
                            "pause_ms": 0,
                            "append_silence_ms": args.append_silence_ms,
                        },
                        timeout=timeout_per_batch,
                    )
                    break
                except RuntimeError as exc:
                    message = str(exc)
                    if "inject queue is full" not in message:
                        raise
                    if time.time() >= inject_deadline:
                        raise
                    event(
                        f"inject queue full for batch {idx + 1}, waiting before retry..."
                    )
                    time.sleep(max(0.5, args.poll_interval))
            event(
                "audio injected: "
                f"batch={idx + 1}/{args.inject_batches} frames={injected.get('injected_frames')} "
                f"segments={injected.get('injected_segments')} elapsed_ms={float(injected.get('elapsed_ms', 0.0)):.1f}"
            )
            latest_metrics = http_json("GET", f"{base_url}/api/v1/metrics/current")
            if int(latest_metrics.get("sample_count", 0)) >= args.min_samples:
                break
            if idx < args.inject_batches - 1 and args.batch_pause_ms > 0:
                time.sleep(args.batch_pause_ms / 1000.0)

        deadline = time.time() + max(1.0, args.max_wait_seconds)
        metrics = latest_metrics
        while time.time() < deadline:
            metrics = http_json("GET", f"{base_url}/api/v1/metrics/current")
            sample_count = int(metrics.get("sample_count", 0))
            event(
                "metrics: "
                f"samples={sample_count} p95_fad={float(metrics.get('p95_fad_ms', 0.0)):.1f}ms "
                f"p95_e2e={float(metrics.get('p95_e2e_ms', 0.0)):.1f}ms"
            )
            if sample_count >= args.min_samples:
                break
            time.sleep(max(0.2, args.poll_interval))

        sample_count = int(metrics.get("sample_count", 0))
        p95_fad = float(metrics.get("p95_fad_ms", 0.0))
        p95_e2e = float(metrics.get("p95_e2e_ms", 0.0))
        status = resolve_status(
            sample_count=sample_count,
            min_samples=args.min_samples,
            p95_fad=p95_fad,
            p95_e2e=p95_e2e,
            target_fad=args.target_fad_ms,
            target_e2e=args.target_e2e_ms,
        )

        summary = {
            "status": status,
            "base_url": base_url,
            "fake_mode": fake_mode,
            "voice_id": selected["id"],
            "created_voice_id": created_voice_id,
            "targets": {
                "p95_fad_ms": args.target_fad_ms,
                "p95_e2e_ms": args.target_e2e_ms,
                "min_samples": args.min_samples,
            },
            "result": {
                "sample_count": sample_count,
                "p95_fad_ms": p95_fad,
                "p95_e2e_ms": p95_e2e,
                "p50_fad_ms": float(metrics.get("p50_fad_ms", 0.0)),
                "p50_e2e_ms": float(metrics.get("p50_e2e_ms", 0.0)),
                "drop_rate": float(metrics.get("drop_rate", 0.0)),
            },
            "inject": {
                "inject_batches": args.inject_batches,
                "batch_pause_ms": args.batch_pause_ms,
                "inject_mode": inject_mode,
                "repeat": args.repeat,
                "realtime_pacing": bool(args.realtime_pacing),
                "append_silence_ms": args.append_silence_ms,
                "vad_silence_ms": args.vad_silence_ms,
                "max_segment_ms": args.max_segment_ms,
                "input_text_chars": len(args.input_text),
            },
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))

        if args.report_json:
            out = Path(args.report_json)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            event(f"report written: {out}")

        if started:
            http_json("POST", f"{base_url}/api/v1/realtime/stop")
            started = False
            event("realtime stopped")

        if status == "PASS":
            return 0
        if status == "UNVERIFIED":
            return 1 if args.strict else 0
        return 1
    except Exception as exc:
        print(f"loopback benchmark failed: {exc}", file=sys.stderr)
        return 2
    finally:
        if started:
            try:
                http_json("POST", f"{base_url}/api/v1/realtime/stop")
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
