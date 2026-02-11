from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib import error, request


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def http_json(
    method: str,
    url: str,
    payload: dict | None = None,
    timeout: float = 15.0,
) -> dict:
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


def choose_voice(voices: list[dict], name_hint: str) -> dict | None:
    lowered = name_hint.strip().lower()
    for voice in voices:
        name = str(voice.get("name", "")).strip().lower()
        if name == lowered:
            return voice
    return voices[0] if voices else None


def check_gate(metrics: dict, target_fad_ms: float, target_e2e_ms: float) -> dict:
    p95_fad = float(metrics.get("p95_fad_ms", 0.0))
    p95_e2e = float(metrics.get("p95_e2e_ms", 0.0))
    sample_count = int(metrics.get("sample_count", 0))
    has_samples = sample_count > 0
    return {
        "sample_count": sample_count,
        "p95_fad_ms": p95_fad,
        "p95_e2e_ms": p95_e2e,
        "fad_gate_pass": has_samples and p95_fad <= target_fad_ms,
        "e2e_gate_pass": has_samples and p95_e2e <= target_e2e_ms,
    }


def resolve_status(
    *,
    sample_count: int,
    min_samples: int,
    fad_gate_pass: bool,
    e2e_gate_pass: bool,
) -> str:
    if sample_count < min_samples:
        return "UNVERIFIED"
    if fad_gate_pass and e2e_gate_pass:
        return "PASS"
    return "FAIL"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime chain smoke test and gate snapshot")
    parser.add_argument("--base-url", default="http://127.0.0.1:8787")
    parser.add_argument("--voice-name", default="smoke_voice")
    parser.add_argument("--language", default="Auto")
    parser.add_argument("--create-voice-if-missing", action="store_true", default=False)
    parser.add_argument("--monitor-enabled", action="store_true", default=False)
    parser.add_argument("--require-virtual-mic", action="store_true", default=False)
    parser.add_argument("--run-seconds", type=float, default=8.0)
    parser.add_argument("--wait-for-min-samples", action="store_true", default=False)
    parser.add_argument("--max-wait-seconds", type=float, default=120.0)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--target-fad-ms", type=float, default=250.0)
    parser.add_argument("--target-e2e-ms", type=float, default=800.0)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--strict", action="store_true", default=False)
    parser.add_argument("--keep-running", action="store_true", default=False)
    parser.add_argument("--report-json", default="")
    parser.add_argument(
        "--simulate-in-fake-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When fake mode is enabled, auto inject simulated segments for metrics sampling.",
    )
    parser.add_argument("--simulate-count-per-poll", type=int, default=2)
    parser.add_argument("--simulate-duration-ms", type=int, default=500)
    parser.add_argument("--simulate-amplitude", type=float, default=0.08)
    parser.add_argument("--simulate-frequency-hz", type=float, default=220.0)
    parser.add_argument(
        "--reset-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reset in-memory metrics before sampling (default: true).",
    )
    parser.add_argument(
        "--reset-clear-log",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When resetting metrics, also truncate metrics.ndjson log.",
    )
    parser.add_argument(
        "--reset-clear-db",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When resetting metrics, also clear metrics rows in sqlite table.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    events: list[str] = []
    created_voice_id: str | None = None
    started = False

    def event(text: str) -> None:
        line = f"[{_now()}] {text}"
        events.append(line)
        print(line)

    try:
        health = http_json("GET", f"{base_url}/api/v1/health")
        fake_mode = bool(health.get("fake_mode"))
        event(f"health: status={health.get('status')} fake_mode={fake_mode}")

        if args.reset_metrics:
            reset = http_json(
                "POST",
                f"{base_url}/api/v1/metrics/reset"
                f"?clear_log={'true' if args.reset_clear_log else 'false'}"
                f"&clear_db={'true' if args.reset_clear_db else 'false'}",
            )
            event(
                "metrics reset: "
                f"sample_count={reset.get('metrics', {}).get('sample_count', 'n/a')}, "
                f"clear_log={reset.get('clear_log')}, clear_db={reset.get('clear_db')}"
            )

        devices = http_json("GET", f"{base_url}/api/v1/audio/devices")
        default_input = devices.get("default_input_id")
        default_output = devices.get("default_output_id")
        virtual_output = devices.get("virtual_mic_output_id")
        event(
            "devices: "
            f"default_input={default_input}, default_output={default_output}, virtual_mic={virtual_output}"
        )
        if args.require_virtual_mic and virtual_output is None:
            raise RuntimeError("virtual mic is required but no VB-CABLE style output device was found")

        voices = http_json("GET", f"{base_url}/api/v1/voices")
        chosen = choose_voice(voices, args.voice_name)
        if chosen is None and args.create_voice_if_missing:
            unique_name = f"{args.voice_name}_{int(time.time())}"
            event(f"creating smoke design voice: {unique_name}")
            create_payload = {
                "name": unique_name,
                "voice_prompt": "Clear, stable, natural speaking voice with medium pace.",
                "preview_text": "This is a realtime voice changer smoke check.",
                "language": args.language,
                "save": True,
            }
            created = http_json("POST", f"{base_url}/api/v1/voices/design", create_payload, timeout=180.0)
            chosen = created["voice"]
            created_voice_id = str(chosen["id"])
            event(f"created voice_id={created_voice_id}")
        elif chosen is None:
            raise RuntimeError(
                "no voices available. Create one in UI or run with --create-voice-if-missing"
            )
        else:
            event(f"using existing voice_id={chosen.get('id')} name={chosen.get('name')}")

        settings_payload = {
            "input_device_id": default_input,
            "monitor_device_id": default_output,
            "virtual_mic_device_id": virtual_output,
            "monitor_enabled": bool(args.monitor_enabled),
            "vad_silence_ms": 240,
            "max_segment_ms": 12000,
        }
        http_json("PATCH", f"{base_url}/api/v1/realtime/settings", settings_payload)
        event("realtime settings applied")

        start_payload = {
            "voice_id": chosen["id"],
            "language": args.language,
        }
        http_json("POST", f"{base_url}/api/v1/realtime/start", start_payload)
        started = True
        if fake_mode and args.simulate_in_fake_mode:
            event("realtime started; fake mode auto simulation is enabled")
        else:
            event("realtime started; speak to microphone now for metrics sampling")

        poll_interval = max(args.poll_interval, 0.2)
        started_at = time.time()
        end_at = started_at + max(args.run_seconds, 0.5)
        if args.wait_for_min_samples:
            end_at = started_at + max(args.max_wait_seconds, poll_interval)
        latest_metrics: dict = {}
        while time.time() < end_at:
            if fake_mode and args.simulate_in_fake_mode:
                current_metrics = http_json("GET", f"{base_url}/api/v1/metrics/current")
                current_samples = int(current_metrics.get("sample_count", 0))
                remaining = max(0, args.min_samples - current_samples)
                simulate_count = max(args.simulate_count_per_poll, 1)
                if args.wait_for_min_samples and remaining > 0:
                    simulate_count = min(simulate_count, remaining)
                if simulate_count > 0:
                    simulated = http_json(
                        "POST",
                        f"{base_url}/api/v1/realtime/simulate",
                        payload={
                            "count": simulate_count,
                            "duration_ms": args.simulate_duration_ms,
                            "amplitude": args.simulate_amplitude,
                            "frequency_hz": args.simulate_frequency_hz,
                        },
                    )
                    event(
                        "simulated segments: "
                        f"{simulated.get('injected_segments')} -> "
                        f"samples={simulated.get('metrics', {}).get('sample_count', 'n/a')}"
                    )

            latest_metrics = http_json("GET", f"{base_url}/api/v1/metrics/current")
            samples = int(latest_metrics.get("sample_count", 0))
            event(
                "metrics: "
                f"samples={samples} p95_fad={latest_metrics.get('p95_fad_ms', 0):.1f}ms "
                f"p95_e2e={latest_metrics.get('p95_e2e_ms', 0):.1f}ms"
            )
            if args.wait_for_min_samples and samples >= args.min_samples:
                event(f"min samples reached: {samples} >= {args.min_samples}")
                break
            time.sleep(poll_interval)

        gate = check_gate(
            latest_metrics,
            target_fad_ms=args.target_fad_ms,
            target_e2e_ms=args.target_e2e_ms,
        )
        enough_samples = gate["sample_count"] >= args.min_samples
        both_pass = gate["fad_gate_pass"] and gate["e2e_gate_pass"]
        status = resolve_status(
            sample_count=gate["sample_count"],
            min_samples=args.min_samples,
            fad_gate_pass=gate["fad_gate_pass"],
            e2e_gate_pass=gate["e2e_gate_pass"],
        )

        elapsed = max(0.0, time.time() - started_at)
        timed_out = args.wait_for_min_samples and not enough_samples and elapsed >= args.max_wait_seconds

        summary = {
            "base_url": base_url,
            "voice_id": chosen["id"],
            "created_voice_id": created_voice_id,
            "status": status,
            "gate": gate,
            "wait_mode": {
                "wait_for_min_samples": bool(args.wait_for_min_samples),
                "run_seconds": args.run_seconds,
                "max_wait_seconds": args.max_wait_seconds,
                "elapsed_seconds": round(elapsed, 3),
                "timed_out": bool(timed_out),
            },
            "targets": {
                "p95_fad_ms": args.target_fad_ms,
                "p95_e2e_ms": args.target_e2e_ms,
                "min_samples": args.min_samples,
            },
            "enough_samples": enough_samples,
            "both_pass": both_pass,
            "strict": bool(args.strict),
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if args.report_json:
            report_path = Path(args.report_json)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            event(f"report written: {report_path}")

        if not args.keep_running and started:
            http_json("POST", f"{base_url}/api/v1/realtime/stop")
            started = False
            event("realtime stopped")

        if status == "UNVERIFIED":
            event("result: UNVERIFIED (insufficient samples)")
            return 1 if args.strict else 0
        if status == "PASS":
            event("result: PASS")
            return 0
        event("result: FAIL")
        return 1 if args.strict else 0
    except Exception as exc:
        print(f"smoke failed: {exc}", file=sys.stderr)
        return 2
    finally:
        if started and not args.keep_running:
            try:
                http_json("POST", f"{base_url}/api/v1/realtime/stop")
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
