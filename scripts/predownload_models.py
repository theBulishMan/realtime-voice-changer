from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

DEFAULT_BASE_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_DESIGN_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
DEFAULT_CUSTOM_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
DEFAULT_ASR_MODEL = "Qwen/Qwen3-ASR-0.6B"
DEFAULT_TEXT_CORRECTION_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

SUPPORTED_PROVIDERS = ("modelscope", "huggingface", "auto")


def _repo_leaf(repo_id: str) -> str:
    return repo_id.rsplit("/", 1)[-1]


def configure_env(cache_dir: Path, download_timeout: int) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir.resolve()))
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(download_timeout))
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")


def _provider_order(provider: str) -> list[str]:
    normalized = (provider or "auto").strip().lower()
    if normalized not in SUPPORTED_PROVIDERS:
        raise ValueError(f"Unsupported provider: {provider}")
    if normalized == "auto":
        # Prefer ModelScope first for Mainland China network.
        return ["modelscope", "huggingface"]
    return [normalized]


def _download_huggingface(*, repo_id: str, cache_dir: Path, local_dir: Path | None) -> str:
    from huggingface_hub import snapshot_download

    if local_dir is not None:
        local_dir.mkdir(parents=True, exist_ok=True)
    return str(
        snapshot_download(
            repo_id=repo_id,
            cache_dir=str(cache_dir),
            local_dir=str(local_dir) if local_dir is not None else None,
            resume_download=True,
        )
    )


def _download_modelscope(*, repo_id: str, cache_dir: Path, local_dir: Path | None) -> str:
    try:
        from modelscope import snapshot_download
    except Exception:
        from modelscope.hub.snapshot_download import snapshot_download

    if local_dir is not None:
        local_dir.mkdir(parents=True, exist_ok=True)
    return str(
        snapshot_download(
            model_id=repo_id,
            cache_dir=str(cache_dir),
            local_dir=str(local_dir) if local_dir is not None else None,
            local_files_only=False,
        )
    )


def _download_once(
    *,
    provider: str,
    repo_id: str,
    cache_dir: Path,
    local_dir: Path | None,
) -> str:
    if provider == "modelscope":
        return _download_modelscope(repo_id=repo_id, cache_dir=cache_dir, local_dir=local_dir)
    if provider == "huggingface":
        return _download_huggingface(repo_id=repo_id, cache_dir=cache_dir, local_dir=local_dir)
    raise ValueError(f"Unsupported provider: {provider}")


def download_with_retry(
    *,
    repo_id: str,
    cache_dir: Path,
    local_dir: Path | None,
    retries: int,
    provider: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    errors: list[str] = []
    providers = _provider_order(provider)
    provider_used = ""
    attempts_total = 0

    for current_provider in providers:
        for attempt in range(1, retries + 1):
            attempts_total += 1
            try:
                snapshot_path = _download_once(
                    provider=current_provider,
                    repo_id=repo_id,
                    cache_dir=cache_dir,
                    local_dir=local_dir,
                )
                provider_used = current_provider
                return {
                    "repo_id": repo_id,
                    "provider": provider_used,
                    "status": "ok",
                    "attempts": attempts_total,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "snapshot_path": snapshot_path,
                    "local_dir": str(local_dir.resolve()) if local_dir is not None else "",
                    "errors": errors,
                }
            except Exception as exc:
                err = f"{current_provider}#{attempt}: {exc}"
                errors.append(err)
                if attempt < retries:
                    sleep_seconds = min(30, attempt * 5)
                    print(
                        f"[predownload] {repo_id}: {current_provider} "
                        f"attempt {attempt}/{retries} failed: {exc}. retrying in {sleep_seconds}s..."
                    )
                    time.sleep(sleep_seconds)

    return {
        "repo_id": repo_id,
        "provider": provider_used or "none",
        "status": "error",
        "attempts": attempts_total,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "snapshot_path": None,
        "local_dir": str(local_dir.resolve()) if local_dir is not None else "",
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pre-download local models for real mode (Qwen3-TTS + Qwen3-ASR)."
    )
    parser.add_argument("--base", action="store_true", help="Download base model.")
    parser.add_argument("--design", action="store_true", help="Download voice design model.")
    parser.add_argument("--custom", action="store_true", help="Download custom voice model.")
    parser.add_argument("--asr", action="store_true", help="Download Qwen3-ASR model.")
    parser.add_argument("--text-corrector", action="store_true", help="Download text correction LLM.")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download base + design + custom + asr + text-corrector models.",
    )
    parser.add_argument(
        "--provider",
        default=os.getenv("RVC_MODEL_PROVIDER", "modelscope"),
        choices=SUPPORTED_PROVIDERS,
        help="Download provider: modelscope, huggingface, or auto fallback.",
    )
    parser.add_argument("--retries", type=int, default=6, help="Max retries per provider per model.")
    parser.add_argument(
        "--download-timeout",
        type=int,
        default=1200,
        help="HF_HUB_DOWNLOAD_TIMEOUT seconds.",
    )
    parser.add_argument(
        "--cache-dir",
        default=os.getenv("HF_HOME", str((Path.home() / ".cache" / "huggingface").resolve())),
        help="Shared download cache root.",
    )
    parser.add_argument(
        "--models-dir",
        default=os.getenv("RVC_MODELS_DIR", str((Path.cwd() / ".cache" / "models").resolve())),
        help="Local models root; each model goes to models_dir/<model_name>.",
    )
    parser.add_argument(
        "--asr-local-dir",
        default="",
        help="Deprecated override for ASR local directory. Prefer --models-dir.",
    )
    parser.add_argument("--report-json", default="", help="Optional output report path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    models_dir = Path(args.models_dir).resolve()
    models_dir.mkdir(parents=True, exist_ok=True)
    configure_env(cache_dir=cache_dir, download_timeout=max(args.download_timeout, 60))

    selected: list[tuple[str, Path | None]] = []
    if args.all or args.base:
        selected.append((DEFAULT_BASE_MODEL, models_dir / _repo_leaf(DEFAULT_BASE_MODEL)))
    if args.all or args.design:
        selected.append((DEFAULT_DESIGN_MODEL, models_dir / _repo_leaf(DEFAULT_DESIGN_MODEL)))
    if args.all or args.custom:
        selected.append((DEFAULT_CUSTOM_MODEL, models_dir / _repo_leaf(DEFAULT_CUSTOM_MODEL)))
    if args.all or args.asr:
        asr_local_dir = (
            Path(args.asr_local_dir).resolve()
            if args.asr_local_dir
            else (models_dir / _repo_leaf(DEFAULT_ASR_MODEL))
        )
        selected.append((DEFAULT_ASR_MODEL, asr_local_dir))
    if args.all or args.text_corrector:
        selected.append(
            (
                DEFAULT_TEXT_CORRECTION_MODEL,
                models_dir / _repo_leaf(DEFAULT_TEXT_CORRECTION_MODEL),
            )
        )
    if not selected:
        selected.append((DEFAULT_BASE_MODEL, models_dir / _repo_leaf(DEFAULT_BASE_MODEL)))
        asr_local_dir = (
            Path(args.asr_local_dir).resolve()
            if args.asr_local_dir
            else (models_dir / _repo_leaf(DEFAULT_ASR_MODEL))
        )
        selected.append((DEFAULT_ASR_MODEL, asr_local_dir))
        print("[predownload] no model flag provided, defaulting to --base --asr.")

    print(f"[predownload] provider={args.provider}")
    print(f"[predownload] cache_dir={cache_dir.resolve()}")
    print(f"[predownload] models_dir={models_dir.resolve()}")
    print(f"[predownload] download_timeout={os.environ.get('HF_HUB_DOWNLOAD_TIMEOUT')}s")
    print(f"[predownload] targets={[repo for repo, _ in selected]}")

    report: dict[str, Any] = {
        "provider": args.provider,
        "cache_dir": str(cache_dir.resolve()),
        "models_dir": str(models_dir.resolve()),
        "download_timeout": int(os.environ.get("HF_HUB_DOWNLOAD_TIMEOUT", "1200")),
        "results": [],
    }

    has_error = False
    for repo_id, local_dir in selected:
        result = download_with_retry(
            repo_id=repo_id,
            cache_dir=cache_dir,
            local_dir=local_dir,
            retries=max(1, int(args.retries)),
            provider=args.provider,
        )
        report["results"].append(result)
        print(
            f"[predownload] {repo_id} -> {result['status']} "
            f"(provider={result['provider']}, attempts={result['attempts']}, "
            f"elapsed={result['elapsed_seconds']}s)"
        )
        if result["status"] != "ok":
            has_error = True

    if args.report_json:
        out = Path(args.report_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[predownload] wrote report: {out}")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if has_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
