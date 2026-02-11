from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
import importlib.util


def run_cmd(cmd: list[str]) -> str:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=8)
        return out.strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def detect_sox() -> str:
    direct = shutil.which("sox")
    if direct:
        return run_cmd([direct, "--version"])

    local = os.getenv("LOCALAPPDATA", "")
    if local:
        winget_root = Path(local) / "Microsoft" / "WinGet" / "Packages"
        if winget_root.exists():
            for pkg in winget_root.glob("ChrisBagwell.SoX*"):
                for candidate in (
                    pkg / "sox-14.4.2" / "sox.exe",
                    pkg / "sox.exe",
                ):
                    if candidate.exists():
                        return run_cmd([str(candidate), "--version"])
    return "ERROR: sox executable not found"


def has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def model_dir_ready(path: Path) -> bool:
    if not path.exists():
        return False
    if not (path / "config.json").exists():
        return False
    return (
        (path / "model.safetensors").exists()
        or (path / "model.safetensors.index.json").exists()
        or (path / "pytorch_model.bin").exists()
    )


def list_audio_devices() -> dict:
    try:
        import sounddevice as sd

        devices = [dict(d) for d in sd.query_devices()]
        vb = [d for d in devices if "cable" in str(d.get("name", "")).lower()]
        return {
            "count": len(devices),
            "vb_cable_candidates": [d.get("name") for d in vb],
        }
    except Exception as exc:
        return {"error": str(exc)}


def main() -> None:
    root = Path.cwd()
    disk = shutil.disk_usage(root)
    models_dir = Path(os.getenv("RVC_MODELS_DIR", str((root / ".cache" / "models").resolve())))
    tts_base_dir = Path(
        os.getenv("RVC_BASE_MODEL_ID", str((models_dir / "Qwen3-TTS-12Hz-1.7B-Base").resolve()))
    )
    tts_design_dir = Path(
        os.getenv(
            "RVC_DESIGN_MODEL_ID",
            str((models_dir / "Qwen3-TTS-12Hz-1.7B-VoiceDesign").resolve()),
        )
    )
    tts_custom_dir = Path(
        os.getenv(
            "RVC_CUSTOM_VOICE_MODEL_ID",
            str((models_dir / "Qwen3-TTS-12Hz-1.7B-CustomVoice").resolve()),
        )
    )
    asr_local_dir = Path(
        os.getenv("RVC_QWEN_ASR_LOCAL_DIR", str((models_dir / "Qwen3-ASR-0.6B").resolve()))
    )
    report = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        "nvidia_smi": run_cmd(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "sox": detect_sox(),
        "ffmpeg": run_cmd(["ffmpeg", "-version"]),
        "git": run_cmd(["git", "--version"]),
        "qwen_asr_installed": has_module("qwen_asr"),
        "modelscope_installed": has_module("modelscope"),
        "disk_free_gb": round(disk.free / (1024**3), 2),
        "audio": list_audio_devices(),
        "model_cache_hint": os.getenv("HF_HOME", str((root / ".cache" / "huggingface").resolve())),
        "models_dir": str(models_dir),
        "qwen_tts_base_local_dir": str(tts_base_dir),
        "qwen_tts_base_model_ready": model_dir_ready(tts_base_dir),
        "qwen_tts_design_local_dir": str(tts_design_dir),
        "qwen_tts_design_model_ready": model_dir_ready(tts_design_dir),
        "qwen_tts_custom_local_dir": str(tts_custom_dir),
        "qwen_tts_custom_model_ready": model_dir_ready(tts_custom_dir),
        "qwen_asr_local_dir": str(asr_local_dir),
        "qwen_asr_local_model_ready": model_dir_ready(asr_local_dir),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
