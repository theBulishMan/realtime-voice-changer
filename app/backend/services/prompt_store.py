from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any


def save_prompt_items(path: Path, prompt_items: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable: list[dict[str, Any]] = []
    for item in prompt_items:
        if isinstance(item, dict):
            serializable.append(
                {
                    "ref_code": item.get("ref_code"),
                    "ref_spk_embedding": item.get("ref_spk_embedding"),
                    "x_vector_only_mode": bool(item.get("x_vector_only_mode", False)),
                    "icl_mode": bool(item.get("icl_mode", False)),
                    "ref_text": item.get("ref_text"),
                }
            )
            continue
        serializable.append(
            {
                "ref_code": getattr(item, "ref_code", None),
                "ref_spk_embedding": getattr(item, "ref_spk_embedding", None),
                "x_vector_only_mode": bool(getattr(item, "x_vector_only_mode", False)),
                "icl_mode": bool(getattr(item, "icl_mode", False)),
                "ref_text": getattr(item, "ref_text", None),
            }
        )
    try:
        import torch

        torch.save(serializable, path)
    except Exception:
        with path.open("wb") as f:
            pickle.dump(serializable, f)


def load_prompt_items(path: Path, *, ref_text_override: str | None = None) -> list[Any]:
    try:
        import torch

        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            # Backward compatibility for older torch versions without weights_only.
            payload = torch.load(path, map_location="cpu")
        except Exception:
            # Compatibility fallback for legacy prompt files; suppress known torch warning.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r".*weights_only=False.*",
                    category=FutureWarning,
                )
                payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        with path.open("rb") as f:
            payload = pickle.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"Invalid prompt payload in {path}")

    try:
        # Lazy import to avoid hard-failing when qwen-tts is unavailable in fake mode.
        from qwen_tts import VoiceClonePromptItem
    except Exception:
        VoiceClonePromptItem = None  # type: ignore[assignment]

    items: list[Any] = []
    for item in payload:
        if VoiceClonePromptItem is None:
            if isinstance(item, dict) and ref_text_override is not None:
                item = dict(item)
                item["ref_text"] = ref_text_override
            items.append(item)
        else:
            items.append(
                VoiceClonePromptItem(
                    ref_code=item.get("ref_code"),
                    ref_spk_embedding=item.get("ref_spk_embedding"),
                    x_vector_only_mode=bool(item.get("x_vector_only_mode", False)),
                    icl_mode=bool(item.get("icl_mode", False)),
                    ref_text=ref_text_override if ref_text_override is not None else item.get("ref_text"),
                )
            )
    return items
