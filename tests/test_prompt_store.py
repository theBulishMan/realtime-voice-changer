from __future__ import annotations

from pathlib import Path
import sys
import warnings

from app.backend.services.prompt_store import load_prompt_items, save_prompt_items


def test_prompt_store_roundtrip_without_future_warning(tmp_path: Path) -> None:
    path = tmp_path / "voice" / "prompt.pt"
    payload = [
        {
            "ref_code": None,
            "ref_spk_embedding": None,
            "x_vector_only_mode": False,
            "icl_mode": True,
            "ref_text": "hello",
        }
    ]
    save_prompt_items(path, payload)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = load_prompt_items(path)

    assert isinstance(loaded, list)
    assert loaded
    torch_load_future = [
        w
        for w in caught
        if issubclass(w.category, FutureWarning)
        and "weights_only=False" in str(w.message)
    ]
    assert torch_load_future == []


def test_prompt_store_legacy_fallback_suppresses_weights_only_warning(monkeypatch, tmp_path: Path) -> None:
    class FakeTorch:
        @staticmethod
        def load(path, map_location=None, weights_only=None):
            if weights_only is True:
                raise RuntimeError("legacy format")
            warnings.warn(
                "You are using `torch.load` with `weights_only=False` (the current default value)",
                FutureWarning,
            )
            return [
                {
                    "ref_code": None,
                    "ref_spk_embedding": None,
                    "x_vector_only_mode": False,
                    "icl_mode": True,
                    "ref_text": "legacy",
                }
            ]

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    path = tmp_path / "voice" / "legacy.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"legacy")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loaded = load_prompt_items(path)

    assert isinstance(loaded, list)
    assert loaded
    torch_load_future = [
        w
        for w in caught
        if issubclass(w.category, FutureWarning)
        and "weights_only=False" in str(w.message)
    ]
    assert torch_load_future == []


def _extract_ref_text(item: object) -> str | None:
    if isinstance(item, dict):
        value = item.get("ref_text")
    else:
        value = getattr(item, "ref_text", None)
    return str(value) if value is not None else None


def test_load_prompt_items_applies_ref_text_override(tmp_path: Path) -> None:
    path = tmp_path / "voice" / "prompt.pt"
    payload = [
        {
            "ref_code": [1, 2, 3],
            "ref_spk_embedding": [4, 5, 6],
            "x_vector_only_mode": False,
            "icl_mode": True,
            "ref_text": "legacy_bad_text",
        }
    ]
    save_prompt_items(path, payload)

    loaded = load_prompt_items(path, ref_text_override="fixed_text")

    assert isinstance(loaded, list)
    assert loaded
    assert _extract_ref_text(loaded[0]) == "fixed_text"
