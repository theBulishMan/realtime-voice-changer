from __future__ import annotations

from difflib import SequenceMatcher
from typing import Any

from app.backend.types import AudioDeviceInfo, AudioDevicesResponse

_NOISY_NAME_HINTS = (
    "microsoft sound mapper",
    "sound mapper",
    "primary sound",
    "主声音",
    "primary audio",
)


def _normalize_name(name: str) -> str:
    return " ".join(name.lower().strip().split())


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _is_noisy_pseudo_device(name: str) -> bool:
    lowered = _normalize_name(name)
    return any(hint in lowered for hint in _NOISY_NAME_HINTS)


def _has_channel(dev: dict[str, Any], *, output: bool) -> bool:
    key = "max_output_channels" if output else "max_input_channels"
    return int(dev.get(key, 0)) > 0


def _is_usable_device(dev: dict[str, Any]) -> bool:
    return _has_channel(dev, output=False) or _has_channel(dev, output=True)


def _hostapi_name(hostapis: list[dict[str, Any]], idx: int) -> str:
    if idx < 0 or idx >= len(hostapis):
        return ""
    return str(hostapis[idx].get("name", ""))


def _hostapi_priority(hostapi_name: str) -> int:
    lowered = hostapi_name.lower()
    if "wasapi" in lowered:
        return 0
    if "wdm-ks" in lowered:
        return 1
    if "directsound" in lowered:
        return 2
    if "mme" in lowered:
        return 3
    return 4


def _find_preferred_hostapi(
    devices: list[dict[str, Any]],
    hostapis: list[dict[str, Any]],
) -> int | None:
    per_api_count: dict[int, int] = {}
    for dev in devices:
        if not _is_usable_device(dev):
            continue
        name = str(dev.get("name", ""))
        if _is_noisy_pseudo_device(name):
            continue
        api_idx = int(dev.get("hostapi", -1))
        if api_idx < 0:
            continue
        per_api_count[api_idx] = per_api_count.get(api_idx, 0) + 1

    if not per_api_count:
        return None

    ranked = sorted(
        per_api_count.items(),
        key=lambda item: (
            _hostapi_priority(_hostapi_name(hostapis, item[0])),
            -item[1],
            item[0],
        ),
    )
    return ranked[0][0]


def _pick_best_fallback_device_id(
    devices: list[dict[str, Any]],
    hostapis: list[dict[str, Any]],
    *,
    output: bool,
    exclude_ids: set[int],
) -> int | None:
    candidates: list[tuple[tuple[int, int], int]] = []
    for idx, dev in enumerate(devices):
        if idx in exclude_ids:
            continue
        if not _has_channel(dev, output=output):
            continue
        name = str(dev.get("name", ""))
        noisy_penalty = 1 if _is_noisy_pseudo_device(name) else 0
        api_idx = int(dev.get("hostapi", -1))
        hostapi_rank = _hostapi_priority(_hostapi_name(hostapis, api_idx))
        candidates.append(((noisy_penalty, hostapi_rank), idx))

    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][1]


def _selected_device_ids(
    devices: list[dict[str, Any]],
    hostapis: list[dict[str, Any]],
    *,
    virtual_mic_output_id: int | None,
) -> list[int]:
    preferred_hostapi = _find_preferred_hostapi(devices, hostapis)
    selected: set[int] = set()

    if preferred_hostapi is not None:
        for idx, dev in enumerate(devices):
            if int(dev.get("hostapi", -1)) != preferred_hostapi:
                continue
            if not _is_usable_device(dev):
                continue
            name = str(dev.get("name", ""))
            if _is_noisy_pseudo_device(name):
                continue
            selected.add(idx)

    if virtual_mic_output_id is not None:
        if 0 <= virtual_mic_output_id < len(devices):
            if _has_channel(devices[virtual_mic_output_id], output=True):
                selected.add(virtual_mic_output_id)

    if not any(_has_channel(devices[idx], output=False) for idx in selected):
        fallback_in = _pick_best_fallback_device_id(
            devices, hostapis, output=False, exclude_ids=selected
        )
        if fallback_in is not None:
            selected.add(fallback_in)

    if not any(_has_channel(devices[idx], output=True) for idx in selected):
        fallback_out = _pick_best_fallback_device_id(
            devices, hostapis, output=True, exclude_ids=selected
        )
        if fallback_out is not None:
            selected.add(fallback_out)

    if not selected:
        for idx, dev in enumerate(devices):
            if _is_usable_device(dev):
                selected.add(idx)

    return sorted(
        selected,
        key=lambda idx: (
            _hostapi_priority(_hostapi_name(hostapis, int(devices[idx].get("hostapi", -1)))),
            idx,
        ),
    )


def _resolve_default_id(
    raw_default_id: int | None,
    *,
    output: bool,
    devices: list[dict[str, Any]],
    hostapis: list[dict[str, Any]],
    selected_ids: list[int],
) -> int | None:
    candidates = [idx for idx in selected_ids if _has_channel(devices[idx], output=output)]
    if not candidates:
        return None

    if raw_default_id is not None:
        if raw_default_id in candidates:
            return raw_default_id
        if 0 <= raw_default_id < len(devices) and _has_channel(devices[raw_default_id], output=output):
            target_name = _normalize_name(str(devices[raw_default_id].get("name", "")))
            best_idx = max(
                candidates,
                key=lambda idx: _similarity(
                    target_name, _normalize_name(str(devices[idx].get("name", "")))
                ),
            )
            if _similarity(
                target_name, _normalize_name(str(devices[best_idx].get("name", "")))
            ) >= 0.4:
                return best_idx

    return sorted(
        candidates,
        key=lambda idx: (
            _hostapi_priority(_hostapi_name(hostapis, int(devices[idx].get("hostapi", -1)))),
            idx,
        ),
    )[0]


def _query_devices() -> tuple[list[dict[str, Any]], list[dict[str, Any]], int | None, int | None]:
    import sounddevice as sd

    devices = [dict(d) for d in sd.query_devices()]
    hostapis = [dict(h) for h in sd.query_hostapis()]
    default_input, default_output = sd.default.device
    default_input_id = int(default_input) if default_input is not None and default_input >= 0 else None
    default_output_id = (
        int(default_output) if default_output is not None and default_output >= 0 else None
    )
    return devices, hostapis, default_input_id, default_output_id


def find_device_id_by_name_hint(
    devices: list[dict[str, Any]],
    hostapis: list[dict[str, Any]],
    hint: str,
    output: bool,
) -> int | None:
    if not hint:
        return None
    target = hint.lower()
    candidates: list[tuple[tuple[float, float, float, float, int], int]] = []
    for idx, dev in enumerate(devices):
        name = str(dev.get("name", "")).lower()
        channels = int(dev.get("max_output_channels" if output else "max_input_channels", 0))
        if channels <= 0 or target not in name:
            continue
        noisy_penalty = 1.0 if _is_noisy_pseudo_device(name) else 0.0
        api_idx = int(dev.get("hostapi", -1))
        hostapi_rank = float(_hostapi_priority(_hostapi_name(hostapis, api_idx)))
        default_rate = float(dev.get("default_samplerate", 0.0) or 0.0)
        preferred_sr_penalty = abs(default_rate - 48000.0)
        score = (noisy_penalty, hostapi_rank, preferred_sr_penalty, -float(channels), idx)
        candidates.append((score, idx))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def list_audio_devices(virtual_mic_hint: str) -> AudioDevicesResponse:
    devices_raw, hostapis, raw_default_input_id, raw_default_output_id = _query_devices()
    virtual_mic_output_id = find_device_id_by_name_hint(
        devices_raw, hostapis, hint=virtual_mic_hint, output=True
    )
    selected_ids = _selected_device_ids(
        devices_raw,
        hostapis,
        virtual_mic_output_id=virtual_mic_output_id,
    )
    default_input_id = _resolve_default_id(
        raw_default_input_id,
        output=False,
        devices=devices_raw,
        hostapis=hostapis,
        selected_ids=selected_ids,
    )
    default_output_id = _resolve_default_id(
        raw_default_output_id,
        output=True,
        devices=devices_raw,
        hostapis=hostapis,
        selected_ids=selected_ids,
    )

    out: list[AudioDeviceInfo] = []
    for idx in selected_ids:
        dev = devices_raw[idx]
        out.append(
            AudioDeviceInfo(
                id=idx,
                name=str(dev.get("name", "")),
                max_input_channels=int(dev.get("max_input_channels", 0)),
                max_output_channels=int(dev.get("max_output_channels", 0)),
                default_samplerate=float(dev.get("default_samplerate", 0.0)),
                is_default_input=default_input_id == idx,
                is_default_output=default_output_id == idx,
            )
        )
    return AudioDevicesResponse(
        devices=out,
        default_input_id=default_input_id,
        default_output_id=default_output_id,
        virtual_mic_output_id=virtual_mic_output_id,
        virtual_mic_hint=virtual_mic_hint,
    )
