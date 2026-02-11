from __future__ import annotations

import importlib


def _dev(
    name: str,
    *,
    hostapi: int,
    in_ch: int = 0,
    out_ch: int = 0,
    sr: float = 48000.0,
) -> dict:
    return {
        "name": name,
        "hostapi": hostapi,
        "max_input_channels": in_ch,
        "max_output_channels": out_ch,
        "default_samplerate": sr,
    }


def _hostapis() -> list[dict]:
    return [
        {"name": "MME"},
        {"name": "Windows DirectSound"},
        {"name": "Windows WASAPI"},
        {"name": "Windows WDM-KS"},
    ]


def test_audio_devices_prefers_wasapi_and_keeps_virtual(monkeypatch):
    mod = importlib.import_module("app.backend.audio.audio_devices")
    mod = importlib.reload(mod)

    devices = [
        _dev("Microsoft Sound Mapper - Input", hostapi=0, in_ch=2),
        _dev("Microphone (USB Mic)", hostapi=0, in_ch=1),
        _dev("Speakers (USB Audio)", hostapi=0, out_ch=2),
        _dev("Microphone (USB Mic)", hostapi=2, in_ch=1),
        _dev("Speakers (USB Audio)", hostapi=2, out_ch=2),
        _dev("Monitor (NVIDIA)", hostapi=2, out_ch=2),
        _dev("CABLE Input (VB-Audio Virtual Cable)", hostapi=1, out_ch=2),
    ]

    monkeypatch.setattr(
        mod,
        "_query_devices",
        lambda: (devices, _hostapis(), 1, 2),
    )

    resp = mod.list_audio_devices("CABLE Input")
    ids = [d.id for d in resp.devices]

    assert set(ids) == {3, 4, 5, 6}
    assert resp.default_input_id == 3
    assert resp.default_output_id == 4
    assert resp.virtual_mic_output_id == 6


def test_audio_devices_adds_fallback_input_when_preferred_has_output_only(monkeypatch):
    mod = importlib.import_module("app.backend.audio.audio_devices")
    mod = importlib.reload(mod)

    devices = [
        _dev("Speakers (USB Audio)", hostapi=2, out_ch=2),
        _dev("Monitor (NVIDIA)", hostapi=2, out_ch=2),
        _dev("Microphone (USB Mic)", hostapi=0, in_ch=1),
    ]

    monkeypatch.setattr(
        mod,
        "_query_devices",
        lambda: (devices, _hostapis(), 2, 0),
    )

    resp = mod.list_audio_devices("CABLE Input")
    ids = [d.id for d in resp.devices]

    assert 2 in ids
    assert resp.default_input_id == 2
    assert resp.default_output_id == 0


def test_audio_devices_filters_noisy_mapper_entries(monkeypatch):
    mod = importlib.import_module("app.backend.audio.audio_devices")
    mod = importlib.reload(mod)

    devices = [
        _dev("Microsoft Sound Mapper - Input", hostapi=0, in_ch=2),
        _dev("Microsoft Sound Mapper - Output", hostapi=0, out_ch=2),
        _dev("主声音捕获驱动程序", hostapi=1, in_ch=1),
        _dev("主声音驱动程序", hostapi=1, out_ch=2),
        _dev("Microphone (USB Mic)", hostapi=2, in_ch=1),
        _dev("Speakers (USB Audio)", hostapi=2, out_ch=2),
    ]

    monkeypatch.setattr(
        mod,
        "_query_devices",
        lambda: (devices, _hostapis(), 0, 1),
    )

    resp = mod.list_audio_devices("CABLE Input")
    names = [d.name.lower() for d in resp.devices]

    assert all("mapper" not in name for name in names)
    assert all("主声音" not in name for name in names)
    assert any("microphone" in name for name in names)
    assert any("speakers" in name for name in names)


def test_audio_devices_picks_best_virtual_mic_by_hostapi(monkeypatch):
    mod = importlib.import_module("app.backend.audio.audio_devices")
    mod = importlib.reload(mod)

    devices = [
        _dev("CABLE Input (VB-Audio Virtual Cable)", hostapi=0, out_ch=8, sr=44100.0),  # MME
        _dev("CABLE Input (VB-Audio Virtual Cable)", hostapi=1, out_ch=8, sr=44100.0),  # DirectSound
        _dev("CABLE Input (VB-Audio Virtual Cable)", hostapi=2, out_ch=2, sr=48000.0),  # WASAPI
        _dev("Microphone (USB Mic)", hostapi=2, in_ch=1),
        _dev("Speakers (USB Audio)", hostapi=2, out_ch=2),
    ]

    monkeypatch.setattr(
        mod,
        "_query_devices",
        lambda: (devices, _hostapis(), 3, 4),
    )

    resp = mod.list_audio_devices("CABLE Input")
    assert resp.virtual_mic_output_id == 2
