from __future__ import annotations

from pathlib import Path

from app.backend.db import AppDatabase


def test_voice_crud(tmp_path: Path):
    db = AppDatabase(tmp_path / "app.db")
    db.init_schema()
    voice = {
        "id": "v1",
        "name": "demo",
        "mode": "clone",
        "language_hint": "Auto",
        "ref_audio_path": "a.wav",
        "ref_text": "hello",
        "prompt_path": "p.pt",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
    }
    db.create_voice(voice)
    rows = db.list_voices()
    assert len(rows) == 1
    assert rows[0]["id"] == "v1"
    got = db.get_voice("v1")
    assert got is not None
    assert got["name"] == "demo"
    deleted = db.delete_voice("v1")
    assert deleted == 1
    db.close()


def test_clear_metrics(tmp_path: Path):
    db = AppDatabase(tmp_path / "app.db")
    db.init_schema()
    db.record_metric("2026-01-01T00:00:00Z", {"a": 1})
    db.record_metric("2026-01-01T00:00:01Z", {"a": 2})
    assert len(db.list_metrics(limit=10)) == 2
    deleted = db.clear_metrics()
    assert deleted == 2
    assert db.list_metrics(limit=10) == []
    db.close()
