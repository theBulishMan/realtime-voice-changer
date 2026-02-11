from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any


class AppDatabase:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS voices (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    language_hint TEXT NOT NULL,
                    ref_audio_path TEXT NOT NULL,
                    ref_text TEXT,
                    prompt_path TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                """
            )

    def upsert_setting(self, key: str, value: Any) -> None:
        payload = json.dumps(value, ensure_ascii=True)
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO settings(key, value) VALUES(?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, payload),
            )

    def get_setting(self, key: str) -> Any | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM settings WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        return json.loads(row["value"])

    def create_voice(self, voice: dict[str, Any]) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO voices(
                    id, name, mode, language_hint, ref_audio_path, ref_text,
                    prompt_path, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    voice["id"],
                    voice["name"],
                    voice["mode"],
                    voice["language_hint"],
                    voice["ref_audio_path"],
                    voice.get("ref_text"),
                    voice["prompt_path"],
                    voice["created_at"],
                    voice["updated_at"],
                ),
            )

    def list_voices(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM voices ORDER BY created_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def get_voice(self, voice_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM voices WHERE id = ?", (voice_id,)
            ).fetchone()
        return None if row is None else dict(row)

    def delete_voice(self, voice_id: str) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM voices WHERE id = ?", (voice_id,))
        return int(cur.rowcount)

    def record_metric(self, ts: str, payload: dict[str, Any]) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO metrics(ts, payload) VALUES (?, ?)",
                (ts, json.dumps(payload, ensure_ascii=True)),
            )

    def list_metrics(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT ts, payload FROM metrics ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "ts": row["ts"],
                    "payload": json.loads(row["payload"]),
                }
            )
        return out

    def clear_metrics(self) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute("DELETE FROM metrics")
        return int(cur.rowcount)
