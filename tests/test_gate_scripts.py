from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_script(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )


def test_latency_benchmark_unverified_strict(tmp_path: Path):
    input_path = tmp_path / "metrics.ndjson"
    output_path = tmp_path / "latency.md"
    input_path.write_text("", encoding="utf-8")

    proc = _run_script(
        [
            "scripts/benchmark_latency.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--min-samples",
            "3",
            "--strict",
        ]
    )
    assert proc.returncode == 1
    content = output_path.read_text(encoding="utf-8")
    assert "Gate: **UNVERIFIED**" in content


def test_latency_benchmark_pass_strict(tmp_path: Path):
    input_path = tmp_path / "metrics.ndjson"
    output_path = tmp_path / "latency.md"
    rows = [
        {"fad_ms": 120.0, "e2e_ms": 450.0},
        {"fad_ms": 180.0, "e2e_ms": 500.0},
        {"fad_ms": 140.0, "e2e_ms": 520.0},
    ]
    input_path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    proc = _run_script(
        [
            "scripts/benchmark_latency.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--min-samples",
            "3",
            "--strict",
        ]
    )
    assert proc.returncode == 0
    content = output_path.read_text(encoding="utf-8")
    assert "Gate: **PASS**" in content


def test_quality_eval_unverified_strict(tmp_path: Path):
    input_path = tmp_path / "quality_input.json"
    output_path = tmp_path / "quality.md"
    input_path.write_text("[]", encoding="utf-8")

    proc = _run_script(
        [
            "scripts/evaluate_quality.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--strict",
        ]
    )
    assert proc.returncode == 1
    content = output_path.read_text(encoding="utf-8")
    assert "Objective gate: **UNVERIFIED**" in content
