from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_metrics(path: Path) -> tuple[list[float], list[float]]:
    fad: list[float] = []
    e2e: list[float] = []
    if not path.exists():
        return fad, e2e
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        fad.append(float(row.get("fad_ms", 0.0)))
        e2e.append(float(row.get("e2e_ms", 0.0)))
    return fad, e2e


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.array(values, dtype=np.float32), p))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute latency gate report from metrics.ndjson")
    parser.add_argument("--input", default="data/logs/metrics.ndjson")
    parser.add_argument("--output", default="reports/latency.md")
    parser.add_argument("--target-fad", type=float, default=250.0)
    parser.add_argument("--target-e2e", type=float, default=800.0)
    parser.add_argument("--min-samples", type=int, default=30)
    parser.add_argument("--strict", action="store_true", default=False)
    args = parser.parse_args()

    src = Path(args.input)
    fad, e2e = read_metrics(src)
    sample_count = min(len(fad), len(e2e))
    p95_fad = pct(fad, 95)
    p95_e2e = pct(e2e, 95)
    enough_samples = sample_count >= args.min_samples
    if not enough_samples:
        gate = "UNVERIFIED"
    else:
        gate = "PASS" if p95_fad <= args.target_fad and p95_e2e <= args.target_e2e else "FAIL"

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(
            [
                "# Latency Benchmark",
                "",
                f"- Samples: {sample_count}",
                f"- Min samples required: {args.min_samples}",
                f"- P50 FAD: {pct(fad, 50):.2f} ms",
                f"- P95 FAD: {p95_fad:.2f} ms (target <= {args.target_fad:.2f} ms)",
                f"- P50 E2E: {pct(e2e, 50):.2f} ms",
                f"- P95 E2E: {p95_e2e:.2f} ms (target <= {args.target_e2e:.2f} ms)",
                f"- Gate: **{gate}**",
            ]
        ),
        encoding="utf-8",
    )
    print(out)
    if args.strict and gate != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
