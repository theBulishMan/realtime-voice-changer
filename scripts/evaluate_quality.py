from __future__ import annotations

import argparse
import json
from pathlib import Path


def levenshtein(a: list[str], b: list[str]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    dp = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev = dp[0]
        dp[0] = i
        for j, y in enumerate(b, 1):
            cur = dp[j]
            cost = 0 if x == y else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[-1]


def cer(reference: str, hypothesis: str) -> float:
    ref = list(reference.strip())
    hyp = list(hypothesis.strip())
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / max(len(ref), 1)


def wer(reference: str, hypothesis: str) -> float:
    ref = reference.strip().split()
    hyp = hypothesis.strip().split()
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / max(len(ref), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate quality gate from refs/hyps json")
    parser.add_argument("--input", default="reports/quality_input.json")
    parser.add_argument("--output", default="reports/quality.md")
    parser.add_argument("--target-zh-cer", type=float, default=0.08)
    parser.add_argument("--target-en-wer", type=float, default=0.12)
    parser.add_argument("--min-zh-samples", type=int, default=5)
    parser.add_argument("--min-en-samples", type=int, default=5)
    parser.add_argument("--strict", action="store_true", default=False)
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Create JSON list with fields: language, reference, hypothesis"
        )
    rows = json.loads(path.read_text(encoding="utf-8"))
    zh_scores = []
    en_scores = []
    for row in rows:
        lang = str(row.get("language", "")).lower()
        ref = str(row.get("reference", ""))
        hyp = str(row.get("hypothesis", ""))
        if lang.startswith("zh"):
            zh_scores.append(cer(ref, hyp))
        elif lang.startswith("en"):
            en_scores.append(wer(ref, hyp))

    zh_avg = sum(zh_scores) / len(zh_scores) if zh_scores else 0.0
    en_avg = sum(en_scores) / len(en_scores) if en_scores else 0.0
    zh_ok = zh_avg <= args.target_zh_cer
    en_ok = en_avg <= args.target_en_wer
    enough_zh = len(zh_scores) >= args.min_zh_samples
    enough_en = len(en_scores) >= args.min_en_samples
    if enough_zh and enough_en:
        gate = "PASS" if zh_ok and en_ok else "FAIL"
    else:
        gate = "UNVERIFIED"

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        "\n".join(
            [
                "# Quality Evaluation",
                "",
                f"- zh samples: {len(zh_scores)} (min required: {args.min_zh_samples})",
                f"- en samples: {len(en_scores)} (min required: {args.min_en_samples})",
                f"- zh CER avg: {zh_avg:.4f} (target <= {args.target_zh_cer:.4f})",
                f"- en WER avg: {en_avg:.4f} (target <= {args.target_en_wer:.4f})",
                f"- Objective gate: **{gate}**",
                "",
                "## Subjective MOS Template",
                "- Naturalness MOS (1-5): _fill manually_",
                "- Similarity MOS (1-5): _fill manually_",
                "- Final subjective gate: _pass/fail_",
            ]
        ),
        encoding="utf-8",
    )
    print(out)
    if args.strict and gate != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
