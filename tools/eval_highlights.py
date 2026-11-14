#!/usr/bin/env python3
"""Compare predicted made-basket times to a hand-labelled CSV.

Examples:
  python tools/eval_highlights.py --pred output/game_highlights.csv \\
      --truth tests/fixtures/clip_list_v2.csv --tolerance 0.75
  python tools/eval_highlights.py --pred pred.csv --truth truth.csv --all-rows

Exit code 0 always when the comparison runs; print precision/recall/F1.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from fhh_core import format_ts, parse_time  # noqa: E402


def _cell(row: dict, *names: str) -> str:
    lower = {str(k).strip().lower(): (v or "").strip() for k, v in row.items() if k is not None}
    for n in names:
        if lower.get(n.lower()):
            return lower[n.lower()]
    return ""


def _parse_t(raw: str) -> float:
    try:
        return parse_time(raw)
    except ValueError:
        return float(raw)


def load_truth_times(path: Path) -> list[float]:
    times: list[float] = []
    with open(path, newline="", encoding="utf-8-sig") as fp:
        for row in csv.DictReader(fp):
            raw = _cell(row, "ball_in", "event_time", "score_time", "time", "event_s")
            if not raw:
                continue
            times.append(_parse_t(raw))
    times.sort()
    return times


_MAKE_TYPES = {
    "made", "make", "basket", "shot", "layup", "three", "jumper", "drive",
    "putback", "bank", "fastbreak", "long shot", "dunk", "floater", "hook",
}


def load_pred_times(path: Path, included_only: bool = True) -> list[float]:
    times: list[float] = []
    with open(path, newline="", encoding="utf-8-sig") as fp:
        reader = csv.DictReader(fp)
        fields = {str(k).strip().lower() for k in (reader.fieldnames or []) if k}
        has_include = "include" in fields
        for row in reader:
            if included_only and has_include:
                inc = _cell(row, "include").lower()
                cid = _cell(row, "id")
                if cid.startswith("#"):
                    continue
                if inc in ("no", "n", "0", "false", "off"):
                    continue
            kind = _cell(row, "type").lower()
            # App clips use type=made|attempt|...; hand labels use layup/shot/etc.
            if kind and kind not in _MAKE_TYPES and kind not in ("", "manual", "play"):
                if kind in ("attempt", "miss", "rejected"):
                    continue
            raw = _cell(row, "event_time", "ball_in", "event_s", "time")
            if not raw:
                continue
            times.append(_parse_t(raw))
    times.sort()
    return times


def match_times(pred: list[float], truth: list[float], tolerance: float
                ) -> tuple[int, list[float], list[float]]:
    """Greedy 1-1 matching within tolerance. Returns (tp, unmatched_truth, unmatched_pred)."""
    used_t = set()
    used_p = set()
    tp = 0
    for i, p in enumerate(pred):
        best_j = None
        best_d = None
        for j, t in enumerate(truth):
            if j in used_t:
                continue
            d = abs(p - t)
            if d <= tolerance and (best_d is None or d < best_d):
                best_d, best_j = d, j
        if best_j is not None:
            used_t.add(best_j)
            used_p.add(i)
            tp += 1
    unmatched_truth = [t for j, t in enumerate(truth) if j not in used_t]
    unmatched_pred = [p for i, p in enumerate(pred) if i not in used_p]
    return tp, unmatched_truth, unmatched_pred


def metrics(tp: int, n_pred: int, n_truth: int) -> dict:
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_truth if n_truth else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return dict(tp=tp, pred=n_pred, truth=n_truth, precision=precision, recall=recall, f1=f1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", required=True, help="predicted highlights CSV")
    ap.add_argument("--truth", required=True, help="hand-labelled CSV with ball_in / event_time")
    ap.add_argument("--tolerance", type=float, default=0.75, help="match window in seconds")
    ap.add_argument("--all-rows", action="store_true",
                    help="score all prediction rows, not only include=yes")
    args = ap.parse_args(argv)

    pred = load_pred_times(Path(args.pred), included_only=not args.all_rows)
    truth = load_truth_times(Path(args.truth))
    tp, miss, fp = match_times(pred, truth, args.tolerance)
    m = metrics(tp, len(pred), len(truth))
    print(f"truth={m['truth']}  pred={m['pred']}  tp={m['tp']}  "
          f"precision={m['precision']:.3f}  recall={m['recall']:.3f}  F1={m['f1']:.3f}  "
          f"tolerance={args.tolerance:.2f}s")
    if miss:
        print("unmatched truth:", ", ".join(format_ts(t) for t in miss[:20])
              + (" ..." if len(miss) > 20 else ""))
    if fp:
        print("unmatched pred:", ", ".join(format_ts(t) for t in fp[:20])
              + (" ..." if len(fp) > 20 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
