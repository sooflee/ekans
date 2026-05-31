"""
Recompute results/_multiple_testing.json across ALL current results.

The stale version (from analyze_signals.py) filters on status=="ok" — but most
older results omit the status field, so they were silently excluded from BH
correction. That makes the winner gate downstream meaningless for any signal
not in the BH table.

This script uses the same permissive status check as pipeline/winner_gate.py
(only `fail`/`failed`/`error` are excluded), and only requires `t_stat` and
`n_days` to be present. Output schema matches analyze_signals.py exactly.

Run after a fresh batch of backtests:
    .venv/bin/python pipeline/refresh_bh.py
"""
from __future__ import annotations

import json
from pathlib import Path

from scipy import stats as sp_stats

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "_multiple_testing.json"

EXCLUDED_STATUSES = {"fail", "failed", "error"}
MIN_N = 30
FDR = 0.05


def load_all_results() -> dict[str, dict]:
    out = {}
    for fp in sorted(RESULTS.glob("*.json")):
        if fp.stem.startswith("_"):
            continue
        try:
            with open(fp) as f:
                out[fp.stem] = json.load(f)
        except Exception:
            continue
    return out


def bh_correct(results: dict[str, dict]) -> tuple[dict[str, dict], int, int]:
    signals = []
    for sid, d in results.items():
        status = (d.get("status") or "").lower()
        if status in EXCLUDED_STATUSES:
            continue
        t = d.get("t_stat")
        n = d.get("n_days")
        if t is None or n is None or n < MIN_N:
            continue
        p = 2 * (1 - sp_stats.t.cdf(abs(t), df=n - 1))
        signals.append({"signal_id": sid, "t_stat": t, "n_days": n, "p_value": float(p)})

    signals.sort(key=lambda x: x["p_value"])
    m = len(signals)

    for i, s in enumerate(signals):
        rank = i + 1
        s["bh_rank"] = rank
        s["bh_threshold"] = FDR * rank / m if m else 0.0

    max_reject = 0
    for i, s in enumerate(signals):
        if s["p_value"] <= s["bh_threshold"]:
            max_reject = i + 1

    for i, s in enumerate(signals):
        s["bh_significant"] = (i + 1) <= max_reject

    return {s["signal_id"]: s for s in signals}, m, max_reject


def main():
    results = load_all_results()
    print(f"Loaded {len(results)} result files.")
    table, m, k = bh_correct(results)
    print(f"BH-tested {m} signals (filtered out {len(results)-m} for missing/short t-stat or failed status)")
    print(f"BH-significant at FDR={FDR}: {k}")
    OUT.write_text(json.dumps(table, indent=2))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
