#!/usr/bin/env python3
"""
pipeline/find_sleepers.py — surface "doesn't yield now, yields a lot later" signals.

Scans results/*.json for sleepers (see sleeper_gate.py): signals with a strong
*conditional* edge (big payoff when they fire) that the standard winner_gate
rejects because their blended Sharpe is diluted by long flat stretches.

Conditional metrics are read straight from the result JSON when present (new
backtests write them via harness.conditional_metrics). For older results that
predate those fields, we backfill from the saved daily PnL in results/pnl/*.parquet,
using non-zero pnl as the "active" proxy — exact for long-only timing overlays,
approximate for always-invested long/short. Backfilled rows are flagged ~.

Usage:
  python pipeline/find_sleepers.py                 # rank hidden sleepers (sleeper & not winner)
  python pipeline/find_sleepers.py --all           # rank all sleepers, winner or not
  python pipeline/find_sleepers.py --top 30        # show N rows (default 25)
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "backtests"))

from sleeper_gate import is_sleeper, is_hidden_sleeper, sleeper_score  # noqa: E402
from winner_gate import is_winner, load_mt_data  # noqa: E402

PNL_DIR = ROOT / "results" / "pnl"
COND_KEYS = ("cond_sharpe", "n_episodes", "cond_oos_sharpe", "mean_episode_ret", "active_frac")


def backfill_conditional(result: dict) -> bool:
    """Add conditional_* fields from the saved PnL parquet. Returns True if it could."""
    import pandas as pd
    from harness import conditional_metrics

    sid = result.get("signal_id") or result.get("id")
    fp = PNL_DIR / f"{sid}.parquet"
    if not fp.exists():
        return False
    pnl = pd.read_parquet(fp)["pnl"].dropna()
    if len(pnl) < 60:
        return False
    result.update(conditional_metrics(pnl, pnl != 0))
    result["_cond_backfilled"] = True
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Find sleeper signals.")
    ap.add_argument("--all", action="store_true", help="include sleepers that are also winners")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args(argv)

    mt = load_mt_data()
    rows, no_cond, backfilled = [], 0, 0

    for fp in glob.glob(str(ROOT / "results" / "*.json")):
        if Path(fp).name.startswith("_"):
            continue
        try:
            r = json.load(open(fp))
        except Exception:
            continue
        if not isinstance(r, dict):
            continue
        if r.get("cond_sharpe") is None and "cond_sharpe" not in r:
            if backfill_conditional(r):
                backfilled += 1
            else:
                no_cond += 1
                continue

        if not is_sleeper(r):
            continue
        hidden = not is_winner(r, mt)
        if not args.all and not hidden:
            continue
        rows.append({
            "id": r.get("signal_id") or r.get("id"),
            "score": sleeper_score(r),
            "cond_sharpe": r.get("cond_sharpe"),
            "cond_oos": r.get("cond_oos_sharpe"),
            "n_ep": r.get("n_episodes"),
            "ep_ret": r.get("mean_episode_ret"),
            "active": r.get("active_frac"),
            "blended_sharpe": r.get("sharpe"),
            "winner": not hidden,
            "bf": r.get("_cond_backfilled", False),
        })

    rows.sort(key=lambda x: x["score"], reverse=True)
    label = "all sleepers" if args.all else "HIDDEN sleepers (pass sleeper_gate, fail winner_gate)"
    print(f"=== {label} ===")
    print(f"(backfilled cond metrics for {backfilled}; {no_cond} results had no cond data and no saved pnl)\n")
    hdr = f"{'rank':>4}  {'signal_id':38s} {'score':>6} {'cSharpe':>7} {'cOOS':>6} {'nEp':>4} {'epRet%':>7} {'active%':>7} {'blendSh':>7}"
    print(hdr)
    print("-" * len(hdr))
    for i, x in enumerate(rows[:args.top], 1):
        flag = "~" if x["bf"] else " "
        win = " [also winner]" if x["winner"] else ""
        print(f"{i:>4}{flag} {str(x['id'])[:38]:38s} {x['score']:6.2f} "
              f"{(x['cond_sharpe'] or 0):7.2f} {(x['cond_oos'] or 0):6.2f} {x['n_ep']:4d} "
              f"{(x['ep_ret'] or 0)*100:7.2f} {(x['active'] or 0)*100:7.1f} {(x['blended_sharpe'] or 0):7.2f}{win}")
    print(f"\n{len(rows)} sleeper(s) total. '~' = conditional metrics backfilled from saved pnl (approximate).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
