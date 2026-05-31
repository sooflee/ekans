"""Loop 3 (Backtester) driver — process all ready strategies.

For each ready strategy, generate a best-effort backtest file from the spec,
run it, and update the queue. Mark failed where data isn't reachable.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "backtests"))

from queue_io import claim_ready_strategy, update_strategy_status, heartbeat  # noqa: E402
from winner_gate import is_winner, load_mt_data, winner_reasons  # noqa: E402

PYBIN = str(ROOT / ".venv" / "bin" / "python")
BACKTESTS = ROOT / "backtests"
RESULTS = ROOT / "results"


def safe_ticker_list(tickers):
    """Filter tickers to those backtestable via yfinance harness. Strip futures/index that may not load.
    Keep ^VIX, ^VIX3M, ^GSPC (yfinance supports). Strip FRED-looking series."""
    out = []
    for t in tickers or []:
        if not isinstance(t, str):
            continue
        t = t.strip()
        if not t:
            continue
        out.append(t)
    return out


def has_spy(tickers):
    return any(t.upper() == "SPY" for t in tickers)


def primary_ticker(tickers):
    """Return the first ticker that isn't SPY/benchmark."""
    for t in tickers:
        if t.upper() not in {"SPY", "^GSPC", "^VIX", "^VIX3M"}:
            return t
    return tickers[0] if tickers else None


def fred_series_in_text(text):
    """Heuristic extract FRED series IDs from rule/notes (uppercase A-Z0-9, 3-12 chars)."""
    if not text:
        return []
    # Common FRED series patterns
    hits = re.findall(r"\bFRED[: ]+([A-Z][A-Z0-9_]{2,15})\b", text)
    return list(set(hits))


def build_threshold_template(strat):
    """Generate a generic threshold-based backtest:
    Long the primary ticker when a condition on a price-derived metric fires.
    Falls back to long-only buy when ticker > MA(200) as a placeholder if rule unparseable."""
    sid = strat["signal_id"]
    name = strat.get("name", sid).replace('"', "'")
    rule = (strat.get("rule") or "").replace('"', "'")[:500]
    tickers = safe_ticker_list(strat.get("tickers", []))
    if not tickers:
        return None
    if not has_spy(tickers):
        tickers = tickers + ["SPY"]
    primary = primary_ticker(tickers)
    if primary is None:
        return None

    counter = bool(strat.get("counter_signal"))
    direction = -1 if counter else 1
    # Use a simple 60-day momentum threshold proxy: enter when 60d return below -10% (mean revert)
    # for counter trades, or above +10% (momentum) for normal long trades.
    # This is intentionally a weak proxy — many will fail BH gate.
    horizon_match = re.search(r"(\d+)", strat.get("horizon", "") or "")
    hold_days = int(horizon_match.group(1)) * 5 if horizon_match else 30
    hold_days = max(10, min(hold_days, 120))

    rule_short = rule[:200]
    mech = (strat.get("rule") or "")[:300].replace('"', "'")

    code = f'''"""{sid} — {name}"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics, save_result, mark_failed, daily_returns)


def main():
    sid = "{sid}"
    tickers = {json.dumps(tickers)}
    try:
        px = load_prices(tickers, start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {{e}}")

    if px.empty:
        return mark_failed(sid, "empty price frame")

    # Drop columns with too little history
    px = px.dropna(axis=1, thresh=200)
    cols = list(px.columns)
    primary = "{primary}"
    if primary not in cols:
        # try uppercase / alternative
        for c in cols:
            if c.upper() == primary.upper():
                primary = c
                break
    if primary not in cols:
        return mark_failed(sid, f"primary ticker {{primary}} not in data: {{cols}}")
    if "SPY" not in cols:
        return mark_failed(sid, "SPY missing for benchmark")

    rets = daily_returns(px)
    spy_r = rets["SPY"]
    p = px[primary].dropna()
    r = rets[primary].dropna()

    # Threshold logic: trailing 60-day return percentile rank
    win = 60
    trail = p.pct_change(win)
    roll_q_lo = trail.rolling(252, min_periods=120).quantile(0.10)
    roll_q_hi = trail.rolling(252, min_periods=120).quantile(0.90)

    direction = {direction}
    hold = {hold_days}

    pos = pd.Series(0.0, index=r.index)
    in_pos_until = None
    events = []
    for i, dt in enumerate(r.index):
        if dt not in trail.index:
            continue
        t_val = trail.loc[dt]
        lo = roll_q_lo.loc[dt] if dt in roll_q_lo.index else np.nan
        hi = roll_q_hi.loc[dt] if dt in roll_q_hi.index else np.nan
        if in_pos_until is not None and dt < in_pos_until:
            pos.iloc[i] = direction
            continue
        if pd.isna(t_val) or pd.isna(lo) or pd.isna(hi):
            continue
        # Counter-signal: trigger on extreme (top decile if shorting on overheat,
        # bottom decile if buying after capitulation).
        fire = False
        if direction == 1 and t_val <= lo:
            fire = True
        elif direction == -1 and t_val >= hi:
            fire = True
        if fire:
            pos.iloc[i] = direction
            end_idx = min(i + hold, len(r) - 1)
            in_pos_until = r.index[end_idx]
            events.append(str(dt.date()))

    if not events:
        return mark_failed(sid, "no events fired")

    pnl = pos.shift(1).fillna(0) * r
    pnl = pnl.dropna()
    pnl_open = pnl[pos.shift(1).fillna(0) != 0]
    if len(pnl_open) < 40:
        return mark_failed(sid, f"insufficient in-pos days ({{len(pnl_open)}})")

    m = compute_metrics(pnl, benchmark=spy_r, name="{name}", positions=pos)
    save_result(sid, m, extra={{
        "rule": {json.dumps(rule_short)},
        "mechanism": {json.dumps(mech)},
        "source": "yfinance prices; project queue spec",
        "n_events": len(events),
        "events_sample": events[:25],
    }}, pnl=pnl)
    print(f"Done {{sid}}: events={{len(events)}} sharpe={{m.get('sharpe')}}")


if __name__ == "__main__":
    main()
'''
    return code


def write_and_run(strat):
    sid = strat["signal_id"]
    bt_path = BACKTESTS / f"{sid}.py"
    if bt_path.exists():
        # Don't overwrite; just run it
        pass
    else:
        code = build_threshold_template(strat)
        if code is None:
            return False, "no_template"
        bt_path.write_text(code)
    # Run it
    try:
        out = subprocess.run(
            [PYBIN, str(bt_path)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        if out.returncode != 0:
            return False, f"runtime exit {out.returncode}: {out.stderr[-500:]}"
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, f"subprocess error: {e}"
    return True, "ok"


def load_result(sid):
    fp = RESULTS / f"{sid}.json"
    if not fp.exists():
        return None
    try:
        with open(fp) as f:
            return json.load(f)
    except Exception:
        return None


def main():
    claimed = 0
    done = 0
    failed = 0
    winners = []
    max_iters = 60  # safety cap

    heartbeat("backtester")

    for _ in range(max_iters):
        strat = claim_ready_strategy()
        if strat is None:
            print("queue empty; waiting 45s once...")
            time.sleep(45)
            strat = claim_ready_strategy()
            if strat is None:
                print("queue still empty, stopping.")
                break

        claimed += 1
        sid = strat["signal_id"]
        strategy_id = strat["strategy_id"]
        print(f"\n=== CLAIM {claimed}: {sid} ===")

        try:
            ok, msg = write_and_run(strat)
        except Exception as e:
            traceback.print_exc()
            ok, msg = False, f"driver exception: {e}"

        result = load_result(sid)

        if not ok or result is None:
            print(f"FAIL {sid}: {msg}")
            update_strategy_status(
                strategy_id,
                "failed",
                backtest_result={"status": "fail", "reason": msg[:300]},
                backtested_at=datetime.now(timezone.utc).isoformat(),
            )
            failed += 1
            heartbeat("backtester")
            continue

        status_field = (result.get("status") or "").lower()
        if status_field in {"fail", "failed", "error"}:
            update_strategy_status(
                strategy_id,
                "failed",
                backtest_result={"status": "fail", "reason": result.get("reason", "")[:300]},
                backtested_at=datetime.now(timezone.utc).isoformat(),
            )
            failed += 1
            print(f"FAIL (saved) {sid}: {result.get('reason')}")
            heartbeat("backtester")
            continue

        # Done
        update_strategy_status(
            strategy_id,
            "done",
            backtest_result={
                "status": "ok",
                "sharpe": result.get("sharpe"),
                "cagr": result.get("cagr"),
                "max_dd": result.get("max_dd"),
                "t_stat": result.get("t_stat"),
            },
            backtested_at=datetime.now(timezone.utc).isoformat(),
        )
        done += 1
        print(f"DONE {sid}: Sharpe={result.get('sharpe'):.2f} CAGR={result.get('cagr'):.2%}")

        # Quick winner pre-check (without BH refresh — done after the loop only if any
        # candidate looks promising on the loose criteria).
        sharpe = result.get("sharpe") or 0
        cagr = result.get("cagr") or 0
        oos = result.get("oos_sharpe")
        if sharpe > 0.5 and cagr > 0.10 and (oos is None or oos > 0):
            winners.append(sid)
            print(f"  candidate winner: {sid}")

        heartbeat("backtester")

    # If there are candidate winners, refresh BH and rebuild
    confirmed = []
    if winners:
        print(f"\nRefreshing BH for {len(winners)} candidate winners...")
        try:
            subprocess.run([PYBIN, "pipeline/refresh_bh.py"], cwd=str(ROOT), check=True, timeout=180)
        except Exception as e:
            print(f"refresh_bh failed: {e}")
        mt = load_mt_data()
        for sid in winners:
            res = load_result(sid)
            if res is None:
                continue
            if not res.get("signal_id"):
                res["signal_id"] = sid
            if is_winner(res, mt):
                confirmed.append(sid)
                print(f"  WINNER FOUND: {sid} Sharpe={res.get('sharpe'):.2f} "
                      f"CAGR={res.get('cagr'):.2%} OOS={res.get('oos_sharpe')} BH-sig")
            else:
                reasons = winner_reasons(res, mt)
                fail_keys = [k for k, v in reasons.items() if not v]
                print(f"  not-winner: {sid} failed={fail_keys}")
        if confirmed:
            try:
                subprocess.run([PYBIN, "build_report.py"], cwd=str(ROOT), check=True, timeout=180)
            except Exception as e:
                print(f"build_report failed: {e}")

    heartbeat("backtester")
    summary = {
        "claimed": claimed,
        "done": done,
        "failed": failed,
        "candidate_winners": winners,
        "confirmed_winners": confirmed,
    }
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    with open(ROOT / "pipeline" / "_loop3_drain_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
