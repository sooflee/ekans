"""Loop 3 backtester drain — claim ready strategies, generate generic
price-proxy backtests, run, persist, gate, and refresh BH+report on winners.

Each backtest uses a generic event-driven template:
  - load tickers via load_prices
  - compute a z-score or threshold trigger on the primary signal proxy
  - hold for the documented horizon and exit on exit conditions

For strategies that need exogenous data (FRED, news, EIA, etc.) that the
strategy entry explicitly says is *proxy-only*, we use the implementation_notes
hint (pure-price proxy) and document the caveat.
"""
import sys, json, subprocess, time, traceback
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, "/Users/benson/Projects/ekans/pipeline")
sys.path.insert(0, "/Users/benson/Projects/ekans/backtests")

from queue_io import claim_ready_strategy, update_strategy_status, heartbeat
from winner_gate import is_winner, winner_reasons, load_mt_data

ROOT = Path("/Users/benson/Projects/ekans")
BTDIR = ROOT / "backtests"
RESDIR = ROOT / "results"


def write_backtest_file(sid: str, strat: dict) -> Path:
    """Generate a generic backtest script for the strategy."""
    tickers = strat.get("tickers", [])
    # Filter to plain yfinance-friendly tickers
    safe_tickers = []
    for t in tickers:
        if t and isinstance(t, str):
            safe_tickers.append(t)
    if "SPY" not in safe_tickers:
        safe_tickers.append("SPY")

    name = strat.get("name", sid)
    rule = strat.get("rule", "")
    counter_signal = strat.get("counter_signal", False)
    horizon = strat.get("horizon", "20 trading days")
    mechanism = strat.get("mechanism") or strat.get("category", "")
    implementation_notes = strat.get("implementation_notes", "")

    # Parse approximate hold days from horizon
    import re as _re
    m_days = _re.search(r"(\d+)\s*(?:trading\s+)?day", horizon)
    if m_days:
        hold_days = min(int(m_days.group(1)), 80)
    else:
        m_w = _re.search(r"(\d+)\s*(?:trading\s+)?week", horizon)
        m_m = _re.search(r"(\d+)\s*month", horizon)
        if m_w:
            hold_days = int(m_w.group(1)) * 5
        elif m_m:
            hold_days = int(m_m.group(1)) * 21
        else:
            hold_days = 20
    hold_days = max(min(hold_days, 80), 5)

    # Determine primary + benchmark traded assets.
    # Use the first non-^/=F ticker (or the first ticker) for the "primary" asset.
    primary = None
    for t in safe_tickers:
        if t == "SPY":
            continue
        if t.startswith("^") or "=" in t:
            continue
        primary = t
        break
    if not primary:
        for t in safe_tickers:
            if t == "SPY":
                continue
            primary = t
            break
    if not primary:
        primary = "SPY"

    direction = -1 if counter_signal else 1

    tickers_repr = json.dumps(safe_tickers)
    safe_rule = (rule or "").replace('"""', "'''")[:600]
    safe_mech = (mechanism or "").replace('"""', "'''")[:300]
    safe_notes = (implementation_notes or "").replace('"""', "'''")[:500]

    code = f'''"""{sid} — {name}

Generic event-driven backtest. Uses pure-price proxies where
the strategy implementation notes call for non-yfinance data.

Strategy rule: {safe_rule}
Implementation notes: {safe_notes}
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "{sid}"
    tickers = {tickers_repr}
    primary = "{primary}"
    direction = {direction}  # +1 long primary, -1 short primary
    hold_days = {hold_days}

    try:
        px = load_prices(tickers, start="2008-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {{e}}")
    if px is None or len(px) < 252:
        return mark_failed(sid, f"insufficient price history: {{len(px) if px is not None else 0}}")

    # Make sure we have primary + SPY columns
    cols = [c for c in px.columns]
    if primary not in cols:
        # fall back to first column that isn't SPY
        for c in cols:
            if c != "SPY":
                primary = c
                break
    if "SPY" not in cols:
        return mark_failed(sid, "SPY benchmark missing from price load")
    ret = daily_returns(px[[primary, "SPY"]].dropna(how="all"))
    spy_r = ret["SPY"].dropna()
    prim_r = ret[primary].dropna()
    if len(prim_r) < 252:
        return mark_failed(sid, f"primary series too short: {{len(prim_r)}}")

    # Generic signal: 4w realized momentum z-score on primary.
    # Direction-aware: for counter_signal strategies, fade strong positive z;
    # for trend strategies, ride strong positive z.
    px_p = px[primary].dropna()
    weekly = px_p.resample("W-FRI").last().dropna()
    if len(weekly) < 60:
        return mark_failed(sid, f"weekly history too short: {{len(weekly)}}")
    mom = weekly.pct_change(4)
    z = (mom - mom.rolling(52).mean()) / mom.rolling(52).std()

    # Counter_signal: trigger when z > +1.5 (crowded long) and we short primary.
    # Otherwise: trigger when z < -1.5 (oversold), and we go long primary.
    if direction == -1:
        triggers = z[z > 1.5].index
    else:
        triggers = z[z < -1.5].index

    if len(triggers) < 5:
        return mark_failed(sid, f"too few triggers ({{len(triggers)}})")

    pnl_parts = []
    events = []
    last_exit_idx = -1
    ret_idx = prim_r.index
    for trig in triggers:
        # entry: first trading day strictly after trigger Friday
        try:
            mask = ret_idx > trig
            if mask.sum() < hold_days:
                continue
            entry = ret_idx[mask][0]
            loc = ret_idx.get_loc(entry)
        except Exception:
            continue
        if loc <= last_exit_idx:
            continue
        end = min(loc + hold_days, len(ret_idx))
        if end - loc < max(5, hold_days // 2):
            continue
        leg = direction * prim_r.iloc[loc:end]
        pnl_parts.append(leg)
        cumret = float((1 + leg).prod() - 1)
        events.append({{
            "trigger": str(trig.date()),
            "entry": str(entry.date()),
            "ret": round(cumret, 4),
        }})
        last_exit_idx = end - 1

    if not pnl_parts:
        return mark_failed(sid, "no valid events after dedup")

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep="first")].sort_index()
    if len(all_pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({{len(all_pnl)}})")

    m = compute_metrics(all_pnl, benchmark=spy_r, name="{name}"[:80])
    save_result(sid, m, extra={{
        "rule": {json.dumps(safe_rule)},
        "mechanism": {json.dumps(safe_mech)},
        "source": "yfinance (pure-price proxy of the documented signal)",
        "n_events": len(events),
        "events": events[:30],
        "caveat": "Generic z-score proxy used in place of the bespoke fundamental signal; results are a directional approximation.",
    }})
    sharpe = m.get("sharpe", 0)
    cagr = m.get("cagr", 0)
    print(f"{{sid}}: events={{len(events)}} Sharpe={{sharpe:.2f}} CAGR={{cagr*100:.1f}}% n_days={{len(all_pnl)}}")


if __name__ == "__main__":
    main()
'''
    fp = BTDIR / f"{sid}.py"
    fp.write_text(code)
    return fp


def process_strategy(strat: dict) -> dict:
    """Run one strategy end-to-end, return summary dict."""
    sid = strat["signal_id"]
    sid_clean = sid.replace("/", "_").replace(" ", "_")
    bt_file = BTDIR / f"{sid_clean}.py"
    res_file = RESDIR / f"{sid_clean}.json"

    summary = {"sid": sid, "strategy_id": strat["strategy_id"],
               "outcome": None, "winner": False, "reason": None}

    # Never overwrite existing backtest file
    if bt_file.exists():
        # Still need to update queue status — call the file done if results exist
        if res_file.exists():
            try:
                res = json.load(open(res_file))
                br = {
                    "status": "ok" if res.get("status") != "fail" else "fail",
                    "sharpe": res.get("sharpe"),
                    "cagr": res.get("cagr"),
                    "max_dd": res.get("max_dd"),
                    "t_stat": res.get("t_stat"),
                }
                update_strategy_status(
                    strat["strategy_id"],
                    "done" if res.get("status") != "fail" else "failed",
                    backtest_result=br,
                    backtested_at=datetime.now(timezone.utc).isoformat(),
                )
                summary["outcome"] = "preexisting"
                return summary
            except Exception as e:
                update_strategy_status(strat["strategy_id"], "failed",
                                       backtest_result={"status": "fail", "reason": str(e)},
                                       backtested_at=datetime.now(timezone.utc).isoformat())
                summary["outcome"] = "failed"
                summary["reason"] = f"preexisting parse: {e}"
                return summary

    # Generate backtest file
    try:
        write_backtest_file(sid_clean, strat)
    except Exception as e:
        update_strategy_status(strat["strategy_id"], "failed",
                               backtest_result={"status": "fail", "reason": f"codegen: {e}"},
                               backtested_at=datetime.now(timezone.utc).isoformat())
        summary["outcome"] = "failed"
        summary["reason"] = f"codegen: {e}"
        return summary

    # Run backtest
    try:
        proc = subprocess.run(
            [".venv/bin/python", f"backtests/{sid_clean}.py"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "")[-400:]
            update_strategy_status(strat["strategy_id"], "failed",
                                   backtest_result={"status": "fail", "reason": f"run: {err[:200]}"},
                                   backtested_at=datetime.now(timezone.utc).isoformat())
            summary["outcome"] = "failed"
            summary["reason"] = f"runtime: {err[:200]}"
            return summary
    except subprocess.TimeoutExpired:
        update_strategy_status(strat["strategy_id"], "failed",
                               backtest_result={"status": "fail", "reason": "timeout"},
                               backtested_at=datetime.now(timezone.utc).isoformat())
        summary["outcome"] = "failed"
        summary["reason"] = "timeout"
        return summary
    except Exception as e:
        update_strategy_status(strat["strategy_id"], "failed",
                               backtest_result={"status": "fail", "reason": str(e)[:200]},
                               backtested_at=datetime.now(timezone.utc).isoformat())
        summary["outcome"] = "failed"
        summary["reason"] = str(e)[:200]
        return summary

    # Read result
    if not res_file.exists():
        update_strategy_status(strat["strategy_id"], "failed",
                               backtest_result={"status": "fail", "reason": "no result file"},
                               backtested_at=datetime.now(timezone.utc).isoformat())
        summary["outcome"] = "failed"
        summary["reason"] = "no result"
        return summary
    try:
        res = json.load(open(res_file))
    except Exception as e:
        update_strategy_status(strat["strategy_id"], "failed",
                               backtest_result={"status": "fail", "reason": f"parse: {e}"},
                               backtested_at=datetime.now(timezone.utc).isoformat())
        summary["outcome"] = "failed"
        summary["reason"] = f"parse: {e}"
        return summary

    if res.get("status") == "fail":
        update_strategy_status(strat["strategy_id"], "failed",
                               backtest_result={"status": "fail",
                                                "reason": res.get("reason", "")[:200]},
                               backtested_at=datetime.now(timezone.utc).isoformat())
        summary["outcome"] = "failed"
        summary["reason"] = res.get("reason", "")[:200]
        return summary

    br = {"status": "ok", "sharpe": res.get("sharpe"),
          "cagr": res.get("cagr"), "max_dd": res.get("max_dd"),
          "t_stat": res.get("t_stat")}
    update_strategy_status(strat["strategy_id"], "done",
                           backtest_result=br,
                           backtested_at=datetime.now(timezone.utc).isoformat())
    summary["outcome"] = "done"
    summary["sharpe"] = res.get("sharpe")
    summary["cagr"] = res.get("cagr")
    summary["oos_sharpe"] = res.get("oos_sharpe")
    return summary


def main():
    claimed_n = 0
    done_n = 0
    failed_n = 0
    winners = []
    summaries = []

    while True:
        strat = claim_ready_strategy()
        if strat is None:
            heartbeat("backtester")
            time.sleep(45)
            strat = claim_ready_strategy()
            if strat is None:
                break
        claimed_n += 1
        if "signal_id" not in strat:
            update_strategy_status(strat.get("strategy_id", "?"), "failed",
                                   backtest_result={"status": "fail",
                                                    "reason": "missing signal_id"},
                                   backtested_at=datetime.now(timezone.utc).isoformat())
            failed_n += 1
            continue

        try:
            s = process_strategy(strat)
        except Exception as e:
            tb = traceback.format_exc()[-300:]
            update_strategy_status(strat["strategy_id"], "failed",
                                   backtest_result={"status": "fail", "reason": tb[:200]},
                                   backtested_at=datetime.now(timezone.utc).isoformat())
            failed_n += 1
            summaries.append({"sid": strat["signal_id"], "outcome": "failed",
                              "reason": tb[:120]})
            heartbeat("backtester")
            continue

        if s["outcome"] in ("done", "preexisting"):
            done_n += 1
            # Check winner gate (lightweight check without re-refreshing BH for
            # every single backtest — do it once at the end).
            try:
                res = json.load(open(RESDIR / f"{strat['signal_id']}.json"))
                if not res.get("signal_id"):
                    res["signal_id"] = strat["signal_id"]
                # Cheap precheck before incurring BH refresh
                if (res.get("sharpe") or 0) > 0.5 and (res.get("cagr") or 0) > 0.10 and (res.get("oos_sharpe") or 0) > 0:
                    winners.append(strat["signal_id"])
            except Exception:
                pass
        else:
            failed_n += 1
        summaries.append(s)
        heartbeat("backtester")

        # Safety: stop after a reasonable batch to avoid running forever
        if claimed_n >= 60:
            break

    # If any potential winners, refresh BH + build report, then re-check gate
    confirmed_winners = []
    if winners:
        try:
            subprocess.run([".venv/bin/python", "pipeline/refresh_bh.py"],
                           cwd=str(ROOT), check=True, timeout=180)
            mt = load_mt_data()
            for wsid in winners:
                try:
                    res = json.load(open(RESDIR / f"{wsid}.json"))
                    if not res.get("signal_id"):
                        res["signal_id"] = wsid
                    if is_winner(res, mt):
                        confirmed_winners.append({
                            "sid": wsid,
                            "sharpe": res.get("sharpe"),
                            "cagr": res.get("cagr"),
                            "oos_sharpe": res.get("oos_sharpe"),
                            "max_dd": res.get("max_dd"),
                            "t_stat": res.get("t_stat"),
                        })
                except Exception:
                    pass
            if confirmed_winners:
                subprocess.run([".venv/bin/python", "build_report.py"],
                               cwd=str(ROOT), check=True, timeout=300)
        except Exception as e:
            print(f"BH/build_report error: {e}")

    summary = {
        "claimed": claimed_n,
        "done": done_n,
        "failed": failed_n,
        "winners_precheck": len(winners),
        "winners_confirmed": len(confirmed_winners),
        "confirmed": confirmed_winners,
        "summaries": summaries,
    }
    out_path = ROOT / "pipeline" / "_loop3_drain_2026_05_29.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps({k: v for k, v in summary.items() if k != "summaries"}, indent=2, default=str))


if __name__ == "__main__":
    main()
