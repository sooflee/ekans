"""Drainer loop — one of 6 parallel backtesters draining strategies_queue.
Uses queue_io for atomic claim/update. Generates a per-signal backtest file
when none exists, runs it, classifies, marks winners.
"""
from __future__ import annotations

import sys
import json
import subprocess
import traceback
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "backtests"))

from queue_io import claim_ready_strategy, update_strategy_status, heartbeat
from winner_gate import is_winner, load_mt_data, winner_reasons
from harness import mark_failed

# Import the generic FRED runner from the batch backtester
import importlib.util
spec = importlib.util.spec_from_file_location(
    "_generate_and_run_pl", str(ROOT / "backtests" / "_generate_and_run_pl.py")
)
batch_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch_mod)
generic_fred_backtest = batch_mod.generic_fred_backtest


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def process_one(strat: dict) -> tuple[str, dict | None]:
    """Run backtest for strat. Returns (outcome, result_dict | None)
    outcome in {"done","failed","winner"}."""
    sid = strat.get("signal_id") or strat["strategy_id"]
    script = ROOT / "backtests" / f"{sid}.py"

    try:
        if script.exists():
            # Run the dedicated script in a subprocess to keep isolation
            cp = subprocess.run(
                [str(ROOT / ".venv/bin/python"), str(script)],
                cwd=str(ROOT), capture_output=True, text=True, timeout=600,
            )
            if cp.returncode != 0:
                err = (cp.stderr or cp.stdout)[-400:]
                mark_failed(sid, f"script exit {cp.returncode}: {err}")
        else:
            # Use the generic FRED-driven runner from the batch module
            generic_fred_backtest(strat)
    except subprocess.TimeoutExpired:
        mark_failed(sid, "timeout running dedicated script")
    except Exception as e:
        tb = traceback.format_exc()[-500:]
        mark_failed(sid, f"runner error: {e!r} :: {tb}")

    rpath = ROOT / "results" / f"{sid}.json"
    if not rpath.exists():
        return "failed", None

    try:
        with open(rpath) as f:
            result = json.load(f)
    except Exception:
        return "failed", None

    if not result.get("signal_id"):
        result["signal_id"] = sid

    status_val = (result.get("status") or "").lower()
    if status_val in {"fail", "failed", "error"}:
        return "failed", result

    return "done", result


def main():
    claimed_count = 0
    done_count = 0
    failed_count = 0
    winners = []

    while True:
        try:
            strat = claim_ready_strategy()
        except Exception as e:
            print(f"[drain] claim error: {e!r}", flush=True)
            break

        if strat is None:
            print("[drain] queue empty.", flush=True)
            break

        claimed_count += 1
        sid = strat.get("signal_id") or strat["strategy_id"]
        strategy_id = strat["strategy_id"]
        print(f"\n[drain] #{claimed_count} claimed {strategy_id} -> {sid}", flush=True)

        try:
            outcome, result = process_one(strat)
        except Exception as e:
            tb = traceback.format_exc()[-500:]
            print(f"[drain] process_one fatal: {e!r}\n{tb}", flush=True)
            try:
                update_strategy_status(
                    strategy_id, "failed",
                    backtested_at=now_iso(),
                    backtest_result={"status": "fail", "reason": f"drain fatal: {e!r}"},
                )
            except Exception:
                pass
            failed_count += 1
            heartbeat("backtester")
            continue

        if outcome == "failed":
            reason = (result or {}).get("reason", "unknown") if result else "no result file"
            try:
                update_strategy_status(
                    strategy_id, "failed",
                    backtested_at=now_iso(),
                    backtest_result={"status": "fail", "reason": str(reason)[:300]},
                )
            except Exception as e:
                print(f"[drain] update_strategy_status failed: {e!r}", flush=True)
            failed_count += 1
            print(f"[drain] FAILED {sid}: {reason}", flush=True)
        else:
            sharpe = result.get("sharpe")
            cagr = result.get("cagr")
            t_stat = result.get("t_stat")
            max_dd = result.get("max_dd")
            try:
                update_strategy_status(
                    strategy_id, "done",
                    backtested_at=now_iso(),
                    backtest_result={
                        "status": "ok",
                        "sharpe": round(sharpe, 4) if sharpe is not None else None,
                        "cagr": round(cagr, 4) if cagr is not None else None,
                        "max_dd": round(max_dd, 4) if max_dd is not None else None,
                        "t_stat": round(t_stat, 4) if t_stat is not None else None,
                        "n_events": result.get("n_events"),
                        "n_days": result.get("n_days"),
                    },
                )
            except Exception as e:
                print(f"[drain] update_strategy_status done failed: {e!r}", flush=True)
            done_count += 1
            print(f"[drain] DONE   {sid}: Sharpe={sharpe}, CAGR={cagr}", flush=True)

            # Loose pre-filter to skip BH refresh cost on obvious non-winners
            if (sharpe is not None and sharpe > 0.5 and
                    cagr is not None and cagr > 0.10):
                # Refresh BH then check the gate
                try:
                    subprocess.run(
                        [str(ROOT / ".venv/bin/python"), str(ROOT / "pipeline/refresh_bh.py")],
                        check=True, cwd=str(ROOT), capture_output=True, timeout=180,
                    )
                except Exception as e:
                    print(f"[drain] BH refresh failed: {e!r}", flush=True)
                mt = load_mt_data()
                if is_winner(result, mt):
                    winners.append({
                        "signal_id": sid,
                        "strategy_id": strategy_id,
                        "sharpe": sharpe, "cagr": cagr,
                        "oos_sharpe": result.get("oos_sharpe"),
                        "max_dd": max_dd, "t_stat": t_stat,
                        "name": strat.get("name"),
                    })
                    print(f"WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, "
                          f"CAGR {cagr*100:.2f}%, "
                          f"OOS {result.get('oos_sharpe')}, BH-sig", flush=True)
                else:
                    reasons = winner_reasons(result, mt)
                    print(f"[drain] {sid} passed loose but missed gate: {reasons}", flush=True)

        heartbeat("backtester")

    print("\n=== DRAIN REPORT ===", flush=True)
    print(f"claimed: {claimed_count}", flush=True)
    print(f"done:    {done_count}", flush=True)
    print(f"failed:  {failed_count}", flush=True)
    print(f"winners: {len(winners)}", flush=True)
    for w in winners:
        print(f"  WIN {w['signal_id']}: Sharpe={w['sharpe']:.2f}, "
              f"CAGR={w['cagr']*100:.1f}%, OOS={w.get('oos_sharpe')}", flush=True)

    # Save report
    report_path = ROOT / "pipeline" / "_drain_loop_4_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "finished_at": now_iso(),
            "claimed": claimed_count,
            "done": done_count,
            "failed": failed_count,
            "winners": winners,
        }, f, indent=2, default=str)
    print(f"[drain] report -> {report_path}", flush=True)


if __name__ == "__main__":
    main()
