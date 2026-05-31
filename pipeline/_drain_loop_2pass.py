"""Second-pass drainer for strategies_queue.

Drains all 'ready' strategies. On empty, waits 45s and retries once before
exiting (per spec) — strategy_developer agents may still be appending.

Three other backtesters are running in parallel; partition via queue_io's
LOCK_EX-backed claim_ready_strategy(). Never overwrites existing
backtests/<sid>.py files (they may carry triage fixes).
"""
from __future__ import annotations

import sys
import json
import subprocess
import time
import traceback
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "backtests"))

from queue_io import claim_ready_strategy, update_strategy_status, heartbeat
from winner_gate import is_winner, load_mt_data, winner_reasons
from harness import mark_failed

# Generic FRED runner for strategies that don't ship a dedicated backtest file
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
    """Run backtest for strat. Returns (outcome, result_dict | None).

    outcome in {"done","failed"}.

    IMPORTANT: never overwrites an existing backtests/<sid>.py — triage agents
    may have corrected it. If no script exists, fall back to the generic FRED
    runner from _generate_and_run_pl.
    """
    sid = strat.get("signal_id") or strat["strategy_id"]
    script = ROOT / "backtests" / f"{sid}.py"

    try:
        if script.exists():
            cp = subprocess.run(
                [str(ROOT / ".venv/bin/python"), str(script)],
                cwd=str(ROOT), capture_output=True, text=True, timeout=600,
            )
            if cp.returncode != 0:
                err = (cp.stderr or cp.stdout)[-400:]
                mark_failed(sid, f"script exit {cp.returncode}: {err}")
        else:
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


def claim_with_retry() -> dict | None:
    """Claim one ready strategy. If queue is empty, wait 45s and retry once."""
    s = claim_ready_strategy()
    if s is not None:
        return s
    print("[drain2] queue appears empty; waiting 45s for strategy_devs...", flush=True)
    time.sleep(45)
    s = claim_ready_strategy()
    return s


def main():
    claimed_count = 0
    done_count = 0
    failed_count = 0
    winners: list[dict] = []

    while True:
        try:
            strat = claim_with_retry()
        except Exception as e:
            print(f"[drain2] claim error: {e!r}", flush=True)
            break

        if strat is None:
            print("[drain2] queue empty after retry; exiting.", flush=True)
            break

        claimed_count += 1
        sid = strat.get("signal_id") or strat["strategy_id"]
        strategy_id = strat["strategy_id"]
        print(f"\n[drain2] #{claimed_count} claimed {strategy_id} -> {sid}", flush=True)

        try:
            outcome, result = process_one(strat)
        except Exception as e:
            tb = traceback.format_exc()[-500:]
            print(f"[drain2] process_one fatal: {e!r}\n{tb}", flush=True)
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
                print(f"[drain2] update_strategy_status failed: {e!r}", flush=True)
            failed_count += 1
            print(f"[drain2] FAILED {sid}: {reason}", flush=True)
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
                print(f"[drain2] update_strategy_status done failed: {e!r}", flush=True)
            done_count += 1
            print(f"[drain2] DONE {sid}: Sharpe={sharpe}, CAGR={cagr}", flush=True)

            # Loose pre-filter: only refresh BH for plausible candidates
            if (sharpe is not None and sharpe > 0.5 and
                    cagr is not None and cagr > 0.10):
                try:
                    subprocess.run(
                        [str(ROOT / ".venv/bin/python"),
                         str(ROOT / "pipeline/refresh_bh.py")],
                        check=True, cwd=str(ROOT),
                        capture_output=True, timeout=180,
                    )
                except Exception as e:
                    print(f"[drain2] BH refresh failed: {e!r}", flush=True)
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
                    print(f"[drain2] {sid} passed loose but missed gate: "
                          f"{reasons}", flush=True)

        heartbeat("backtester")

    print("\n=== DRAIN 2PASS REPORT ===", flush=True)
    print(f"claimed: {claimed_count}", flush=True)
    print(f"done:    {done_count}", flush=True)
    print(f"failed:  {failed_count}", flush=True)
    print(f"winners: {len(winners)}", flush=True)
    for w in winners:
        print(f"  WIN {w['signal_id']}: Sharpe={w['sharpe']:.2f}, "
              f"CAGR={w['cagr']*100:.1f}%, OOS={w.get('oos_sharpe')}",
              flush=True)

    report_path = ROOT / "pipeline" / "_drain_loop_2pass_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "finished_at": now_iso(),
            "claimed": claimed_count,
            "done": done_count,
            "failed": failed_count,
            "winners": winners,
        }, f, indent=2, default=str)
    print(f"[drain2] report -> {report_path}", flush=True)

    # If we found winners, refresh BH and rebuild full_catalog.
    # DO NOT modify index.html — promotion sweep runs after.
    if winners:
        try:
            subprocess.run(
                [str(ROOT / ".venv/bin/python"),
                 str(ROOT / "pipeline/refresh_bh.py")],
                check=True, cwd=str(ROOT), timeout=180,
            )
        except Exception as e:
            print(f"[drain2] final BH refresh failed: {e!r}", flush=True)
        try:
            subprocess.run(
                [str(ROOT / ".venv/bin/python"),
                 str(ROOT / "build_report.py")],
                check=True, cwd=str(ROOT), timeout=300,
            )
        except Exception as e:
            print(f"[drain2] build_report failed: {e!r}", flush=True)


if __name__ == "__main__":
    main()
