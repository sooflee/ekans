"""Run all 'ready' backtests sequentially, updating strategies_queue.json
via the queue_io locking helper.

Winner detection uses pipeline/winner_gate.is_winner (BH-significance + OOS
Sharpe + sample-size floor on top of the raw Sharpe/CAGR thresholds), with a
single refresh_bh.py call at the end.
"""
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/Users/benson/Projects/ekans")
PYTHON = str(ROOT / ".venv/bin/python")
PIPELINE = ROOT / "pipeline"

sys.path.insert(0, str(PIPELINE))
from queue_io import (
    STRATEGIES, claim_ready_strategy, update_strategy_status,
)
from winner_gate import is_winner, load_mt_data, winner_reasons


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run_one(strat: dict) -> dict:
    """Run one already-claimed (status=in_progress) strategy. Returns the
    backtest_result dict written into the queue."""
    sid = strat["signal_id"]
    strategy_id = strat.get("strategy_id", sid)
    bt_file = ROOT / "backtests" / f"{sid}.py"
    result_file = ROOT / "results" / f"{sid}.json"

    if not bt_file.exists():
        bt_result = {"status": "fail", "reason": "no backtest file"}
        update_strategy_status(
            strategy_id, "failed",
            backtest_result=bt_result, backtested_at=_now_iso(),
        )
        print(f"  SKIP {sid}: no backtest file")
        return bt_result

    print(f"  Running {sid}...", end=" ", flush=True)
    try:
        result = subprocess.run(
            [PYTHON, str(bt_file)],
            capture_output=True, text=True, timeout=120,
            cwd=str(ROOT),
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        if stdout:
            for line in stdout.split("\n")[-3:]:
                print(line)
        if result.returncode != 0 and stderr:
            print(f"  ERR: {stderr[-200:]}")
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        bt_result = {"status": "fail", "reason": "timeout"}
        update_strategy_status(
            strategy_id, "failed",
            backtest_result=bt_result, backtested_at=_now_iso(),
        )
        return bt_result
    except Exception as e:
        print(f"EXCEPTION: {e}")
        bt_result = {"status": "fail", "reason": str(e)}
        update_strategy_status(
            strategy_id, "failed",
            backtest_result=bt_result, backtested_at=_now_iso(),
        )
        return bt_result

    if not result_file.exists():
        bt_result = {"status": "fail", "reason": "no result file generated"}
        update_strategy_status(
            strategy_id, "failed",
            backtest_result=bt_result, backtested_at=_now_iso(),
        )
        return bt_result

    with open(result_file) as f:
        res = json.load(f)

    if res.get("status") == "fail":
        bt_result = {
            "status": "fail",
            "reason": res.get("reason", "unknown"),
            "sharpe": None, "cagr": None, "max_dd": None, "t_stat": None,
        }
        update_strategy_status(
            strategy_id, "failed",
            backtest_result=bt_result, backtested_at=_now_iso(),
        )
        return bt_result

    sharpe = res.get("sharpe", 0) or 0
    cagr = res.get("cagr", 0) or 0
    max_dd = res.get("max_dd", 0) or 0
    t_stat = res.get("t_stat", 0) or 0
    bt_result = {
        "status": "ok",
        "sharpe": round(sharpe, 4),
        "cagr": round(cagr, 4),
        "max_dd": round(max_dd, 4),
        "t_stat": round(t_stat, 4),
        "n_events": res.get("n_events"),
        "avg_event_return": res.get("avg_event_return"),
        "event_win_rate": res.get("event_win_rate"),
    }
    update_strategy_status(
        strategy_id, "done",
        backtest_result=bt_result, backtested_at=_now_iso(),
    )
    print(f"    -> Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%, MaxDD={max_dd*100:.1f}%, t={t_stat:.2f}")
    return bt_result


def main():
    total_initial = len(json.loads(STRATEGIES.read_text()))
    print(f"Strategies queue: {total_initial} total. Running ready ones...")

    ran = 0
    while True:
        strat = claim_ready_strategy()
        if strat is None:
            break
        run_one(strat)
        ran += 1

    print(f"\nRan {ran} backtests.")

    if ran == 0:
        return

    # Refresh BH across the catalog once at the end (cheap, ~1s).
    print("\nRefreshing BH multiple-testing table...")
    subprocess.run([PYTHON, str(PIPELINE / "refresh_bh.py")], cwd=str(ROOT))
    mt = load_mt_data()

    # Now apply the canonical gate to every done strategy and print winners.
    strats = json.loads(STRATEGIES.read_text())
    winners = []
    for s in strats:
        if s.get("status") != "done":
            continue
        sid = s.get("signal_id")
        if not sid:
            continue
        rp = ROOT / "results" / f"{sid}.json"
        if not rp.exists():
            continue
        try:
            res = json.loads(rp.read_text())
        except Exception:
            continue
        res.setdefault("signal_id", sid)
        if is_winner(res, mt):
            winners.append((sid, res))

    print(f"\n=== SUMMARY ===")
    print(f"Total winners (canonical gate): {len(winners)}")
    print(f"\nCANONICAL WINNERS:")
    for sid, res in winners:
        sh = res.get("sharpe", 0) or 0
        cg = res.get("cagr", 0) or 0
        md = res.get("max_dd", 0) or 0
        oos = res.get("oos_sharpe", 0) or 0
        print(f"  {sid}: Sharpe={sh:.2f}, CAGR={cg*100:.1f}%, MaxDD={md*100:.1f}%, OOS={oos:.2f}")


if __name__ == "__main__":
    main()
