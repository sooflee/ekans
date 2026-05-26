"""Run all ready backtests, update strategies_queue.json with results."""
import json, os, sys, subprocess, datetime as dt
from pathlib import Path

ROOT = Path("/Users/benson/Projects/ekans")
PYTHON = str(ROOT / ".venv/bin/python")

with open(ROOT / "pipeline/strategies_queue.json") as f:
    strategies = json.load(f)

ready = [s for s in strategies if s['status'] == 'ready']
print(f"Running {len(ready)} backtests...")

for strat in ready:
    sid = strat['signal_id']
    bt_file = ROOT / "backtests" / f"{sid}.py"
    result_file = ROOT / "results" / f"{sid}.json"
    
    if not bt_file.exists():
        print(f"  SKIP {sid}: no backtest file")
        strat['status'] = 'failed'
        strat['backtest_result'] = {"status": "fail", "reason": "no backtest file"}
        continue
    
    print(f"  Running {sid}...", end=" ", flush=True)
    try:
        result = subprocess.run(
            [PYTHON, str(bt_file)],
            capture_output=True, text=True, timeout=120,
            cwd=str(ROOT)
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()
        if stdout:
            # Print just last 2 lines
            lines = stdout.split('\n')
            for l in lines[-3:]:
                print(l)
        if result.returncode != 0 and stderr:
            print(f"  ERR: {stderr[-200:]}")
    except subprocess.TimeoutExpired:
        print("TIMEOUT")
        strat['status'] = 'failed'
        strat['backtest_result'] = {"status": "fail", "reason": "timeout"}
        strat['backtested_at'] = dt.datetime.utcnow().isoformat() + "+00:00"
        continue
    except Exception as e:
        print(f"EXCEPTION: {e}")
        strat['status'] = 'failed'
        strat['backtest_result'] = {"status": "fail", "reason": str(e)}
        strat['backtested_at'] = dt.datetime.utcnow().isoformat() + "+00:00"
        continue
    
    # Read result
    if result_file.exists():
        with open(result_file) as f:
            res = json.load(f)
        
        if res.get('status') == 'fail':
            strat['status'] = 'failed'
            strat['backtest_result'] = {
                "status": "fail",
                "reason": res.get('reason', 'unknown'),
                "sharpe": None, "cagr": None, "max_dd": None, "t_stat": None
            }
        else:
            sharpe = res.get('sharpe', 0) or 0
            cagr = res.get('cagr', 0) or 0
            max_dd = res.get('max_dd', 0) or 0
            t_stat = res.get('t_stat', 0) or 0
            
            strat['status'] = 'done'
            strat['backtest_result'] = {
                "status": "ok",
                "sharpe": round(sharpe, 4),
                "cagr": round(cagr, 4),
                "max_dd": round(max_dd, 4),
                "t_stat": round(t_stat, 4),
                "n_events": res.get('n_events'),
                "avg_event_return": res.get('avg_event_return'),
                "event_win_rate": res.get('event_win_rate'),
            }
            
            # Check winner
            is_winner = sharpe > 0.5 and cagr > 0.10
            tag = "WINNER" if is_winner else "done"
            print(f"    -> {tag}: Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%, MaxDD={max_dd*100:.1f}%, t={t_stat:.2f}")
    else:
        strat['status'] = 'failed'
        strat['backtest_result'] = {"status": "fail", "reason": "no result file generated"}
    
    strat['backtested_at'] = dt.datetime.utcnow().isoformat() + "+00:00"

# Save updated strategies
with open(ROOT / "pipeline/strategies_queue.json", "w") as f:
    json.dump(strategies, f, indent=2)

# Summary
winners = [s for s in strategies if s['status'] == 'done' and s.get('backtest_result', {}).get('status') == 'ok'
           and (s.get('backtest_result', {}).get('sharpe', 0) or 0) > 0.5 
           and (s.get('backtest_result', {}).get('cagr', 0) or 0) > 0.10
           and s['signal_id'].startswith('PL1')]
print(f"\n=== SUMMARY ===")
print(f"New backtests run: {len(ready)}")
new_done = [s for s in ready if s['status'] == 'done']
new_failed = [s for s in ready if s['status'] == 'failed']
print(f"Done: {len(new_done)}, Failed: {len(new_failed)}")
print(f"\nNEW WINNERS (Sharpe > 0.5 AND CAGR > 10%):")
for s in strategies:
    if s['status'] == 'done' and s.get('backtest_result', {}).get('status') == 'ok':
        sh = s['backtest_result'].get('sharpe', 0) or 0
        cg = s['backtest_result'].get('cagr', 0) or 0
        if sh > 0.5 and cg > 0.10 and s['signal_id'].startswith('PL1') and int(s['signal_id'][2:5].replace('_','').replace('0','0')) >= 102:
            print(f"  {s['signal_id']}: Sharpe={sh:.2f}, CAGR={cg*100:.1f}%, MaxDD={s['backtest_result'].get('max_dd',0)*100:.1f}%")
