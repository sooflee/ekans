"""Batch backtester for PL-series FRED-based strategies.
Reads strategies_queue.json, generates and runs backtests for each 'ready' strategy.
"""
import sys
import json
import datetime
import traceback
import importlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backtests"))

from harness import (
    load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns,
    ROOT as HARNESS_ROOT
)
import numpy as np
import pandas as pd


def run_fred_yoy_threshold_strategy(sid, fred_series, threshold, below_threshold,
                                     months_below, ticker_basket, hold_days,
                                     benchmark="SPY", start_prices="1998-01-01",
                                     start_fred="1996-01-01", name="Strategy",
                                     rule="", mechanism="", cooldown_months=6,
                                     resample="M", yoy_periods=12,
                                     direction="long_above"):
    """Generic FRED YoY threshold crossing strategy.
    direction: 'long_above' = long when crossing above threshold
               'long_below' = long when crossing below threshold
    """
    try:
        all_tickers = list(set(ticker_basket + [benchmark]))
        px = load_prices(all_tickers, start=start_prices)
        fred_data = load_fred(fred_series, start=start_fred).squeeze()
    except Exception as e:
        return mark_failed(sid, f"data load: {e}"), None

    ret = daily_returns(px)
    bench_r = ret[benchmark]

    avail_basket = [t for t in ticker_basket if t in ret.columns]
    if not avail_basket:
        return mark_failed(sid, f"No basket tickers available from {ticker_basket}"), None
    basket_r = ret[avail_basket].mean(axis=1)

    # Resample and compute YoY
    fred_m = fred_data.resample(resample).last().dropna()
    fred_yoy = fred_m.pct_change(yoy_periods)

    # Find threshold crossings
    count_below = 0
    triggers = []
    cooldown = 0

    for i in range(1, len(fred_yoy)):
        val = float(fred_yoy.iloc[i])
        if np.isnan(val):
            continue
        if cooldown > 0:
            cooldown -= 1
            if val < below_threshold:
                count_below += 1
            else:
                count_below = 0
            continue

        if direction == "long_above":
            if val < below_threshold:
                count_below += 1
            elif val >= threshold and count_below >= months_below:
                triggers.append(fred_yoy.index[i])
                count_below = 0
                cooldown = cooldown_months
            else:
                count_below = 0
        else:  # long_below
            if val > below_threshold:
                count_below += 1
            elif val <= threshold and count_below >= months_below:
                triggers.append(fred_yoy.index[i])
                count_below = 0
                cooldown = cooldown_months
            else:
                count_below = 0

    events = []
    pnl_parts = []

    for trig_date in triggers:
        entry_mask = ret.index >= trig_date
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = ret.index[entry_mask][0]
        entry_loc = ret.index.get_loc(entry_idx)
        exit_loc = min(entry_loc + hold_days, len(ret.index) - 1)

        window = slice(entry_loc, exit_loc)
        basket_window = basket_r.iloc[window]
        bench_window = bench_r.iloc[window]
        pnl_parts.append(basket_window)

        bask_cum = float((1 + basket_window).prod() - 1)
        bench_cum = float((1 + bench_window).prod() - 1)

        events.append({
            "trigger_date": str(trig_date.date()),
            "fred_yoy": round(float(fred_yoy.loc[trig_date]), 4),
            "basket_return": round(bask_cum, 4),
            "bench_return": round(bench_cum, 4),
            "excess": round(bask_cum - bench_cum, 4),
        })

    if not events:
        return mark_failed(sid, f"No threshold crossing events found for {fred_series}"), None

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]
    m = compute_metrics(all_pnl, benchmark=bench_r.reindex(all_pnl.index).dropna(), name=name)

    avg_basket = np.mean([e["basket_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["basket_return"] > 0)

    save_result(sid, m, extra={
        "rule": rule,
        "mechanism": mechanism,
        "source": f"FRED {fred_series} + yfinance",
        "n_events": len(events),
        "avg_basket_return": round(avg_basket, 4),
        "avg_excess_vs_spy": round(avg_excess, 4),
        "win_rate": f"{win_count}/{len(events)}",
        "events": events,
    })

    sharpe = m.get('sharpe', 0)
    cagr = m.get('cagr', 0)
    is_winner = sharpe > 0.5 and cagr > 0.10
    print(f"  {sid}: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%, Win={win_count}/{len(events)}"
          f"{' *** WINNER ***' if is_winner else ''}")
    return m, events


def process_strategy_with_existing_script(sid):
    """If a backtest script already exists, run it."""
    script = ROOT / "backtests" / f"{sid}.py"
    if not script.exists():
        return False
    spec = importlib.util.spec_from_file_location(sid, str(script))
    mod = importlib.util.import_module_from_spec(spec)
    spec.loader.exec_module(mod)
    if hasattr(mod, 'main'):
        mod.main()
    return True


def update_queue(strategy_id, status, result=None):
    """Update a strategy's status in the queue."""
    qpath = ROOT / "pipeline" / "strategies_queue.json"
    with open(qpath) as f:
        q = json.load(f)
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    for s in q:
        if s['strategy_id'] == strategy_id:
            s['status'] = status
            s['backtested_at'] = now
            if result:
                s['backtest_result'] = result
            break
    with open(qpath, 'w') as f:
        json.dump(q, f, indent=2, ensure_ascii=False)


def update_heartbeat():
    spath = ROOT / "pipeline" / "status.json"
    with open(spath) as f:
        st = json.load(f)
    st['loops']['backtester']['last_run'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    st['loops']['backtester']['status'] = 'running'
    with open(spath, 'w') as f:
        json.dump(st, f, indent=2)


if __name__ == "__main__":
    qpath = ROOT / "pipeline" / "strategies_queue.json"
    with open(qpath) as f:
        q = json.load(f)

    ready = [s for s in q if s.get('status') == 'ready']
    print(f"Found {len(ready)} ready strategies")

    for s in ready:
        sid = s.get('signal_id', s['strategy_id'])
        print(f"\nProcessing {s['strategy_id']}: {s['name']}")
        update_queue(s['strategy_id'], 'in_progress')

        try:
            # Try running existing script first
            script = ROOT / "backtests" / f"{sid}.py"
            if script.exists():
                spec = importlib.util.spec_from_file_location(sid, str(script))
                mod = importlib.util.import_module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, 'main'):
                    mod.main()
            else:
                # No script exists -- try generic approach based on strategy config
                rule = s.get('rule', '')
                tickers = s.get('tickers', [])
                mark_failed(sid, f"No backtest script exists at {script.name}")

            # Read result
            rpath = ROOT / "results" / f"{sid}.json"
            if rpath.exists():
                with open(rpath) as f:
                    r = json.load(f)
                if r.get('status') == 'fail':
                    update_queue(s['strategy_id'], 'failed', {
                        'status': 'fail', 'reason': r.get('reason', 'unknown')
                    })
                else:
                    sharpe = r.get('sharpe', 0)
                    cagr = r.get('cagr', 0)
                    winner = sharpe > 0.5 and cagr > 0.10
                    update_queue(s['strategy_id'], 'done', {
                        'status': 'ok',
                        'sharpe': round(sharpe, 4),
                        'cagr': round(cagr, 4),
                        'max_dd': round(r.get('max_dd', 0), 4),
                        't_stat': round(r.get('t_stat', 0), 4),
                        'n_events': r.get('n_events', 0),
                        'verdict': f"{'WINNER' if winner else 'Not a winner'} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}%"
                    })
            else:
                update_queue(s['strategy_id'], 'failed', {
                    'status': 'fail', 'reason': 'No result file produced'
                })
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            try:
                mark_failed(sid, str(e)[:200])
                update_queue(s['strategy_id'], 'failed', {
                    'status': 'fail', 'reason': str(e)[:200]
                })
            except:
                pass

    update_heartbeat()
    print(f"\nDone processing {len(ready)} strategies")
