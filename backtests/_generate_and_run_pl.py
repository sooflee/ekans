"""Generate and run FRED-based PL backtests from strategies_queue.json.
Creates backtest scripts dynamically based on strategy rules, then runs them.
"""
import sys
import json
import datetime
import traceback
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backtests"))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def parse_fred_series(strategy):
    """Extract FRED series from strategy data_sources_concrete or rule."""
    rule = strategy.get('rule', '')
    sources = strategy.get('data_sources_concrete', {})
    fundamental = sources.get('fundamental', '')

    # Look for FRED series IDs (all caps, possibly with numbers)
    import re
    series = []
    for text in [rule, fundamental]:
        matches = re.findall(r'FRED\s+(\w+)', text)
        series.extend(matches)
    return list(set(series))


def generic_fred_backtest(strategy):
    """Run a generic FRED-based backtest based on strategy configuration."""
    sid = strategy.get('signal_id', strategy['strategy_id'])
    rule = strategy.get('rule', '')
    tickers = strategy.get('tickers', [])
    fred_series_list = parse_fred_series(strategy)

    if not fred_series_list:
        return mark_failed(sid, "Could not parse FRED series from strategy rule")

    fred_series = fred_series_list[0]  # Primary FRED series
    benchmark = "SPY"
    basket = [t for t in tickers if t != "SPY" and "=" not in t and t not in
              ("TLT","IEF","SHY","BIL","SHV","GLD","TIP","VWO")]

    # If basket is empty or contains ETFs that ARE the trade, use first non-SPY ticker
    if not basket:
        basket = [t for t in tickers if t != "SPY"]
    if not basket:
        basket = ["SPY"]

    # Determine hold days from horizon
    horizon = strategy.get('horizon', '3-6 months')
    if '1-2 week' in horizon or '2-4 week' in horizon:
        hold_days = 20
    elif '4-8 week' in horizon or '2-4 month' in horizon:
        hold_days = 63
    elif '3-6 month' in horizon:
        hold_days = 126
    elif '6-12 month' in horizon or '12' in horizon:
        hold_days = 252
    else:
        hold_days = 63

    # Try to load data
    try:
        all_tickers = list(set(basket + [benchmark]))
        px = load_prices(all_tickers, start="1998-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    try:
        fred_data = load_fred(fred_series, start="1996-01-01").squeeze()
    except Exception as e:
        return mark_failed(sid, f"FRED {fred_series} load: {e}")

    ret = daily_returns(px)
    bench_r = ret[benchmark] if benchmark in ret.columns else ret.iloc[:, 0]

    avail_basket = [t for t in basket if t in ret.columns]
    if not avail_basket:
        return mark_failed(sid, f"No basket tickers available from {basket}")
    basket_r = ret[avail_basket].mean(axis=1)

    # Determine strategy type from rule text
    rule_lower = rule.lower()

    # Try to detect the signal logic
    fred_resampled = fred_data.resample("M").last().dropna()

    # Strategy: YoY crossing / threshold
    if 'yoy' in rule_lower:
        fred_metric = fred_resampled.pct_change(12)
        metric_name = "YoY"
    elif 'mom' in rule_lower or 'qoq' in rule_lower:
        fred_metric = fred_resampled.pct_change(1)
        metric_name = "MoM"
    elif '3mo' in rule_lower or '3-month' in rule_lower:
        fred_metric = fred_resampled.pct_change(3)
        metric_name = "3mo_chg"
    else:
        fred_metric = fred_resampled.pct_change(12)
        metric_name = "YoY"

    # Determine direction and thresholds
    # Look for patterns like "> +5%", "above 50", "crosses above", "decline", "below"
    import re

    is_decline_signal = any(w in rule_lower for w in ['decline', 'negative', 'below', 'trough', 'collapse', 'decelerat'])
    is_rise_signal = any(w in rule_lower for w in ['above', 'surge', 'acceleration', 'recovery', 'positive', 'rises', 'increase'])

    # Generic approach: use z-score or percentile of the metric
    # Find events where metric is in extreme territory
    fred_metric = fred_metric.dropna()
    if len(fred_metric) < 24:
        return mark_failed(sid, f"Insufficient FRED data: only {len(fred_metric)} observations")

    # Rolling z-score approach
    zscore = (fred_metric - fred_metric.rolling(36, min_periods=12).mean()) / fred_metric.rolling(36, min_periods=12).std()
    zscore = zscore.dropna()

    triggers = []
    cooldown = 0
    cooldown_periods = max(hold_days // 21, 3)  # months

    if is_decline_signal:
        # Trigger on extreme low z-score then recovery
        for i in range(1, len(zscore)):
            if cooldown > 0:
                cooldown -= 1
                continue
            # Recovery from extreme: previous month z < -1, current month improving
            if float(zscore.iloc[i-1]) < -1.0 and float(zscore.iloc[i]) > float(zscore.iloc[i-1]):
                triggers.append(zscore.index[i])
                cooldown = cooldown_periods
    elif is_rise_signal:
        # Trigger on crossing above positive threshold
        for i in range(1, len(zscore)):
            if cooldown > 0:
                cooldown -= 1
                continue
            if float(zscore.iloc[i-1]) < 1.0 and float(zscore.iloc[i]) >= 1.0:
                triggers.append(zscore.index[i])
                cooldown = cooldown_periods
    else:
        # Default: trigger on any significant move (absolute z > 1.5)
        for i in range(1, len(zscore)):
            if cooldown > 0:
                cooldown -= 1
                continue
            if abs(float(zscore.iloc[i])) > 1.5 and abs(float(zscore.iloc[i-1])) <= 1.5:
                triggers.append(zscore.index[i])
                cooldown = cooldown_periods

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
            "metric_val": round(float(fred_metric.loc[trig_date]) if trig_date in fred_metric.index else 0, 4),
            "basket_return": round(bask_cum, 4),
            "bench_return": round(bench_cum, 4),
            "excess": round(bask_cum - bench_cum, 4),
        })

    if not events:
        return mark_failed(sid, f"No events found for {fred_series} {metric_name} signal")

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]
    m = compute_metrics(all_pnl, benchmark=bench_r.reindex(all_pnl.index).dropna(),
                        name=strategy.get('name', sid))

    avg_basket = np.mean([e["basket_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["basket_return"] > 0)

    save_result(sid, m, extra={
        "rule": rule[:200],
        "mechanism": strategy.get('backtest_approach', '')[:200],
        "source": f"FRED {fred_series} + yfinance",
        "n_events": len(events),
        "avg_basket_return": round(avg_basket, 4),
        "avg_excess_vs_spy": round(avg_excess, 4),
        "win_rate": f"{win_count}/{len(events)}",
        "events": events[:20],  # cap events list
    })

    sharpe = m.get('sharpe', 0)
    cagr = m.get('cagr', 0)
    is_winner = sharpe > 0.5 and cagr > 0.10
    tag = " *** WINNER ***" if is_winner else ""
    print(f"  {sid}: N={len(events)}, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%, "
          f"Win={win_count}/{len(events)}{tag}")
    return m


def update_queue(strategy_id, status, result=None):
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


def main():
    qpath = ROOT / "pipeline" / "strategies_queue.json"
    with open(qpath) as f:
        q = json.load(f)

    ready = [s for s in q if s.get('status') == 'ready']
    print(f"=== Batch backtester: {len(ready)} ready strategies ===")

    winners = []
    processed = 0

    for s in ready:
        sid = s.get('signal_id', s['strategy_id'])
        print(f"\n[{processed+1}/{len(ready)}] {s['strategy_id']}: {s['name'][:60]}")
        update_queue(s['strategy_id'], 'in_progress')

        try:
            # Check if dedicated script exists
            script = ROOT / "backtests" / f"{sid}.py"
            if script.exists():
                import importlib.util
                spec = importlib.util.spec_from_file_location(sid, str(script))
                mod = importlib.util.import_module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, 'main'):
                    mod.main()
            else:
                # Run generic FRED-based backtest
                generic_fred_backtest(s)

            # Read result
            rpath = ROOT / "results" / f"{sid}.json"
            if rpath.exists():
                with open(rpath) as f:
                    r = json.load(f)
                if r.get('status') == 'fail':
                    update_queue(s['strategy_id'], 'failed', {
                        'status': 'fail', 'reason': r.get('reason', 'unknown')[:200]
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
                    if winner:
                        winners.append((s, r))
            else:
                update_queue(s['strategy_id'], 'failed', {
                    'status': 'fail', 'reason': 'No result file produced'
                })
        except Exception as e:
            err = str(e)[:200]
            print(f"  ERROR: {err}")
            try:
                mark_failed(sid, err)
                update_queue(s['strategy_id'], 'failed', {
                    'status': 'fail', 'reason': err
                })
            except:
                pass

        processed += 1

    update_heartbeat()
    print(f"\n=== Done: {processed} processed, {len(winners)} winners ===")
    for s, r in winners:
        print(f"  WINNER: {s['strategy_id']} — Sharpe={r.get('sharpe',0):.2f}, CAGR={r.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
