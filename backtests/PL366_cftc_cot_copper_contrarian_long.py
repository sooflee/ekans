"""PL366_cftc_cot_copper_contrarian_long — CFTC COT Managed Money Net Short Extreme in Copper -> Contrarian Long COPX/FCX
Without direct CFTC COT API parsing, use a proxy: copper price drawdown from 52-week high > 25%
as a proxy for extreme speculative bearishness. When HG=F (copper futures) drops > 25% from
52-week high, long COPX+FCX+SCCO for 20 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL366_cftc_cot_copper_contrarian_long"
    try:
        px = load_prices(["COPX", "FCX", "SCCO", "SPY", "HG=F"], start="2006-01-01")
    except Exception as e:
        return mark_failed(sid, f"price data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    # Copper price proxy for extreme positioning
    if "HG=F" not in px.columns:
        return mark_failed(sid, "Copper futures (HG=F) data not available")

    copper = px["HG=F"].dropna()
    copper_52w_high = copper.rolling(252, min_periods=126).max()
    copper_drawdown = (copper / copper_52w_high) - 1

    # Build basket — COPX launched 2010, use FCX+SCCO pre-2010
    basket_tickers_post2010 = [t for t in ["COPX", "FCX", "SCCO"] if t in ret.columns]
    basket_tickers_pre2010 = [t for t in ["FCX", "SCCO"] if t in ret.columns]
    if not basket_tickers_pre2010:
        return mark_failed(sid, "No copper miner tickers available")

    # Find drawdown > 25% events
    triggers = []
    last_trigger = None
    for i in range(len(copper_drawdown)):
        dt = copper_drawdown.index[i]
        dd = float(copper_drawdown.iloc[i])
        if np.isnan(dd):
            continue
        if dd < -0.25:
            if last_trigger is None or (dt - last_trigger).days >= 60:
                triggers.append(dt)
                last_trigger = dt

    if not triggers:
        return mark_failed(sid, "No copper drawdown > 25% events found")

    events = []
    pnl_parts = []
    hold_days = 20

    for trig_date in triggers:
        # Pick basket based on date
        if trig_date >= pd.Timestamp("2010-05-01"):
            tickers = basket_tickers_post2010
        else:
            tickers = basket_tickers_pre2010
        if not tickers:
            continue
        basket_r = ret[tickers].mean(axis=1)

        entry_mask = ret.index >= trig_date
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = ret.index[entry_mask][0]
        entry_loc = ret.index.get_loc(entry_idx)
        exit_loc = min(entry_loc + hold_days, len(ret.index) - 1)

        window = slice(entry_loc, exit_loc)
        basket_window = basket_r.iloc[window]
        spy_window = spy_r.iloc[window]
        pnl_parts.append(basket_window)

        bask_cum = float((1 + basket_window).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)

        events.append({
            "trigger_date": str(trig_date.date()),
            "copper_drawdown": round(float(copper_drawdown.loc[trig_date]), 4),
            "basket_20d_return": round(bask_cum, 4),
            "spy_20d_return": round(spy_cum, 4),
            "excess": round(bask_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No tradeable copper contrarian events")

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="Copper Drawdown Contrarian -> Long COPX+FCX+SCCO")

    avg_basket = np.mean([e["basket_20d_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["basket_20d_return"] > 0)

    save_result(sid, m, extra={
        "rule": "When copper (HG=F) drawdown from 52-week high > 25% (proxy for CFTC COT extreme net short), long COPX+FCX+SCCO 20 days",
        "mechanism": "Extreme speculative shorts create crowded trade vulnerable to short-covering rally on any positive catalyst",
        "source": "CFTC COT (proxy via copper price drawdown) + yfinance",
        "n_events": len(events),
        "avg_basket_return": round(avg_basket, 4),
        "avg_excess_vs_spy": round(avg_excess, 4),
        "win_rate": f"{win_count}/{len(events)}",
        "events": events,
    })
    sharpe = m.get('sharpe', 0)
    cagr = m.get('cagr', 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    print(f"  Avg basket: {avg_basket*100:.1f}%  Avg excess: {avg_excess*100:.1f}%  Win: {win_count}/{len(events)}")
    for e in events:
        flag = "+" if e["basket_20d_return"] > 0 else "-"
        print(f"  {flag} {e['trigger_date']}: dd={e['copper_drawdown']*100:.0f}%, basket {e['basket_20d_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} -- Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
