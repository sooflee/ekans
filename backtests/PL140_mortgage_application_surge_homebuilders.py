"""PL140_mortgage_application_surge_homebuilders — Mortgage Rate Drop >50bps from 13wk High → Long Homebuilders
When FRED MORTGAGE30US drops >50bps from trailing 13-week high (proxy for application surge),
long LEN+DHI+NVR with 1-month lag for 126 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL140_mortgage_application_surge_homebuilders"
    try:
        px = load_prices(["LEN", "DHI", "NVR", "SPY"], start="1999-01-01")
        mtg = load_fred("MORTGAGE30US", start="1997-01-01").squeeze()
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    # Build basket from available tickers
    basket_tickers = [t for t in ["LEN", "DHI", "NVR"] if t in ret.columns]
    if not basket_tickers:
        return mark_failed(sid, "No homebuilder tickers available")
    basket_r = ret[basket_tickers].mean(axis=1)

    # Weekly mortgage rate — compute 13-week trailing high
    mtg = mtg.dropna()
    mtg_13wk_high = mtg.rolling(13, min_periods=1).max()
    rate_drop = mtg_13wk_high - mtg  # positive = rate dropped from high

    # Find dates where rate drop > 50bps (0.50 percentage points)
    signal_dates = []
    cooldown = 0
    for i in range(13, len(rate_drop)):
        if cooldown > 0:
            cooldown -= 1
            continue
        if float(rate_drop.iloc[i]) > 0.50:
            signal_dates.append(rate_drop.index[i])
            cooldown = 26  # ~6 months cooldown

    events = []
    pnl_parts = []
    hold_days = 126
    entry_lag = 21  # 1-month lag for applications-to-closings pipeline

    for sig_date in signal_dates:
        # Find entry date = 21 trading days after signal
        entry_mask = ret.index >= sig_date
        if entry_mask.sum() < (entry_lag + hold_days):
            continue
        lag_idx = ret.index[entry_mask][entry_lag]
        entry_loc = ret.index.get_loc(lag_idx)
        exit_loc = min(entry_loc + hold_days, len(ret.index) - 1)

        window = slice(entry_loc, exit_loc)
        basket_window = basket_r.iloc[window]
        spy_window = spy_r.iloc[window]
        pnl_parts.append(basket_window)

        bask_cum = float((1 + basket_window).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)

        events.append({
            "signal_date": str(sig_date.date()),
            "entry_date": str(lag_idx.date()),
            "rate": round(float(mtg.loc[sig_date]), 2) if sig_date in mtg.index else None,
            "rate_13wk_high": round(float(mtg_13wk_high.loc[sig_date]), 2) if sig_date in mtg_13wk_high.index else None,
            "basket_6m_return": round(bask_cum, 4),
            "spy_6m_return": round(spy_cum, 4),
            "excess": round(bask_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No mortgage rate drop events found")

    all_pnl = pd.concat(pnl_parts)
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="Mortgage Rate Drop → Long Homebuilders")

    avg_basket = np.mean([e["basket_6m_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["basket_6m_return"] > 0)

    save_result(sid, m, extra={
        "rule": "FRED MORTGAGE30US drops >50bps from 13-week high → long LEN+DHI+NVR (1-month lag) 126 days",
        "mechanism": "Mortgage rate drops drive application surges → new orders pipeline fills → homebuilder revenue accelerates 1-2 quarters later",
        "source": "FRED MORTGAGE30US + yfinance",
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
        flag = "+" if e["basket_6m_return"] > 0 else "-"
        print(f"  {flag} {e['signal_date']}→{e['entry_date']} (rate={e['rate']}%): basket {e['basket_6m_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
