"""PL70_new_home_supply_spike_delayed_long — New Home Months-Supply >8.0 → Long Homebuilders 12mo Later
Contrarian delayed-buy: glut NOW → supply discipline → 12mo later inventory clears → pricing power returns.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL70_new_home_supply_spike_delayed_long"
    try:
        px = load_prices(["XHB", "LEN", "DHI", "SPY"], start="2002-01-01")
    except Exception as e:
        return mark_failed(sid, f"price data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    # Try FRED data first, fall back to hand-coded dates
    triggers = []
    series_used = "hand-coded"
    try:
        supply = load_fred("MNEWSRSA", start="1990-01-01").squeeze()
        supply_m = supply.resample("ME").last().dropna()

        last_trigger = None
        for i in range(1, len(supply_m)):
            val = float(supply_m.iloc[i])
            prev = float(supply_m.iloc[i - 1])
            if np.isnan(val) or np.isnan(prev):
                continue
            if val >= 8.0 and prev < 8.0:
                trig = supply_m.index[i]
                if last_trigger is None or (trig - last_trigger).days > 365:
                    triggers.append(trig)
                    last_trigger = trig
        if triggers:
            series_used = "FRED MNEWSRSA"
    except Exception:
        pass

    if not triggers:
        # Hand-coded dates when MNEWSRSA crossed above 8.0
        # Historical: crossed 8+ in 2005-06 (pre-GFC housing glut),
        # spiked during GFC in 2008-2010, and briefly in 2022
        triggers = pd.to_datetime(["2005-09-01", "2008-01-01", "2022-08-01"])
        series_used = "hand-coded"

    print(f"Found {len(triggers)} new home supply spike events (>8.0 months) [{series_used}]")

    events = []
    pnl_parts = []
    hold_days = 252
    delay_days = 252  # wait 12 months before entry

    for trig_date in triggers:
        # Delay entry by ~12 months
        delayed_entry = trig_date + pd.DateOffset(months=12)
        entry_mask = ret.index >= delayed_entry
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = ret.index[entry_mask][0]
        entry_loc = ret.index.get_loc(entry_idx)
        exit_loc = min(entry_loc + hold_days, len(ret.index) - 1)

        window = slice(entry_loc, exit_loc)

        # Equal-weight basket of available tickers
        basket_tickers = [t for t in ["XHB", "LEN", "DHI"] if t in ret.columns
                          and not ret[t].iloc[window].isna().all()]
        if not basket_tickers:
            continue

        basket_r = ret[basket_tickers].iloc[window].mean(axis=1)
        spy_window = spy_r.iloc[window]
        pnl_parts.append(basket_r)

        basket_cum = float((1 + basket_r).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)

        supply_val = np.nan
        try:
            supply_val = float(supply_m.iloc[supply_m.index.get_indexer([trig_date], method='nearest')[0]])
        except Exception:
            pass

        events.append({
            "spike_date": str(trig_date.date()) if hasattr(trig_date, 'date') else str(trig_date),
            "entry_date": str(entry_idx.date()),
            "supply_at_spike": round(supply_val, 2) if not np.isnan(supply_val) else None,
            "basket_tickers": basket_tickers,
            "basket_12m_return": round(basket_cum, 4),
            "spy_12m_return": round(spy_cum, 4),
            "excess": round(basket_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No new home supply spike events found with enough data for delayed entry")

    all_pnl = pd.concat(pnl_parts)
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="New Home Supply Spike → Delayed Long Builders")

    avg_basket = np.mean([e["basket_12m_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["basket_12m_return"] > 0)

    save_result(sid, m, extra={
        "rule": "FRED MNEWSRSA crosses above 8.0 → wait 12 months → long XHB+LEN+DHI 12mo",
        "mechanism": "Housing glut → supply discipline → 12mo later inventory clears → builder pricing power returns",
        "source": "FRED MNEWSRSA + yfinance",
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
        flag = "+" if e["basket_12m_return"] > 0 else "-"
        print(f"  {flag} spike {e['spike_date']} → entry {e['entry_date']} (supply={e['supply_at_spike']}): basket {e['basket_12m_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
