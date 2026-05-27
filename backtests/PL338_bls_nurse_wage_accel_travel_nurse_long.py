"""PL338_bls_nurse_wage_accel_travel_nurse_long — BLS Healthcare Wage Acceleration + JOLTS Openings -> Long Travel Nurse Agencies
When FRED CES healthcare hourly earnings YoY > 5% AND JOLTS healthcare openings above 24-month median,
long AMN+CCRN equal-weight for 63 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL338_bls_nurse_wage_accel_travel_nurse_long"
    try:
        px = load_prices(["AMN", "CCRN", "SPY"], start="2001-01-01")
    except Exception as e:
        return mark_failed(sid, f"price data load: {e}")

    # Load FRED healthcare wages — try SA then NSA (broader healthcare & social assistance sector)
    wages = None
    for series in ["CES6562000008", "CEU6562000008", "CES6500000008", "CEU6500000008"]:
        try:
            wages = load_fred(series, start="2001-01-01").squeeze()
            if wages.dropna().empty:
                wages = None
                continue
            break
        except Exception:
            continue
    if wages is None:
        return mark_failed(sid, "Could not load FRED healthcare wages (CES6562000008 / CEU6562000008)")

    # Load JOLTS healthcare openings
    jolts = None
    for series in ["JTS6200JOL", "JTS620000000000000JOL"]:
        try:
            jolts = load_fred(series, start="2001-01-01").squeeze()
            if jolts.dropna().empty:
                jolts = None
                continue
            break
        except Exception:
            continue
    if jolts is None:
        return mark_failed(sid, "Could not load FRED JOLTS healthcare openings (JTS6200JOL)")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    basket_tickers = [t for t in ["AMN", "CCRN"] if t in ret.columns]
    if not basket_tickers:
        return mark_failed(sid, "No travel nurse tickers available")
    basket_r = ret[basket_tickers].mean(axis=1)

    # Monthly healthcare wages, YoY growth
    wages_m = wages.resample("M").last().dropna()
    wages_yoy = wages_m.pct_change(12)  # 12-month YoY

    # Monthly JOLTS, 24-month rolling median
    jolts_m = jolts.resample("M").last().dropna()
    jolts_median_24 = jolts_m.rolling(24).median()

    # Find signal months: wages YoY > 5% AND jolts above 24m median
    # Align on common monthly dates
    common_idx = wages_yoy.dropna().index.intersection(jolts_median_24.dropna().index).intersection(jolts_m.index)
    signal_months = []
    last_signal = None

    for dt in sorted(common_idx):
        wyoy = float(wages_yoy.loc[dt])
        jval = float(jolts_m.loc[dt])
        jmed = float(jolts_median_24.loc[dt])

        if np.isnan(wyoy) or np.isnan(jval) or np.isnan(jmed):
            continue

        if wyoy > 0.05 and jval > jmed:
            # Enforce 90 calendar day gap
            if last_signal is not None and (dt - last_signal).days < 90:
                continue
            signal_months.append(dt)
            last_signal = dt

    if not signal_months:
        return mark_failed(sid, "No signal events found (wages YoY > 5% AND JOLTS above 24m median)")

    events = []
    pnl_parts = []
    hold_days = 63

    for trig_date in signal_months:
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
            "basket_63d_return": round(bask_cum, 4),
            "spy_63d_return": round(spy_cum, 4),
            "excess": round(bask_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No tradeable events after filtering")

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="BLS Healthcare Wage Accel + JOLTS -> Long AMN+CCRN")

    avg_basket = np.mean([e["basket_63d_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["basket_63d_return"] > 0)

    save_result(sid, m, extra={
        "rule": "FRED CES healthcare hourly earnings YoY > 5% AND JOLTS healthcare openings above 24m median -> long AMN+CCRN 63 days",
        "mechanism": "Travel nurse agencies benefit from wage acceleration — they charge markup over permanent wages, bill-pay spread widens when demand is acute",
        "source": "FRED CES6562000008 + JTS6200JOL + yfinance",
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
        flag = "+" if e["basket_63d_return"] > 0 else "-"
        print(f"  {flag} {e['trigger_date']}: basket {e['basket_63d_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} -- Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
