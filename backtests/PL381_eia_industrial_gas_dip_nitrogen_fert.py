"""PL381_eia_industrial_gas_dip_nitrogen_fert — EIA Industrial Gas Consumption Dip -> Nitrogen Fertilizer Price Spike
When FRED N3035US3M (industrial natural gas consumption) drops > 10% MoM during March-May,
long CF+NTR for 20 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL381_eia_industrial_gas_dip_nitrogen_fert"
    try:
        px = load_prices(["CF", "NTR", "SPY"], start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"price data load: {e}")

    # Load EIA industrial natural gas consumption from FRED
    gas = None
    for series in ["N3035US3M", "NGMPINDUS"]:
        try:
            gas = load_fred(series, start="2002-01-01").squeeze()
            if gas.dropna().empty:
                gas = None
                continue
            break
        except Exception:
            continue
    if gas is None:
        return mark_failed(sid, "Could not load FRED industrial gas consumption (N3035US3M / NGMPINDUS)")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    basket_tickers = [t for t in ["CF", "NTR"] if t in ret.columns]
    if not basket_tickers:
        return mark_failed(sid, "No fertilizer tickers available")
    basket_r = ret[basket_tickers].mean(axis=1)

    # Monthly MoM change
    gas_m = gas.resample("M").last().dropna()
    gas_mom = gas_m.pct_change()

    # Find March-May MoM drops > 10%
    triggers = []
    last_year = None
    for dt in gas_mom.dropna().index:
        if dt.month not in (3, 4, 5):
            continue
        mom = float(gas_mom.loc[dt])
        if np.isnan(mom):
            continue
        if mom < -0.10:
            # One signal per spring season
            if last_year == dt.year:
                continue
            triggers.append(dt)
            last_year = dt.year

    if not triggers:
        return mark_failed(sid, "No industrial gas consumption dips > 10% MoM in spring")

    events = []
    pnl_parts = []
    hold_days = 20

    for trig_date in triggers:
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
            "gas_mom_change": round(float(gas_mom.loc[trig_date]), 4),
            "basket_20d_return": round(bask_cum, 4),
            "spy_20d_return": round(spy_cum, 4),
            "excess": round(bask_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No tradeable gas consumption dip events")

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="EIA Industrial Gas Dip -> Long CF+NTR")

    avg_basket = np.mean([e["basket_20d_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["basket_20d_return"] > 0)

    save_result(sid, m, extra={
        "rule": "EIA industrial gas consumption drops > 10% MoM during Mar-May -> long CF+NTR 20 days",
        "mechanism": "Spring ammonia plant turnarounds reduce nitrogen fertilizer supply heading into peak application season",
        "source": "FRED N3035US3M + yfinance",
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
        print(f"  {flag} {e['trigger_date']}: gas_mom={e['gas_mom_change']*100:.0f}%, basket {e['basket_20d_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} -- Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
