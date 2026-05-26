"""PL262_noaa_pdsi_drought_hydro_collapse_natgas — NOAA PDSI Extreme Drought → Long Natgas Producers
When extreme drought hits hydro-dependent regions (Pacific NW, California), hydroelectric
generation collapses and natgas must fill the gap. Long EQT+AR+RRC for 60 trading days.
Uses hand-coded drought events from NOAA PDSI data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL262_noaa_pdsi_drought_hydro_collapse_natgas"
    try:
        px = load_prices(["EQT", "AR", "RRC", "SPY"], start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    basket_tickers = [t for t in ["EQT", "AR", "RRC"] if t in ret.columns]
    if len(basket_tickers) < 2:
        return mark_failed(sid, f"Not enough natgas producer tickers: {basket_tickers}")
    basket_r = ret[basket_tickers].mean(axis=1)

    # Major drought events affecting hydro generation (PDSI extreme, <-4 in Pacific NW/CA)
    # Source: NOAA PDSI historical data + EIA hydroelectric generation reports
    drought_dates = [
        "2007-07-01",  # California drought, low hydro
        "2012-07-01",  # Severe US drought, low river/reservoir levels
        "2014-07-01",  # California mega-drought peak
        "2015-07-01",  # CA drought continued, Oroville reservoir critical
        "2020-08-01",  # Western US drought intensifying
        "2021-06-01",  # Pacific NW extreme heat dome + drought (Lake Mead critical)
        "2022-06-01",  # Western mega-drought continued, CA hydro at historic lows
        "2024-07-01",  # Renewed western drought conditions
    ]

    events = []
    pnl_parts = []
    hold_days = 60

    for date_str in drought_dates:
        trig_date = pd.Timestamp(date_str)
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
            "trigger_date": date_str,
            "basket_60d_return": round(bask_cum, 4),
            "spy_60d_return": round(spy_cum, 4),
            "excess": round(bask_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No drought events found in price data range")

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="NOAA PDSI Drought → Long Natgas Producers")

    avg_basket = np.mean([e["basket_60d_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["basket_60d_return"] > 0)

    save_result(sid, m, extra={
        "rule": "Long EQT+AR+RRC equal-weight 60d when NOAA PDSI shows extreme drought in hydro-dependent regions",
        "mechanism": "Extreme drought → hydroelectric generation collapses → natgas power demand surges → natgas producers benefit",
        "source": "NOAA PDSI + yfinance",
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
        flag = "+" if e["basket_60d_return"] > 0 else "-"
        print(f"  {flag} {e['trigger_date']}: basket {e['basket_60d_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
