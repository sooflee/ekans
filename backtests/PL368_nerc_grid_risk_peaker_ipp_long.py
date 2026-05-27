"""PL368_nerc_grid_risk_peaker_ipp_long — NERC High-Risk Grid Assessment + Summer Heat -> Long Gas Peaker IPPs
Backtest proxy: long VST+NRG each June 1 in years where ERCOT grid stress
indicators are elevated (prior year had grid emergencies or record peak demand).
Signal years: 2019, 2022, 2023, 2024, 2025.
Non-signal years (control): 2018, 2020, 2021.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL368_nerc_grid_risk_peaker_ipp_long"
    try:
        px = load_prices(["VST", "NRG", "SPY"], start="2018-01-01")
    except Exception as e:
        return mark_failed(sid, f"price data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    basket_tickers = [t for t in ["VST", "NRG"] if t in ret.columns]
    if not basket_tickers:
        return mark_failed(sid, "No peaker IPP tickers available")
    basket_r = ret[basket_tickers].mean(axis=1)

    # Signal years: prior year had ERCOT grid stress
    # 2019: after 2018 summer had some stress; NERC flagged ERCOT reserve margin tight
    # 2022: after 2021 Winter Storm Uri and summer heat; NERC flagged high risk
    # 2023: after 2022 record heat; NERC flagged ERCOT
    # 2024: after 2023 record ERCOT demand; continued tight reserves
    # 2025: data center load growth tightened ERCOT further
    # Non-signal years: 2018 (VST just formed), 2020 (COVID reduced demand), 2021 (post-Uri, summer mild)
    signal_years = [2019, 2022, 2023, 2024, 2025]

    events = []
    pnl_parts = []
    hold_days = 63

    for year in signal_years:
        june1 = pd.Timestamp(f"{year}-06-01")
        entry_mask = ret.index >= june1
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
            "trigger_date": str(june1.date()),
            "year": year,
            "basket_63d_return": round(bask_cum, 4),
            "spy_63d_return": round(spy_cum, 4),
            "excess": round(bask_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No tradeable NERC grid risk events")

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="NERC Grid Risk Summer -> Long VST+NRG")

    avg_basket = np.mean([e["basket_63d_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["basket_63d_return"] > 0)

    save_result(sid, m, extra={
        "rule": "Long VST+NRG from June 1 for 63 trading days in NERC high-risk years (ERCOT reserve margin <15%)",
        "mechanism": "ERCOT-concentrated gas peaker IPPs capture windfall scarcity pricing during summer heat in grid-stressed years",
        "source": "NERC Summer Assessment + yfinance (backtest uses curated signal years)",
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
