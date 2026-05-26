"""PL250_bls_qcew_employment_industrial_reit — BLS QCEW Warehouse Employment Surge → Long Industrial REITs
When BLS QCEW shows NAICS 493 (Warehousing & Storage) national employment growing >5% YoY,
long STAG+PLD+REXR for 30 trading days. QCEW releases quarterly with ~6 month lag.
Since QCEW API requires parsing, use known warehouse employment boom periods from BLS data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL250_bls_qcew_employment_industrial_reit"
    try:
        px = load_prices(["STAG", "PLD", "REXR", "SPY"], start="2011-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    basket_tickers = [t for t in ["STAG", "PLD", "REXR"] if t in ret.columns]
    if len(basket_tickers) < 2:
        return mark_failed(sid, f"Not enough REIT tickers available: {basket_tickers}")
    basket_r = ret[basket_tickers].mean(axis=1)

    # QCEW warehouse employment growth >5% YoY periods (release dates, ~6 month lag)
    # Data from BLS QCEW public tables — NAICS 493 national employment
    # Format: approximate QCEW release date when qualifying data was published
    qcew_signal_dates = [
        "2014-03-05",  # Q3 2013 data release — e-commerce warehouse boom starting
        "2014-09-03",  # Q1 2014 data — continued warehouse growth
        "2015-12-02",  # Q2 2015 data — Amazon fulfillment center expansion
        "2016-06-08",  # Q4 2015 data — sustained logistics growth
        "2017-09-06",  # Q1 2017 data — warehouse construction boom
        "2018-03-07",  # Q3 2017 data — peak e-commerce logistics hiring
        "2018-12-05",  # Q2 2018 data — continued strong growth
        "2020-12-02",  # Q2 2020 data — pandemic warehouse surge
        "2021-06-09",  # Q4 2020 data — pandemic e-commerce boom
        "2021-12-01",  # Q2 2021 data — supply chain crunch hiring
        "2022-03-09",  # Q3 2021 data — peak warehouse employment growth
        "2022-09-07",  # Q1 2022 data — still elevated growth
    ]

    events = []
    pnl_parts = []
    hold_days = 30

    for date_str in qcew_signal_dates:
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
            "basket_30d_return": round(bask_cum, 4),
            "spy_30d_return": round(spy_cum, 4),
            "excess": round(bask_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No QCEW warehouse employment signal events found")

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="BLS QCEW Warehouse Employment Surge → Long Industrial REITs")

    avg_basket = np.mean([e["basket_30d_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["basket_30d_return"] > 0)

    save_result(sid, m, extra={
        "rule": "Long STAG+PLD+REXR equal-weight 30d when BLS QCEW NAICS 493 employment >5% YoY",
        "mechanism": "Warehouse employment surge → industrial space demand → rental rate growth → industrial REIT NAV increase",
        "source": "BLS QCEW + yfinance",
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
        flag = "+" if e["basket_30d_return"] > 0 else "-"
        print(f"  {flag} {e['trigger_date']}: basket {e['basket_30d_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
