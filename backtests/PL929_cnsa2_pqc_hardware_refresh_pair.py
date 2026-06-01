"""PL929_cnsa2_pqc_hardware_refresh_pair — NSA CNSA 2.0 PQC Mandate: Long CSCO/HPE vs Short PANW/FTNT"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL929_cnsa2_pqc_hardware_refresh_pair"

    try:
        px = load_prices(["CSCO", "HPE", "PANW", "FTNT", "SPY"], start="2012-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    for t in ["CSCO", "HPE", "PANW", "FTNT", "SPY"]:
        if t not in px.columns or px[t].dropna().shape[0] < 100:
            return mark_failed(sid, f"{t} data unavailable or insufficient")

    spy_r = daily_returns(px[["SPY"]]).iloc[:, 0].dropna()
    csco_r = daily_returns(px[["CSCO"]]).iloc[:, 0].dropna()
    hpe_r = daily_returns(px[["HPE"]]).iloc[:, 0].dropna()
    panw_r = daily_returns(px[["PANW"]]).iloc[:, 0].dropna()
    ftnt_r = daily_returns(px[["FTNT"]]).iloc[:, 0].dropna()

    common_idx = (spy_r.index
                  .intersection(csco_r.index)
                  .intersection(hpe_r.index)
                  .intersection(panw_r.index)
                  .intersection(ftnt_r.index))

    spy_r = spy_r.reindex(common_idx)
    csco_r = csco_r.reindex(common_idx)
    hpe_r = hpe_r.reindex(common_idx)
    panw_r = panw_r.reindex(common_idx)
    ftnt_r = ftnt_r.reindex(common_idx)

    # Pair return: long 60% CSCO + 40% HPE vs short 50% PANW + 50% FTNT
    # Dollar-neutral: long leg return - short leg return
    long_r = 0.6 * csco_r + 0.4 * hpe_r
    short_r = 0.5 * panw_r + 0.5 * ftnt_r
    pair_r = long_r - short_r

    # Known PQC milestone event dates (from strategy spec)
    known_events = [
        pd.Timestamp("2022-09-07"),   # NSA CNSA 2.0 advisory published
        pd.Timestamp("2024-08-13"),   # NIST FIPS 203/204/205 finalized
    ]

    hold_days = 126  # ~6 months (2 earnings cycles)

    pnl = pd.Series(0.0, index=common_idx)
    events = []

    for event_date in known_events:
        future = common_idx[common_idx >= event_date]
        if len(future) == 0:
            print(f"  No trading days available at or after {event_date.date()}")
            continue
        entry_date = future[0]
        entry_idx = common_idx.get_loc(entry_date)
        exit_idx = min(entry_idx + hold_days, len(common_idx))

        pair_slice = pair_r.iloc[entry_idx:exit_idx]
        spy_slice = spy_r.iloc[entry_idx:exit_idx]

        # Stop-loss: exit if pair spread moves >20% adverse
        cum_pair = pair_slice.cumsum()
        stop_hit = cum_pair < -0.20

        if stop_hit.any():
            exit_point = stop_hit.idxmax()
            pair_slice = pair_slice.loc[:exit_point]
            spy_slice = spy_r.reindex(pair_slice.index).fillna(0)
            print(f"  Stop-loss triggered at {exit_point.date()}")

        actual_exit_idx = entry_idx + len(pair_slice)
        pnl.iloc[entry_idx:actual_exit_idx] = pair_slice.values

        cum_pair_total = float((1 + pair_slice).prod() - 1)
        cum_spy_total = float((1 + spy_slice).prod() - 1)
        cum_long_total = float((1 + long_r.reindex(pair_slice.index).fillna(0)).prod() - 1)
        cum_short_total = float((1 + short_r.reindex(pair_slice.index).fillna(0)).prod() - 1)

        events.append({
            "event_date": str(event_date.date()),
            "entry_date": str(entry_date.date()),
            "hold_days": len(pair_slice),
            "pair_return": round(cum_pair_total, 4),
            "long_leg_return": round(cum_long_total, 4),
            "short_leg_return": round(cum_short_total, 4),
            "spy_return": round(cum_spy_total, 4),
            "alpha": round(cum_pair_total - cum_spy_total, 4),
            "stop_hit": bool(stop_hit.any() if len(stop_hit) else False),
        })
        print(f"  {event_date.date()}: pair={cum_pair_total*100:.1f}%, long={cum_long_total*100:.1f}%, short={cum_short_total*100:.1f}%, SPY={cum_spy_total*100:.1f}%")

    # Also run rolling pair spread analysis for additional sample
    # Compute rolling 252-day pair spread starting from PANW IPO (2012)
    # This gives us a longer time series for metrics
    rolling_pnl = pair_r.copy()
    rolling_active = rolling_pnl[(rolling_pnl.index >= "2022-01-01")]

    if not events and len(rolling_active) < 30:
        return mark_failed(sid, "no valid events and insufficient rolling data")

    # Primary PnL: event-driven
    active_pnl = pnl[pnl != 0]

    if len(active_pnl) < 20:
        # Fallback: use rolling pair spread post-2022
        print(f"Event-driven PnL insufficient ({len(active_pnl)} days), using rolling pair spread 2022+")
        active_pnl = rolling_active
        if len(active_pnl) < 30:
            return mark_failed(sid, f"insufficient data for backtest ({len(active_pnl)} days)")

    print(f"Active PnL days: {len(active_pnl)}")
    m = compute_metrics(active_pnl, benchmark=spy_r, name="PQC Mandate → Long CSCO/HPE / Short PANW/FTNT")
    m["n_events"] = len(events)

    avg_alpha = float(np.mean([e["alpha"] for e in events])) if events else float('nan')
    win_rate = float(np.mean([1 if e["pair_return"] > 0 else 0 for e in events])) if events else float('nan')

    save_result(sid, m, extra={
        "rule": "At each NSA/NIST PQC milestone (CNSA 2.0 advisory Sept 2022, NIST FIPS 203/204/205 Aug 2024), enter dollar-neutral pair: long 60% CSCO + 40% HPE vs short 50% PANW + 50% FTNT. Hold ~126 trading days (2 earnings cycles). Stop-loss: -20% on pair spread.",
        "mechanism": "PQC mandates require hardware refresh cycles for classified/federal networks (routers, switches must support PQC-ready TLS). CSCO and HPE (post-JNPR acquisition) are primary beneficiaries of federal networking refresh cycles. PANW and FTNT have software-heavy revenue models less exposed to hardware refresh spend.",
        "source": "yfinance (CSCO, HPE, PANW, FTNT, SPY); NSA CNSA 2.0 advisory (nsa.gov); NIST FIPS 203/204/205 (nist.gov)",
        "known_events": [e["event_date"] for e in events],
        "n_events": len(events),
        "avg_event_alpha": round(avg_alpha, 4) if not np.isnan(avg_alpha) else None,
        "event_win_rate": round(win_rate, 4) if not np.isnan(win_rate) else None,
        "events": events,
    })
    print(f"Done: Sharpe={m.get('sharpe', 'N/A'):.2f}, CAGR={m.get('cagr', 'N/A')*100:.1f}%, MaxDD={m.get('max_dd', 'N/A')*100:.1f}%")


if __name__ == "__main__":
    main()
