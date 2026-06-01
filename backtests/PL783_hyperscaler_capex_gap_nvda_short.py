"""PL783_hyperscaler_capex_gap_nvda_short
Hyperscaler 10-Q Capex Run-Rate Overshoot -> Short NVDA Counter-Signal

Event study: When all 4 hyperscalers (MSFT, GOOGL, META, AMZN) have filed
their quarterly 10-Q with capex pace > 105% of annual FY guide, short NVDA.
Known events: 2023-08-02 (Q2 2023), 2024-08-07 (Q2 2024).
Exit: NVDA next earnings (8-week proxy), trailing 12% stop.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL783_hyperscaler_capex_gap_nvda_short"
    tickers = ["NVDA", "SOXX", "SPY"]

    try:
        px = load_prices(tickers, start="2020-01-01")
    except Exception as e:
        try:
            px = load_prices(tickers, start="2020-01-01")
        except Exception as e2:
            return mark_failed(sid, f"data load: {e2}")

    for t in tickers:
        if t not in px.columns:
            return mark_failed(sid, f"missing ticker: {t}")

    px = px.sort_index().ffill(limit=3)
    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()
    nvda_ret = ret["NVDA"]

    # Hardcoded event dates: day after 4th hyperscaler files 10-Q, capex > 105% guide pace
    # 2023-08-02: Q2 2023 -- META revised H2 capex down after spending run-rate exceeded guide
    # 2024-08-07: Q2 2024 -- aggregate capex front-loaded, NVDA DC decel signaled
    event_dates_str = ["2023-08-02", "2024-08-07"]
    hold_days = 40  # ~8 weeks of trading days

    pnl_records = []
    event_details = []

    for ev_str in event_dates_str:
        ev_date = pd.Timestamp(ev_str)
        future_idx = ret.index[ret.index >= ev_date]
        if len(future_idx) < 2:
            continue

        entry_date = future_idx[0]
        entry_px = px["NVDA"].get(entry_date) if entry_date in px.index else None
        if entry_px is None or np.isnan(entry_px):
            continue

        # Hold for up to hold_days or until 12% trailing stop
        peak_from_entry = 0.0
        window_pnl = []
        stop_hit = False
        exit_reason = "time_stop"

        for j in range(min(hold_days, len(future_idx))):
            d = future_idx[j]
            nvda_p = px["NVDA"].get(d) if d in px.index else None
            if nvda_p is None or np.isnan(nvda_p):
                break

            cum_move = nvda_p / entry_px - 1.0  # NVDA price change since entry
            # Track trailing stop: NVDA moves 12% against short (i.e., rises 12% from entry)
            if cum_move > peak_from_entry:
                peak_from_entry = cum_move
            if peak_from_entry >= 0.12:
                stop_hit = True
                exit_reason = "trailing_stop"
                break

            # PnL on short: -1 * NVDA daily return
            if d in nvda_ret.index:
                window_pnl.append((d, -nvda_ret[d]))

        for d, p in window_pnl:
            pnl_records.append({"date": d, "pnl": p})

        # Compute cumulative pair return
        if window_pnl:
            pnl_series_ev = pd.Series([p for _, p in window_pnl])
            cum_ret = (1 + pnl_series_ev).cumprod().iloc[-1] - 1
        else:
            cum_ret = 0.0

        event_details.append({
            "event_date": ev_str,
            "n_hold_days": len(window_pnl),
            "cum_short_nvda_ret": float(cum_ret),
            "exit_reason": exit_reason,
        })

    if not pnl_records:
        return mark_failed(sid, "no valid events found in price data")

    pnl_df = pd.DataFrame(pnl_records).set_index("date")["pnl"]
    pnl_series = pnl_df.groupby(level=0).sum()

    full_idx = ret.index[ret.index >= pnl_series.index.min()]
    pnl_full = pnl_series.reindex(full_idx).fillna(0)

    spy_aligned = spy_r.reindex(full_idx).dropna()
    pnl_aligned = pnl_full.reindex(spy_aligned.index).fillna(0)

    m = compute_metrics(pnl_aligned, benchmark=spy_aligned, name="Hyperscaler CapEx Gap Short NVDA")
    m["n_events"] = len(event_dates_str)

    save_result(sid, m, extra={
        "rule": "After all 4 hyperscalers (MSFT/GOOGL/META/AMZN) file Q2/Q3 10-Q with aggregate capex pace >105% of FY guide, short NVDA for up to 8 weeks or 12% trailing stop.",
        "mechanism": "NVDA earns ~50% of DC revenue from top-4 hyperscalers. Capex overshoot vs guide signals deceleration risk into next quarter. Market prices in DC demand disappointment at NVDA next earnings.",
        "source": "SEC EDGAR XBRL us-gaap:PaymentsToAcquirePropertyPlantAndEquipment; 8-K earnings transcripts for FY guides. Hardcoded events: 2023-08-02, 2024-08-07.",
        "event_details": event_details,
        "caveats": "Only 2 known events; event count too small for robust statistics. Counter-signal vs long-semis/long-NVDA.",
        "status": "ok",
    }, pnl=pnl_aligned)

    print(f"\n=== {sid} ===")
    print(f"Events: {len(event_dates_str)}, total active days: {int((pnl_full != 0).sum())}")
    for ev in event_details:
        print(f"  {ev['event_date']}: held {ev['n_hold_days']}d, short-NVDA cum={ev['cum_short_nvda_ret']:.2%}, exit={ev['exit_reason']}")
    print(f"Sharpe: {m.get('sharpe', 'N/A'):.3f}  CAGR: {m.get('cagr', 0):.2%}  MaxDD: {m.get('max_dd', 0):.2%}  t-stat: {m.get('t_stat', 0):.3f}")


if __name__ == "__main__":
    main()
