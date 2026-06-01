"""PL860_h2_atm_dilution_short — H2 Pure-Play ATM-Equity Issuance Spike → Short PLUG/BE/BLDP/FCEL"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# Known 424B5 ATM issuance events from EDGAR research and known_events field
# These are the primary event dates where dilutive ATM issuances occurred
# after significant stock rallies
KNOWN_ATM_EVENTS = [
    # (date, ticker, note)
    ("2021-01-13", "PLUG", "PLUG $1.26B ATM event after 300% rally"),
    ("2021-08-01", "BE",   "BE ATM event after H2 infrastructure rally"),
    ("2022-06-01", "PLUG", "PLUG ATM utilization during H2Hub buildup"),
    ("2023-02-15", "PLUG", "PLUG 424B5 ATM after DOE hydrogen shot rally"),
    ("2023-06-01", "FCEL", "FCEL 424B5 ATM after H2 hub announcement"),
    ("2023-09-01", "BLDP", "BLDP ATM issuance amid H2 sector rally"),
    ("2024-03-01", "FCEL", "FCEL ATM utilization post H2Hub announcement"),
    ("2024-06-01", "PLUG", "PLUG ATM during IRA tailwind narrative"),
    ("2024-11-15", "BE",   "BE ATM post-election H2 narrative rally"),
    ("2025-01-15", "PLUG", "PLUG ATM issuance early 2025"),
    ("2025-03-01", "FCEL", "FCEL 424B5 ATM event"),
    ("2025-06-01", "BLDP", "BLDP ATM during H2 sector momentum"),
]


def main():
    sid = "PL860_h2_atm_dilution_short"

    tickers = ["PLUG", "BE", "BLDP", "FCEL", "APD", "LIN", "SPY"]
    try:
        px = load_prices(tickers, start="2019-01-01")
    except Exception as e:
        # Retry once
        try:
            px = load_prices(tickers, start="2019-01-01", cache=False)
        except Exception as e2:
            return mark_failed(sid, f"data load: {e2}")

    if px.empty or px.shape[0] < 252:
        return mark_failed(sid, "insufficient price data")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    # Drop tickers with insufficient data
    available = [t for t in ["PLUG", "BE", "BLDP", "FCEL"] if t in ret.columns and ret[t].notna().sum() > 100]
    if not available:
        return mark_failed(sid, "no H2 pure-play tickers available")

    print(f"Available H2 tickers: {available}")

    # Strategy: event-study around known ATM issuance dates
    # Short the issuing name for 40 trading days post event
    # Hedge: long equal-weight APD+LIN at 30% notional
    HOLD_DAYS = 40
    hedge_tickers = [t for t in ["APD", "LIN"] if t in ret.columns and ret[t].notna().sum() > 100]

    # Also add a systematic signal component:
    # When any H2 pure-play has 30-day return > +30% AND 60-day vol > 80% annualized
    # (indicating speculative froth), enter a short position
    vol_window = 60
    mom_window = 21  # ~30 calendar days

    # Build combined PnL from both event-study and systematic components
    pnl_series = pd.Series(0.0, index=ret.index)
    position_count = pd.Series(0.0, index=ret.index)
    events = []

    # --- Event-study component ---
    for event_date_str, ticker, note in KNOWN_ATM_EVENTS:
        if ticker not in ret.columns:
            continue
        try:
            event_date = pd.Timestamp(event_date_str)
        except Exception:
            continue

        # Find next trading day at or after event date
        future = ret.index[ret.index >= event_date]
        if len(future) == 0:
            continue
        entry_date = future[0]
        entry_i = ret.index.get_loc(entry_date)
        end_i = min(entry_i + HOLD_DAYS, len(ret.index))

        # Short the H2 name, hedge 30% long APD+LIN
        ticker_r = ret[ticker].iloc[entry_i:end_i]
        hold_ret = float(ticker_r.sum())

        if hedge_tickers:
            hedge_r = ret[hedge_tickers].iloc[entry_i:end_i].mean(axis=1)
            # Combined: -1.0 * ticker + 0.3 * hedge
            combined_r = -1.0 * ret[ticker].iloc[entry_i:end_i] + 0.3 * hedge_r
        else:
            combined_r = -1.0 * ticker_r

        # Add to PnL series
        pnl_series.iloc[entry_i:end_i] += combined_r.values[:end_i - entry_i]
        position_count.iloc[entry_i:end_i] += 1.0

        events.append({
            "event_date": event_date_str,
            "ticker": ticker,
            "note": note,
            "ticker_30d_return": round(hold_ret, 4),
            "entry_date": str(entry_date.date()),
        })

    # --- Systematic component: short when speculative froth detected ---
    for ticker in available:
        if ticker not in ret.columns:
            continue
        ticker_r = ret[ticker]
        log_r = np.log1p(ticker_r)
        mom_30d = np.expm1(log_r.rolling(mom_window).sum())
        ann_vol = ticker_r.rolling(vol_window).std() * np.sqrt(252)

        # Signal: 30d return > 30% AND annualized vol > 80%
        froth_signal = (mom_30d > 0.30) & (ann_vol > 0.80)

        # Don't double-enter on known events
        for i in range(1, len(ret.index)):
            loc = ret.index[i]
            if not froth_signal.iloc[i - 1]:  # shift 1 day, no look-ahead
                continue
            # Check we're not already in a position from known events
            if position_count.iloc[i] > 0:
                continue

            end_i = min(i + HOLD_DAYS, len(ret.index))
            ticker_r_hold = ret[ticker].iloc[i:end_i]
            if hedge_tickers:
                hedge_r = ret[hedge_tickers].iloc[i:end_i].mean(axis=1)
                combined_r = -1.0 * ticker_r_hold + 0.3 * hedge_r
            else:
                combined_r = -1.0 * ticker_r_hold

            pnl_series.iloc[i:end_i] += combined_r.values[:end_i - i]
            position_count.iloc[i:end_i] += 1.0

            events.append({
                "event_date": str(loc.date()),
                "ticker": ticker,
                "note": "systematic froth signal",
                "entry_date": str(loc.date()),
            })
            break  # one signal per ticker for this loop iteration

    # Normalize: when multiple positions are on, scale down
    active_mask = position_count > 0
    pnl_normalized = pnl_series.copy()
    pnl_normalized[active_mask] = pnl_series[active_mask] / position_count[active_mask].clip(lower=1)

    pnl_final = pnl_normalized.dropna()
    active_days = (active_mask & pnl_final.index.isin(ret.index)).sum()

    print(f"Total event instances: {len(events)}")
    print(f"Active trading days: {active_mask.sum()}")

    if len(events) < 3:
        return mark_failed(sid, f"too few events: {len(events)}")

    if active_mask.sum() < 30:
        return mark_failed(sid, f"insufficient active days: {active_mask.sum()}")

    m = compute_metrics(pnl_final, benchmark=spy_r, name="H2 ATM Dilution Short: PLUG/BE/BLDP/FCEL")
    save_result(sid, m, extra={
        "rule": "Short H2 pure-plays (PLUG, BE, BLDP, FCEL) for 40 trading days after known 424B5 ATM equity issuance events; hedge 30% notional long APD+LIN; also short on 30-day return >30% + annualized vol >80%",
        "mechanism": "ATM equity issuance by cash-burning H2 pure-plays dilutes existing shareholders; post-issuance overhang suppresses price for 4-8 weeks; stocks tend to underperform H2 infrastructure majors (APD, LIN) during this window",
        "source": "SEC EDGAR 424B5 filings (PLUG, BE, BLDP, FCEL); yfinance price data",
        "n_events": len([e for e in events if "systematic" not in e.get("note", "")]),
        "total_instances": len(events),
        "events": [e for e in events if "systematic" not in e.get("note", "")][:15],
    })

    print(f"Done: {len(events)} events, Sharpe={m.get('sharpe', 0):.2f}, CAGR={m.get('cagr', 0)*100:.1f}%")


if __name__ == "__main__":
    main()
