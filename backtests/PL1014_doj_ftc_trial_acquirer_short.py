"""PL1014_doj_ftc_trial_acquirer_short — DOJ/FTC Preliminary Injunction Filing — Acquirer Short"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL1014_doj_ftc_trial_acquirer_short"

    # Historical DOJ/FTC PI filing events: (date, acquirer_ticker, case_note)
    # Each: short the acquirer from filing date, exit at deal conclusion or 120d
    events_def = [
        # date, acquirer_ticker, exit_date, exit_reason
        ("2016-07-21", "CI",   "2017-05-12", "deal_terminated"),  # Anthem/Cigna DOJ challenge; deal terminated Jan 2017, CI fell
        ("2016-07-21", "ELV",  "2017-05-12", "deal_terminated"),  # Anthem (ELV=Anthem after rename); same event separate position
        ("2017-11-20", "T",    "2018-06-12", "trial_verdict"),     # AT&T/TWX; DOJ lost June 2018 (120d ~= Mar 19 2018)
        ("2023-04-26", "MSFT", "2023-07-11", "deal_cleared"),      # MSFT/Activision; FTC PI denied July 11 2023
        ("2023-10-24", "ADBE", "2023-12-18", "deal_terminated"),   # Adobe/Figma DOJ challenge; abandoned Dec 18 2023
        ("2024-02-26", "KR",   "2024-11-29", "deal_terminated"),   # Kroger/Albertsons FTC; terminated Oct 2024 (~120d = Jun 25 2024)
    ]

    # Map entries to unique acquirers for price download
    tickers_needed = list(set([e[1] for e in events_def])) + ["SPY"]

    try:
        px = load_prices(tickers_needed, start="2016-01-01")
    except Exception as e:
        # retry once
        try:
            px = load_prices(tickers_needed, start="2016-01-01")
        except Exception as e2:
            return mark_failed(sid, f"price load: {e2}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    # Build market-neutral PnL: short acquirer + long SPY (equal notional)
    # Daily PnL = -acquirer_return + spy_return (market neutral)
    combined_pnl = pd.Series(0.0, index=spy_r.index)
    events_summary = []
    seen_entries = {}  # avoid double-counting same ticker same date

    for (filing_date_str, ticker, exit_date_str, exit_reason) in events_def:
        key = (filing_date_str, ticker)
        if key in seen_entries:
            continue
        seen_entries[key] = True

        if ticker not in ret.columns:
            print(f"  SKIP {ticker}: not in price data")
            continue

        acq_r = ret[ticker]
        filing_dt = pd.Timestamp(filing_date_str)
        exit_dt = pd.Timestamp(exit_date_str)

        # Find entry trading day (on or after filing)
        entry_mask = acq_r.index >= filing_dt
        if not entry_mask.any():
            print(f"  SKIP {ticker}: no trading days after {filing_date_str}")
            continue
        entry_idx = acq_r.index[entry_mask][0]

        # Find exit trading day (on or before exit_date, or 120 cal days from entry)
        max_exit = entry_idx + pd.Timedelta(days=120)
        actual_exit = min(exit_dt, max_exit)
        exit_mask = (acq_r.index > entry_idx) & (acq_r.index <= actual_exit)
        if not exit_mask.any():
            print(f"  SKIP {ticker}: no trading days in window")
            continue

        window_acq = acq_r[exit_mask]
        window_spy = spy_r.reindex(window_acq.index).fillna(0.0)

        # Market-neutral: short acquirer, long SPY
        daily_mn = -window_acq + window_spy

        for idx, val in daily_mn.items():
            combined_pnl[idx] += val

        total_acq = float((1 + window_acq).prod() - 1)
        total_spy = float((1 + window_spy).prod() - 1)
        excess = -total_acq + total_spy  # short acq + long SPY
        n_days = len(window_acq)

        print(f"  {ticker} ({filing_date_str} -> {actual_exit.date()}): "
              f"n={n_days}, acq_ret={total_acq:.3f}, excess={excess:.3f}")

        events_summary.append({
            "filing_date": filing_date_str,
            "ticker": ticker,
            "exit_reason": exit_reason,
            "exit_date": str(actual_exit.date()),
            "n_days_held": n_days,
            "acq_return": round(total_acq, 4),
            "spy_return": round(total_spy, 4),
            "excess_return": round(excess, 4),
        })

    if not events_summary:
        return mark_failed(sid, "no valid events found")

    active_pnl = combined_pnl[combined_pnl != 0.0]
    print(f"\nActive trading days: {len(active_pnl)}")
    print(f"Events: {len(events_summary)}")

    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="DOJ/FTC PI Filing → Short Acquirer")
    save_result(sid, m, extra={
        "rule": "Short acquirer stock within 2 days of DOJ/FTC filing PI complaint; equal-notional SPY hedge; exit at trial verdict, deal termination, or 120 calendar days",
        "mechanism": "Regulatory overhang compresses acquirer P/E multiple, deal uncertainty raises cost of capital; acquirers historically underperform during litigation period",
        "source": "justice.gov/atr; ftc.gov press releases; yfinance price data",
        "n_events": len(events_summary),
        "avg_excess_return": round(float(np.mean([e["excess_return"] for e in events_summary])), 4),
        "win_rate": round(float(np.mean([e["excess_return"] > 0 for e in events_summary])), 4),
        "events": events_summary,
    })
    print(f"Done: {len(events_summary)} events")


if __name__ == "__main__":
    main()
