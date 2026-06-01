"""PL799_scotus_cfpb_cert_subprime_fintech_pair
SCOTUS CFPB Cert Grant -> Long Subprime Fintech (UPST/OPRT/LC) Short XLF Pair

Event-study pair trade: when SCOTUS grants cert on CFPB authority challenge,
enter long subprime fintech basket vs short XLF. Hold up to 90 calendar days.
Two historical cert events: Seila Law (Oct 2019), CFSA v CFPB (Feb 2023).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL799_scotus_cfpb_cert_subprime_fintech_pair"
    tickers = ["UPST", "OPRT", "LC", "XLF", "SPY"]

    try:
        px = load_prices(tickers, start="2019-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in ["OPRT", "LC", "XLF", "SPY"] if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    # Cert grant events with known opinion dates:
    # Event 1: Seila Law v CFPB cert granted 2019-10-18, opinion 2020-06-29
    #   UPST not available until Dec 2020 - use OPRT + LC only
    # Event 2: CFSA v CFPB cert granted 2023-02-27, opinion 2024-05-16
    #   All three tickers available
    events = [
        {"cert_date": "2019-10-18", "opinion_date": "2020-06-29", "use_upst": False},
        {"cert_date": "2023-02-27", "opinion_date": "2024-05-16", "use_upst": True},
    ]

    hold_calendar_days = 90
    long_notional = 1.0
    short_notional = 0.7

    positions_long = pd.Series(0.0, index=ret.index)
    positions_short = pd.Series(0.0, index=ret.index)
    # Track which events' long basket to use (event-specific)
    # We'll use a single positions series per event

    all_pnl = pd.Series(0.0, index=ret.index)

    for ev in events:
        cert_dt = pd.Timestamp(ev["cert_date"])
        opinion_dt = pd.Timestamp(ev["opinion_date"])
        hard_stop_dt = cert_dt + pd.Timedelta(days=hold_calendar_days)
        exit_dt = min(opinion_dt, hard_stop_dt)

        # Find entry day (cert_date close = next bar in our model)
        future = ret.index[ret.index >= cert_dt]
        if len(future) == 0:
            continue
        entry_date = future[0]
        exit_dates = ret.index[ret.index >= exit_dt]
        if len(exit_dates) == 0:
            exit_date = ret.index[-1]
        else:
            exit_date = exit_dates[0]

        hold_mask = (ret.index >= entry_date) & (ret.index <= exit_date)
        hold_dates = ret.index[hold_mask]

        # Build long basket return for this event
        if ev["use_upst"] and "UPST" in ret.columns:
            long_tickers = [t for t in ["UPST", "OPRT", "LC"] if t in ret.columns]
        else:
            long_tickers = [t for t in ["OPRT", "LC"] if t in ret.columns]

        if not long_tickers:
            continue

        basket_ret = ret[long_tickers].fillna(0).mean(axis=1)
        xlf_ret = ret["XLF"].fillna(0)

        # Pair PnL on hold dates: long basket - 0.7 * XLF
        ev_pnl = (long_notional * basket_ret - short_notional * xlf_ret)
        all_pnl.loc[hold_dates] += ev_pnl.loc[hold_dates]

    all_pnl = all_pnl.reindex(spy_r.index).fillna(0)

    m = compute_metrics(all_pnl, benchmark=spy_r, name="SCOTUS CFPB Cert Subprime Fintech Pair")
    save_result(sid, m, extra={
        "rule": "Long UPST+OPRT+LC equal-weight (1x notional) short XLF (0.7x) on SCOTUS cert grant for CFPB authority challenge. Exit at opinion or 90-day hard stop. N=2 events.",
        "mechanism": "CFPB authority challenge creates regulatory uncertainty that disproportionately benefits subprime fintechs (lighter CFPB burden) vs broad financials. Market prices in relief premium at cert grant.",
        "source": "SCOTUSblog cert grants: Seila Law v CFPB cert granted 2019-10-18; CFSA v CFPB cert granted 2023-02-27. Very sparse N=2 — results indicative only.",
    })


if __name__ == "__main__":
    main()
