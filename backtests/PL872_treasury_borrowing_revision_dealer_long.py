"""PL872_treasury_borrowing_revision_dealer_long — Treasury Borrowing Estimate Upward Revision → Long JPM/GS vs XLF"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# Known quarterly Treasury refunding borrowing estimate announcement dates
# Format: (date_str, revision_flag, revision_desc)
# revision_flag: True = upward revision >$100B (enter long JPM+GS vs short XLF)
#                False = no significant revision or downward (skip or inverse)
# Based on: Treasury ODM press releases, TBAC meeting records
TREASURY_REFUNDING_EVENTS = [
    # Large upward revisions (>$100B) = Long dealer basket signal
    ("2020-04-27", True,  "COVID-19 fiscal surge; Treasury +$3T borrowing revision"),
    ("2021-02-01", True,  "Biden stimulus bill; massive fiscal expansion borrowing"),
    ("2021-10-29", True,  "Infrastructure bill; continued deficit expansion"),
    ("2022-05-02", True,  "Post-QT announcement; Q2 2022 large revision"),
    ("2022-10-31", True,  "Fiscal year-end; elevated post-rate-hike borrowing"),
    ("2023-01-30", True,  "Debt ceiling suspension end; refunding elevated"),
    ("2023-07-31", True,  "Largest modern revision: $1.007T vs ~$700B expected"),
    ("2024-01-29", True,  "FY2024 deficit elevated; Jan refunding upward revision"),
    ("2024-07-29", True,  "Summer 2024; Treasury maintained elevated long-end issuance"),
    ("2025-01-27", True,  "FY2025 deficit projections; elevated refunding estimate"),
    ("2025-07-28", True,  "Continued elevated deficit; summer 2025 refunding"),

    # Stable/downward revisions (small or negative) — no signal or inverse
    ("2019-04-29", False, "Q2 2019: modest revision, no large surprise"),
    ("2019-07-29", False, "Q3 2019: steady state, no significant revision"),
    ("2019-10-28", False, "Q4 2019: small revision"),
    ("2020-10-26", False, "Q4 2020: declining from COVID peak"),
    ("2021-05-03", False, "Q2 2021: declining from peak fiscal response"),
    ("2021-07-26", False, "Q3 2021: normalization; downward revision"),
    ("2023-04-24", False, "Q2 2023: debt ceiling uncertainty; lower estimate"),
    ("2023-10-30", False, "Q4 2023: post-surge normalization"),
    ("2024-04-29", False, "Q2 2024: slightly below expectations"),
    ("2024-10-28", False, "Q4 2024: modest revision"),
    ("2025-04-28", False, "Q2 2025: stable estimate"),
]


def main():
    sid = "PL872_treasury_borrowing_revision_dealer_long"

    try:
        px = load_prices(["JPM", "GS", "MS", "XLF", "SPY"], start="2018-01-01")
    except Exception as e:
        try:
            px = load_prices(["JPM", "GS", "MS", "XLF", "SPY"], start="2018-01-01", cache=False)
        except Exception as e2:
            return mark_failed(sid, f"data load: {e2}")

    if px.empty or px.shape[0] < 252:
        return mark_failed(sid, "insufficient price data")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    HOLD_DAYS = 25  # 5 trading weeks

    # Dealer basket vs XLF spread
    dealer_r = (ret["JPM"] + ret["GS"]) / 2.0  # equal-weight dealer basket
    xlf_r = ret["XLF"]
    spread_r = dealer_r - xlf_r  # positive when dealers outperform broad financials

    pnl = pd.Series(0.0, index=ret.index)
    position = pd.Series(0.0, index=ret.index)
    events = []

    for date_str, is_upward_revision, desc in TREASURY_REFUNDING_EVENTS:
        try:
            event_date = pd.Timestamp(date_str)
        except Exception:
            continue

        # Find next trading day
        future = ret.index[ret.index >= event_date]
        if len(future) == 0:
            continue
        entry_date = future[0]
        entry_i = ret.index.get_loc(entry_date)
        end_i = min(entry_i + HOLD_DAYS, len(ret.index))

        if is_upward_revision:
            # Long JPM+GS, short XLF → positive when dealers outperform
            sign = 1.0
        else:
            # No trade on non-revision quarters
            continue

        trade_r = sign * spread_r.iloc[entry_i:end_i]
        hold_return = float(trade_r.sum())

        pnl.iloc[entry_i:end_i] += trade_r.values[:end_i - entry_i]
        position.iloc[entry_i:end_i] += 1.0

        events.append({
            "event_date": date_str,
            "entry_date": str(entry_date.date()),
            "description": desc,
            "25d_spread_return": round(hold_return, 4),
        })

    print(f"Events: {len(events)}")
    active_mask = position > 0
    print(f"Active trading days: {active_mask.sum()}")

    if len(events) < 3:
        return mark_failed(sid, f"too few events: {len(events)}")

    if active_mask.sum() < 30:
        return mark_failed(sid, f"insufficient active days: {active_mask.sum()}")

    # Normalize overlapping positions
    pnl_norm = pnl.copy()
    overlap = position > 1
    pnl_norm[overlap] = pnl[overlap] / position[overlap]

    pnl_final = pnl_norm.dropna()
    win_rate = float((pd.Series([e["25d_spread_return"] for e in events]) > 0).mean())
    print(f"Win rate: {win_rate:.1%}")

    m = compute_metrics(pnl_final, benchmark=spy_r, name="Treasury Borrowing Revision: Long JPM+GS vs XLF")
    save_result(sid, m, extra={
        "rule": "Long JPM+GS equal-weight basket, short XLF 100% notional for 25 trading sessions when Treasury ODM quarterly borrowing estimate shows upward revision >$100B vs prior quarter estimate",
        "mechanism": "Large upward borrowing revisions signal elevated primary dealer FICC activity (underwriting, market-making); JPM and GS earn outsized revenue from increased auction activity and credit distribution, outperforming broad XLF which includes rate-sensitive banks and insurers",
        "source": "Treasury ODM Quarterly Refunding press releases (fiscaldata.treasury.gov); yfinance JPM, GS, XLF, SPY",
        "n_events": len(events),
        "event_win_rate": round(win_rate, 4),
        "events": events,
        "anchor_event": "2023-07-31 $1.007T revision (+307B vs expectations)",
    })

    print(f"Done: {len(events)} events, Sharpe={m.get('sharpe', 0):.2f}, CAGR={m.get('cagr', 0)*100:.1f}%")


if __name__ == "__main__":
    main()
