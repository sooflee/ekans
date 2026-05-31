"""
PL607 — CMS IPPS Final Rule Net Market Basket Update >3% -> Long HCA

Event-study using hardcoded historical CMS IPPS Final Rule net market basket
updates (per spec). Fire on calendar Aug 2 of each year where that fiscal year's
update exceeds 3.0%. Hold 40 trading days OR until -8% stop OR until Oct 25
Q3 earnings proxy.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics, save_result,
                     mark_failed, daily_returns)


SIGNAL_ID = "PL607_ipps_final_rule_long_hca"
NAME = "IPPS Final Rule Net Update >3% -> Long HCA (40d)"

# Hardcoded historical net market basket update values per spec
# FY -> (announce_year, net_update_pct)
# Entry year = announce_year (the August before the fiscal year)
IPPS_HISTORY = {
    2015: (2014, 1.4),
    2016: (2015, 0.9),
    2017: (2016, 0.95),
    2018: (2017, 1.2),
    2019: (2018, 1.85),
    2020: (2019, 3.0),
    2021: (2020, 2.9),
    2022: (2021, 2.5),
    2023: (2022, 4.3),  # fires
    2024: (2023, 3.1),  # fires
    2025: (2024, 2.9),
}


def main():
    sid = SIGNAL_ID
    try:
        px = load_prices(["HCA", "XLV", "SPY"], start="2012-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    needed = ["HCA", "SPY"]
    if not all(t in px.columns for t in needed):
        return mark_failed(sid, f"missing tickers; got {list(px.columns)}")

    px = px.sort_index().ffill(limit=3)
    ret = daily_returns(px)
    hca_r = ret["HCA"]
    spy_r = ret["SPY"]

    hold_days = 40
    stop_loss = -0.08
    trading_idx = ret.index

    positions = pd.Series(0.0, index=trading_idx)
    events = []

    # Determine fire years
    fire_years = sorted([
        announce_year for (announce_year, upd) in IPPS_HISTORY.values()
        if upd > 3.0
    ])

    for announce_year in fire_years:
        ev_date = pd.Timestamp(announce_year, 8, 2)
        loc = trading_idx.searchsorted(ev_date)
        if loc >= len(trading_idx) - 1:
            continue
        # entry at next trading day's close-to-close return
        entry_loc = loc + 1 if trading_idx[loc] < ev_date else loc
        if entry_loc >= len(trading_idx):
            continue
        if pd.isna(px["HCA"].iloc[entry_loc]):
            continue
        end_loc = min(entry_loc + hold_days, len(trading_idx) - 1)
        # Q3 earnings proxy: Oct 25 same year
        q3_date = pd.Timestamp(announce_year, 10, 25)
        q3_loc = trading_idx.searchsorted(q3_date)
        if q3_loc < end_loc:
            end_loc = max(entry_loc + 1, q3_loc)

        entry_price = px["HCA"].iloc[entry_loc]
        hca_window = px["HCA"].iloc[entry_loc:end_loc + 1]
        if len(hca_window) == 0 or pd.isna(entry_price):
            continue
        cumret = hca_window / entry_price - 1.0
        exit_loc = end_loc
        exit_reason = "max_hold_or_q3"
        for k, val in enumerate(cumret.values):
            if pd.notna(val) and val <= stop_loss:
                exit_loc = entry_loc + k
                exit_reason = "stop_loss"
                break

        positions.iloc[entry_loc:exit_loc + 1] = 1.0
        events.append({
            "announce_year": announce_year,
            "net_update_pct": next(u for (a, u) in IPPS_HISTORY.values() if a == announce_year),
            "entry_date": str(trading_idx[entry_loc].date()),
            "exit_date": str(trading_idx[exit_loc].date()),
            "exit_reason": exit_reason,
            "hca_return": float(cumret.iloc[min(exit_loc - entry_loc, len(cumret) - 1)]),
        })

    if not events:
        return mark_failed(
            sid,
            "no IPPS >3% events fired in price coverage",
            extra={
                "rule": "Long HCA 40d when IPPS net update > 3% (Aug 2 event)",
                "mechanism": "Above-trend hospital reimbursement boosts margins",
                "source": "PL607 catalog; CMS IPPS Final Rule",
            },
        )

    pnl = positions.shift(1).fillna(0.0) * hca_r.reindex(positions.index).fillna(0.0)
    pnl = pnl.dropna()

    # Trim to first event for active metrics
    first_entry = pd.Timestamp(events[0]["entry_date"])
    pnl = pnl.loc[pnl.index >= first_entry]
    active_pos = positions.loc[positions.index >= first_entry]

    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient PnL coverage: {len(pnl)} days")

    m = compute_metrics(
        pnl,
        benchmark=spy_r.reindex(pnl.index),
        name=NAME,
        positions=active_pos,
        cost_bps=10,
    )
    m["n_events"] = len(events)
    m["pct_in_market"] = float(active_pos.mean()) if len(active_pos) else 0.0
    m["events"] = events
    m["status"] = "ok"

    save_result(
        sid,
        m,
        extra={
            "rule": "On Aug 2 each year, if CMS IPPS Final Rule net market basket update for that fiscal year >3.0% (from hardcoded historical CMS Final Rule fact-sheet table), go long HCA at next-day close for up to 40 trading days; exit early on -8% stop or Oct 25 Q3 earnings proxy.",
            "mechanism": "Above-trend (>3%) CMS reimbursement rate updates flow directly into HCA top-line and gross margin for the upcoming fiscal year (Oct-start). Hospital operators are price-takers on CMS; surprise upside in net update is one of the few exogenous shocks that mechanically lifts HCA earnings expectations.",
            "source": "PL607 idea catalog; CMS Federal Register IPPS Final Rule fact sheets (hardcoded historical net update table)",
            "caveats": "Very small N (only FY2023 and FY2024 firings in 2014-2025 window). Calendar proxy Aug 2 substitutes for the actual variable Federal Register release date. Q3 earnings exit uses Oct 25 proxy. Single-name idiosyncratic risk in HCA.",
        },
        pnl=pnl,
    )
    print(f"Saved {sid}: n_events={len(events)}  Sharpe={m.get('sharpe',0):.2f}  CAGR={m.get('cagr',0)*100:.2f}%")


if __name__ == "__main__":
    main()
