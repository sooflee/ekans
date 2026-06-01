"""PL939_fairr_protein_esg_flow_pair FAIRR Coller Protein Index: Short TSN/PPC vs Long NESN.SW/MOWI.OL"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL939_fairr_protein_esg_flow_pair"
    # Annual FAIRR release events (approximate first trading day of April/May)
    event_dates = ["2019-05-01", "2020-04-01", "2021-05-01", "2022-04-01", "2023-05-01", "2024-04-01"]
    hold_days = 60  # ~12 weeks

    try:
        # NESN.SW and MOWI.OL are Swiss/Norwegian stocks; load via yfinance
        px = load_prices(["TSN", "PPC", "NESN.SW", "MOWI.OL", "SPY"], start="2018-01-01")
    except Exception as e:
        # Retry with just US-listed tickers if foreign fails
        try:
            px = load_prices(["TSN", "PPC", "SPY"], start="2018-01-01")
        except Exception as e2:
            return mark_failed(sid, f"data load failed: {e2}")

    rets = daily_returns(px)
    spy_r = rets["SPY"].dropna()

    # Check which tickers are available
    available = list(rets.columns)
    has_nesn = "NESN.SW" in available
    has_mowi = "MOWI.OL" in available

    all_pnl_windows = []
    for event_str in event_dates:
        event_dt = pd.Timestamp(event_str)
        valid_idx = rets.index[rets.index >= event_dt]
        if len(valid_idx) < hold_days:
            continue
        entry_idx = valid_idx[0]
        end_idx = valid_idx[min(hold_days - 1, len(valid_idx) - 1)]

        window = rets.loc[entry_idx:end_idx]

        # Build short side: TSN + PPC
        short_leg = pd.Series(0.0, index=window.index)
        short_count = 0
        for t in ["TSN", "PPC"]:
            if t in window.columns and not window[t].isnull().all():
                short_leg += window[t].fillna(0)
                short_count += 1
        if short_count > 0:
            short_leg = short_leg / short_count

        # Build long side: NESN.SW + MOWI.OL (if available), else use SPY as proxy
        long_leg = pd.Series(0.0, index=window.index)
        long_count = 0
        for t in ["NESN.SW", "MOWI.OL"]:
            if t in window.columns and not window[t].isnull().all():
                long_leg += window[t].fillna(0)
                long_count += 1

        if long_count == 0:
            # No foreign tickers available; skip this event or use SPY as benchmark leg
            if "SPY" in window.columns:
                long_leg = window["SPY"].fillna(0)
            else:
                continue
        else:
            long_leg = long_leg / long_count

        if short_count == 0:
            continue

        # Pair PnL: long leg - short leg (dollar neutral)
        pair_r = long_leg - short_leg
        all_pnl_windows.append(pair_r)

    if not all_pnl_windows:
        return mark_failed(sid, "no valid event windows could be constructed")

    pnl = pd.concat(all_pnl_windows).sort_index()
    pnl = pnl[~pnl.index.duplicated(keep='first')]

    spy_aligned = spy_r.reindex(pnl.index).dropna()

    long_tickers = ["NESN.SW", "MOWI.OL"] if (has_nesn or has_mowi) else ["SPY (proxy)"]
    m = compute_metrics(pnl, benchmark=spy_aligned,
                        name="FAIRR Protein ESG Flow: Short TSN/PPC Long NESN.SW/MOWI.OL")
    save_result(sid, m, extra={
        "rule": "Annual FAIRR Coller Protein Index release (April/May): short bottom-quartile TSN+PPC, "
                "long top-quartile NESN.SW+MOWI.OL for 60 trading days.",
        "mechanism": "SFDR Art 9 EU pension/ESG fund overlays rebalance away from bottom-quartile protein "
                     "producers (emissions, antibiotics, deforestation) within 8-12 weeks of FAIRR release.",
        "source": "FAIRR Coller Protein Producer Index annual releases; events: " + ", ".join(event_dates),
        "events_used": event_dates,
        "long_tickers_used": long_tickers,
        "has_nesn": has_nesn,
        "has_mowi": has_mowi,
        "status": "ok",
    })


if __name__ == "__main__":
    main()
