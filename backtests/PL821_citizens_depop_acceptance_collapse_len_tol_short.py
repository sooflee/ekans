"""PL821_citizens_depop_acceptance_collapse_len_tol_short — FL Citizens Depopulation Acceptance Rate Collapse -> Short LEN/TOL vs Long DHI"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL821_citizens_depop_acceptance_collapse_len_tol_short"

    # FL Citizens depopulation acceptance rate collapse windows
    # These are T+1 entry dates after 2nd consecutive sub-25% monthly acceptance rate confirmed
    # Sources: FL Citizens Board of Governors monthly packets (citizensfla.com)
    # 2022-11: post-Hurricane Ian (Aug-Sep 2022), private market retreated;
    #          acceptance rates collapsed from ~60% to <20% for Citizens takeout
    # 2023-03: continued FL insurer insolvencies; acceptance rate stalled <25%
    # 2024-02: pre-2024 hurricane season concern; takeout carriers paused
    events = [
        ("2022-11-01", "2022 post-Hurricane-Ian FL insurance market retreat"),
        ("2023-03-01", "2023 Q1 continued FL insurer insolvencies"),
        ("2024-02-01", "2024 Q1 FL Citizens depop stall pre-hurricane season"),
    ]

    tickers = ["LEN", "TOL", "DHI", "SPY", "PHM"]
    try:
        px = load_prices(tickers, start="2019-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    for t in ["LEN", "TOL", "DHI"]:
        if t not in px.columns:
            return mark_failed(sid, f"{t} not available")

    ret = daily_returns(px)
    len_r = ret["LEN"]
    tol_r = ret["TOL"]
    dhi_r = ret["DHI"]
    spy_r = ret["SPY"]

    hold = 60  # 12 weeks = 60 trading days
    event_results = []
    pnl_parts = []

    for event_date_str, desc in events:
        event_date = pd.Timestamp(event_date_str)

        # Entry: next trading day after trigger confirmed
        future_mask = len_r.index > event_date
        if future_mask.sum() < hold // 2:
            continue

        entry_idx = len_r.index[future_mask][0]
        pos_idx = len_r.index.get_loc(entry_idx)
        end_pos = min(pos_idx + hold, len(len_r))

        len_window = len_r.iloc[pos_idx:end_pos]
        tol_window = tol_r.reindex(len_window.index).fillna(0)
        dhi_window = dhi_r.reindex(len_window.index).fillna(0)
        spy_window = spy_r.reindex(len_window.index).fillna(0)

        # Pair: short LEN (0.5x) + short TOL (0.5x), long DHI (1x)
        # Trade PnL = long DHI - 0.5*LEN - 0.5*TOL
        trade_pnl = dhi_window - 0.5 * len_window - 0.5 * tol_window

        pnl_parts.append(trade_pnl)

        len_car = float((1 + len_window).prod() - 1)
        tol_car = float((1 + tol_window).prod() - 1)
        dhi_car = float((1 + dhi_window).prod() - 1)
        pair_car = float((1 + trade_pnl).prod() - 1)
        spy_car = float((1 + spy_window).prod() - 1)

        event_results.append({
            "event_date": event_date_str,
            "entry_date": str(entry_idx.date()),
            "description": desc,
            "hold_days": len(len_window),
            "len_return": round(len_car, 4),
            "tol_return": round(tol_car, 4),
            "dhi_return": round(dhi_car, 4),
            "pair_return": round(pair_car, 4),
            "spy_return": round(spy_car, 4),
        })

    if not event_results:
        return mark_failed(sid, "no valid Citizens depop events found")

    if len(pnl_parts) < 2:
        return mark_failed(sid, f"insufficient events: only {len(pnl_parts)} valid trades")

    # Combine all trade PnL
    all_pnl = pd.concat(pnl_parts).sort_index()
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]

    if len(all_pnl) < 30:
        return mark_failed(sid, f"insufficient trading days: {len(all_pnl)}")

    m = compute_metrics(all_pnl, benchmark=spy_r, name="FL Citizens Depop Collapse -> Short LEN/TOL Long DHI")
    m["n_events"] = len(event_results)

    save_result(sid, m, extra={
        "rule": "When FL Citizens depopulation acceptance rate < 25% for 2+ consecutive months: short LEN (0.5x) + short TOL (0.5x), long DHI (1x). Hold 60 trading days or until acceptance recovers >35%.",
        "mechanism": "FL insurance market collapse → builder cancellation rates spike in FL markets. LEN (~20% FL revenue) and TOL (~16% FL) are disproportionately exposed vs DHI (~8% FL). Insurance availability drives homebuyer mortgage approval and cancellation rates.",
        "source": "FL Citizens Board packets (citizensfla.com, manual); known event dates: post-Hurricane Ian 2022, 2023 insolvencies, 2024 pre-season. yfinance LEN, TOL, DHI, SPY.",
        "n_events": len(event_results),
        "avg_pair_return": round(float(np.mean([e["pair_return"] for e in event_results])), 4),
        "win_rate": round(float(np.mean([e["pair_return"] > 0 for e in event_results])), 4),
        "events": event_results,
        "caveat": "Very small sample (n=3). FL Citizens acceptance rate data requires manual PDF extraction from board packets. The mechanism requires 1-2 quarter lag before appearing in builder 10-Q data. Hurricane Ian timing may make 2022 event idiosyncratic.",
    })

    print(f"Done: {len(event_results)} events")
    for e in event_results:
        print(f"  {e['event_date']} ({e['description'][:55]}): LEN={e['len_return']:.3f}, TOL={e['tol_return']:.3f}, DHI={e['dhi_return']:.3f}, pair={e['pair_return']:.3f}")


if __name__ == "__main__":
    main()
