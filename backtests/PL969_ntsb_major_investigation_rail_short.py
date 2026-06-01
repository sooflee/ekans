"""PL969_ntsb_major_investigation_rail_short
NTSB Major Rail Derailment Investigation -> Short Named Class I Carrier vs Long Peer Basket

Event study: On NTSB major rail investigation openings, short named Class I carrier
and long equal-weighted basket of peer Class I carriers for 60 trading days.
Known events: East Palestine 2023-02-10 (NSC), UNP Loup City 2021-09,
CSX Howard Siding 2017-02, NS 2016-01.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL969_ntsb_major_investigation_rail_short"

    tickers = ["NSC", "UNP", "CSX", "CP", "CNI", "SPY"]
    try:
        px = load_prices(tickers, start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    missing = [t for t in tickers if t not in px.columns]
    if missing:
        try:
            px = load_prices(tickers, start="2010-01-01")
            missing = [t for t in tickers if t not in px.columns]
        except Exception as e:
            return mark_failed(sid, f"data reload: {e}")
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()
    all_carriers = ["NSC", "UNP", "CSX", "CP", "CNI"]

    # Known NTSB major investigation events: (date, named_carrier)
    known_events = [
        ("2023-02-10", "NSC"),   # East Palestine OH - NTSB RRD23MR005
        ("2021-09-14", "UNP"),   # Loup City NE derailment
        ("2017-02-04", "CSX"),   # Howard Siding collision investigation
        ("2016-01-12", "NSC"),   # NS derailment investigation
        ("2015-05-12", "NSC"),   # Amtrak Philadelphia (NSC track)
    ]

    hold_days = 60
    stop_loss_pct = 0.08    # exit if named carrier outperforms peers by 8%
    take_profit_pct = 0.20  # exit if named carrier underperforms peers by 20%

    all_dates = px.index.sort_values()
    pnl_series = pd.Series(0.0, index=all_dates)
    n_events = 0

    for (evt_date_str, named_carrier) in known_events:
        evt_date = pd.Timestamp(evt_date_str)
        if named_carrier not in px.columns:
            continue

        # Find nearest trading day
        diffs = abs(all_dates - evt_date)
        entry_date = all_dates[diffs.argmin()]
        if entry_date < all_dates[0] or entry_date > all_dates[-1]:
            continue

        # Peer basket: all carriers except the named one
        peers = [c for c in all_carriers if c != named_carrier and c in px.columns]
        if len(peers) == 0:
            continue

        entry_iloc = all_dates.get_loc(entry_date)
        end_iloc = min(entry_iloc + hold_days, len(all_dates) - 1)
        n_events += 1
        cumulative_pair = 0.0

        for j in range(entry_iloc + 1, end_iloc + 1):
            date_j = all_dates[j]
            named_r_j = ret[named_carrier].get(date_j, 0.0) if date_j in ret.index else 0.0
            peer_r_j = np.mean([ret[p].get(date_j, 0.0) for p in peers if date_j in ret.index])

            # Short named carrier, long peer basket (equal notional)
            day_pnl = peer_r_j - named_r_j
            pnl_series[date_j] = pnl_series[date_j] + day_pnl
            cumulative_pair += day_pnl

            # Exit conditions
            if cumulative_pair < -stop_loss_pct:   # named outperforms peers (adverse)
                break
            if cumulative_pair > take_profit_pct:   # named underperforms peers (profit)
                break

    first_nonzero = pnl_series[pnl_series != 0].first_valid_index()
    if first_nonzero is None:
        return mark_failed(sid, "no trades executed")
    pnl = pnl_series.loc[first_nonzero:]
    spy_r_aligned = spy_r.reindex(pnl.index).fillna(0)

    if len(pnl) < 30:
        return mark_failed(sid, f"too few trading days: {len(pnl)}")

    m = compute_metrics(pnl, benchmark=spy_r_aligned,
                        name="NTSB Rail Investigation: Short Named Carrier / Long Peers")
    m["n_events"] = n_events

    save_result(sid, m, extra={
        "rule": "Short named Class I rail carrier / long equal-weight peer basket for 60 days after NTSB major investigation docket opening",
        "mechanism": "NTSB major investigation triggers regulatory scrutiny, legal reserve charges, and potential FRA emergency orders that penalize the named carrier relative to unaffected peers",
        "source": "NTSB Surface Accident Database (data.ntsb.gov/carol-main-public); known events: East Palestine 2023-02-10 (NSC), Loup City 2021-09 (UNP), Howard Siding 2017-02 (CSX), NS 2016-01 (NSC), Amtrak Philadelphia 2015-05 (NSC track)",
        "status": "ok",
    }, pnl=pnl)


if __name__ == "__main__":
    main()
