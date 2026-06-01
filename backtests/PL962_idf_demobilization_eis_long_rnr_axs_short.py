"""PL962_idf_demobilization_eis_long_rnr_axs_short
IDF Multi-Brigade Demobilization + Northern School Resumption:
Long EIS, Short RNR/AXS (Counter Defense/War-Risk)

Event study: On major IDF demobilization / ceasefire dates, enter long EIS
and short equal-weight RNR+AXS for 30 trading days. Dollar-neutral.
Known event dates: 2024-11-27, 2021-05-21, 2014-08-26, 2012-11-21, 2009-01-17
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL962_idf_demobilization_eis_long_rnr_axs_short"

    tickers = ["EIS", "RNR", "AXS", "SPY"]
    try:
        px = load_prices(tickers, start="2008-03-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    missing = [t for t in tickers if t not in px.columns]
    if missing:
        try:
            px = load_prices(tickers, start="2008-03-01")
            missing = [t for t in tickers if t not in px.columns]
        except Exception as e:
            return mark_failed(sid, f"data reload: {e}")
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()
    eis_r = ret["EIS"].dropna()
    rnr_r = ret["RNR"].dropna()
    axs_r = ret["AXS"].dropna()

    # Known IDF demobilization / ceasefire dates (long-EIS trigger)
    known_event_dates = pd.to_datetime([
        "2024-11-27",  # Lebanon ceasefire
        "2021-05-21",  # Gaza ceasefire
        "2014-08-26",  # Gaza ceasefire
        "2012-11-21",  # Operation Pillar of Defense ceasefire
        "2009-01-17",  # Operation Cast Lead ceasefire
    ])

    # Find nearest trading day for each event
    all_dates = px.index.sort_values()
    deduped_events = []
    for d in known_event_dates:
        diffs = abs(all_dates - d)
        nearest = all_dates[diffs.argmin()]
        deduped_events.append(nearest)

    hold_days = 30
    hard_stop_pct = -0.08

    pnl_series = pd.Series(0.0, index=all_dates)
    positions = pd.Series(0.0, index=all_dates)
    n_events = 0

    for entry_date in deduped_events:
        if entry_date not in px.index:
            continue
        entry_iloc = all_dates.get_loc(entry_date)
        end_iloc = min(entry_iloc + hold_days, len(all_dates) - 1)

        entry_eis = px["EIS"].iloc[entry_iloc]
        n_events += 1
        cumulative_pnl = 0.0

        for j in range(entry_iloc + 1, end_iloc + 1):
            date_j = all_dates[j]

            eis_r_j = eis_r.get(date_j, 0.0)
            rnr_r_j = rnr_r.get(date_j, 0.0) if date_j in rnr_r.index else 0.0
            axs_r_j = axs_r.get(date_j, 0.0) if date_j in axs_r.index else 0.0

            # Long EIS (0.5 weight), short RNR (0.25), short AXS (0.25)
            # Dollar-neutral: EIS notional = 1/2 * (RNR + AXS)
            day_pnl = 0.5 * eis_r_j - 0.25 * rnr_r_j - 0.25 * axs_r_j
            pnl_series[date_j] = pnl_series[date_j] + day_pnl
            positions[date_j] = 1.0
            cumulative_pnl += eis_r_j  # track EIS for hard stop

            if cumulative_pnl < hard_stop_pct:
                break

    first_nonzero = pnl_series[pnl_series != 0].first_valid_index()
    if first_nonzero is None:
        return mark_failed(sid, "no trades executed")
    pnl = pnl_series.loc[first_nonzero:]
    spy_r_aligned = spy_r.reindex(pnl.index).fillna(0)

    if len(pnl) < 30:
        return mark_failed(sid, f"too few trading days: {len(pnl)}")

    m = compute_metrics(pnl, benchmark=spy_r_aligned,
                        name="IDF Demob EIS Long / RNR AXS Short")
    m["n_events"] = n_events

    save_result(sid, m, extra={
        "rule": "Long EIS / Short RNR+AXS for 30 trading days on IDF major demobilization or ceasefire announcement",
        "mechanism": "IDF stand-down reduces war-risk premium in Israeli equities (EIS rerate higher) while reinsurers' conflict exposure pricing reverts down",
        "source": "Known ceasefire/demobilization dates: Nov 2024 Lebanon, May 2021 Gaza, Aug 2014 Gaza, Nov 2012 Gaza, Jan 2009 Gaza",
        "status": "ok",
    }, pnl=pnl)


if __name__ == "__main__":
    main()
