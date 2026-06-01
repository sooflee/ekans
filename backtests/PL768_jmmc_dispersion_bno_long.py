"""PL768_jmmc_dispersion_bno_long -- JMMC Analyst Dispersion -> BNO Long Pre-Meeting

Event study: around OPEC/JMMC ministerial meetings, go long BNO from M-5 to M+3 trading days.
Proxy for "high analyst dispersion" (stddev >= 300 kb/d): use known surprise meetings where
actual OPEC policy change deviated from consensus (surprise = large cut or large hike).
For backtest, use all identified JMMC/OPEC+ ministerial meeting dates 2016-2025 with known outcomes.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# OPEC/JMMC ministerial meeting dates with known surprise classification.
# surprise=True when actual output change deviated >200 kb/d from pre-meeting consensus.
# Sources: OPEC press releases, Reuters/Bloomberg historical reports.
OPEC_MEETINGS = [
    # 2016 - OPEC returns to quotas after 2-year hiatus
    {"date": "2016-11-30", "surprise": True,  "note": "First cut deal in 8 years, -1.2mb/d"},
    # 2017
    {"date": "2017-05-25", "surprise": False, "note": "Extended 2016 deal, in-line"},
    {"date": "2017-11-30", "surprise": False, "note": "Extended to end 2018, expected"},
    # 2018
    {"date": "2018-06-22", "surprise": True,  "note": "Surprise +1mb/d hike vs expected hold"},
    {"date": "2018-12-07", "surprise": True,  "note": "Cut 1.2mb/d agreed, surprise depth"},
    # 2019
    {"date": "2019-07-02", "surprise": False, "note": "Extended cuts, expected"},
    {"date": "2019-12-06", "surprise": True,  "note": "Deepened cuts +0.5mb/d surprise"},
    # 2020
    {"date": "2020-03-06", "surprise": True,  "note": "Meeting collapse, Saudi flood (COVID)"},
    {"date": "2020-04-09", "surprise": True,  "note": "Historic -9.7mb/d cut, largest ever"},
    {"date": "2020-06-06", "surprise": False, "note": "Extended cuts, expected"},
    {"date": "2020-12-03", "surprise": True,  "note": "Surprise +0.5mb/d partial taper vs hold"},
    # 2021
    {"date": "2021-03-04", "surprise": True,  "note": "Saudi unilateral extra -1mb/d hold"},
    {"date": "2021-07-01", "surprise": False, "note": "Gradual taper schedule, expected"},
    {"date": "2021-10-04", "surprise": False, "note": "Kept to schedule, in-line"},
    {"date": "2021-12-02", "surprise": False, "note": "Omicron, continued taper anyway"},
    # 2022
    {"date": "2022-02-02", "surprise": False, "note": "Kept +400kb/d, expected"},
    {"date": "2022-04-01", "surprise": False, "note": "Kept +400kb/d, expected"},
    {"date": "2022-06-02", "surprise": False, "note": "+648kb/d July, expected"},
    {"date": "2022-09-05", "surprise": True,  "note": "Surprise -100kb/d cut amid price pressure"},
    {"date": "2022-10-05", "surprise": True,  "note": "Surprise -2mb/d cut, largest since 2020"},
    {"date": "2022-12-04", "surprise": False, "note": "Held steady, in-line"},
    # 2023
    {"date": "2023-02-01", "surprise": False, "note": "No change, expected"},
    {"date": "2023-04-02", "surprise": True,  "note": "Surprise voluntary -1.66mb/d cut"},
    {"date": "2023-06-04", "surprise": False, "note": "Saudi extra -1mb/d, partly expected"},
    {"date": "2023-11-26", "surprise": True,  "note": "Additional voluntary cuts ~0.9mb/d"},
    # 2024
    {"date": "2024-02-01", "surprise": False, "note": "Maintained existing cuts"},
    {"date": "2024-06-02", "surprise": False, "note": "Extended cuts + phase-out plan"},
    {"date": "2024-11-03", "surprise": True,  "note": "Delayed taper again, surprised market"},
    # 2025
    {"date": "2025-02-02", "surprise": False, "note": "Kept June taper delay"},
    {"date": "2025-05-04", "surprise": True,  "note": "+411kb/d surprise large hike vs consensus"},
]


def main():
    sid = "PL768_jmmc_dispersion_bno_long"

    # Load prices: BNO available since 2007; use BNO primarily
    try:
        px = load_prices(["BNO", "SPY"], start="2014-01-01")
    except Exception as e:
        # Retry once
        try:
            px = load_prices(["BNO", "SPY"], start="2014-01-01")
        except Exception as e2:
            return mark_failed(sid, f"price load: {e2}")

    if px is None or px.empty:
        return mark_failed(sid, "price data empty after load")

    ret = daily_returns(px)
    if "BNO" not in ret.columns:
        return mark_failed(sid, "BNO not in price data")

    spy_r = ret["SPY"] if "SPY" in ret.columns else None
    bno_r = ret["BNO"]

    hold_pre = 5   # enter M-5 trading days
    hold_post = 3  # exit M+3 trading days

    pnl_parts_all = []
    pnl_parts_surprise = []
    event_results = []

    for meeting in OPEC_MEETINGS:
        meeting_date = pd.Timestamp(meeting["date"])
        is_surprise = meeting["surprise"]

        # Find meeting index in price data
        # Meeting date may be weekend/holiday - find nearest following trading day
        meeting_mask = ret.index >= meeting_date
        if meeting_mask.sum() == 0:
            continue
        meeting_idx_loc = ret.index.get_loc(ret.index[meeting_mask][0])

        # Entry: M-5 trading days before meeting
        entry_loc = max(0, meeting_idx_loc - hold_pre)
        # Exit: M+3 trading days after meeting
        exit_loc = min(len(ret.index) - 1, meeting_idx_loc + hold_post)

        if exit_loc - entry_loc < hold_pre + hold_post - 3:
            # Window too short (might be near end of data)
            continue

        window = slice(entry_loc, exit_loc)
        bno_window = bno_r.iloc[window]

        if bno_window.isna().all():
            continue

        bno_cum = float((1 + bno_window.dropna()).prod() - 1)
        spy_cum = None
        if spy_r is not None:
            spy_window = spy_r.iloc[window]
            spy_cum = float((1 + spy_window.dropna()).prod() - 1)

        event_results.append({
            "meeting_date": str(meeting_date.date()),
            "entry_date": str(ret.index[entry_loc].date()),
            "exit_date": str(ret.index[exit_loc].date()),
            "surprise": is_surprise,
            "note": meeting["note"],
            "bno_return": round(bno_cum, 4),
            "spy_return": round(spy_cum, 4) if spy_cum is not None else None,
        })

        pnl_parts_all.append(bno_window)
        if is_surprise:
            pnl_parts_surprise.append(bno_window)

    if not event_results:
        return mark_failed(sid, "no valid OPEC meeting events found in price data")

    # Primary PnL: all meetings (tests pre-meeting drift unconditionally)
    all_pnl = pd.concat(pnl_parts_all)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep="first")].dropna()

    if len(all_pnl) < 30:
        return mark_failed(sid, f"insufficient in-position days ({len(all_pnl)})")

    bench = spy_r.reindex(all_pnl.index).dropna() if spy_r is not None else None
    m = compute_metrics(all_pnl, benchmark=bench,
                        name="OPEC/JMMC Meeting Pre-Event Drift -> Long BNO")

    # Surprise-only sub-group stats
    n_surprise = sum(1 for e in event_results if e["surprise"])
    n_all = len(event_results)
    all_rets = [e["bno_return"] for e in event_results]
    surprise_rets = [e["bno_return"] for e in event_results if e["surprise"]]
    non_surprise_rets = [e["bno_return"] for e in event_results if not e["surprise"]]

    win_rate_all = sum(1 for r in all_rets if r > 0) / len(all_rets) if all_rets else 0
    win_rate_surprise = sum(1 for r in surprise_rets if r > 0) / len(surprise_rets) if surprise_rets else 0

    save_result(sid, m, extra={
        "rule": "Long BNO from M-5 to M+3 trading days around OPEC/JMMC ministerial meetings; focus on surprise meetings (analyst dispersion proxy)",
        "mechanism": "Pre-meeting uncertainty premium and position unwinding creates directional drift; surprise decisions (high analyst dispersion) amplify post-event move in long direction",
        "source": "OPEC.org meeting calendar; BNO/SPY prices from yfinance; survey dispersion proxied by known surprise events",
        "n_events": n_all,
        "n_surprise_events": n_surprise,
        "avg_return_all": round(float(np.mean(all_rets)), 4),
        "avg_return_surprise": round(float(np.mean(surprise_rets)), 4) if surprise_rets else None,
        "avg_return_non_surprise": round(float(np.mean(non_surprise_rets)), 4) if non_surprise_rets else None,
        "win_rate_all": round(win_rate_all, 3),
        "win_rate_surprise": round(win_rate_surprise, 3),
        "events": event_results,
    })

    sharpe = m.get("sharpe", 0)
    cagr = m.get("cagr", 0)
    print(f"Done: {n_all} meetings ({n_surprise} surprise), Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    print(f"  All avg: {np.mean(all_rets)*100:.1f}%  Win: {win_rate_all*100:.0f}%")
    if surprise_rets:
        print(f"  Surprise avg: {np.mean(surprise_rets)*100:.1f}%  Win: {win_rate_surprise*100:.0f}%")
    for e in event_results:
        flag = "+" if e["bno_return"] > 0 else "-"
        surp = "S" if e["surprise"] else " "
        print(f"  {flag}{surp} {e['meeting_date']}: BNO {e['bno_return']*100:+.1f}%")


if __name__ == "__main__":
    main()
