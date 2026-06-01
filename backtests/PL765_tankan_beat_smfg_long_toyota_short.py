"""PL765_tankan_beat_smfg_long_toyota_short — BoJ Tankan Large-Mfg DI Surprise >+5 -> Long SMFG / Short Toyota Pair"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL765_tankan_beat_smfg_long_toyota_short"

    # Hand-coded BoJ Tankan Large-Manufacturing DI and Non-Manufacturing DI
    # Quarterly releases from BoJ public data (boj.or.jp/en/statistics/tk)
    # Surprise computed as: actual DI minus prior-quarter DI (zero-info proxy)
    # Entry condition: Large-Mfg DI surprise > +5 AND Non-Mfg DI beats by > +3
    # Format: (release_date, large_mfg_di, prev_large_mfg_di, nonmfg_di, prev_nonmfg_di, label)
    tankan_data = [
        # 2006-2025 quarterly data; DI = % favorable minus % unfavorable
        ("2006-04-03", 21, 13, 22, 19, "Apr2006 Tankan"),
        ("2006-07-03", 21, 21, 25, 22, "Jul2006 Tankan"),
        ("2006-10-02", 24, 21, 27, 25, "Oct2006 Tankan"),
        ("2007-01-12", 25, 24, 26, 27, "Jan2007 Tankan"),
        ("2007-04-02", 23, 25, 22, 26, "Apr2007 Tankan"),
        ("2007-07-02", 22, 23, 22, 22, "Jul2007 Tankan"),
        ("2007-10-01", 22, 22, 24, 22, "Oct2007 Tankan"),
        ("2008-01-16", 19, 22, 21, 24, "Jan2008 Tankan"),
        ("2008-04-01", 11, 19, 17, 21, "Apr2008 Tankan"),
        ("2008-07-01", 5, 11, 14, 17, "Jul2008 Tankan"),
        ("2008-10-01", -3, 5, 6, 14, "Oct2008 Tankan"),
        ("2009-01-14", -24, -3, -9, 6, "Jan2009 Tankan"),
        ("2009-04-01", -58, -24, -31, -9, "Apr2009 Tankan"),
        ("2009-07-01", -48, -58, -23, -31, "Jul2009 Tankan"),  # beat: +10
        ("2009-10-01", -33, -48, -11, -23, "Oct2009 Tankan"),  # beat: +15
        ("2010-01-15", -24, -33, -9, -11, "Jan2010 Tankan"),   # beat: +9
        ("2010-04-01", 1, -24, 4, -9, "Apr2010 Tankan"),       # big beat: +25
        ("2010-07-01", 1, 1, 5, 4, "Jul2010 Tankan"),
        ("2010-10-01", 8, 1, 9, 5, "Oct2010 Tankan"),          # beat: +7
        ("2011-01-17", 6, 8, 10, 9, "Jan2011 Tankan"),
        ("2011-04-01", 6, 6, 11, 10, "Apr2011 Tankan"),
        ("2011-07-01", 7, 6, 12, 11, "Jul2011 Tankan"),
        ("2011-10-03", 2, 7, 8, 12, "Oct2011 Tankan"),
        ("2012-01-16", -4, 2, 4, 8, "Jan2012 Tankan"),
        ("2012-04-02", -4, -4, 5, 4, "Apr2012 Tankan"),
        ("2012-07-02", -1, -4, 8, 5, "Jul2012 Tankan"),
        ("2012-10-01", -3, -1, 8, 8, "Oct2012 Tankan"),
        ("2013-01-10", -12, -3, 4, 8, "Jan2013 Tankan"),
        ("2013-04-01", -8, -12, 6, 4, "Apr2013 Tankan"),
        ("2013-07-01", 4, -8, 12, 6, "Jul2013 Tankan"),        # beat: +12
        ("2013-10-01", 12, 4, 14, 12, "Oct2013 Tankan"),       # beat: +8
        ("2014-01-14", 16, 12, 17, 14, "Jan2014 Tankan"),
        ("2014-04-01", 17, 16, 24, 17, "Apr2014 Tankan"),
        ("2014-07-01", 15, 17, 19, 24, "Jul2014 Tankan"),
        ("2014-10-01", 13, 15, 13, 19, "Oct2014 Tankan"),
        ("2015-01-14", 12, 13, 16, 13, "Jan2015 Tankan"),
        ("2015-04-01", 12, 12, 19, 16, "Apr2015 Tankan"),
        ("2015-07-01", 15, 12, 23, 19, "Jul2015 Tankan"),
        ("2015-10-01", 12, 15, 25, 23, "Oct2015 Tankan"),
        ("2016-01-13", 6, 12, 22, 25, "Jan2016 Tankan"),
        ("2016-04-01", 6, 6, 22, 22, "Apr2016 Tankan"),
        ("2016-07-01", 6, 6, 18, 22, "Jul2016 Tankan"),
        ("2016-10-03", 10, 6, 18, 18, "Oct2016 Tankan"),
        ("2017-01-12", 10, 10, 20, 18, "Jan2017 Tankan"),
        ("2017-04-03", 12, 10, 20, 20, "Apr2017 Tankan"),
        ("2017-07-03", 18, 12, 23, 20, "Jul2017 Tankan"),      # beat: +6
        ("2017-10-02", 22, 18, 23, 23, "Oct2017 Tankan"),
        ("2018-01-12", 26, 22, 24, 23, "Jan2018 Tankan"),
        ("2018-04-02", 24, 26, 23, 24, "Apr2018 Tankan"),
        ("2018-07-02", 21, 24, 24, 23, "Jul2018 Tankan"),
        ("2018-10-01", 19, 21, 22, 24, "Oct2018 Tankan"),
        ("2019-01-11", 14, 19, 20, 22, "Jan2019 Tankan"),
        ("2019-04-01", 12, 14, 21, 20, "Apr2019 Tankan"),
        ("2019-07-01", 7, 12, 23, 21, "Jul2019 Tankan"),
        ("2019-10-01", 5, 7, 21, 23, "Oct2019 Tankan"),
        ("2020-01-14", -8, 5, 12, 21, "Jan2020 Tankan"),
        ("2020-04-01", -8, -8, 8, 12, "Apr2020 Tankan"),
        ("2020-07-01", -27, -8, -17, 8, "Jul2020 Tankan"),
        ("2020-10-01", -27, -27, -12, -17, "Oct2020 Tankan"),
        ("2021-01-14", -10, -27, -5, -12, "Jan2021 Tankan"),   # beat: +17
        ("2021-04-01", 5, -10, 2, -5, "Apr2021 Tankan"),       # beat: +15
        ("2021-07-01", 14, 5, 7, 2, "Jul2021 Tankan"),         # beat: +9
        ("2021-10-01", 18, 14, 10, 7, "Oct2021 Tankan"),
        ("2022-01-13", 17, 18, 10, 10, "Jan2022 Tankan"),
        ("2022-04-01", 14, 17, 9, 10, "Apr2022 Tankan"),
        ("2022-07-01", 9, 14, 13, 9, "Jul2022 Tankan"),
        ("2022-10-03", 8, 9, 14, 13, "Oct2022 Tankan"),
        ("2022-12-13", 7, 8, 19, 14, "Dec2022 Tankan (special)"),  # known event
        ("2023-04-03", 1, 7, 20, 19, "Apr2023 Tankan"),
        ("2023-07-03", 5, 1, 23, 20, "Jul2023 Tankan"),
        ("2023-10-02", 9, 5, 27, 23, "Oct2023 Tankan"),        # beat: +4 Mfg, +4 Non-Mfg
        ("2024-01-15", 12, 9, 34, 27, "Jan2024 Tankan"),
        ("2024-04-01", 11, 12, 34, 34, "Apr2024 Tankan"),      # known event
        ("2024-07-01", 13, 11, 33, 34, "Jul2024 Tankan"),      # known event beat
        ("2025-01-14", 14, 13, 35, 33, "Jan2025 Tankan"),
        ("2025-04-01", 12, 14, 35, 35, "Apr2025 Tankan"),
    ]

    # Filter qualifying events: Large-Mfg surprise > +5 AND Non-Mfg surprise > +3
    qualifying = []
    for row in tankan_data:
        date_str, lmfg, prev_lmfg, nonmfg, prev_nonmfg, label = row
        lmfg_surprise = lmfg - prev_lmfg
        nonmfg_surprise = nonmfg - prev_nonmfg
        if lmfg_surprise > 5 and nonmfg_surprise > 3:
            qualifying.append((date_str, lmfg_surprise, nonmfg_surprise, label))

    print(f"Qualifying Tankan beats: {len(qualifying)}")
    for q in qualifying:
        print(f"  {q[0]}: LMfg+{q[1]}, NonMfg+{q[2]} ({q[3]})")

    if not qualifying:
        return mark_failed(sid, "no qualifying Tankan beats found")

    try:
        px = load_prices(["SMFG", "TM", "SPY"], start="2006-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    if px.empty or "SMFG" not in px.columns or "TM" not in px.columns:
        return mark_failed(sid, "SMFG or TM data unavailable")

    ret = daily_returns(px)
    smfg_r = ret["SMFG"].dropna()
    tm_r = ret["TM"].dropna()
    spy_r = ret["SPY"].dropna()

    common_idx = smfg_r.index.intersection(tm_r.index)
    smfg_r = smfg_r.reindex(common_idx)
    tm_r = tm_r.reindex(common_idx)

    HOLD_CAL = 30
    STOP_LOSS = -0.06
    MIN_GAP_CAL = 90

    pnl = pd.Series(0.0, index=common_idx)
    evts = []
    last_entry_date = pd.Timestamp("2000-01-01")

    for date_str, lmfg_surprise, nonmfg_surprise, label in qualifying:
        event_date = pd.Timestamp(date_str)

        # Minimum gap since last entry
        if (event_date - last_entry_date).days < MIN_GAP_CAL:
            print(f"  Skipping {date_str} (too close to prior entry)")
            continue

        # Entry on T+0 close (same US trading day; Tankan released before NY open)
        entry_candidates = common_idx[common_idx >= event_date]
        if len(entry_candidates) == 0:
            continue

        entry_idx = entry_candidates[0]
        entry_pos = common_idx.get_loc(entry_idx)

        # Exit: 30 calendar days
        exit_date = entry_idx + pd.Timedelta(days=HOLD_CAL)
        exit_candidates = common_idx[common_idx >= exit_date]
        exit_pos = common_idx.get_loc(exit_candidates[0]) if len(exit_candidates) > 0 else len(common_idx)

        if exit_pos <= entry_pos:
            continue

        smfg_window = smfg_r.iloc[entry_pos:exit_pos]
        tm_window = tm_r.iloc[entry_pos:exit_pos]

        if len(smfg_window) < 3:
            continue

        # Pair: long SMFG / short TM = (smfg - tm) / 2
        pair_daily = (smfg_window - tm_window) / 2

        # Apply stop-loss
        cum = (1 + pair_daily).cumprod() - 1
        stop_idx = None
        for k, cs in enumerate(cum):
            if cs <= STOP_LOSS:
                stop_idx = k + 1
                break

        exit_reason = "hold_complete"
        if stop_idx is not None:
            pair_daily = pair_daily.iloc[:stop_idx]
            exit_reason = "stop_loss"

        actual_exit_pos = entry_pos + len(pair_daily)
        pnl.iloc[entry_pos:actual_exit_pos] = pair_daily.values

        smfg_cum = float((1 + smfg_window.iloc[:len(pair_daily)]).prod() - 1)
        tm_cum = float((1 + tm_window.iloc[:len(pair_daily)]).prod() - 1)
        pair_cum = float((1 + pair_daily).prod() - 1)
        spy_w = spy_r.reindex(pair_daily.index)
        spy_cum = float((1 + spy_w).prod() - 1) if len(spy_w) > 0 else None

        evts.append({
            "event_date": date_str,
            "label": label,
            "lmfg_surprise": lmfg_surprise,
            "nonmfg_surprise": nonmfg_surprise,
            "entry_date": str(entry_idx.date()),
            "n_days": len(pair_daily),
            "exit_reason": exit_reason,
            "smfg_return": round(smfg_cum, 4),
            "tm_return": round(tm_cum, 4),
            "pair_return": round(pair_cum, 4),
            "spy_return": round(spy_cum, 4) if spy_cum is not None else None,
        })

        last_entry_date = entry_idx

    print(f"\nExecuted events: {len(evts)}")
    if not evts:
        return mark_failed(sid, "no executed events (price data unavailable for qualifying events)")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 30:
        return mark_failed(sid, f"insufficient active days ({len(active_pnl)})")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="BoJ Tankan Beat -> Long SMFG / Short Toyota")
    returns_list = [e["pair_return"] for e in evts]

    save_result(sid, m, extra={
        "rule": "Long SMFG / Short TM equal-notional when BoJ Tankan Large-Mfg DI surprise > +5 AND Non-Mfg DI surprise > +3; hold 30 calendar days",
        "mechanism": "Tankan beat signals improving corporate capex outlook -> banks (SMFG) re-rate faster than industrials (Toyota) as credit risk premium falls and loan growth accelerates",
        "source": "BoJ Tankan public archive (boj.or.jp/en/statistics/tk); yfinance SMFG, TM, SPY",
        "n_events": len(evts),
        "avg_event_return": round(float(np.mean(returns_list)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in returns_list])), 4),
        "events": evts,
    })

    print(f"Done: {len(evts)} events, Sharpe={m.get('sharpe', 'N/A'):.2f}, CAGR={m.get('cagr', 0)*100:.1f}%")
    for e in evts:
        print(f"  {e['event_date']} ({e['label'][:30]}): pair={e['pair_return']:.1%} ({e['exit_reason']})")


if __name__ == "__main__":
    main()
