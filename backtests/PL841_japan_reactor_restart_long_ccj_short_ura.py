"""PL841_japan_reactor_restart_long_ccj_short_ura — Japan Reactor Restart NRA Approval -> Long CCJ / Short URA Pair"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL841_japan_reactor_restart_long_ccj_short_ura"

    # Strategy: Long CCJ (60%) / Short URA (40%) on Japan-specific reactor restart events
    # Focus on NRA operational gate events (NOT policy announcements = sell-the-news).
    #
    # Key event types:
    # (a) Policy announcements -> BUY-the-rumor, then FADE = NEGATIVE for long CCJ
    # (b) NRA operational gates (no-objection to anti-terror measures) -> POSITIVE
    # (c) Actual fuel loading / sync -> mixed (uranium bear overlay in 2015-2016)
    #
    # Historical events:
    # 2015-07-16: Sendai Unit 1 NRA approval for actual restart (fuel load scheduled)
    #   -> NRA operational gate, POSITIVE signal
    # 2015-08-11: Sendai Unit 1 actual restart (fuel load complete)
    #   -> Actual restart, MIXED (uranium bear market)
    # 2022-09-08: Kishida "restart all reactors" policy announcement
    #   -> Policy announcement, BUY RUMOR/FADE = negative for CCJ after T+3
    # 2023-04-19: Kashiwazaki-Kariwa NRA no-objection to anti-terrorism countermeasures
    #   -> NRA operational gate, POSITIVE signal
    # 2024-02-07: Shimane Unit 2 NRA operational safety inspection pass (pre-restart)
    #   -> NRA operational gate, POSITIVE signal
    # 2024-06-26: Kashiwazaki-Kariwa full NRA final approval (all gates cleared)
    #   -> NRA final approval, POSITIVE signal
    #
    # Treatment = NRA operational gate events (b-type)
    # Control = policy announcements and actual restarts (a-type, c-type)

    events = [
        # (date, event_type, is_treatment, label)
        # NRA operational gate events -> POSITIVE for CCJ/URA spread
        ("2015-07-16", "nra_gate", True, "Sendai Unit 1 NRA fuel-load approval"),
        ("2023-04-19", "nra_gate", True, "Kashiwazaki-Kariwa NRA anti-terror no-objection"),
        ("2024-02-07", "nra_gate", True, "Shimane Unit 2 NRA safety inspection pass"),
        ("2024-06-26", "nra_gate", True, "Kashiwazaki-Kariwa NRA final all-gates approval"),
        # Policy announcement (NEGATIVE/control - buy rumor fade)
        ("2022-09-08", "policy", False, "Kishida restart-all policy announcement (fade)"),
        # Actual restart (mixed/control)
        ("2015-08-11", "actual_restart", False, "Sendai Unit 1 actual restart fuel load"),
    ]

    hold_days = 40  # T+8 weeks per spec

    try:
        px = load_prices(["CCJ", "URA", "SPY"], start="2014-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    ret = daily_returns(px)
    needed = ["CCJ", "URA", "SPY"]
    for col in needed:
        if col not in ret.columns:
            return mark_failed(sid, f"{col} not in price data")

    ccj_r = ret["CCJ"]
    ura_r = ret["URA"]
    spy_r = ret["SPY"]

    pnl = pd.Series(0.0, index=spy_r.index)
    event_results = []

    for entry_str, event_type, is_treatment, label in events:
        entry_dt = pd.Timestamp(entry_str)
        mask = ccj_r.index >= entry_dt
        if mask.sum() < hold_days:
            continue
        entry_idx = ccj_r.index[mask][0]
        pos = ccj_r.index.get_loc(entry_idx)
        end_pos = min(pos + hold_days, len(ccj_r))
        if end_pos - pos < 10:
            continue

        ccj_rets = ccj_r.iloc[pos:end_pos]
        ura_rets = ura_r.iloc[pos:end_pos]

        # Pair: 60% long CCJ, 40% short URA
        pair_rets = 0.6 * ccj_rets.values - 0.4 * ura_rets.values

        ccj_cumret = float((1 + ccj_rets).prod() - 1)
        ura_cumret = float((1 + ura_rets).prod() - 1)
        pair_cumret = float(np.prod(1 + pair_rets) - 1)
        spy_cumret = float((1 + spy_r.iloc[pos:end_pos]).prod() - 1)

        if is_treatment:
            pnl.iloc[pos:end_pos] += pair_rets

        event_results.append({
            "entry_date": entry_str,
            "actual_entry": str(entry_idx.date()),
            "event_type": event_type,
            "is_treatment": is_treatment,
            "label": label,
            "n_days": end_pos - pos,
            "ccj_cumret": round(ccj_cumret, 4),
            "ura_cumret": round(ura_cumret, 4),
            "pair_cumret": round(pair_cumret, 4),
            "spy_cumret": round(spy_cumret, 4),
        })

    print(f"Events: {len(event_results)}")
    for e in event_results:
        flag = "TREAT" if e["is_treatment"] else "ctrl"
        print(f"  [{flag}][{e['event_type']}] {e['entry_date']} -> CCJ={e['ccj_cumret']:.2%} URA={e['ura_cumret']:.2%} pair={e['pair_cumret']:.2%} SPY={e['spy_cumret']:.2%}")

    treatment_events = [e for e in event_results if e["is_treatment"]]
    if len(treatment_events) < 3:
        return mark_failed(sid, f"insufficient treatment events ({len(treatment_events)})")

    in_pos = pnl[pnl != 0]
    if len(in_pos) < 30:
        return mark_failed(sid, f"insufficient in-position days ({len(in_pos)})")

    m = compute_metrics(in_pos, benchmark=spy_r, name="Japan Reactor NRA Gate -> Long CCJ Short URA Pair")
    pair_rets_list = [e["pair_cumret"] for e in treatment_events]

    save_result(sid, m, extra={
        "rule": "Long CCJ 60% / Short URA 40% for 40 trading days when NRA issues no-objection to anti-terrorism countermeasures or operational gate approval for Japan reactor restart (NOT policy announcements); exit T+40 or on CCJ earnings",
        "mechanism": "CCJ has Japan-specific long-term contracts at premium prices; NRA operational gate events signal near-term fuel delivery pull-forward benefiting CCJ; URA is Kazatomprom-weighted so shorts Kazakhstan spot exposure while preserving Japan-specific CCJ alpha",
        "source": "NRA (Nuclear Regulation Authority Japan) press releases; METI nuclear briefings; yfinance CCJ/URA daily prices",
        "n_events": len(treatment_events),
        "avg_pair_return": round(float(np.mean(pair_rets_list)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in pair_rets_list])), 4),
        "events": event_results,
    })
    print(f"Done: {len(treatment_events)} NRA gate events, avg pair={np.mean(pair_rets_list)*100:.2f}%, win_rate={np.mean([r>0 for r in pair_rets_list]):.0%}")


if __name__ == "__main__":
    main()
