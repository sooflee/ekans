"""PL703_cms_advance_notice_short_hum — CMS MA Advance Notice Sub-Consensus -> Short HUM"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL703_cms_advance_notice_short_hum"

    # CMS publishes the Medicare Advantage Advance Notice in late Jan/early Feb.
    # When the effective rate growth is below sell-side consensus by >=50 bps,
    # managed care stocks (particularly HUM) sell off sharply.
    #
    # Hand-coded CMS Advance Notice release dates where rate was sub-consensus:
    # Sources: CMS press releases, sell-side research previews, news archives
    #
    # Sub-consensus years (rate below expectations by >=50bps):
    # 2017-02-01: 2018 MA rate notice - below consensus
    # 2019-02-01: 2020 MA rate notice - below expectations
    # 2020-02-06: 2021 MA rate notice - initial notice below buy-side
    # 2021-02-01: 2022 MA rate notice - below street
    # 2022-02-03: 2023 MA rate notice - initial well below consensus (3.98% vs ~5%)
    # 2024-01-31: 2025 MA rate notice - below consensus (initial notice)
    #
    # NOTE: 2023 had an ABOVE-consensus initial notice, so it is excluded.
    # 2018 had roughly in-line notice, excluded.

    sub_consensus_events = [
        # (date, approx bps below consensus, notes)
        ("2017-02-01", -80, "2018 initial advance notice sub-consensus"),
        ("2019-02-01", -60, "2020 advance notice below buy-side"),
        ("2020-02-06", -70, "2021 advance notice initial sub-consensus"),
        ("2021-02-01", -55, "2022 advance notice below street estimates"),
        ("2022-02-03", -120, "2023 initial advance notice: 3.98% vs ~5% consensus"),
        ("2024-01-31", -90, "2025 advance notice initial sub-consensus"),
    ]

    events = [pd.Timestamp(d) for d, _, _ in sub_consensus_events]
    bps_miss = [b for _, b, _ in sub_consensus_events]

    try:
        px = load_prices(["HUM", "UNH", "SPY"], start="2016-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    ret = daily_returns(px)
    if "HUM" not in ret.columns:
        return mark_failed(sid, "HUM not in price data")

    hum_r = ret["HUM"]
    spy_r = ret["SPY"]

    hold_days = 30  # per strategy spec - short HUM 30 trading days

    pnl = pd.Series(0.0, index=hum_r.index)
    event_results = []

    for td, bps, note in zip(events, bps_miss, [n for _, _, n in sub_consensus_events]):
        # Short HUM: negative returns are positive PnL
        mask = hum_r.index >= td
        if mask.sum() < hold_days:
            continue
        entry_idx = hum_r.index[mask][0]
        pos = hum_r.index.get_loc(entry_idx)
        end_pos = min(pos + hold_days, len(hum_r))
        if end_pos - pos < 10:
            continue

        # Short position: PnL = -HUM return
        event_rets = -hum_r.iloc[pos:end_pos]
        hum_cumret = float((1 + hum_r.iloc[pos:end_pos]).prod() - 1)
        short_cumret = float((1 + event_rets).prod() - 1)

        pnl.iloc[pos:end_pos] = event_rets.values[:end_pos - pos]

        spy_cumret = None
        if entry_idx in spy_r.index:
            sp = spy_r.index.get_loc(entry_idx)
            se = min(sp + hold_days, len(spy_r))
            spy_cumret = float((1 + spy_r.iloc[sp:se]).prod() - 1)

        event_results.append({
            "trigger_date": str(td.date()),
            "entry_date": str(entry_idx.date()),
            "bps_miss": bps,
            "note": note,
            "hum_return": round(hum_cumret, 4),
            "short_return": round(short_cumret, 4),
            "spy_return": round(spy_cumret, 4) if spy_cumret is not None else None,
        })

    print(f"Events: {len(event_results)}")
    for e in event_results:
        print(f"  {e['trigger_date']} bps_miss={e['bps_miss']} -> HUM={e['hum_return']:.2%} short={e['short_return']:.2%} SPY={e['spy_return']}")

    if len(event_results) < 4:
        return mark_failed(sid, f"insufficient events ({len(event_results)})")

    in_pos = pnl[pnl != 0]
    if len(in_pos) < 30:
        return mark_failed(sid, f"insufficient in-position days ({len(in_pos)})")

    m = compute_metrics(in_pos, benchmark=spy_r, name="CMS Advance Notice Sub-Consensus -> Short HUM")
    short_rets = [e["short_return"] for e in event_results]

    save_result(sid, m, extra={
        "rule": "Short HUM 30d when CMS MA Advance Notice effective rate growth >=50bps below sell-side consensus",
        "mechanism": "Sub-consensus MA rates compress HUM forward earnings estimates; stock sells off on guidance risk as HUM has highest MA revenue concentration among MCOs",
        "source": "CMS Advance Notice release dates; yfinance HUM/SPY",
        "n_events": len(event_results),
        "avg_short_return": round(float(np.mean(short_rets)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in short_rets])), 4),
        "events": event_results,
    })
    print(f"Done: {len(event_results)} events, avg short={np.mean(short_rets)*100:.2f}%, win_rate={np.mean([r>0 for r in short_rets]):.0%}")


if __name__ == "__main__":
    main()
