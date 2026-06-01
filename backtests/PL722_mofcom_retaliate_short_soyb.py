"""PL722_mofcom_retaliate_short_soyb
MOFCOM Retaliation -> Short SOYB

When China MOFCOM announces retaliatory tariff or punitive measures on US
agricultural exports (notably soybeans), short SOYB for 20 trading days.

Events hand-coded from MOFCOM official announcements 2018-2025:
  - 2018-04-04: MOFCOM announces 25% tariff list on 106 US products incl. soybeans
  - 2018-07-06: 25% tariffs on US soybeans take effect
  - 2019-05-13: China tariff retaliation escalation on $60B US goods (ag included)
  - 2020-08-05: China suspends ag purchases amid tensions (partial)
  - 2022-08-05: China suspends ag imports in response to Pelosi Taiwan visit
  - 2024-01-30: MOFCOM anti-dumping investigation on soybean meal (indirect)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL722_mofcom_retaliate_short_soyb"
    tickers = ["SOYB", "SPY"]

    try:
        px = load_prices(tickers, start="2012-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()
    soyb_r = ret["SOYB"].dropna()

    # Hand-coded MOFCOM retaliation event dates
    # Source: MOFCOM official press releases, Reuters/Bloomberg coverage
    event_signal_dates = [
        "2018-04-04",  # MOFCOM announces 25% tariff list on 106 US goods incl. soybeans
        "2018-07-06",  # 25% tariffs on US soybeans take effect
        "2019-05-13",  # China escalates on $60B US goods, ag included
        "2020-08-05",  # China suspends ag purchases amid tensions
        "2022-08-05",  # China suspends ag imports post-Pelosi Taiwan visit
        "2024-01-30",  # MOFCOM anti-dumping probe on soybean meal
    ]

    hold_days = 20
    events = []
    pnl_parts = []
    positions_parts = []

    idx = soyb_r.index

    for sig_str in event_signal_dates:
        sig_date = pd.Timestamp(sig_str)
        # Entry at next trading day after signal
        future_idx = idx[idx > sig_date]
        if len(future_idx) < hold_days:
            continue
        entry_date = future_idx[0]
        entry_loc = idx.get_loc(entry_date)
        exit_loc = min(entry_loc + hold_days, len(idx))

        window = slice(entry_loc, exit_loc)
        soyb_window = soyb_r.iloc[window]
        spy_window = spy_r.reindex(soyb_window.index).fillna(0)

        if soyb_window.isna().all():
            continue

        # Short SOYB: pnl = -soyb_r
        pnl_window = -soyb_window.fillna(0)
        pos_window = pd.Series(-1.0, index=soyb_window.index)

        soyb_cum = float((1 + soyb_window.fillna(0)).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)

        pnl_parts.append(pnl_window)
        positions_parts.append(pos_window)

        events.append({
            "signal_date": sig_str,
            "entry_date": str(entry_date.date()),
            "soyb_20d_return": round(soyb_cum, 4),
            "spy_20d_return": round(spy_cum, 4),
            "short_pnl": round(-soyb_cum, 4),
        })

    if len(events) < 5:
        return mark_failed(
            sid,
            f"insufficient events: {len(events)} (MOFCOM events not sufficient for statistical test)",
        )

    all_pnl = pd.concat(pnl_parts)
    all_pos = pd.concat(positions_parts)

    m = compute_metrics(
        all_pnl,
        benchmark=spy_r.reindex(all_pnl.index).fillna(0),
        name="MOFCOM Retaliation -> Short SOYB",
        positions=all_pos,
        cost_bps=10,
    )

    n_events = len(events)
    short_rets = [e["short_pnl"] for e in events]
    win_rate = float(np.mean([r > 0 for r in short_rets]))
    avg_short_ret = float(np.mean(short_rets))

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "When MOFCOM announces retaliatory tariff or punitive measure on "
                "US ag exports (notably soybeans), short SOYB for 20 trading days."
            ),
            "mechanism": (
                "MOFCOM retaliation on US soybeans directly targets the largest "
                "US agricultural export. China is the largest soy importer; "
                "retaliation triggers Brazilian soy substitution, crushing demand "
                "for US soybeans and depressing SOYB immediately."
            ),
            "source": (
                "yfinance SOYB, SPY; event dates from MOFCOM official announcements "
                "and Reuters/Bloomberg trade-war coverage 2018-2024"
            ),
            "tickers": tickers,
            "n_events": n_events,
            "event_win_rate": round(win_rate, 4),
            "avg_short_return": round(avg_short_ret, 4),
            "events": events,
            "caveats": (
                "MOFCOM announcements are discrete geopolitical events; small sample. "
                "Some events may be partially priced in before announcement. "
                "SOYB tracks soybean futures prices; roll costs and contango not "
                "fully reflected in yfinance data."
            ),
        },
        pnl=all_pnl,
    )

    print(f"Done: {sid}")
    print(f"  n_events: {n_events}, win_rate: {win_rate:.0%}, avg_short_return: {avg_short_ret:.4f}")
    sharpe = m.get("sharpe", 0)
    cagr = m.get("cagr", 0)
    print(
        f"  Sharpe: {sharpe:.2f}  CAGR: {cagr*100:.2f}%  "
        f"MaxDD: {m.get('max_dd', 0)*100:.2f}%  t-stat: {m.get('t_stat', 0):.2f}"
    )
    for e in events:
        flag = "+" if e["short_pnl"] > 0 else "-"
        print(f"  {flag} {e['signal_date']}: short_pnl={e['short_pnl']*100:+.1f}%, SPY={e['spy_20d_return']*100:+.1f}%")


if __name__ == "__main__":
    main()
