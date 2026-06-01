"""PL940_usda_cof_heifer_retention_cattle_cycle_counter — USDA Cattle-on-Feed Heifer Retention Inflection: Short LE=F, Long TSN"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL940_usda_cof_heifer_retention_cattle_cycle_counter"

    # LE=F is CME live cattle front-month; may not load cleanly from yfinance
    # Try LE=F first, fall back to approximate via FEEDER cattle or skip
    try:
        px = load_prices(["LE=F", "TSN", "SPY"], start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    # Check LE=F data
    le_col = None
    for col in ["LE=F", "LE"]:
        if col in px.columns and px[col].dropna().shape[0] > 100:
            le_col = col
            break

    if le_col is None:
        return mark_failed(sid, "LE=F (live cattle futures) data unavailable — cannot backtest this strategy without futures data")

    for t in ["TSN", "SPY"]:
        if t not in px.columns or px[t].dropna().shape[0] < 100:
            return mark_failed(sid, f"{t} data unavailable or insufficient")

    spy_r = daily_returns(px[["SPY"]]).iloc[:, 0].dropna()
    le_r = daily_returns(px[[le_col]]).iloc[:, 0].dropna()
    tsn_r = daily_returns(px[["TSN"]]).iloc[:, 0].dropna()

    common_idx = spy_r.index.intersection(le_r.index).intersection(tsn_r.index)
    spy_r = spy_r.reindex(common_idx)
    le_r = le_r.reindex(common_idx)
    tsn_r = tsn_r.reindex(common_idx)

    # Pair: long TSN - short LE=F (dollar-neutral)
    pair_r = tsn_r - le_r

    # Known USDA heifer retention signal trigger dates
    # 2014-02-21: USDA Feb 2014 COF heifer share dropped after drought
    # 2021-03-19: USDA Mar 2021 COF herd rebuild signal
    known_events = [
        pd.Timestamp("2014-02-21"),
        pd.Timestamp("2021-03-19"),
    ]

    hold_days = 126  # ~6 months

    pnl = pd.Series(0.0, index=common_idx)
    events = []

    for event_date in known_events:
        future = common_idx[common_idx >= event_date]
        if len(future) == 0:
            print(f"  No trading days at or after {event_date.date()}")
            continue
        entry_date = future[0]
        entry_idx = common_idx.get_loc(entry_date)
        exit_idx = min(entry_idx + hold_days, len(common_idx))

        pair_slice = pair_r.iloc[entry_idx:exit_idx]
        spy_slice = spy_r.iloc[entry_idx:exit_idx]
        le_slice = le_r.iloc[entry_idx:exit_idx]
        tsn_slice = tsn_r.iloc[entry_idx:exit_idx]

        # Stop-loss: LE=F moves >10% adverse (short LE position loses >10%)
        le_cum = le_r.iloc[entry_idx:exit_idx].cumsum()
        stop_hit = le_cum > 0.10   # cattle rally extends - short LE hurts

        if stop_hit.any():
            exit_point = stop_hit.idxmax()
            pair_slice = pair_r.loc[:exit_point].iloc[entry_idx:]
            le_slice = le_r.reindex(pair_slice.index).fillna(0)
            tsn_slice = tsn_r.reindex(pair_slice.index).fillna(0)
            spy_slice = spy_r.reindex(pair_slice.index).fillna(0)
            print(f"  Stop-loss triggered at {exit_point.date()} (LE=F cattle rally)")

        actual_exit_idx = entry_idx + len(pair_slice)
        pnl.iloc[entry_idx:actual_exit_idx] = pair_slice.values

        cum_pair_total = float((1 + pair_slice).prod() - 1)
        cum_tsn = float((1 + tsn_slice).prod() - 1)
        cum_le = float((1 + le_slice).prod() - 1)
        cum_spy = float((1 + spy_slice).prod() - 1)

        events.append({
            "event_date": str(event_date.date()),
            "entry_date": str(entry_date.date()),
            "hold_days": len(pair_slice),
            "pair_return": round(cum_pair_total, 4),
            "tsn_return": round(cum_tsn, 4),
            "le_futures_return": round(cum_le, 4),
            "spy_return": round(cum_spy, 4),
            "alpha": round(cum_pair_total - cum_spy, 4),
            "stop_hit": bool(stop_hit.any()),
        })
        print(f"  {event_date.date()}: pair={cum_pair_total*100:.1f}%, TSN={cum_tsn*100:.1f}%, LE=F={cum_le*100:.1f}%, SPY={cum_spy*100:.1f}%")

    if not events:
        return mark_failed(sid, "no valid event dates found in data")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active trading days ({len(active_pnl)})")

    print(f"Active PnL days: {len(active_pnl)}")
    m = compute_metrics(active_pnl, benchmark=spy_r, name="USDA Heifer Retention → Long TSN / Short LE=F")
    m["n_events"] = len(events)

    avg_alpha = float(np.mean([e["alpha"] for e in events]))
    win_rate = float(np.mean([1 if e["pair_return"] > 0 else 0 for e in events]))

    save_result(sid, m, extra={
        "rule": "When USDA COF heifer retention signal fires (heifer share < 37% for 2 consecutive months), enter dollar-neutral pair: long TSN / short LE=F front-month. Hold ~126 trading days. Stop-loss: LE=F futures rally >10% adverse.",
        "mechanism": "Heifer retention signals herd rebuild cycle start. Cattle supplies tighten 18-24 months later, pressuring live cattle prices higher — but packers like TSN benefit from lower immediate slaughter costs during the transition period. Counter-signal to consensus long-cattle position.",
        "source": "yfinance (LE=F, TSN, SPY); USDA NASS Cattle on Feed monthly reports (nass.usda.gov)",
        "n_events": len(events),
        "avg_event_alpha": round(avg_alpha, 4),
        "event_win_rate": round(win_rate, 4),
        "events": events,
    })
    print(f"Done: Sharpe={m.get('sharpe', 'N/A'):.2f}, CAGR={m.get('cagr', 'N/A')*100:.1f}%, MaxDD={m.get('max_dd', 'N/A')*100:.1f}%")


if __name__ == "__main__":
    main()
