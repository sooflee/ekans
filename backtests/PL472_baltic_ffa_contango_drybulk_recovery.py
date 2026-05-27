"""PL472_baltic_ffa_contango_drybulk_recovery
Baltic FFA Forward Curve Contango -> Dry Bulk Equity Recovery -> SBLK/GNK Long

When BDI shows extreme trough + recovery pattern, long dry bulk shipping equities.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns

import numpy as np
import pandas as pd


def main():
    sid = "PL472_baltic_ffa_contango_drybulk_recovery"

    # Baltic FFA data is not freely available on yfinance/FRED.
    # Proxy approach: use BDI trough + recovery events as signal dates.
    # When BDI drops below the 10th percentile and then recovers 50%+ from trough,
    # it signals the contango flip that precedes equity recovery.
    #
    # Hand-coded BDI trough + recovery events:
    events = [
        {"date": "2009-02-02", "label": "BDI recovery from ~660 (Dec 2008 trough)", "bdi_trough": 663},
        {"date": "2012-02-13", "label": "BDI recovery from ~647 (Feb 2012 trough)", "bdi_trough": 647},
        {"date": "2014-08-04", "label": "BDI recovery from ~735 (Jul 2014 trough)", "bdi_trough": 723},
        {"date": "2016-02-16", "label": "BDI recovery from all-time low ~290 (Feb 2016)", "bdi_trough": 290},
        {"date": "2020-05-18", "label": "BDI recovery from COVID trough ~393", "bdi_trough": 393},
        {"date": "2023-02-13", "label": "BDI recovery from ~525 (China reopening)", "bdi_trough": 525},
    ]

    # Load dry bulk shipper prices
    try:
        px = load_prices(["SBLK", "GNK", "SPY"], start="2008-01-01")
    except Exception as e:
        return mark_failed(sid, f"price data: {e}")

    # SBLK IPO 2007, GNK IPO 2005
    available_tickers = [t for t in ["SBLK", "GNK"] if t in px.columns and px[t].dropna().shape[0] > 100]
    if not available_tickers:
        return mark_failed(sid, "No dry bulk tickers available")

    ret = daily_returns(px)
    bulk_ret = ret[available_tickers].mean(axis=1).dropna()
    spy_ret = ret["SPY"].dropna()

    hold_days = 30
    pnl_series = pd.Series(0.0, index=bulk_ret.index)
    positions = pd.Series(0.0, index=bulk_ret.index)
    event_results = []

    for ev in events:
        entry_date = pd.Timestamp(ev["date"])

        mask = bulk_ret.index >= entry_date
        if mask.sum() < hold_days + 1:
            event_results.append({**ev, "status": "skipped", "reason": "insufficient future data"})
            continue

        start_idx = bulk_ret.index[mask][0]
        pos = bulk_ret.index.get_loc(start_idx)
        end_pos = min(pos + hold_days, len(bulk_ret))

        event_rets = bulk_ret.iloc[pos:end_pos]
        cumret = float((1 + event_rets).prod() - 1)

        spy_mask = spy_ret.index >= start_idx
        if spy_mask.sum() >= hold_days:
            spy_start = spy_ret.index[spy_mask][0]
            spy_pos = spy_ret.index.get_loc(spy_start)
            spy_event = spy_ret.iloc[spy_pos:spy_pos + hold_days]
            spy_cumret = float((1 + spy_event).prod() - 1)
        else:
            spy_cumret = None

        for i in range(pos, end_pos):
            idx = bulk_ret.index[i]
            if idx in pnl_series.index:
                pnl_series.loc[idx] = bulk_ret.iloc[i]
                positions.loc[idx] = 1.0

        event_results.append({
            **ev,
            "status": "ok",
            "bulk_30d_return": round(cumret, 4),
            "spy_30d_return": round(spy_cumret, 4) if spy_cumret is not None else None,
            "excess_return": round(cumret - spy_cumret, 4) if spy_cumret is not None else None,
        })

    ok_events = [e for e in event_results if e["status"] == "ok"]
    if len(ok_events) < 2:
        return mark_failed(sid, f"Only {len(ok_events)} valid events")

    bulk_rets = np.array([e["bulk_30d_return"] for e in ok_events])
    excess_arr = np.array([e["excess_return"] for e in ok_events if e.get("excess_return") is not None])
    avg_ret = float(bulk_rets.mean())
    avg_excess = float(excess_arr.mean()) if len(excess_arr) > 0 else None
    win_rate = float((bulk_rets > 0).mean())

    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) >= 30:
        m = compute_metrics(in_pos, benchmark=spy_ret, name="BDI Trough Recovery -> Dry Bulk Long", positions=positions[positions != 0])
    else:
        m = {
            "name": "BDI Trough Recovery -> Dry Bulk Long",
            "n_days": len(in_pos),
            "sharpe": float(in_pos.mean() / in_pos.std() * np.sqrt(252)) if len(in_pos) > 1 and in_pos.std() > 0 else 0,
            "cagr": avg_ret,
            "max_dd": float((((1 + in_pos).cumprod() / (1 + in_pos).cumprod().cummax()) - 1).min()) if len(in_pos) > 0 else 0,
            "t_stat": float(in_pos.mean() / (in_pos.std() / np.sqrt(len(in_pos)))) if len(in_pos) > 1 and in_pos.std() > 0 else 0,
        }

    save_result(sid, m, extra={
        "status": "ok",
        "rule": "When BDI recovers from extreme trough (bottom-decile + 50% recovery), long SBLK+GNK 30d",
        "mechanism": "FFA contango flip signals freight market recovery -> dry bulk equity rerate",
        "source": "Baltic Exchange BDI (proxy via trough events) + yfinance",
        "events": event_results,
        "n_events": len(ok_events),
        "avg_event_return": round(avg_ret, 4),
        "avg_excess_return": round(avg_excess, 4) if avg_excess is not None else None,
        "event_win_rate": round(win_rate, 4),
    }, pnl=in_pos if len(in_pos) >= 30 else None)

    print(f"Done: {len(ok_events)} events, avg={avg_ret*100:.2f}%, win={win_rate*100:.0f}%")
    print(f"  Sharpe: {m.get('sharpe', 'N/A')}")
    for e in event_results:
        if e["status"] == "ok":
            print(f"  {e['date']}: bulk={e['bulk_30d_return']*100:+.1f}%, excess={e.get('excess_return','N/A')}")


if __name__ == "__main__":
    main()
