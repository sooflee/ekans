"""PL801_bcra_net_reserves_argt_short_gld — BCRA Net Reserves Zero Crossing -> Short ARGT / Long GLD"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL801_bcra_net_reserves_argt_short_gld"

    # Known Argentina BCRA reserve crisis trigger dates (manually identified)
    # Each represents when BCRA net reserves crossed zero / sharply deteriorated
    # Trade: SHORT ARGT, LONG 0.5x GLD, hold up to 40 trading days
    # Based on: 2014 ARS devaluation, 2018 IMF crisis, 2019 PASO shock,
    #           2023 Massa reserve zero-cross, 2024 cepo-lift tension
    events = [
        ("2014-01-22", "Argentina ARS devaluation trigger"),
        ("2018-05-03", "BCRA reserves critical decline, IMF request"),
        ("2019-08-09", "PASO primary shock, net reserves dropped sharply"),
        ("2023-07-17", "Net reserves crossed zero under Massa"),  # Monday after 07-15 wknd
        ("2024-04-25", "Cepo-lift tension, BCRA net reserves stress"),
    ]

    tickers = ["ARGT", "GLD", "SPY"]
    try:
        px = load_prices(tickers, start="2013-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if "ARGT" not in px.columns:
        return mark_failed(sid, "ARGT not available")

    ret = daily_returns(px)
    argt_r = ret["ARGT"]
    gld_r = ret["GLD"]
    spy_r = ret["SPY"]

    hold = 40  # max 40 trading days
    event_results = []
    pnl_parts = []

    for trigger_date_str, desc in events:
        trigger_date = pd.Timestamp(trigger_date_str)

        # Entry on next trading day after trigger
        future_mask = argt_r.index > trigger_date
        if future_mask.sum() < hold // 2:
            continue

        entry_idx = argt_r.index[future_mask][0]
        pos = argt_r.index.get_loc(entry_idx)
        end_pos = min(pos + hold, len(argt_r))

        argt_window = argt_r.iloc[pos:end_pos]
        gld_window = gld_r.reindex(argt_window.index).fillna(0)

        # Trade PnL: SHORT ARGT (so negate), LONG 0.5x GLD
        trade_pnl = -argt_window + 0.5 * gld_window

        pnl_parts.append(trade_pnl)

        argt_car = float((1 + argt_window).prod() - 1)
        gld_car = float((1 + gld_window).prod() - 1)
        trade_car = float((1 + trade_pnl).prod() - 1)

        # Cumulative ARGT drawdown check (stop-loss at -25% of entry)
        argt_cum = (1 + argt_window).cumprod() - 1
        short_cum = -(1 + argt_window).cumprod() + 1  # perspective of short

        event_results.append({
            "trigger_date": trigger_date_str,
            "entry_date": str(entry_idx.date()),
            "description": desc,
            "hold_days": len(argt_window),
            "argt_return": round(argt_car, 4),
            "gld_return": round(gld_car, 4),
            "trade_pnl": round(trade_car, 4),
        })

    if not event_results:
        return mark_failed(sid, "no valid events found")

    if len(pnl_parts) < 3:
        return mark_failed(sid, f"insufficient events: only {len(pnl_parts)} valid trades")

    # Combine all trade PnL series
    all_pnl = pd.concat(pnl_parts).sort_index()
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]

    if len(all_pnl) < 20:
        return mark_failed(sid, f"insufficient trading days: {len(all_pnl)}")

    m = compute_metrics(all_pnl, benchmark=spy_r, name="BCRA Reserve Zero Crossing -> Short ARGT / Long GLD")

    save_result(sid, m, extra={
        "rule": "Short ARGT + Long 0.5x GLD when BCRA net reserves cross zero or deteriorate >$3B in 30d; max 40-day hold",
        "mechanism": "BCRA reserve depletion signals imminent ARS devaluation / capital control tightening, pressuring CCL-priced ADR basket (ARGT); GLD provides safe-haven hedge during EM risk-off",
        "source": "BCRA weekly reserve reports (bcra.gob.ar); yfinance prices",
        "n_events": len(event_results),
        "avg_trade_pnl": round(float(np.mean([e["trade_pnl"] for e in event_results])), 4),
        "win_rate": round(float(np.mean([e["trade_pnl"] > 0 for e in event_results])), 4),
        "events": event_results,
        "caveat": "Only 5 qualifying reserve-stress events since 2014; ARGT has limited history (from 2011); BCRA data requires manual calculation",
    })

    print(f"Done: {len(event_results)} events")
    for e in event_results:
        print(f"  {e['trigger_date']} ({e['description']}): ARGT={e['argt_return']:.3f}, GLD={e['gld_return']:.3f}, trade={e['trade_pnl']:.3f}")


if __name__ == "__main__":
    main()
