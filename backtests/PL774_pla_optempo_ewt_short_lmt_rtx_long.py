"""PL774_pla_optempo_ewt_short_lmt_rtx_long
PLA Taiwan Strait OPTEMPO Surge -> Short EWT, Long LMT/RTX Pair

On named PLA exercise (Joint Sword-type) with elevated activity, enter:
- Short EWT (iShares MSCI Taiwan ETF)
- Long equal-weight LMT + RTX basket (dollar-neutral)

Hard-coded known event onset dates:
  2021-06-15 (ADIZ cluster), 2022-08-04 (Joint Sword precursor / Pelosi drills),
  2023-04-10 (post-Tsai-McCarthy drills), 2024-05-23 (Joint Sword-2024A).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# Known Taiwan Strait escalation onset dates (Eastern Theater Command exercises)
KNOWN_EVENTS = [
    pd.Timestamp("2021-06-15"),
    pd.Timestamp("2022-08-04"),
    pd.Timestamp("2023-04-10"),
    pd.Timestamp("2024-05-23"),
]

HOLD_DAYS = 20
STOP_LOSS_EWT = 0.04  # exit short if EWT rises 4% from entry

def main():
    sid = "PL774_pla_optempo_ewt_short_lmt_rtx_long"
    tickers = ["EWT", "LMT", "RTX", "SPY"]

    try:
        px = load_prices(tickers, start="2020-04-06")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=5)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()
    ewt_r = ret["EWT"].dropna()
    lmt_r = ret["LMT"].dropna()
    rtx_r = ret["RTX"].dropna()

    # Long basket: equal-weight LMT + RTX
    defense_r = lmt_r.add(rtx_r, fill_value=0) / 2

    # Dollar-neutral: $1 short EWT, $1 long defense basket
    ewt_weight = -1.0
    def_weight = 1.0

    positions_ewt = pd.Series(0.0, index=ret.index)
    positions_def = pd.Series(0.0, index=ret.index)
    events = []

    for event_dt in KNOWN_EVENTS:
        # Find next trading day on or after event date
        future = ret.index[ret.index >= event_dt]
        if len(future) == 0:
            continue
        entry_date = future[0]
        entry_loc = ret.index.get_loc(entry_date)
        exit_loc = min(entry_loc + HOLD_DAYS, len(ret.index) - 1)

        # Get entry price for EWT stop-loss
        if entry_date not in px.index:
            continue
        ewt_entry_price = px.loc[entry_date, "EWT"]

        # Walk holding period with stop-loss check
        cum_pnl = 0.0
        actual_exit_loc = entry_loc
        exit_reason = "scheduled_20d"

        for j in range(entry_loc, exit_loc + 1):
            d = ret.index[j]
            ewt_day = ewt_r.get(d, 0.0) or 0.0
            def_day = defense_r.get(d, 0.0) or 0.0
            day_pnl = ewt_weight * ewt_day + def_weight * def_day
            cum_pnl += day_pnl
            positions_ewt.iloc[j] = ewt_weight
            positions_def.iloc[j] = def_weight

            # Stop-loss: EWT risen 4% from entry (short squeeze on EWT)
            ewt_cum = px.loc[d, "EWT"] / ewt_entry_price - 1
            if ewt_cum > STOP_LOSS_EWT:
                actual_exit_loc = j
                exit_reason = "stop_loss_ewt_squeeze"
                break
        else:
            actual_exit_loc = exit_loc
            exit_reason = "scheduled_20d"

        exit_date = ret.index[actual_exit_loc]

        # Compute component returns
        ewt_event = ewt_r.loc[entry_date:exit_date]
        def_event = defense_r.loc[entry_date:exit_date]
        ewt_cum_ret = float((1 + ewt_event).prod() - 1) if len(ewt_event) else 0.0
        def_cum_ret = float((1 + def_event).prod() - 1) if len(def_event) else 0.0

        events.append({
            "event_date": str(event_dt.date()),
            "entry_date": str(entry_date.date()),
            "exit_date": str(exit_date.date()),
            "exit_reason": exit_reason,
            "ewt_return": round(ewt_cum_ret, 4),
            "defense_basket_return": round(def_cum_ret, 4),
            "pair_pnl": round(cum_pnl, 4),
        })

    if not events:
        return mark_failed(sid, "no events within price data range (post-2020-04-06)")

    # Build daily PnL (no look-ahead: shift positions by 1 day)
    pos_ewt_shifted = positions_ewt.shift(1)
    pos_def_shifted = positions_def.shift(1)

    pnl = (pos_ewt_shifted.fillna(0) * ewt_r.reindex(ret.index).fillna(0) +
           pos_def_shifted.fillna(0) * defense_r.reindex(ret.index).fillna(0))
    pnl = pnl.dropna()

    if (pnl != 0).sum() < 15:
        return mark_failed(sid, f"insufficient active days: {(pnl!=0).sum()}, events: {len(events)}")

    combined_pos = (positions_ewt.abs() + positions_def.abs()) / 2
    m = compute_metrics(
        pnl,
        benchmark=spy_r,
        name="PLA Escalation Short EWT / Long LMT+RTX Pair",
        positions=combined_pos.reindex(pnl.index).fillna(0),
        cost_bps=10,
    )

    n_events = len(events)
    ev_rets = [e["pair_pnl"] for e in events]
    win_rate = float(np.mean([r > 0 for r in ev_rets])) if ev_rets else None
    avg_event = float(np.mean(ev_rets)) if ev_rets else None

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "On named PLA Joint Sword-type exercise (4 known events 2021-2024): "
                "Short EWT (Taiwan ETF) + Long equal-weight LMT+RTX (defense basket), "
                "dollar-neutral, hold 20 trading days. Stop-loss if EWT +4% from entry."
            ),
            "mechanism": (
                "Taiwan Strait escalation depresses Taiwan equity risk premium (EWT falls) "
                "while boosting US defense contractors (FMS pipeline updates, arms sales). "
                "Pair trade removes broad market exposure and isolates geopolitical risk channel."
            ),
            "source": "Taiwan MND / PLA Eastern Theater Command known events; yfinance EWT/LMT/RTX/SPY",
            "tickers": ["EWT", "LMT", "RTX"],
            "n_events": n_events,
            "event_win_rate": round(win_rate, 4) if win_rate is not None else None,
            "avg_event_return": round(avg_event, 4) if avg_event is not None else None,
            "events": events,
            "caveats": (
                "Only 4 events in 5-year sample — extremely low sample size. "
                "Wide confidence intervals; treat as directional evidence only. "
                "EWT embeds TWD/USD FX. RTX data starts 2020-04-06 (merger). "
                "Market may front-run well-publicized exercises."
            ),
        },
        pnl=pnl,
    )

    print(f"Done: {sid}")
    print(f"  n_events: {n_events}, win_rate: {win_rate}, avg_event_return: {avg_event}")
    print(
        f"  Sharpe: {m.get('sharpe', float('nan')):.2f}  "
        f"CAGR: {m.get('cagr', float('nan'))*100:.2f}%  "
        f"MaxDD: {m.get('max_dd', float('nan'))*100:.2f}%  "
        f"t-stat: {m.get('t_stat', float('nan')):.2f}"
    )
    for e in events:
        print(f"  Event {e['event_date']}: ewt={e['ewt_return']:.2%} def={e['defense_basket_return']:.2%} pair_pnl={e['pair_pnl']:.4f} [{e['exit_reason']}]")


if __name__ == "__main__":
    main()
