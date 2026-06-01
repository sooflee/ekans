"""PL800_scotus_nlrb_union_heavy_short
SCOTUS NLRB-Authority Ruling -> Short Union-Heavy Airlines/Auto/Logistics Basket

On a SCOTUS ruling that narrows NLRB authority (or grants cert on such a case),
wait 3 trading days for initial market reaction, then enter SHORT equal-weight
basket of F, GM, UAL, AAL, UPS for up to 40 trading days. Also tested with
major strike-event entries as secondary corroboration events.

Known events per strategy spec:
  - 2024-06-13: Starbucks v. McKinney (NLRB injunction standard narrowed)
  - 2024-06-28: Loper Bright v. Raimondo (Chevron overturned -> NLRB weakened)
  - 2023-09-15: UAW Big Three strike begins
  - 2023-07-25: UPS Teamsters near-strike
  - 2024-09-13: Boeing IAM strike begins (BA, not in basket, but market signal)

Benchmark: SPY. Hold: 40 trading days from Day+3. Stop: basket +15%.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# -----------------------------------------------------------------------
# Hard-coded event dates
# announcement_date: the SCOTUS decision / strike announcement date
# entry_lag: trading days to wait before entering short
# hold_days: max hold from entry
# event_type: "scotus_nlrb" or "strike_escalation"
# -----------------------------------------------------------------------
EVENTS = [
    {
        "announcement_date": "2024-06-13",
        "event_type": "scotus_nlrb",
        "description": "Starbucks v. McKinney: SCOTUS narrows NLRB preliminary injunction standard",
        "entry_lag": 3,
        "hold_days": 40,
    },
    {
        "announcement_date": "2024-06-28",
        "event_type": "scotus_nlrb",
        "description": "Loper Bright v. Raimondo: Chevron overturned, NLRB authority broadly weakened",
        "entry_lag": 3,
        "hold_days": 40,
    },
    {
        "announcement_date": "2023-09-15",
        "event_type": "strike_escalation",
        "description": "UAW Big Three strike begins (F/GM direct exposure)",
        "entry_lag": 0,  # immediate entry on announcement
        "hold_days": 40,
    },
    {
        "announcement_date": "2023-07-25",
        "event_type": "strike_escalation",
        "description": "UPS Teamsters near-strike (UPS direct exposure)",
        "entry_lag": 0,
        "hold_days": 40,
    },
    {
        "announcement_date": "2024-09-13",
        "event_type": "strike_escalation",
        "description": "Boeing IAM strike begins (sector signal, not in basket)",
        "entry_lag": 0,
        "hold_days": 40,
    },
]

STOP_LOSS_PCT = 0.15  # exit if basket gains 15% (hurts short)


def run_event_study(events, ret, basket, stop_loss_pct=0.15):
    """
    For each event, enter SHORT equal-weight basket `entry_lag` trading days
    after announcement. Hold up to `hold_days` trading days. Stop if basket
    cumulative return from entry exceeds `stop_loss_pct` (basket up = loss on short).
    Returns (pnl_series, positions_series, event_log).
    """
    idx = ret.index
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)

    basket_ret = ret[basket].fillna(0).mean(axis=1)

    event_log = []
    for ev in events:
        ann_dt = pd.Timestamp(ev["announcement_date"])
        entry_lag = ev["entry_lag"]
        hold_days = ev["hold_days"]

        # Find entry: `entry_lag` trading days after announcement date
        future = idx[idx > ann_dt]
        if len(future) < max(entry_lag, 1):
            ev_record = dict(ev)
            ev_record["status"] = "no_data_after_announcement"
            event_log.append(ev_record)
            continue

        if entry_lag == 0:
            # Same day entry (or next day if announcement is after market)
            # Per spec: entry at close on announcement day + 0 lag
            future_inc = idx[idx >= ann_dt]
            if len(future_inc) == 0:
                ev_record = dict(ev)
                ev_record["status"] = "no_data"
                event_log.append(ev_record)
                continue
            entry_dt = future_inc[0]
        else:
            entry_dt = future[entry_lag - 1]

        entry_pos = idx.get_loc(entry_dt)
        max_exit_pos = min(entry_pos + hold_days, len(idx))

        # Walk forward checking stop-loss
        cumulative = 1.0
        actual_exit_pos = max_exit_pos
        exit_reason = "scheduled_40d"

        for j in range(entry_pos, max_exit_pos):
            day_basket_ret = float(basket_ret.iloc[j])
            cumulative *= (1 + day_basket_ret)
            basket_gain = cumulative - 1.0
            # Stop-loss: basket is UP more than stop_loss_pct from entry
            if basket_gain > stop_loss_pct:
                exit_reason = "stop_loss"
                actual_exit_pos = j + 1  # include this day, then stop
                break

        # Record PnL (short basket = negative of basket return)
        for j in range(entry_pos, actual_exit_pos):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = -1.0
                pnl.iloc[j] = float(-basket_ret.iloc[j])

        # Event return: from entry to exit (on the basket itself)
        basket_cum = float((1 + basket_ret.iloc[entry_pos:actual_exit_pos]).prod() - 1)
        short_cum = float((1 / (1 + basket_ret.iloc[entry_pos:actual_exit_pos])).prod() - 1)

        exit_dt = idx[actual_exit_pos - 1] if actual_exit_pos > entry_pos else entry_dt

        ev_record = dict(ev)
        ev_record["entry_date"] = str(entry_dt.date())
        ev_record["exit_date"] = str(exit_dt.date())
        ev_record["exit_reason"] = exit_reason
        ev_record["n_hold_days"] = int(actual_exit_pos - entry_pos)
        ev_record["basket_return"] = round(basket_cum, 4)
        ev_record["short_basket_return"] = round(short_cum, 4)

        # Per-name returns
        for name in basket:
            if name in ret.columns:
                slice_r = ret[name].iloc[entry_pos:actual_exit_pos]
                cum_r = float((1 + slice_r.fillna(0)).prod() - 1)
                ev_record[f"{name.lower()}_return"] = round(cum_r, 4)

        event_log.append(ev_record)

    return pnl, positions, event_log


def main():
    sid = "PL800_scotus_nlrb_union_heavy_short"
    basket = ["F", "GM", "UAL", "AAL", "UPS"]
    tickers = basket + ["SPY"]

    try:
        px = load_prices(tickers, start="2022-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    # --- Run event study on all events ---
    pnl_all, positions_all, log_all = run_event_study(
        EVENTS, ret, basket, stop_loss_pct=STOP_LOSS_PCT
    )

    # --- Run SCOTUS-only subset ---
    scotus_events = [e for e in EVENTS if e["event_type"] == "scotus_nlrb"]
    pnl_scotus, positions_scotus, log_scotus = run_event_study(
        scotus_events, ret, basket, stop_loss_pct=STOP_LOSS_PCT
    )

    n_all = sum(1 for e in log_all if e.get("entry_date"))
    n_scotus = sum(1 for e in log_scotus if e.get("entry_date"))

    print(f"All events traded: {n_all}")
    print(f"SCOTUS-only events traded: {n_scotus}")
    for e in log_all:
        if e.get("entry_date"):
            print(f"  {e['announcement_date']} [{e['event_type']}] -> entry {e['entry_date']}, "
                  f"short_return={e.get('short_basket_return'):.4f}, reason={e.get('exit_reason')}")

    # Use all events as primary (more data points)
    primary_pnl = pnl_all
    primary_positions = positions_all
    primary_log = log_all
    primary_label = "all_events"

    # Metrics on held days only
    held_pnl = primary_pnl[primary_positions != 0]
    held_spy = spy_r.reindex(held_pnl.index).fillna(0)

    if len(held_pnl) < 20:
        short_returns = [e.get("short_basket_return") for e in primary_log if e.get("short_basket_return") is not None]
        avg_ret = float(np.mean(short_returns)) if short_returns else None
        return mark_failed(
            sid,
            f"insufficient held days: {len(held_pnl)} (n_events={n_all})",
            extra={
                "events_all": log_all,
                "events_scotus": log_scotus,
                "avg_event_return": avg_ret,
            },
        )

    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name="SCOTUS/NLRB-Trigger Short Union Basket (held-days)",
        positions=primary_positions.reindex(held_pnl.index).fillna(0),
        cost_bps=10,
    )

    short_returns = [e.get("short_basket_return") for e in primary_log if e.get("short_basket_return") is not None]
    win_rate = float(np.mean([r > 0 for r in short_returns])) if short_returns else None
    avg_ret = float(np.mean(short_returns)) if short_returns else None

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "On a SCOTUS ruling narrowing NLRB authority (Starbucks v. McKinney, "
                "Loper Bright), enter SHORT equal-weight basket of F/GM/UAL/AAL/UPS "
                "3 trading days after announcement (to let initial reaction settle). "
                "On major union-strike escalation events, enter SHORT at the open "
                "of announcement day. Hold up to 40 trading days; stop-loss if "
                "basket gains >15% from entry."
            ),
            "mechanism": (
                "A SCOTUS ruling weakening NLRB authority removes the legal backstop "
                "for union organizing and enforcement, but the COUNTER-signal thesis "
                "is that operational disruption risk is now higher (employers emboldened "
                "to push harder, escalating strike probability). Strike escalation "
                "directly raises input costs and disrupts production at F/GM (UAW) "
                "and delivery capacity at UPS (Teamsters). Airlines (UAL/AAL) face "
                "pilot/cabin crew work-to-rule slowdowns in sympathy actions."
            ),
            "source": (
                "SCOTUSblog / supremecourt.gov (public); BLS Work Stoppages Program; "
                "prices via yfinance (auto_adjust=True)"
            ),
            "tickers": basket,
            "primary_label": primary_label,
            "n_events_all": n_all,
            "n_events_scotus_only": n_scotus,
            "event_win_rate": round(win_rate, 4) if win_rate is not None else None,
            "avg_short_event_return": round(avg_ret, 4) if avg_ret is not None else None,
            "events_all": log_all,
            "events_scotus_only": log_scotus,
            "caveats": (
                "Extremely small sample (3-5 events). "
                "AAL is financially distressed and may be driven by unrelated factors. "
                "SCOTUS/labor-law causal chain to short equity is indirect; "
                "initial market reaction to Loper Bright was minimal for these names. "
                "Boeing IAM strike included as corroborating event though BA not in basket. "
                "Metrics are on held-days only and not meaningful at n<=5 events."
            ),
            "stop_loss_pct": STOP_LOSS_PCT,
        },
        pnl=held_pnl,
    )

    print(f"\nDone: {sid}")
    print(f"  n_events: {n_all} (scotus-only: {n_scotus}), win_rate: {win_rate}")
    if m.get("sharpe") is not None:
        print(
            f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
            f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
        )
    if m.get("oos_sharpe") is not None:
        print(f"  OOS Sharpe: {m.get('oos_sharpe'):.2f}")


if __name__ == "__main__":
    main()
