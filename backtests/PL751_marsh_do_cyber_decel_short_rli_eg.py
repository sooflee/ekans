"""PL751_marsh_do_cyber_decel_short_rli_eg
Marsh D&O+Cyber Decel -> Short RLI/EG (Counter)

When Marsh Commercial Insurance Pricing Index quarterly D&O+Cyber composite
decelerates by >=5pp QoQ, short equal-weight RLI + EG for 45 trading days.
Counter-signal to long_specialty_insurer.

Hand-coded Marsh quarterly insurance pricing data 2018-2025.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# Hand-coded Marsh Commercial Insurance Pricing Index: D&O + Cyber composite
# quarterly YoY growth rates (approximate, based on Marsh quarterly reports)
# Source: Marsh Global Insurance Market Index quarterly reports 2018-2025
# Format: (quarter_end_date, do_yoy_pct, cyber_yoy_pct)
# D&O = Directors & Officers liability; Cyber = Cyber liability lines
MARSH_QUARTERLY = [
    # (quarter_end, D&O YoY%, Cyber YoY%)
    ("2018-03-31",  5.0,  12.0),
    ("2018-06-30",  6.0,  14.0),
    ("2018-09-30",  7.0,  15.0),
    ("2018-12-31",  8.0,  17.0),

    ("2019-03-31", 10.0,  20.0),
    ("2019-06-30", 11.0,  22.0),
    ("2019-09-30", 12.0,  25.0),
    ("2019-12-31", 14.0,  27.0),

    ("2020-03-31", 18.0,  30.0),
    ("2020-06-30", 20.0,  35.0),
    ("2020-09-30", 25.0,  40.0),
    ("2020-12-31", 30.0,  50.0),

    ("2021-03-31", 40.0,  70.0),
    ("2021-06-30", 45.0,  80.0),
    ("2021-09-30", 42.0,  85.0),
    ("2021-12-31", 40.0,  82.0),

    ("2022-03-31", 35.0,  75.0),
    ("2022-06-30", 25.0,  65.0),   # decel: D&O -10pp, Cyber -10pp (composite -10pp)
    ("2022-09-30", 18.0,  50.0),   # decel: D&O -7pp, Cyber -15pp
    ("2022-12-31", 10.0,  35.0),   # decel: D&O -8pp, Cyber -15pp

    ("2023-03-31",  5.0,  20.0),   # decel: D&O -5pp, Cyber -15pp
    ("2023-06-30",  2.0,  10.0),   # decel: D&O -3pp, Cyber -10pp
    ("2023-09-30", -2.0,   5.0),   # D&O now negative; Cyber still positive
    ("2023-12-31", -5.0,   2.0),

    ("2024-03-31", -7.0,   0.0),   # D&O decel continues (known event Q1 2024)
    ("2024-06-30", -5.0,   2.0),
    ("2024-09-30", -3.0,   3.0),
    ("2024-12-31", -2.0,   5.0),

    ("2025-03-31",  0.0,   5.0),
]

DECEL_THRESHOLD_PP = 5.0  # composite decel >= 5pp QoQ triggers signal


def compute_composite_yoy(do_yoy, cyber_yoy):
    """Equal-weight composite of D&O and Cyber YoY rates."""
    return (do_yoy + cyber_yoy) / 2.0


def main():
    sid = "PL751_marsh_do_cyber_decel_short_rli_eg"
    tickers = ["RLI", "EG", "SPY"]

    try:
        px = load_prices(tickers, start="2018-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)

    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    # Compute composite YoY and QoQ change
    composites = []
    for q_end, do_yoy, cyber_yoy in MARSH_QUARTERLY:
        comp = compute_composite_yoy(do_yoy, cyber_yoy)
        composites.append((pd.Timestamp(q_end), comp))

    # Find deceleration quarters: composite drops by >= DECEL_THRESHOLD_PP vs prior quarter
    trigger_dates = []
    for i in range(1, len(composites)):
        ts_curr, comp_curr = composites[i]
        ts_prev, comp_prev = composites[i - 1]
        decel = comp_prev - comp_curr  # positive = deceleration
        if decel >= DECEL_THRESHOLD_PP:
            # Marsh reports are usually released ~6 weeks after quarter end
            # Approximate release date: 6 weeks after quarter-end
            release_date = ts_curr + pd.Timedelta(weeks=6)
            trigger_dates.append((release_date, ts_curr.strftime("%YQ%q"), decel))

    if not trigger_dates:
        return mark_failed(sid, "no deceleration triggers found")

    hold_days = 45
    idx_list = list(ret.index)
    n = len(idx_list)

    daily_pnl = pd.Series(0.0, index=ret.index)
    positions = pd.Series(0.0, index=ret.index)

    basket_legs = [t for t in ["RLI", "EG"] if t in ret.columns]

    events = []
    active_until_idx = -1

    for release_date, quarter_label, decel_pp in trigger_dates:
        # Entry: first trading day strictly after release date
        entry_candidates = [d for d in idx_list if d > release_date]
        if not entry_candidates:
            continue
        entry_date = entry_candidates[0]
        entry_idx = idx_list.index(entry_date)

        # No overlap
        if entry_idx <= active_until_idx:
            continue

        end_idx = min(entry_idx + hold_days, n)
        active_until_idx = end_idx - 1

        event_pnl = []
        for j in range(entry_idx, end_idx):
            d = idx_list[j]
            leg_rets = []
            for t in basket_legs:
                r = ret[t].get(d, np.nan)
                if not np.isnan(r):
                    leg_rets.append(r)
            if leg_rets:
                basket_r = np.mean(leg_rets)
                short_r = -basket_r  # short
                daily_pnl.iloc[j] = short_r
                positions.iloc[j] = -1.0
                event_pnl.append(short_r)

        if event_pnl:
            cum_event = float((1 + pd.Series(event_pnl)).prod() - 1)
            events.append({
                "quarter": quarter_label,
                "decel_pp": round(decel_pp, 1),
                "release_date": str(release_date.date()),
                "entry_date": str(entry_date.date()),
                "exit_date": str(idx_list[min(end_idx - 1, n - 1)].date()),
                "event_return": round(cum_event, 4),
            })

    pnl = daily_pnl.dropna()
    n_events = len(events)

    if n_events < 3:
        return mark_failed(sid, f"too few events: {n_events}")

    m = compute_metrics(
        pnl,
        benchmark=spy_r,
        name="Marsh D&O+Cyber Decel Short RLI/EG",
        positions=positions.reindex(pnl.index).fillna(0),
        cost_bps=10,
    )
    m["n_events"] = n_events

    ev_returns = [e["event_return"] for e in events]
    win_rate = float(np.mean([r > 0 for r in ev_returns])) if ev_returns else None
    avg_event = float(np.mean(ev_returns)) if ev_returns else None

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "When Marsh quarterly D&O+Cyber composite YoY rate decelerates by >=5pp "
                "QoQ, short equal-weight RLI + EG for 45 trading days after report release."
            ),
            "mechanism": (
                "Specialty insurers RLI and EG earn premium from D&O and cyber lines. "
                "When pricing power decelerates sharply, forward earned premium and "
                "combined ratio guidance deteriorate. Markets typically reprice insurers "
                "on the 2-3 quarters following peak-rate confirmation, creating a short "
                "window for counter-directional positioning."
            ),
            "source": "Marsh Global Insurance Market Index quarterly reports (hand-coded 2018-2025)",
            "tickers": basket_legs,
            "n_events": n_events,
            "event_win_rate": round(win_rate, 4) if win_rate is not None else None,
            "avg_event_return": round(avg_event, 4) if avg_event is not None else None,
            "events": events,
            "caveats": (
                "Hand-coded Marsh data introduces approximation; actual report timing "
                "varies. RLI and EG have diversified revenue beyond D&O/cyber; the "
                "pricing signal may be diluted. Short positions in quality insurance "
                "names carry high opportunity cost in rising-rate environments."
            ),
        },
        pnl=pnl,
    )

    print(f"Done: {sid}")
    print(f"  n_events={n_events}, win_rate={win_rate}, avg_event_return={avg_event}")
    print(
        f"  Sharpe={m.get('sharpe', 0):.2f}  CAGR={m.get('cagr', 0)*100:.2f}%  "
        f"MaxDD={m.get('max_dd', 0)*100:.2f}%  t-stat={m.get('t_stat', 0):.2f}"
    )
    if "oos_sharpe" in m:
        print(f"  OOS Sharpe={m['oos_sharpe']:.2f}")


if __name__ == "__main__":
    main()
