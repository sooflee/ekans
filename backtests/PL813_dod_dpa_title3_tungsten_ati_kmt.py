"""PL813_dod_dpa_title3_tungsten_ati_kmt
DoD DPA Title III Tungsten/Tantalum Award Cluster -> Long ATI / KMT Specialty Alloys

When DoD publishes >=2 DPA Title III awards or obligations for tungsten or tantalum
domestic processing/refining within a 30-calendar-day window, go long equal-weight
ATI + KMT at close. Hold for 40 trading days. Stop-loss at -15%.
Benchmark: SPY.

Known DPA Title III award cluster dates (approximate, from MCEIP archive):
  - 2020-01-15: FY2020 W/Ta awards cluster
  - 2021-09-01: FY2022 tungsten carbide awards
  - 2022-09-15: FY2023 tungsten refining grants
  - 2023-07-01: FY2024 Ta/W awards
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


EVENTS = [
    {
        "announcement_date": "2020-01-15",
        "description": "FY2020 DPA Title III W/Ta award cluster",
        "hold_days": 40,
        "stop_loss": 0.15,
    },
    {
        "announcement_date": "2021-09-01",
        "description": "FY2022 tungsten carbide DPA Title III awards",
        "hold_days": 40,
        "stop_loss": 0.15,
    },
    {
        "announcement_date": "2022-09-15",
        "description": "FY2023 MCEIP tungsten refining grants",
        "hold_days": 40,
        "stop_loss": 0.15,
    },
    {
        "announcement_date": "2023-07-01",
        "description": "FY2024 Ta/W DPA Title III awards cluster",
        "hold_days": 40,
        "stop_loss": 0.15,
    },
]


def run_event_study(events, ret, basket, spy_r):
    """Long equal-weight basket from entry through hold or stop-loss."""
    idx = ret.index
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)

    basket_ret = ret[basket].fillna(0).mean(axis=1)

    event_log = []
    for ev in events:
        ann_dt = pd.Timestamp(ev["announcement_date"])
        hold_days = ev["hold_days"]
        stop_loss = ev["stop_loss"]

        # entry: first trading day on or after announcement
        future = idx[idx >= ann_dt]
        if len(future) == 0:
            ev_record = dict(ev)
            ev_record["status"] = "no_data"
            event_log.append(ev_record)
            continue

        entry_dt = future[0]
        entry_pos = idx.get_loc(entry_dt)
        max_exit_pos = min(entry_pos + hold_days, len(idx))

        # Walk forward checking stop-loss
        cumulative = 1.0
        actual_exit_pos = max_exit_pos
        exit_reason = "scheduled_40d"

        for j in range(entry_pos, max_exit_pos):
            day_basket_ret = float(basket_ret.iloc[j])
            cumulative *= (1 + day_basket_ret)
            basket_dd = cumulative - 1.0
            # Stop-loss: basket is DOWN more than stop_loss from entry
            if basket_dd < -stop_loss:
                exit_reason = "stop_loss"
                actual_exit_pos = j + 1
                break

        # Record PnL (long basket)
        for j in range(entry_pos, actual_exit_pos):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = 1.0
                pnl.iloc[j] = float(basket_ret.iloc[j])

        # Event summary
        basket_cum = float((1 + basket_ret.iloc[entry_pos:actual_exit_pos]).prod() - 1)
        spy_cum = float((1 + spy_r.reindex(idx[entry_pos:actual_exit_pos]).fillna(0)).prod() - 1)
        exit_dt = idx[actual_exit_pos - 1] if actual_exit_pos > entry_pos else entry_dt

        ev_record = dict(ev)
        ev_record["status"] = "traded"
        ev_record["entry_date"] = str(entry_dt.date())
        ev_record["exit_date"] = str(exit_dt.date())
        ev_record["exit_reason"] = exit_reason
        ev_record["n_hold_days"] = int(actual_exit_pos - entry_pos)
        ev_record["basket_return"] = round(basket_cum, 4)
        ev_record["spy_return"] = round(spy_cum, 4)
        ev_record["excess_return"] = round(basket_cum - spy_cum, 4)

        for name in basket:
            if name in ret.columns:
                slice_r = ret[name].iloc[entry_pos:actual_exit_pos]
                cum_r = float((1 + slice_r.fillna(0)).prod() - 1)
                ev_record[f"{name.lower()}_return"] = round(cum_r, 4)

        event_log.append(ev_record)

    return pnl, positions, event_log


def main():
    sid = "PL813_dod_dpa_title3_tungsten_ati_kmt"
    basket = ["ATI", "KMT"]
    tickers = basket + ["XLI", "SPY"]

    try:
        px = load_prices(tickers, start="2019-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    pnl, positions, event_log = run_event_study(EVENTS, ret, basket, spy_r)

    n_traded = sum(1 for e in event_log if e.get("status") == "traded")
    print(f"Events traded: {n_traded} / {len(EVENTS)}")
    for e in event_log:
        if e.get("status") == "traded":
            print(
                f"  {e['announcement_date']} -> entry {e['entry_date']}, "
                f"basket={e.get('basket_return'):.4f}, excess={e.get('excess_return'):.4f}, "
                f"reason={e.get('exit_reason')}"
            )

    if n_traded < 3:
        return mark_failed(
            sid,
            f"insufficient traded events: {n_traded} (need >=3)",
            extra={"event_log": event_log},
        )

    held_pnl = pnl[positions != 0]
    held_spy = spy_r.reindex(held_pnl.index).fillna(0)

    if len(held_pnl) < 20:
        return mark_failed(
            sid,
            f"insufficient held days: {len(held_pnl)}",
            extra={"event_log": event_log},
        )

    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name="DoD DPA Title III W/Ta Cluster Long ATI+KMT (held-days)",
        positions=positions.reindex(held_pnl.index).fillna(0),
        cost_bps=10,
    )

    event_returns = [e.get("basket_return") for e in event_log if e.get("basket_return") is not None]
    win_rate = float(np.mean([r > 0 for r in event_returns])) if event_returns else None
    avg_ret = float(np.mean(event_returns)) if event_returns else None
    avg_excess = float(np.mean([e.get("excess_return", 0) for e in event_log if e.get("excess_return") is not None]))

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "When DoD publishes >=2 DPA Title III awards or obligations for tungsten "
                "or tantalum domestic processing/refining within a 30-calendar-day window, "
                "enter long equal-weight ATI + KMT at close. Hold 40 trading days. "
                "Stop-loss: exit if either combined position falls >15% from entry on close."
            ),
            "mechanism": (
                "DoD DPA Title III awards directly fund domestic rare/critical mineral "
                "processing capacity. ATI (specialty alloys, tungsten carbide) and Kennametal "
                "(tungsten cutting tools, tantalum coatings) are direct beneficiaries. "
                "Award clusters signal increased US government commitment and revenue "
                "flow to these producers, often ahead of defense procurement upticks."
            ),
            "source": (
                "DoD MCEIP/DIU press releases; defense.gov/MCEIP archive; "
                "DPA Title III Annual Obligations Reports; prices via yfinance"
            ),
            "tickers": basket,
            "n_events_traded": n_traded,
            "event_win_rate": round(win_rate, 4) if win_rate is not None else None,
            "avg_event_return": round(avg_ret, 4) if avg_ret is not None else None,
            "avg_excess_vs_spy": round(avg_excess, 4),
            "event_log": event_log,
            "caveats": (
                "Very small sample (n=4 events, approximate dates). "
                "Event dates are hand-curated approximations from MCEIP archive; "
                "true press release dates may differ by days to weeks. "
                "Metrics are on held-days only. Statistical confidence is very limited. "
                "ATI underwent significant corporate restructuring (spun off Allegheny Technologies "
                "Specialty Alloys) — price history reflects legacy entity pre-2021. "
                "KMT is ~50% industrial cutting tools; tungsten raw material cost pass-through "
                "may lag award announcements by 6-12 months."
            ),
            "stop_loss_pct": 0.15,
        },
        pnl=held_pnl,
    )

    print(f"\nDone: {sid}")
    print(f"  n_events: {n_traded}, win_rate: {win_rate}, avg_excess_vs_spy: {avg_excess:.4f}")
    if m.get("sharpe") is not None:
        print(
            f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
            f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
        )
    if m.get("oos_sharpe") is not None:
        print(f"  OOS Sharpe: {m.get('oos_sharpe'):.2f}")


if __name__ == "__main__":
    main()
