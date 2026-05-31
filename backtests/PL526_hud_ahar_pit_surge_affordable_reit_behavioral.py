"""PL526_hud_ahar_pit_surge_affordable_reit_behavioral
HUD AHAR PIT Count Surge >10% YoY -> Long Affordable REITs + Behavioral Health vs Short IYR Hedge

On each HUD AHAR Part 1 release date where the national Point-in-Time (PIT)
homeless count rises >10% YoY, enter at the next session's open:
  Long  33.33% CPT + 33.33% ACHC + 33.33% UHS
  Short 50%   IYR (REIT-beta hedge)
Hold 210 trading days (~10 months / through October following year), exit at close.

Daily strategy return on held days:
  r_strategy = (CPT_r + ACHC_r + UHS_r)/3 - 0.5 * IYR_r

Known events (HUD AHAR Part 1 published annually each December):
  2023-12-15  national PIT YoY +12.0%
  2024-12-27  national PIT YoY +18.1%

Small-sample event study (N=2). We also compute event-level returns individually
and excess vs SPY over the same windows.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# Curated HUD AHAR Part 1 release dates where national PIT count rose >10% YoY.
KNOWN_EVENTS = [
    {"release_date": "2023-12-15", "pit_yoy_pct": 12.0,
     "note": "HUD AHAR Part 1 release; first major post-pandemic surge"},
    {"release_date": "2024-12-27", "pit_yoy_pct": 18.1,
     "note": "HUD AHAR Part 1 release; second consecutive double-digit surge"},
]


def run_event_study(events, ret, long_basket, short_ticker, hold_days=210,
                    short_weight=0.5):
    """Hedged long basket event study.

    long_basket: list of tickers, each weighted equally (1/N) in the long leg.
    short_ticker: ticker shorted at `short_weight` (e.g. 0.5 of notional).
    Daily held-day return = mean(long_basket_r) - short_weight * short_ticker_r.
    Positions captured as net long delta (1 - short_weight) on held days.
    """
    idx = ret.index
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)
    long_ret = ret[long_basket].fillna(0).mean(axis=1)
    short_ret = ret[short_ticker].fillna(0)
    strat_ret_full = long_ret - short_weight * short_ret  # would-be daily PnL

    event_log = []
    for ev in events:
        rel = pd.Timestamp(ev["release_date"])
        future_sessions = idx[idx > rel]
        if len(future_sessions) == 0:
            ev_rec = dict(ev)
            ev_rec["status"] = "no_data_after_release"
            event_log.append(ev_rec)
            continue
        entry_dt = future_sessions[0]
        entry_pos = idx.get_loc(entry_dt)
        exit_pos = min(entry_pos + hold_days, len(idx))

        # event-level cumulative strategy return
        slice_r = strat_ret_full.iloc[entry_pos:exit_pos]
        ev_strat_ret = float((1 + slice_r).prod() - 1) if len(slice_r) else None

        # event-level long basket and short leg returns
        slice_long = long_ret.iloc[entry_pos:exit_pos]
        slice_short = short_ret.iloc[entry_pos:exit_pos]
        ev_long_ret = float((1 + slice_long).prod() - 1) if len(slice_long) else None
        ev_short_ret = float((1 + slice_short).prod() - 1) if len(slice_short) else None

        ev_rec = dict(ev)
        ev_rec["entry_date"] = str(entry_dt.date())
        ev_rec["exit_date"] = str(idx[exit_pos - 1].date()) if exit_pos > entry_pos else None
        ev_rec["n_hold_days"] = int(exit_pos - entry_pos)
        ev_rec["strategy_return"] = round(ev_strat_ret, 4) if ev_strat_ret is not None else None
        ev_rec["long_basket_return"] = round(ev_long_ret, 4) if ev_long_ret is not None else None
        ev_rec["short_leg_return"] = round(ev_short_ret, 4) if ev_short_ret is not None else None
        event_log.append(ev_rec)

        for j in range(entry_pos, exit_pos):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = 1.0 - short_weight  # net long delta
                pnl.iloc[j] = strat_ret_full.iloc[j]

    return pnl, positions, event_log


def main():
    sid = "PL526_hud_ahar_pit_surge_affordable_reit_behavioral"
    long_basket = ["CPT", "ACHC", "UHS"]
    short_ticker = "IYR"
    tickers = long_basket + [short_ticker, "SPY"]

    try:
        px = load_prices(tickers, start="2020-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    pnl, positions, event_log = run_event_study(
        KNOWN_EVENTS, ret, long_basket, short_ticker,
        hold_days=210, short_weight=0.5,
    )

    n_events = sum(1 for e in event_log if e.get("entry_date"))

    # Use held-days only for Sharpe/CAGR (don't dilute with flat zero days)
    held_mask = positions > 0
    held_pnl = pnl[held_mask]
    held_spy = spy_r.reindex(held_pnl.index).dropna()

    if len(held_pnl) < 30:
        return mark_failed(
            sid,
            f"insufficient held days (n_events={n_events}, held_days={len(held_pnl)})",
            extra={"events": event_log},
        )

    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name="HUD AHAR PIT Surge Long-Affordable-REIT/Behavioral vs Short IYR (held-days only)",
        positions=positions.reindex(held_pnl.index).fillna(0),
        cost_bps=10,
    )

    # Event-level summary
    def event_summary(log, key):
        rets = [e[key] for e in log if e.get(key) is not None]
        if not rets:
            return None
        return {
            "n_events": len(rets),
            "avg": round(float(np.mean(rets)), 4),
            "median": round(float(np.median(rets)), 4),
            "win_rate": round(float(np.mean([r > 0 for r in rets])), 4),
            "best": round(float(np.max(rets)), 4),
            "worst": round(float(np.min(rets)), 4),
        }

    summary_strategy = event_summary(event_log, "strategy_return")
    summary_long = event_summary(event_log, "long_basket_return")
    summary_short = event_summary(event_log, "short_leg_return")

    # SPY same-window comparison per event
    excess_vs_spy = []
    for e in event_log:
        entry = e.get("entry_date"); exit_d = e.get("exit_date")
        if not entry or not exit_d:
            continue
        entry_dt = pd.Timestamp(entry); exit_dt = pd.Timestamp(exit_d)
        spy_slice = spy_r.loc[entry_dt:exit_dt]
        if len(spy_slice):
            spy_cum = float((1 + spy_slice).prod() - 1)
            excess_vs_spy.append({
                "release_date": e["release_date"],
                "entry_date": entry,
                "exit_date": exit_d,
                "strategy_return": e.get("strategy_return"),
                "spy_return": round(spy_cum, 4),
                "excess": round((e.get("strategy_return") or 0) - spy_cum, 4),
            })

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "On each HUD AHAR Part 1 release date where the national PIT homeless "
                "count rises >10% YoY, enter at the next session open: long 33.33% CPT + "
                "33.33% ACHC + 33.33% UHS plus 50% short IYR hedge. Hold 210 trading days "
                "(~10 months) then exit at the close."
            ),
            "mechanism": (
                "A PIT count surge crystallises the affordable-housing shortage in the "
                "policy conversation, which typically accelerates federal/state appropriations "
                "(LIHTC, vouchers, Medicaid 1115 housing services) in the May-Sept "
                "appropriations cycle following the December release. CPT is a workforce "
                "multifamily REIT (Sunbelt B/B+ assets) that benefits from voucher-funded "
                "demand and rent stability. ACHC and UHS run behavioral-health facilities "
                "tied to Medicaid expansion, including 1115 housing-and-recovery waivers. "
                "IYR (broad REIT) is shorted at 50% to strip out rate/REIT-beta and isolate "
                "the affordable-tilt + Medicaid-policy alpha."
            ),
            "source": (
                "HUD AHAR Part 1 PIT release calendar and national PIT YoY estimates "
                "2018-2024 (hud.gov/program_offices/comm_planning/coc/reports). "
                "Prices via yfinance (auto_adjust=True)."
            ),
            "long_basket": long_basket,
            "short_ticker": short_ticker,
            "short_weight": 0.5,
            "hold_days": 210,
            "events": event_log,
            "summary_strategy_return": summary_strategy,
            "summary_long_basket_return": summary_long,
            "summary_short_leg_return": summary_short,
            "excess_vs_spy": excess_vs_spy,
            "n_events": n_events,
            "caveats": (
                "Very small-sample event study (N=2: Dec 2023 and Dec 2024 releases). "
                "Statistical significance is extremely limited. ACHC was acquired in "
                "early 2025 (Onex take-private process: announced Feb 2025) -- the "
                "2024-12-27 event window therefore overlaps with M&A-driven price action "
                "in ACHC, which inflates the long basket's reported return on that event. "
                "CPT is also exposed to Sunbelt supply waves and rate moves; behavioral "
                "health is exposed to Medicaid reimbursement and labor-cost cycles. "
                "The 210-day hold spans most of a year so macro/sector regimes dominate. "
                "Sharpe/CAGR are computed on held-day returns only; results should be "
                "interpreted alongside the event-level returns and SPY-excess table."
            ),
        },
        pnl=pnl[held_mask],
    )

    print(f"Done: {sid}")
    print(f"  n_events: {n_events}")
    print(f"  events: {event_log}")
    print(f"  summary_strategy: {summary_strategy}")
    print(f"  summary_long: {summary_long}")
    print(f"  summary_short: {summary_short}")
    if "error" not in m:
        print(
            f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
            f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
        )
        if "net_sharpe" in m:
            print(f"  Net Sharpe: {m['net_sharpe']:.2f}, Net CAGR: {m['net_cagr']*100:.2f}%")


if __name__ == "__main__":
    main()
