"""PL755 — SAVE Vacatur / IDR Restart -> Long SLM/NAVI

On dates when DOE announces SAVE-plan vacatur, IDR repayment restart,
or forbearance-end events, go long equal-weight SLM + NAVI for 45 trading
days. These events signal tighter loan repayment conditions, improving
the revenue outlook for private student loan servicers.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)

# ---------------------------------------------------------------------------
# Hand-coded DOE/Court student loan policy events
# Sources: DOE press releases, court filings, media coverage
# ---------------------------------------------------------------------------
POLICY_EVENTS = [
    # (date_str, event_type, description)
    ("2023-06-30", "forbearance_end_reversal",
     "Supreme Court strikes down Biden broad forgiveness; restart uncertainty"),
    ("2023-08-16", "repayment_restart_announced",
     "Biden announces new SAVE plan as replacement but repayments to restart Oct 2023"),
    ("2023-10-01", "forbearance_end_official",
     "COVID forbearance officially ends; payments due again (40M borrowers)"),
    ("2024-02-22", "idr_restart_announced",
     "DOE restarts IDR recertification timeline; affects servicer fee income"),
    ("2024-06-26", "save_vacatur_court",
     "8th Circuit blocks SAVE plan; DOE puts enrolled borrowers in forbearance"),
    ("2024-07-18", "save_blocked_extended",
     "Court extends SAVE block; longer uncertainty for 8M enrolled borrowers"),
    ("2024-08-09", "doe_responds_forbearance",
     "DOE announces interest-free forbearance for SAVE borrowers; servicer impact negative"),
    ("2025-01-20", "trump_executive_action",
     "Trump EO rescinding SAVE; immediate restart of standard repayment required"),
    ("2025-03-05", "doe_announces_idr_reset",
     "DOE confirms IDR payments to resume; servicers to receive delinquency fees"),
]

HOLD_DAYS = 45


def main():
    sid = "PL755_save_idr_restart_long_slm_navi"

    try:
        px = load_prices(["SLM", "NAVI", "SPY"], start="2023-01-01")
    except Exception as e:
        return mark_failed(sid, f"price data load: {e}")

    for t in ["SLM", "NAVI", "SPY"]:
        if t not in px.columns:
            return mark_failed(sid, f"missing ticker: {t}")

    px = px.sort_index().ffill(limit=3)

    # Filter to repayment-restart / payment-resumption events (positive for SLM/NAVI)
    # Events where borrowers must resume payments = better for servicers
    # Exclude forbearance-extension events (negative for servicers)
    POSITIVE_TYPES = {
        "forbearance_end_reversal",
        "repayment_restart_announced",
        "forbearance_end_official",
        "idr_restart_announced",
        "trump_executive_action",
        "doe_announces_idr_reset",
    }

    events = []
    last_trigger = pd.Timestamp("1900-01-01")
    cooldown_days = 30

    for date_str, event_type, description in POLICY_EVENTS:
        if event_type not in POSITIVE_TYPES:
            continue
        dt = pd.Timestamp(date_str)
        if (dt - last_trigger).days < cooldown_days:
            continue
        # Entry: next trading session after announcement
        future = px.index[px.index > dt]
        if len(future) == 0:
            continue
        entry_dt = future[0]
        events.append({
            "announcement_date": date_str,
            "event_type": event_type,
            "description": description,
            "entry_date": str(entry_dt.date()),
        })
        last_trigger = dt

    print(f"Student loan policy events (repayment-positive): {len(events)}")
    if not events:
        return mark_failed(sid, "no qualifying DOE student loan policy events")

    # Build daily PnL
    ret = daily_returns(px)
    spy_r = ret["SPY"]
    slm_r = ret["SLM"]
    navi_r = ret["NAVI"]
    pnl = pd.Series(0.0, index=ret.index)

    event_log = []
    for ev in events:
        entry_dt = pd.Timestamp(ev["entry_date"])
        if entry_dt not in ret.index:
            future = ret.index[ret.index >= entry_dt]
            if len(future) == 0:
                continue
            entry_dt = future[0]

        ep = ret.index.get_loc(entry_dt)
        ex = min(ep + HOLD_DAYS, len(ret))

        slm_sl = slm_r.iloc[ep:ex]
        navi_sl = navi_r.iloc[ep:ex]
        spy_sl = spy_r.iloc[ep:ex]

        basket_sl = (slm_sl.values[:ex-ep] + navi_sl.values[:ex-ep]) / 2.0

        overlap = (pnl.iloc[ep:ex] != 0).sum()
        if overlap > HOLD_DAYS * 0.5:
            continue

        pnl.iloc[ep:ex] = pnl.iloc[ep:ex] + basket_sl

        slm_ret = float((1 + slm_sl).prod() - 1)
        navi_ret = float((1 + navi_sl).prod() - 1)
        basket_ret = (slm_ret + navi_ret) / 2.0
        spy_ret = float((1 + spy_sl).prod() - 1)

        event_log.append({
            **ev,
            "exit_date": str(ret.index[ex - 1].date()),
            "slm_return": round(slm_ret, 4),
            "navi_return": round(navi_ret, 4),
            "basket_return": round(basket_ret, 4),
            "spy_return": round(spy_ret, 4),
            "excess_vs_spy": round(basket_ret - spy_ret, 4),
        })

    n_valid = len(event_log)
    print(f"Valid events: {n_valid}")
    if n_valid == 0:
        return mark_failed(sid, "no valid events")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="DOE SAVE/IDR Restart Long SLM/NAVI")

    basket_rets = [e["basket_return"] for e in event_log]
    save_result(sid, m, extra={
        "rule": (
            f"On DOE/court student loan repayment-restart or IDR-restart events, "
            f"long equal-weight SLM + NAVI for {HOLD_DAYS} trading days from announcement."
        ),
        "mechanism": (
            "Payment resumption events directly increase servicer processing volume "
            "and associated fees for SLM (Sallie Mae, private) and NAVI (Navient). "
            "IDR restarts also reduce regulatory uncertainty. Markets price-in higher "
            "expected fee income and lower credit risk for servicers."
        ),
        "source": (
            "Hand-coded DOE/court student loan policy events (DOE press releases, "
            "8th Circuit rulings 2023-2025); yfinance SLM, NAVI, SPY."
        ),
        "caveats": (
            "Short backtest period (2023-2025 only; SAVE plan created 2023). "
            "Few events limit statistical power. SLM is primarily private loans "
            "while NAVI services federal loans — different exposure to DOE policy. "
            "Political risk is binary and hard to predict."
        ),
        "n_events": n_valid,
        "avg_basket_return": round(float(np.mean(basket_rets)), 4),
        "win_rate": round(float(np.mean([r > 0 for r in basket_rets])), 4),
        "events": event_log,
    })

    print(f"Done: {n_valid} events")
    if "error" not in m:
        print(
            f"  Sharpe: {m.get('sharpe', float('nan')):.2f}  "
            f"CAGR: {m.get('cagr', float('nan'))*100:.2f}%  "
            f"MaxDD: {m.get('max_dd', float('nan'))*100:.2f}%  "
            f"t-stat: {m.get('t_stat', float('nan')):.2f}"
        )


if __name__ == "__main__":
    main()
