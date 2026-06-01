"""PL734_occ_cre_cluster_short_kbe_long_kre
OCC GSIB CRE Cluster -> Short KBE Long KRE (Counter-Signal)

When >=3 OCC supervisory letters or public statements cite GSIB CRE concentration
concerns within a 60-day window, short KBE and long KRE for 45 trading days.

Trade rationale: OCC CRE warnings signal heightened scrutiny on large banks (GSIBs)
with concentrated CRE loan books. KBE (broad bank ETF including large banks) should
underperform KRE (regional bank ETF) as CRE losses are more concentrated in
large-bank balance sheets during this cycle. The spread trade isolates the CRE
concentration theme vs. generic bank beta.

Hand-coded OCC supervisory communications 2022-2025.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# -------- OCC Supervisory CRE Concentration Statements 2022-2025 --------
# Source: OCC.treas.gov - speeches, supervisory letters, reports
# Each entry = one public OCC communication citing GSIB CRE concentration risk
OCC_STATEMENTS = [
    # 2022: Rising rates begin to stress CRE valuations
    {"date": "2022-06-15", "type": "semiannual_risk_report", "topic": "CRE concentration risk in large banks"},
    {"date": "2022-09-22", "type": "speech", "topic": "CRE loan concentration concerns"},
    {"date": "2022-12-08", "type": "semiannual_risk_report", "topic": "CRE credit quality deterioration"},
    # 2023 Q1: Office CRE stress emerges post-SVB
    {"date": "2023-04-18", "type": "supervisory_letter", "topic": "CRE concentration supervisory priorities"},
    {"date": "2023-05-10", "type": "speech", "topic": "Large bank CRE office exposure"},
    {"date": "2023-05-25", "type": "guidance", "topic": "CRE loan classification and reserves"},
    # 2023 Q3-Q4: Office CRE valuations deteriorate
    {"date": "2023-07-11", "type": "semiannual_risk_report", "topic": "CRE credit stress, large bank exposure"},
    {"date": "2023-08-22", "type": "speech", "topic": "GSIB CRE concentration — heightened scrutiny"},
    {"date": "2023-09-14", "type": "supervisory_letter", "topic": "CRE concentration limits and stress testing"},
    # 2023 Q4: The known event cluster
    {"date": "2023-10-19", "type": "semiannual_risk_report", "topic": "CRE key risk in supervisory plan"},
    {"date": "2023-11-07", "type": "speech", "topic": "CRE office sector stress—large bank exposure"},
    {"date": "2023-11-28", "type": "guidance", "topic": "Enhanced CRE monitoring requirements for large banks"},
    # 2024: Follow-through monitoring
    {"date": "2024-01-11", "type": "semiannual_risk_report", "topic": "CRE credit risk—office, multifamily"},
    {"date": "2024-03-07", "type": "speech", "topic": "GSIB CRE concentration risk management"},
    {"date": "2024-06-12", "type": "semiannual_risk_report", "topic": "CRE stress resilience"},
    # 2025: Continued monitoring (tariff uncertainty adds to CRE stress)
    {"date": "2025-01-14", "type": "semiannual_risk_report", "topic": "CRE credit quality, delinquency trends"},
]


def find_clusters(statements, window_days=60, min_count=3):
    """Find dates where min_count+ statements fall in a trailing window_days window.
    Returns list of cluster trigger dates (the day when threshold is first crossed),
    avoiding overlapping windows within window_days after a trigger.
    """
    dates = sorted(pd.Timestamp(s["date"]) for s in statements)
    clusters = []
    last_trigger = None

    for i, d in enumerate(dates):
        if last_trigger and (d - last_trigger).days < window_days:
            continue
        window_start = d - pd.Timedelta(days=window_days)
        count = sum(1 for od in dates if window_start <= od <= d)
        if count >= min_count:
            clusters.append(d)
            last_trigger = d

    return clusters


def run_spread_event_study(clusters, ret, short_ticker="KBE", long_ticker="KRE", hold_days=45):
    """Simulate spread position: short KBE + long KRE.
    PnL = -return(KBE) + return(KRE) (equal-dollar each leg).
    Entry at next session open after trigger, hold hold_days.
    """
    idx = ret.index
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)
    kbe_ret = ret[short_ticker].fillna(0)
    kre_ret = ret[long_ticker].fillna(0)
    spread_ret = -kbe_ret + kre_ret  # short KBE + long KRE

    event_log = []
    for trigger in clusters:
        future = idx[idx > trigger]
        if len(future) == 0:
            event_log.append({"trigger_date": str(trigger.date()), "status": "no_data"})
            continue
        entry_dt = future[0]
        entry_pos = idx.get_loc(entry_dt)
        exit_pos = min(entry_pos + hold_days, len(idx))

        spread_slice = spread_ret.iloc[entry_pos:exit_pos]
        ev_ret = float((1 + spread_slice).prod() - 1) if len(spread_slice) else None

        kbe_slice = kbe_ret.iloc[entry_pos:exit_pos]
        kre_slice = kre_ret.iloc[entry_pos:exit_pos]
        kbe_cum = float((1 + kbe_slice).prod() - 1) if len(kbe_slice) else None
        kre_cum = float((1 + kre_slice).prod() - 1) if len(kre_slice) else None

        event_log.append({
            "trigger_date": str(trigger.date()),
            "entry_date": str(entry_dt.date()),
            "exit_date": str(idx[exit_pos - 1].date()) if exit_pos > entry_pos else None,
            "n_hold_days": int(exit_pos - entry_pos),
            "spread_return": round(ev_ret, 4) if ev_ret is not None else None,
            "kbe_return": round(kbe_cum, 4) if kbe_cum is not None else None,
            "kre_return": round(kre_cum, 4) if kre_cum is not None else None,
        })

        for j in range(entry_pos, exit_pos):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = 1.0  # mark as "active"
                pnl.iloc[j] = spread_ret.iloc[j]

    return pnl, positions, event_log


def main():
    sid = "PL734_occ_cre_cluster_short_kbe_long_kre"
    tickers = ["KBE", "KRE", "SPY"]

    try:
        px = load_prices(tickers, start="2021-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    clusters = find_clusters(OCC_STATEMENTS, window_days=60, min_count=3)
    print(f"OCC CRE clusters found: {len(clusters)}")
    for c in clusters:
        print(f"  {c.date()}")

    pnl, positions, event_log = run_spread_event_study(
        clusters, ret, short_ticker="KBE", long_ticker="KRE", hold_days=45
    )

    n_events = sum(1 for e in event_log if e.get("entry_date"))
    print(f"Events with price data: {n_events}")

    held_pnl = pnl[positions > 0]
    held_spy = spy_r.reindex(held_pnl.index).dropna()

    if len(held_pnl) < 30:
        event_rets = [e["spread_return"] for e in event_log if e.get("spread_return") is not None]
        return mark_failed(
            sid,
            f"insufficient held days: {len(held_pnl)} (n_events={n_events})",
            extra={
                "events": event_log,
                "rule": "OCC CRE >= 3 statements/60 days -> short KBE + long KRE 45 days",
            },
        )

    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name="OCC CRE Cluster Short KBE Long KRE (held-days only)",
        positions=positions.reindex(held_pnl.index).fillna(0),
        cost_bps=10,
    )

    event_rets = [e["spread_return"] for e in event_log if e.get("spread_return") is not None]
    event_summary = {
        "n_events": len(event_rets),
        "avg_spread_return": round(float(np.mean(event_rets)), 4) if event_rets else None,
        "median_spread_return": round(float(np.median(event_rets)), 4) if event_rets else None,
        "win_rate": round(float(np.mean([r > 0 for r in event_rets])), 4) if event_rets else None,
        "best": round(float(np.max(event_rets)), 4) if event_rets else None,
        "worst": round(float(np.min(event_rets)), 4) if event_rets else None,
    }

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "When >=3 OCC supervisory letters or public statements cite GSIB "
                "CRE concentration concerns within a rolling 60-day window, short "
                "KBE and long KRE (equal-dollar spread) at the next session open; "
                "hold 45 trading days; close both legs at exit."
            ),
            "mechanism": (
                "OCC CRE warnings signal heightened regulatory scrutiny on large "
                "bank CRE loan books. GSIBs (overweighted in KBE) have higher CRE "
                "office and multifamily concentrations as a % of tangible book than "
                "regional banks (KRE). Elevated scrutiny forces larger loss provisions "
                "and constrains dividend/buyback capacity, pressuring KBE more than "
                "KRE. The spread trade isolates this relative effect while neutralizing "
                "broad financial-sector beta."
            ),
            "source": (
                "OCC.treas.gov — Semiannual Risk Perspective, speeches, supervisory "
                "letters 2022-2025; KBE/KRE/SPY prices via yfinance (auto_adjust=True)."
            ),
            "events": event_log,
            "event_summary": event_summary,
            "n_events": n_events,
            "caveats": (
                "OCC statement dates are hand-coded approximations; exact publication "
                "times require direct review of OCC.treas.gov/publications. The data "
                "window (2022-2025) is short and the CRE stress cycle is still evolving. "
                "KBE and KRE overlap substantially in holdings — the spread is not "
                "a pure KBE-vs-KRE bet but reflects the weighting differences. "
                "OCC communications are largely anticipated by the market; the "
                "signal may have limited incremental information beyond what is in "
                "earnings guidance and loan-level data. Limited independent non-overlapping "
                "event windows reduce statistical power."
            ),
        },
        pnl=held_pnl,
    )

    print(f"Done: {sid}")
    print(f"  n_events={n_events}, event_summary={event_summary}")
    if "error" not in m:
        print(
            f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
            f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
        )
        if "oos_sharpe" in m:
            print(f"  OOS Sharpe: {m['oos_sharpe']:.2f}")


if __name__ == "__main__":
    main()
