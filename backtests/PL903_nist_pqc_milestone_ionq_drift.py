"""PL903 — NIST PQC FIPS Milestone -> IONQ/RGTI Narrative Drift Long"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL903_nist_pqc_milestone_ionq_drift"

    try:
        px = load_prices(["IONQ", "RGTI", "QUBT", "SPY"], start="2021-10-01")
        spy_r = daily_returns(px[["SPY"]]).iloc[:, 0].dropna()
        ionq_r = daily_returns(px[["IONQ"]]).iloc[:, 0].dropna()
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    # Check RGTI and QUBT availability
    rgti_r = None
    qubt_r = None
    if "RGTI" in px.columns and px["RGTI"].dropna().shape[0] > 50:
        rgti_r = daily_returns(px[["RGTI"]]).iloc[:, 0].dropna()
    if "QUBT" in px.columns and px["QUBT"].dropna().shape[0] > 50:
        qubt_r = daily_returns(px[["QUBT"]]).iloc[:, 0].dropna()

    # NIST PQC milestone event dates
    # Source: csrc.nist.gov/news
    nist_events = [
        {
            "date": "2022-07-05",
            "name": "NIST PQC Round 3 finalists selected (CRYSTALS-Kyber, Dilithium, FALCON, SPHINCS+)",
            "type": "selection_announcement",
        },
        {
            "date": "2023-08-24",
            "name": "NIST FIPS 203/204/205 draft for public comment (initial public draft)",
            "type": "draft_fips",
        },
        {
            "date": "2024-08-13",
            "name": "NIST FIPS 203/204/205 final standards published (CRYSTALS-Kyber/Dilithium/SPHINCS+)",
            "type": "final_fips",
        },
        {
            "date": "2024-11-19",
            "name": "NIST NISTIR 8547 draft: hybrid PQC migration guidance",
            "type": "hybrid_migration_draft",
            "note": "overlaps Q4 2024 quantum bubble",
        },
    ]

    hold_days = 10
    stop_loss = -0.15  # IONQ -15% from entry

    # Align series
    common_idx = ionq_r.index.intersection(spy_r.index)
    ionq_r = ionq_r.reindex(common_idx)
    spy_r_c = spy_r.reindex(common_idx)
    if rgti_r is not None:
        rgti_r = rgti_r.reindex(common_idx)
    if qubt_r is not None:
        qubt_r = qubt_r.reindex(common_idx)

    events = []

    for ev in nist_events:
        ev_date = pd.Timestamp(ev["date"])
        future = common_idx[common_idx >= ev_date]
        if len(future) < 2:
            continue

        # Entry at close on event date (or next open if after market)
        entry_date = future[0]
        entry_loc = common_idx.get_loc(entry_date)
        exit_loc = min(entry_loc + hold_days, len(common_idx) - 1)

        slice_idx = common_idx[entry_loc:exit_loc + 1]
        if len(slice_idx) < 2:
            continue

        ionq_slice = ionq_r.reindex(slice_idx).fillna(0)
        spy_slice = spy_r_c.reindex(slice_idx).fillna(0)

        # Build portfolio: 1.0x IONQ + 0.5x RGTI (if available)
        if rgti_r is not None and not rgti_r.reindex(slice_idx).isna().all():
            rgti_slice = rgti_r.reindex(slice_idx).fillna(0)
            # Weight-normalize: IONQ 2/3, RGTI 1/3 (matching 1.0 and 0.5 relative weights)
            port_r = (2/3) * ionq_slice + (1/3) * rgti_slice
        else:
            port_r = ionq_slice

        # Stop loss based on IONQ cumulative return
        ionq_cum = ionq_slice.cumsum()
        stop = ionq_cum < stop_loss
        if stop.any():
            stop_idx = stop.idxmax()
            port_r = port_r.loc[:stop_idx]
            spy_slice = spy_slice.loc[:stop_idx]

        cum_ret = float((1 + port_r).prod() - 1)
        cum_spy = float((1 + spy_slice).prod() - 1)
        cum_ionq = float((1 + ionq_slice.loc[:port_r.index[-1]]).prod() - 1)

        ev_result = {
            "event_date": ev["date"],
            "event_name": ev["name"],
            "event_type": ev["type"],
            "entry_date": str(entry_date.date()),
            "exit_date": str(port_r.index[-1].date()),
            "hold_days": len(port_r),
            "ionq_return": round(cum_ionq, 4),
            "port_return": round(cum_ret, 4),
            "spy_return": round(cum_spy, 4),
            "alpha": round(cum_ret - cum_spy, 4),
        }
        if "note" in ev:
            ev_result["note"] = ev["note"]
        events.append(ev_result)

    print(f"Found {len(events)} NIST PQC events")

    # Results with and without the Nov 2024 bubble event
    events_excl_nov24 = [e for e in events if e["event_date"] != "2024-11-19"]
    print(f"Events ex-Nov 2024: {len(events_excl_nov24)}")

    if len(events) < 3:
        return mark_failed(sid, f"too few events ({len(events)}) — need at least 3 NIST milestones")

    # Build full daily PnL
    def build_pnl_series(event_list):
        pnl = pd.Series(0.0, index=common_idx)
        for ev in event_list:
            entry = pd.Timestamp(ev["entry_date"])
            exit_d = pd.Timestamp(ev["exit_date"])
            if rgti_r is not None:
                rgti_slice = rgti_r.loc[entry:exit_d].fillna(0)
                ionq_slice = ionq_r.loc[entry:exit_d].fillna(0)
                trade_slice = (2/3) * ionq_slice + (1/3) * rgti_slice
            else:
                trade_slice = ionq_r.loc[entry:exit_d].fillna(0)
            pnl.loc[entry:exit_d] += trade_slice
        return pnl

    pnl_all = build_pnl_series(events)
    pnl_excl = build_pnl_series(events_excl_nov24)

    active_all = pnl_all[pnl_all != 0]
    active_excl = pnl_excl[pnl_excl != 0]

    print(f"Active days (all): {len(active_all)}, ex-Nov24: {len(active_excl)}")

    if len(active_excl) < 10:
        # Use all events if excl doesn't have enough
        active_pnl = active_all
        note = "all events including Nov 2024 quantum bubble"
    else:
        active_pnl = active_all
        note = "all events"

    if len(active_pnl) < 15:
        return mark_failed(sid, f"insufficient active days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r_c,
                        name="NIST PQC FIPS Milestone -> Long IONQ+RGTI (10-day hold)")
    m["n_events"] = len(events)

    # Also compute ex-Nov24 Sharpe for diagnostics
    if len(active_excl) >= 10:
        m_excl = compute_metrics(active_excl, benchmark=spy_r_c, name="ex-Nov24")
        excl_sharpe = m_excl.get("sharpe")
        excl_cagr = m_excl.get("cagr")
    else:
        excl_sharpe = None
        excl_cagr = None

    save_result(sid, m, extra={
        "rule": "Long IONQ (1.0x) + RGTI (0.5x, half weight) from close on NIST PQC FIPS milestone date, hold 10 trading days. Stop: IONQ -15% from entry. Sources: NIST CSRC publication RSS.",
        "mechanism": "NIST PQC standard milestones generate narrative catalysts for quantum computing stocks as institutional investors interpret post-quantum cryptography deployment timelines as proof-of-concept validation for quantum hardware demand. The 10-day window captures media cycle before mean reversion.",
        "source": "yfinance (IONQ, RGTI, QUBT, SPY); NIST CSRC news (csrc.nist.gov/news)",
        "n_events": len(events),
        "events": events,
        "ex_nov24_sharpe": excl_sharpe,
        "ex_nov24_cagr": excl_cagr,
        "note": f"4 events only; small N. {note}. Nov 2024 event confounded by Q4 retail quantum bubble.",
    })

    avg_alpha = float(np.mean([e["alpha"] for e in events]))
    win_rate = float(np.mean([1 if e["port_return"] > 0 else 0 for e in events]))
    print(f"Sharpe={m.get('sharpe','N/A'):.2f}, CAGR={m.get('cagr','N/A')*100:.1f}%, events={len(events)}")
    print(f"Win rate={win_rate:.0%}, avg alpha={avg_alpha:.3f}")
    print(f"Ex-Nov24: Sharpe={excl_sharpe}, CAGR={excl_cagr}")


if __name__ == "__main__":
    main()
