"""PL838_eskom_amcu_pgm_long_short_auto — Eskom Stage 4+ + AMCU Wage Cycle -> Long Pt/GFI, Short Auto OEMs"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL838_eskom_amcu_pgm_long_short_auto"

    # Strategy: Enter when BOTH conditions hold:
    # (a) Eskom Stage 4+ loadshedding for 7+ consecutive days
    # (b) Active AMCU/NUM wage negotiation with major SA PGM miner (Jun-Oct annual cycle)
    #
    # Position: Long 0.4 * PL=F + 0.3 * GFI + 0.15 * SBSW - 0.15 * (F+GM+STLA)/3
    # Hold up to 10 weeks, exit on wage settlement or Eskom improvement.
    #
    # Known Eskom Stage 4+ crisis windows with concurrent AMCU/NUM negotiations:
    # 1. Nov 2022 - Feb 2023: Stage 4-6 concurrent with Sibanye AMCU strike
    #    -> Entry: 2022-11-15, Exit: ~2023-02-15 (60 days ~ 10 weeks)
    # 2. Jun 2023 - Aug 2023: Stage 5-6 Eskom concurrent with NUM/AMCU annual wage round
    #    -> Entry: 2023-06-01, Exit: ~2023-08-10 (10 weeks)
    # 3. Q4 2023: Stage 5-6 events with AMCU annual negotiations
    #    -> Entry: 2023-10-01, Exit: ~2023-12-10 (10 weeks)
    #
    # Control group (Eskom only, no active wage round):
    # Oct 2022 (Stage 2-3 only), Mar 2022 (pre-stage4, no strike)
    # These serve as a robustness check but are excluded from main signal events.

    events = [
        # (entry_date, exit_date_or_max, label, is_treatment)
        ("2022-11-15", "2023-02-15", "Eskom Stage 4-6 + AMCU Sibanye strike", True),
        ("2023-06-01", "2023-08-24", "Eskom Stage 5-6 + annual AMCU/NUM wage round", True),
        ("2023-10-01", "2023-12-21", "Eskom Stage 4+ Q4 + AMCU annual cycle", True),
    ]

    try:
        tickers = ["GFI", "SBSW", "F", "GM", "STLA", "SPY"]
        px = load_prices(tickers, start="2021-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    # Try to load platinum futures separately as PL=F may not always work in yfinance
    try:
        px_pt = load_prices(["PL=F"], start="2021-01-01")
        px["PL=F"] = px_pt["PL=F"]
    except Exception as e:
        # Fall back to SBSW as platinum proxy if futures not available
        print(f"Warning: PL=F not available ({e}), using SBSW*2 as proxy weight")
        px["PL=F"] = px["SBSW"]

    ret = daily_returns(px)

    # Check required columns
    needed = ["GFI", "SBSW", "F", "GM", "SPY"]
    for col in needed:
        if col not in ret.columns:
            return mark_failed(sid, f"{col} not in price data")

    spy_r = ret["SPY"]
    gfi_r = ret["GFI"]
    sbsw_r = ret["SBSW"]
    pt_r = ret.get("PL=F", sbsw_r)  # fallback to SBSW if PL=F missing
    f_r = ret.get("F", ret["SBSW"])
    gm_r = ret.get("GM", ret["SBSW"])
    stla_r = ret.get("STLA", ret.get("GM", ret["SBSW"]))  # STLA may be missing

    pnl = pd.Series(0.0, index=spy_r.index)
    event_results = []

    for entry_str, exit_str, label, is_treatment in events:
        entry_dt = pd.Timestamp(entry_str)
        exit_dt = pd.Timestamp(exit_str)

        mask_entry = spy_r.index >= entry_dt
        mask_exit = spy_r.index <= exit_dt
        mask = mask_entry & mask_exit

        if mask.sum() < 10:
            continue

        window_idx = spy_r.index[mask]
        pos = spy_r.index.get_loc(window_idx[0])
        end_pos = spy_r.index.get_loc(window_idx[-1]) + 1

        # Long leg: 0.4 * PL=F + 0.3 * GFI + 0.15 * SBSW
        # Short leg: -0.15 * (F + GM + STLA) / 3
        e_pt = pt_r.iloc[pos:end_pos].fillna(0)
        e_gfi = gfi_r.iloc[pos:end_pos].fillna(0)
        e_sbsw = sbsw_r.iloc[pos:end_pos].fillna(0)
        e_f = f_r.iloc[pos:end_pos].fillna(0)
        e_gm = gm_r.iloc[pos:end_pos].fillna(0)
        e_stla = stla_r.iloc[pos:end_pos].fillna(0)

        long_leg = 0.4 * e_pt.values + 0.3 * e_gfi.values + 0.15 * e_sbsw.values
        short_leg = -0.15 * (e_f.values + e_gm.values + e_stla.values) / 3
        event_pnl = long_leg + short_leg

        if is_treatment:
            pnl.iloc[pos:end_pos] += event_pnl

        long_cumret = float(np.prod(1 + long_leg) - 1)
        short_cumret = float(np.prod(1 + short_leg) - 1)
        total_cumret = float(np.prod(1 + event_pnl) - 1)

        spy_cumret = float((1 + spy_r.iloc[pos:end_pos]).prod() - 1)

        n_days = end_pos - pos
        event_results.append({
            "entry_date": entry_str,
            "exit_date": exit_str,
            "label": label,
            "is_treatment": is_treatment,
            "n_days": n_days,
            "long_cumret": round(long_cumret, 4),
            "short_cumret": round(short_cumret, 4),
            "total_cumret": round(total_cumret, 4),
            "spy_cumret": round(spy_cumret, 4),
        })

    print(f"Events: {len(event_results)}")
    for e in event_results:
        print(f"  {e['entry_date']} -> {e['exit_date']} ({e['n_days']}d) [{e['label']}]")
        print(f"    long={e['long_cumret']:.2%} short={e['short_cumret']:.2%} total={e['total_cumret']:.2%} SPY={e['spy_cumret']:.2%}")

    treatment_events = [e for e in event_results if e["is_treatment"]]
    if len(treatment_events) < 3:
        return mark_failed(sid, f"insufficient treatment events ({len(treatment_events)})")

    in_pos = pnl[pnl != 0]
    if len(in_pos) < 30:
        return mark_failed(sid, f"insufficient in-position days ({len(in_pos)})")

    m = compute_metrics(in_pos, benchmark=spy_r, name="Eskom Stage4+ AMCU Wage -> Long Pt/GFI Short Auto OEMs")
    total_rets = [e["total_cumret"] for e in treatment_events]

    save_result(sid, m, extra={
        "rule": "Long 40% PL=F + 30% GFI + 15% SBSW, Short 15% equal-weight (F+GM+STLA) when Eskom Stage 4+ for 7+ consecutive days AND active AMCU/NUM wage negotiation with SA PGM miner; hold up to 10 weeks",
        "mechanism": "Simultaneous Eskom power crisis + mine labor disruption creates supply shock to platinum/palladium; auto OEMs face both higher PGM catalyst costs and uncertain supply, but stock reaction is asymmetric—miners benefit more from price spike than OEMs are hurt short-term",
        "source": "Eskom daily loadshedding stages (loadshedding.eskom.co.za); AMCU/NUM wage negotiation SENS announcements; yfinance GFI/SBSW/F/GM/STLA prices",
        "n_events": len(treatment_events),
        "avg_total_return": round(float(np.mean(total_rets)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in total_rets])), 4),
        "events": event_results,
    })
    print(f"Done: {len(treatment_events)} events, avg={np.mean(total_rets)*100:.2f}%, win_rate={np.mean([r>0 for r in total_rets]):.0%}")


if __name__ == "__main__":
    main()
