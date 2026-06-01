"""PL795 — FDIC DIF Reserve Ratio Breach -> Special Assessment -> Short Large-Bank EPS"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL795_fdic_dif_special_assessment_short_kbe"
    # Key events:
    # 1. 2023-05-11: FDIC Q1 QBP released (confirms SVB/Signature DIF breach) -> entry signal
    #    Exit at 40 trading days OR FDIC Board vote on Nov 16, 2023
    # 2. 2009-12-30: FDIC prepaid assessment announcement (DIF deeply negative)
    #    Use KBE alone vs SPY for 2009 (KBWR not available until Jun 2010)
    # Strategy: Short KBE (1.5x) + Long KBWR (1.0x) for each event
    # For 2009: Short KBE only vs SPY
    HOLD_DAYS = 40

    # Events: (entry_date, exit_date_or_None_for_40d, use_kbwr)
    # For 2023: entry on QBP release 2023-05-11, exit on FDIC Board vote 2023-11-16 (capped at 40d)
    # For 2009: entry 2009-12-30, KBWR not available, use KBE short vs SPY
    events = [
        {
            "date": "2023-05-11",
            "label": "2023 SVB/Signature DIF breach (Q1 QBP release)",
            "use_kbwr": True,
            "max_hold": 40,  # cap at 40 trading days
        },
        {
            "date": "2009-12-30",
            "label": "2009 DIF deeply negative prepaid assessment",
            "use_kbwr": False,
            "max_hold": 40,
        },
    ]
    KBE_WEIGHT = -1.5
    KBWR_WEIGHT = 1.0

    try:
        # Load KBE and SPY (both available from 2005)
        px_kbe_spy = load_prices(["KBE", "SPY"], start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load KBE/SPY: {e}")

    if px_kbe_spy.empty or "KBE" not in px_kbe_spy.columns:
        return mark_failed(sid, "missing KBE price data")

    try:
        # Load KBWR (available from Jun 2010)
        px_kbwr = load_prices(["KBWR"], start="2010-01-01")
        kbwr_available = "KBWR" in px_kbwr.columns and not px_kbwr["KBWR"].dropna().empty
    except Exception as e:
        print(f"Warning: KBWR load failed ({e}), will skip KBWR leg")
        kbwr_available = False
        px_kbwr = pd.DataFrame()

    ret_kbe_spy = daily_returns(px_kbe_spy)
    kbe_r = ret_kbe_spy["KBE"].dropna()
    spy_r = ret_kbe_spy["SPY"].dropna()

    if kbwr_available:
        ret_kbwr = daily_returns(px_kbwr)
        kbwr_r = ret_kbwr["KBWR"].dropna()
    else:
        kbwr_r = pd.Series(dtype=float)

    # Build combined PnL on KBE/SPY common index
    all_idx = spy_r.index
    pnl = pd.Series(0.0, index=all_idx)
    event_details = []

    for ev in events:
        ev_date = pd.Timestamp(ev["date"])
        use_kbwr = ev["use_kbwr"] and kbwr_available
        max_hold = ev["max_hold"]

        # Pick common index for this event
        if use_kbwr:
            common = kbe_r.index.intersection(kbwr_r.index).intersection(spy_r.index)
        else:
            common = kbe_r.index.intersection(spy_r.index)

        future = common[common >= ev_date]
        if len(future) < 5:
            print(f"Event {ev_date.date()}: no data, skipping")
            continue

        entry_pos = common.get_loc(future[0])
        end_pos = min(entry_pos + max_hold, len(common) - 1)
        window = common[entry_pos:end_pos + 1]

        if len(window) < 5:
            print(f"Event {ev_date.date()}: window too small ({len(window)} days), skipping")
            continue

        # Pair PnL
        if use_kbwr:
            pair_ret = (KBE_WEIGHT * kbe_r.reindex(window) +
                        KBWR_WEIGHT * kbwr_r.reindex(window))
            kbwr_cum = float((1 + kbwr_r.reindex(window)).prod() - 1)
        else:
            # 2009: just short KBE (1x) vs SPY benchmark
            pair_ret = KBE_WEIGHT / abs(KBE_WEIGHT) * (-kbe_r.reindex(window))  # short KBE 1x
            kbwr_cum = None

        pnl.loc[window] = pair_ret.values

        kbe_cum = float((1 + kbe_r.reindex(window)).prod() - 1)
        spy_cum = float((1 + spy_r.reindex(window)).prod() - 1)
        pair_cum = float((1 + pair_ret).prod() - 1)

        det = {
            "event_date": str(ev_date.date()),
            "entry_date": str(common[entry_pos].date()),
            "label": ev["label"],
            "kbe_return": round(kbe_cum, 4),
            "spy_return": round(spy_cum, 4),
            "pair_return": round(pair_cum, 4),
            "days_held": len(window),
            "used_kbwr": use_kbwr,
        }
        if kbwr_cum is not None:
            det["kbwr_return"] = round(kbwr_cum, 4)
        event_details.append(det)
        print(f"Event {ev_date.date()} [{ev['label'][:30]}]: KBE {kbe_cum:.2%}, pair {pair_cum:.2%}, SPY {spy_cum:.2%}")

    active_pnl = pnl[pnl != 0]
    print(f"Active trading days: {len(active_pnl)}")

    if len(active_pnl) < 5:
        return mark_failed(sid, f"insufficient active days ({len(active_pnl)})")

    spy_bench = spy_r.reindex(active_pnl.index).dropna()
    m = compute_metrics(active_pnl, benchmark=spy_bench,
                        name="FDIC DIF Breach → Short KBE / Long KBWR")
    m["n_events"] = len(event_details)

    save_result(sid, m, extra={
        "rule": "Short KBE (1.5x) + Long KBWR (1.0x) for 40 trading days after FDIC QBP shows DIF reserve ratio < 1.35% following GSIB-category bank failure with systemic-risk exception. Known events: May 2023 (post-SVB), Dec 2009 (prepaid assessment).",
        "mechanism": "FDIC special assessments are charged against large banks with >$5B uninsured deposits (GSIBs) but exempt community banks under $5B threshold. This creates a direct KBE (GSIB-heavy) vs KBWR (community banks) spread compression, with KBE lagging as EPS estimates are cut for assessment charges.",
        "source": "FDIC Quarterly Banking Profile (fdic.gov/bank/statistical/); yfinance KBE, KBWR, SPY",
        "known_events": [ev["date"] for ev in events],
        "events": event_details,
        "caveats": "Only 2 events; KBWR not available for 2009 event. 2009 event uses KBE short only. 2023 event is highest-quality analog. Assessment carve-out is statutory (12 CFR 327.6(b)).",
    })
    print(f"Saved result: Sharpe={m.get('sharpe', 'N/A'):.2f}, CAGR={m.get('cagr', 'N/A'):.2%}")


if __name__ == "__main__":
    main()
