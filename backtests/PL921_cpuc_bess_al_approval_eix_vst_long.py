"""PL921 — CPUC BESS Resource Adequacy AL Approval (CAISO LA Basin) -> Long EIX + VST"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL921_cpuc_bess_al_approval_eix_vst_long"

    # CPUC BESS Advice Letter approval seed events:
    # Dec 2021: CPUC D.21-12-014 Aliso Canyon mitigation order
    # Jun 2022: SCE AL 4805-E series approvals for BESS RA contracts
    # Mar 2023: CAISO 2023 LCR Procurement Results published
    # Feb 2024: Additional SCE BESS RA contract approvals
    known_events = [
        "2021-12-20",   # CPUC D.21-12-014 Aliso Canyon mitigation order
        "2022-06-15",   # SCE AL 4805-E series BESS RA approval
        "2023-03-01",   # CAISO 2023 LCR Procurement Results
        "2024-02-01",   # Additional BESS RA contract approvals
    ]
    event_dates = pd.to_datetime(known_events)

    try:
        px = load_prices(["EIX", "VST", "NRG", "SRE", "SPY"], start="2021-01-01")
    except Exception as e:
        # Retry without NRG/SRE
        try:
            px = load_prices(["EIX", "VST", "SPY"], start="2021-01-01")
        except Exception as e2:
            return mark_failed(sid, f"data load: {e2}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]
    eix_r = ret["EIX"]
    vst_r = ret["VST"]

    hold_td = 90  # ~4.5 months trading days (within 180 cal day max)

    # Long equal-weighted EIX + VST
    pair_r = 0.5 * eix_r + 0.5 * vst_r

    pnl = pd.Series(0.0, index=pair_r.index)
    evts = []

    for ed in event_dates:
        # Enter T+2 after CPUC AL approval (strategy spec says T+2)
        mask = pair_r.index > ed
        if mask.sum() < 3:
            continue
        entry_idx = pair_r.index[mask][2]  # T+2 (skip first 2 trading days)
        p = pair_r.index.get_loc(entry_idx)
        ep = min(p + hold_td, len(pair_r))
        seg = pair_r.iloc[p:ep]
        cr = float((1 + seg).prod() - 1)
        pnl.iloc[p:ep] += seg.values[:ep - p]

        # SPY over same window
        sp_mask = spy_r.index >= entry_idx
        sc = None
        if sp_mask.sum() >= hold_td:
            spi = spy_r.index.get_loc(spy_r.index[sp_mask][0])
            sc = float((1 + spy_r.iloc[spi:spi + hold_td]).prod() - 1)

        # Individual leg returns
        eix_mask = eix_r.index >= entry_idx
        eix_cr = None
        if eix_mask.sum() >= hold_td:
            ei = eix_r.index.get_loc(eix_r.index[eix_mask][0])
            eix_cr = float((1 + eix_r.iloc[ei:ei + hold_td]).prod() - 1)

        vst_mask = vst_r.index >= entry_idx
        vst_cr = None
        if vst_mask.sum() >= hold_td:
            vi = vst_r.index.get_loc(vst_r.index[vst_mask][0])
            vst_cr = float((1 + vst_r.iloc[vi:vi + hold_td]).prod() - 1)

        evts.append({
            "event_date": str(ed.date()),
            "entry_date": str(entry_idx.date()),
            "pair_return": round(cr, 4),
            "eix_return": round(eix_cr, 4) if eix_cr is not None else None,
            "vst_return": round(vst_cr, 4) if vst_cr is not None else None,
            "spy_return": round(sc, 4) if sc is not None else None,
        })

    if not evts:
        return mark_failed(sid, "no valid events after filtering")

    ip = pnl[pnl != 0]
    if len(ip) < 30:
        return mark_failed(sid, f"insufficient active trading days ({len(ip)})")

    m = compute_metrics(ip, benchmark=spy_r,
                        name="CPUC BESS AL Approval → Long EIX+VST")
    pr = [e["pair_return"] for e in evts]
    save_result(sid, m, extra={
        "rule": "Long EIX + VST equal-weight for 90 trading days when CPUC approves SCE/SDG&E BESS RA contract in CAISO LA Basin LCR zone",
        "mechanism": "BESS RA approvals lock in capacity revenue for CAISO BESS fleet operators (VST Moss Landing) and rate-base capex for SCE parent EIX; forward earnings visibility improves",
        "source": "CPUC Advice Letter portal; CAISO LCR Technical Study; yfinance",
        "n_events": len(evts),
        "avg_pair_return": round(float(np.mean(pr)), 4),
        "pair_win_rate": round(float(np.mean([r > 0 for r in pr])), 4),
        "events": evts,
    })
    print(f"Done: {len(evts)} events, avg pair return={np.mean(pr):.2%}")


if __name__ == "__main__":
    main()
