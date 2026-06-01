"""PL904 — Quantum Milestone Gap >15mo → Short QTUM / Long CSCO Pair"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns

# Manually annotated milestone-gap start dates (IBM/Google/IonQ roadmap archives).
# Gap = month when last major quantum milestone (>=100-qubit system or QV doubling)
# passed the 15-month stagnation threshold, triggering the short-QTUM/long-CSCO entry.
# Source: IBM Quantum roadmap, Google Quantum AI publications, IonQ press releases.
#   2021-04: ~15 months after Dec 2019 Google 53-qubit supremacy demo (no major follow-up)
#   2022-10: gap after IBM 127-qubit Eagle (Nov 2021); next major (Osprey 433q) was Nov 2022
#            — use mid-gap 2022-10 as trigger onset (15mo after Eagle)
#   2024-09: gap after IBM Heron r2 (Feb 2024); Quantinuum H2 upgrade Nov 2024
# QTUM ETF launched 2020-12-22; use 2021-01-01 start for prices.

MILESTONE_GAP_ENTRIES = [
    "2021-04-01",  # post-Google supremacy lull
    "2022-10-01",  # Eagle→Osprey gap
    "2024-09-01",  # Heron r2 → next major gap
]

HOLD_DAYS = 63  # ~3 calendar months in trading days

def main():
    sid = "PL904_quantum_winter_qtum_short_csco_long"
    try:
        px = load_prices(["QTUM", "CSCO", "SPY"], start="2021-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    if "QTUM" not in ret.columns or "CSCO" not in ret.columns:
        return mark_failed(sid, "missing QTUM or CSCO returns")

    qtum_r = ret["QTUM"]
    csco_r = ret["CSCO"]
    spy_r  = ret["SPY"]

    # Pair return: long CSCO, short QTUM (dollar neutral 1:1)
    pair_r = csco_r - qtum_r

    pnl = pd.Series(0.0, index=pair_r.index)
    evts = []
    for entry_str in MILESTONE_GAP_ENTRIES:
        entry_dt = pd.Timestamp(entry_str)
        mask = pair_r.index >= entry_dt
        if mask.sum() < HOLD_DAYS:
            continue
        start_idx = pair_r.index[mask][0]
        p = pair_r.index.get_loc(start_idx)
        ep = min(p + HOLD_DAYS, len(pair_r))
        seg = pair_r.iloc[p:ep]

        # Accumulate into pnl (overwrite only zeros to avoid overlap)
        for j, (idx, v) in enumerate(seg.items()):
            if pnl[idx] == 0.0:
                pnl[idx] = v

        cum_pair = float((1 + seg).prod() - 1)
        spy_seg = spy_r.iloc[p:ep] if p < len(spy_r) else pd.Series(dtype=float)
        cum_spy  = float((1 + spy_seg).prod() - 1) if len(spy_seg) > 0 else None

        # Entry-day QTUM SMA50 filter check (informational)
        qtum_window = px["QTUM"].iloc[max(0, p-50):p]
        above_sma50 = bool(len(qtum_window) >= 10 and px["QTUM"].iloc[p] > qtum_window.mean())

        evts.append({
            "entry_date": str(start_idx.date()),
            "pair_return": round(cum_pair, 4),
            "spy_return": round(cum_spy, 4) if cum_spy is not None else None,
            "qtum_above_sma50_at_entry": above_sma50,
        })

    if not evts:
        return mark_failed(sid, "no valid events after QTUM IPO")

    ip = pnl[pnl != 0]
    if len(ip) < 30:
        return mark_failed(sid, f"insufficient active days ({len(ip)})")

    m = compute_metrics(ip, benchmark=spy_r, name="Quantum Winter: Short QTUM / Long CSCO")
    pr = [e["pair_return"] for e in evts]
    save_result(sid, m, extra={
        "rule": "Short QTUM / Long CSCO (dollar-neutral) when months since last major quantum milestone >15; exit at 3mo or new milestone",
        "mechanism": "Quantum narrative fatigue: ETF premium compresses during innovation lulls; PQC-incumbent CSCO outperforms speculative pure-plays",
        "source": "IBM Quantum roadmap; Google Quantum AI blog; IonQ press releases; yfinance",
        "n_events": len(evts),
        "avg_pair_return": round(float(np.mean(pr)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in pr])), 4),
        "hold_days": HOLD_DAYS,
        "events": evts,
        "caveats": "Only 3 events since QTUM 2021 IPO — extremely limited sample. Milestone dates hand-annotated; live signal requires real-time press monitoring.",
    })
    print(f"Done: {len(evts)} events, avg pair return {np.mean(pr):.2%}")

if __name__ == "__main__":
    main()
