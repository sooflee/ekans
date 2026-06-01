"""PL912 — DOC Solar SE Asia Circumvention Affirmative Final -> Short JKS/CSIQ Long FSLR Pair"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL912_doc_circumvention_short_jks_csiq_long_fslr"

    # Known DOC circumvention ruling affirmative events:
    # Aug 2023: DOC preliminary affirmative circumvention determination for Cambodia/Malaysia/Thailand/Vietnam
    #   Federal Register publication ~2023-08-18 (preliminary)
    # Dec 2023 / early 2024: Final affirmative determinations by country
    #   Country-specific finals published late 2023 - early 2024
    # May 2022: DOC initiated investigation (Biden tariff relief period); market reaction when circumvention confirmed
    # Key events based on public record and strategy spec:
    known_events = [
        "2022-06-06",   # Biden admin tariff relief announcement (anti-dumping/CVD stayed 2 yrs) - inverted signal: short FSLR, long JKS+CSIQ (NOT our signal)
        # Circumvention affirmative events:
        "2023-08-18",   # DOC preliminary affirmative circumvention determination
        "2024-01-19",   # DOC Final affirmative determination for Cambodia, Malaysia, Thailand, Vietnam
        "2024-06-04",   # Solar tariff reinstatement / anti-circumvention enforcement escalation
    ]
    # Only affirmative events (where we short JKS+CSIQ, long FSLR)
    affirm_events = pd.to_datetime([
        "2023-08-18",
        "2024-01-19",
        "2024-06-04",
    ])

    try:
        px = load_prices(["JKS", "CSIQ", "FSLR", "TAN", "SPY"], start="2022-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]
    jks_r = ret["JKS"]
    csiq_r = ret["CSIQ"]
    fslr_r = ret["FSLR"]

    hold_cal = 30  # 30 calendar days ≈ ~21 trading days
    hold_td = 21

    # Pair P&L: short JKS (0.5 unit), short CSIQ (0.5 unit), long FSLR (1 unit)
    # Daily return = fslr_r - 0.5*jks_r - 0.5*csiq_r (sector-neutral vs Chinese/SE-Asia solar)
    pair_r = fslr_r - 0.5 * jks_r - 0.5 * csiq_r

    pnl = pd.Series(0.0, index=pair_r.index)
    evts = []

    for ed in affirm_events:
        # Enter T+1 after Federal Register publication
        mask = pair_r.index > ed
        if mask.sum() < hold_td:
            continue
        entry_idx = pair_r.index[mask][0]
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
        fslr_mask = fslr_r.index >= entry_idx
        fslr_cr = None
        if fslr_mask.sum() >= hold_td:
            fi = fslr_r.index.get_loc(fslr_r.index[fslr_mask][0])
            fslr_cr = float((1 + fslr_r.iloc[fi:fi + hold_td]).prod() - 1)

        jks_mask = jks_r.index >= entry_idx
        jks_cr = None
        if jks_mask.sum() >= hold_td:
            ji = jks_r.index.get_loc(jks_r.index[jks_mask][0])
            jks_cr = float((1 + jks_r.iloc[ji:ji + hold_td]).prod() - 1)

        evts.append({
            "event_date": str(ed.date()),
            "entry_date": str(entry_idx.date()),
            "pair_return": round(cr, 4),
            "fslr_return": round(fslr_cr, 4) if fslr_cr is not None else None,
            "jks_return": round(jks_cr, 4) if jks_cr is not None else None,
            "spy_return": round(sc, 4) if sc is not None else None,
        })

    if not evts:
        return mark_failed(sid, "no valid events after filtering")

    ip = pnl[pnl != 0]
    if len(ip) < 20:
        return mark_failed(sid, f"insufficient active trading days ({len(ip)})")

    m = compute_metrics(ip, benchmark=spy_r,
                        name="DOC Solar Circumvention Affirmative → Short JKS+CSIQ / Long FSLR")
    pr = [e["pair_return"] for e in evts]
    save_result(sid, m, extra={
        "rule": "Short JKS + short CSIQ (equal notional each), long FSLR (combined notional) for 21 trading days when DOC ITA publishes affirmative circumvention determination under A-570-979/C-570-980",
        "mechanism": "DOC affirmative circumvention ruling extends AD/CVD duties to SE-Asia assembled Chinese solar modules (JKS/CSIQ sourcing hubs), raising landed cost 40-200% vs FSLR US-manufactured CdTe modules; pair isolates China-vs-domestic regulatory wedge",
        "source": "DOC ITA access.trade.gov; Federal Register; yfinance prices",
        "n_events": len(evts),
        "avg_pair_return": round(float(np.mean(pr)), 4),
        "pair_win_rate": round(float(np.mean([r > 0 for r in pr])), 4),
        "events": evts,
    })
    print(f"Done: {len(evts)} events, avg pair return={np.mean(pr):.2%}")


if __name__ == "__main__":
    main()
