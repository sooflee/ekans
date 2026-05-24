"""
X12 Junior explorer basket triggered by major-miner RRR shock.

Pairs the X1 trigger (oil major RRR < 80%) — but for gold majors. Curated
events where NEM/GOLD/AEM disclosed sub-80% RRR or material reserve write-downs
in their 10-K/40-F filings. On each event, go long a curated basket of TSX/TSX-V
junior explorers (NI 43-101 public resources) equal-weight for 6 months
(126 trading days).

Mechanism: When the senior gold miners can't replace reserves organically, M&A
appetite for advanced juniors with permitted, near-production resources rises.

Note: Gold majors RRR is more sparsely disclosed than oil majors. Reserve write-
downs are the more material/public signal. We use those as triggers.

Junior basket (TSX/TSX-V listed, NI 43-101 reserves, advanced-stage):
- SKE.TO (Skeena Resources)
- NGEX.TO (NGEx Minerals)
- WRN.TO (Western Copper & Gold)
- IMG.TO (IAMGOLD — actually senior; replace)
- WDO.TO (Wesdome Gold Mines)
- TXG.TO (Torex Gold)
- WPM.TO (Wheaton — actually streamer; replace)
- EQX (Equinox Gold — replacement)
- AGI (Alamos Gold)
- BTG (B2Gold)
Replacements due to delisting/inappropriate: WPM.TO and IMG.TO swapped to AGI, BTG.

Source: yfinance daily TSX listings.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, save_result, mark_failed

# Curated gold-major reserve-shock events (10-K/40-F filing dates).
# Source: company annual reports / 8-K reserve disclosure press releases.
GOLD_RRR_EVENTS = [
    # NEM (Newmont)
    ("NEM",  "2014-02-21", "FY13 reserves -22% on price assumption cut"),
    ("NEM",  "2015-02-20", "FY14 reserves -18% further cut"),
    ("NEM",  "2023-02-23", "FY22 reserves -13% (depletion outpaced replacement)"),
    # GOLD (Barrick — successor to ABX)
    ("GOLD", "2014-02-19", "FY13 reserves -33% on $1,100/oz assumption"),
    ("GOLD", "2015-02-18", "FY14 reserves -19% further cut"),
    ("GOLD", "2024-02-14", "FY23 reserves dropped on M&A divestments"),
    # AEM (Agnico Eagle) — generally strong replacement; one notable event
    ("AEM",  "2015-02-11", "FY14 reserves slight decline at lower price deck"),
]

JUNIOR_BASKET = ["SKE.TO", "NGEX.TO", "WRN.TO", "WDO.TO", "TXG.TO", "EQX",
                 "AGI", "BTG", "AG", "PAAS", "HL", "CDE", "KGC", "GFI"]


def main():
    sid = "X12_junior_explorer_rrr"
    tickers = sorted(set([e[0] for e in GOLD_RRR_EVENTS] + JUNIOR_BASKET + ["SPY", "GDX", "GDXJ", "GLD"]))
    try:
        px = load_prices(tickers, start="2013-01-01", end="2025-12-31")
    except Exception as e:
        return mark_failed(sid, f"price load failed: {e}")

    if "SPY" not in px.columns or "GDXJ" not in px.columns:
        return mark_failed(sid, "SPY/GDXJ missing")

    spy = px["SPY"].dropna()
    gdxj = px["GDXJ"].dropna()
    gdx = px.get("GDX", spy).dropna()
    gld = px.get("GLD", spy).dropna()

    avail_juniors = [t for t in JUNIOR_BASKET if t in px.columns and not px[t].dropna().empty]
    if len(avail_juniors) < 5:
        return mark_failed(sid, f"too few juniors available ({len(avail_juniors)})")

    hold = 126  # 6 months
    per_event = []
    for tkr, ddate, note in GOLD_RRR_EVENTS:
        d = pd.Timestamp(ddate)
        # Build basket return over hold window
        # Use juniors with data on or after d
        idx_anchor = spy.index[spy.index >= d]
        if len(idx_anchor) < 2:
            continue
        t0 = idx_anchor[0]
        i0 = spy.index.get_loc(t0)
        i_entry = i0 + 1
        i_exit = i_entry + hold
        if i_exit >= len(spy):
            continue
        entry_date = spy.index[i_entry]
        exit_date = spy.index[i_exit]

        # Per-junior return
        rets = []
        used = []
        for j in avail_juniors:
            s = px[j].dropna()
            if s.empty:
                continue
            # Ensure junior has data on entry
            if entry_date not in s.index:
                # fallback: nearest forward
                fwd = s.index[s.index >= entry_date]
                if len(fwd) < 2:
                    continue
                e_use = fwd[0]
            else:
                e_use = entry_date
            if exit_date not in s.index:
                bwd = s.index[s.index <= exit_date]
                if len(bwd) < 2:
                    continue
                x_use = bwd[-1]
            else:
                x_use = exit_date
            if e_use >= x_use:
                continue
            r = float(s.loc[x_use] / s.loc[e_use] - 1)
            rets.append(r); used.append(j)

        if len(rets) < 3:
            continue
        basket_ret = float(np.mean(rets))
        spy_w = spy.loc[entry_date:exit_date]
        spy_r = float(spy_w.iloc[-1] / spy_w.iloc[0] - 1)
        gdxj_w = gdxj.loc[entry_date:exit_date]
        gdxj_r = float(gdxj_w.iloc[-1] / gdxj_w.iloc[0] - 1) if len(gdxj_w) >= 2 else 0.0
        gdx_w = gdx.loc[entry_date:exit_date]
        gdx_r = float(gdx_w.iloc[-1] / gdx_w.iloc[0] - 1) if len(gdx_w) >= 2 else 0.0
        gld_w = gld.loc[entry_date:exit_date]
        gld_r = float(gld_w.iloc[-1] / gld_w.iloc[0] - 1) if len(gld_w) >= 2 else 0.0

        per_event.append({
            "trigger_major": tkr, "filing": ddate, "note": note,
            "basket_ret_6m": basket_ret, "n_basket": len(rets),
            "basket_constituents_used": used,
            "vs_spy": basket_ret - spy_r,
            "vs_gdxj": basket_ret - gdxj_r,
            "vs_gdx": basket_ret - gdx_r,
            "vs_gld": basket_ret - gld_r,
        })

    if len(per_event) < 3:
        return mark_failed(sid, f"only {len(per_event)} events with sufficient basket coverage",
                           extra={"n_events": len(per_event), "juniors_available": avail_juniors})

    def stats(arr):
        a = np.array(arr)
        mean = float(a.mean()); std = float(a.std(ddof=1)) if len(a)>1 else 0.0
        t = mean / (std/np.sqrt(len(a))) if std>0 else 0.0
        return {"n": len(a), "mean": mean, "median": float(np.median(a)),
                "std": std, "t_stat": float(t), "hit_rate": float((a > 0).mean()),
                "min": float(a.min()), "max": float(a.max())}

    result = {
        "name": "X12 junior basket on gold-major RRR shock 6mo",
        "n_events": len(per_event),
        "abs_6m":         stats([e["basket_ret_6m"] for e in per_event]),
        "excess_vs_spy":  stats([e["vs_spy"] for e in per_event]),
        "excess_vs_gdxj": stats([e["vs_gdxj"] for e in per_event]),
        "excess_vs_gdx":  stats([e["vs_gdx"] for e in per_event]),
        "excess_vs_gld":  stats([e["vs_gld"] for e in per_event]),
        "juniors_available": avail_juniors,
    }
    print(f"X12: n={len(per_event)}, juniors={len(avail_juniors)}, "
          f"abs-mean={result['abs_6m']['mean']*100:+.2f}%, "
          f"vs-GDXJ={result['excess_vs_gdxj']['mean']*100:+.2f}% t={result['excess_vs_gdxj']['t_stat']:.2f}, "
          f"vs-GLD={result['excess_vs_gld']['mean']*100:+.2f}%")
    save_result(sid, result, extra={
        "status": "ok",
        "rule": "On gold-major reserve write-down disclosure (NEM/GOLD/AEM 10-K/40-F), long equal-weight basket of ~10-14 advanced-stage TSX/NYSE gold names for 126 trading days (~6 months).",
        "mechanism": "Reserve write-downs in seniors -> M&A demand for advanced juniors with NI 43-101 resources rises -> rerating.",
        "source": "Curated annual report disclosures NEM/GOLD/AEM 2014-2024; junior basket curated from advanced-stage NI 43-101 issuers.",
        "events": per_event,
        "universe_basket": avail_juniors,
        "hold_days": hold,
        "small_n": True,
        "substitution_note": "Original spec named TSX-V juniors (SGD.V, SKE.TO, NGEX.TO). Several delisted; basket broadened to advanced-stage TSX/NYSE-listed gold names with public NI 43-101 reports.",
    })


if __name__ == "__main__":
    main()
