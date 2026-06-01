"""PL913 — India DGTR AD + ALMM Enforcement → Short JKS+CSIQ / Long TAN"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns

# Manually annotated DGTR/MNRE event dates (T+1 entry after publication).
# Sources: dgtr.gov.in solar PV cases; mnre.gov.in ALMM orders; Mercom India news.
#   2018-07-30: India Safeguard Duty 25% on solar modules from China/Malaysia announced
#   2019-09-16: DGTR initiation of AD investigation on solar cells from China/Malaysia/Thailand/Vietnam
#   2022-04-01: MNRE ALMM List-I (modules) enforcement begins — modules must be from approved list
#   2023-10-01: MNRE tightened ALMM enforcement; proxy for ongoing List-II extension pressure

TRIGGER_DATES = ["2018-07-31", "2019-09-17", "2022-04-04", "2023-10-02"]
HOLD_DAYS = 32  # ~45 calendar days in trading days

def main():
    sid = "PL913_india_dgtr_ad_almm_phase2_short_jks_csiq"
    try:
        px = load_prices(["JKS", "CSIQ", "TAN", "SPY"], start="2015-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    missing = [t for t in ["JKS", "CSIQ", "TAN"] if t not in ret.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    jks_r  = ret["JKS"]
    csiq_r = ret["CSIQ"]
    tan_r  = ret["TAN"]
    spy_r  = ret["SPY"]

    # Strategy: short JKS + short CSIQ (equal notional), long TAN (combined equal notional)
    # Net: (-0.5 JKS - 0.5 CSIQ) + TAN = TAN - 0.5*(JKS+CSIQ)
    short_basket = 0.5 * jks_r + 0.5 * csiq_r
    strat_r = tan_r - short_basket   # long TAN, short China solar names

    pnl = pd.Series(0.0, index=strat_r.index)
    evts = []

    for entry_str in TRIGGER_DATES:
        entry_dt = pd.Timestamp(entry_str)
        mask = strat_r.index >= entry_dt
        if mask.sum() < HOLD_DAYS:
            continue
        start_idx = strat_r.index[mask][0]
        p = strat_r.index.get_loc(start_idx)
        ep = min(p + HOLD_DAYS, len(strat_r))
        seg = strat_r.iloc[p:ep]

        # Accumulate pnl without overlap
        for idx, v in seg.items():
            if pnl[idx] == 0.0:
                pnl[idx] = v

        cum_strat = float((1 + seg).prod() - 1)
        cum_jks   = float((1 + jks_r.iloc[p:ep]).prod() - 1)
        cum_csiq  = float((1 + csiq_r.iloc[p:ep]).prod() - 1)
        cum_tan   = float((1 + tan_r.iloc[p:ep]).prod() - 1)
        cum_spy   = float((1 + spy_r.iloc[p:ep]).prod() - 1) if p < len(spy_r) else None

        evts.append({
            "entry_date": str(start_idx.date()),
            "strat_return": round(cum_strat, 4),
            "jks_return": round(cum_jks, 4),
            "csiq_return": round(cum_csiq, 4),
            "tan_return": round(cum_tan, 4),
            "spy_return": round(cum_spy, 4) if cum_spy is not None else None,
        })

    if not evts:
        return mark_failed(sid, "no valid DGTR/ALMM events found")

    ip = pnl[pnl != 0]
    if len(ip) < 30:
        return mark_failed(sid, f"insufficient active days ({len(ip)})")

    m = compute_metrics(ip, benchmark=spy_r, name="India DGTR/ALMM: Short JKS+CSIQ / Long TAN")
    sr = [e["strat_return"] for e in evts]
    save_result(sid, m, extra={
        "rule": "Short JKS+CSIQ (equal notional), long TAN on India DGTR AD or MNRE ALMM enforcement event; hold ~45 calendar days",
        "mechanism": "India regulatory headwinds reduce addressable market for China-origin solar exporters; TAN captures global solar beta; spread isolates India-drag alpha",
        "source": "dgtr.gov.in; mnre.gov.in; Mercom India; yfinance",
        "n_events": len(evts),
        "avg_strat_return": round(float(np.mean(sr)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in sr])), 4),
        "hold_days": HOLD_DAYS,
        "events": evts,
        "caveats": "DGTR event dates hand-annotated. India revenue segment not separately disclosed in older JKS/CSIQ filings. 2018 SGD analog from different policy regime. Very small sample (4 events).",
    })
    print(f"Done: {len(evts)} events, avg strat return {np.mean(sr):.2%}")

if __name__ == "__main__":
    main()
