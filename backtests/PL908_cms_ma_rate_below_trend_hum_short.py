"""PL908 — CMS MA Final Rate Below Industry Cost Trend -> Short HUM / Long UNH"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL908_cms_ma_rate_below_trend_hum_short"

    # Known CMS Final Rate shock events: rate materially below sell-side cost trend
    # April 2024: CMS Final Rate +3.7% vs ~5.0% consensus trend (gap ~-130 bps)
    #   Publication date: April 1, 2024
    # April 2023: CMS Final Rate +3.32% vs ~4.5% trend (gap ~-118 bps)
    #   Publication date: April 3, 2023
    # April 2022: CMS Final Rate +5.0% (roughly in-line; NOT a breach event)
    # April 2025: CMS Final Rate +5.06% vs ~6%+ elevated post-COVID trend
    #   Publication date: April 7, 2025
    # We use 2023 and 2024 as primary events; 2025 included with caveat
    known_events = [
        "2023-04-03",  # CMS 2024 MA rate notice: +3.32% vs ~4.5% trend
        "2024-04-01",  # CMS 2025 MA rate notice: +3.7% vs ~5.0% trend (V28 shock)
        "2025-04-07",  # CMS 2026 MA rate notice: +5.06% vs elevated trend
    ]
    event_dates = pd.to_datetime(known_events)

    try:
        px = load_prices(["HUM", "UNH", "SPY"], start="2022-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]
    hum_r = ret["HUM"]
    unh_r = ret["UNH"]

    hold_days = 65  # max hold per strategy spec

    # Pair P&L: short HUM (1 unit), long UNH (1 unit), equal notional
    # Daily pair return = -1 * hum_r + 1 * unh_r
    pair_r = -1.0 * hum_r + 1.0 * unh_r

    pnl = pd.Series(0.0, index=pair_r.index)
    evts = []

    for ed in event_dates:
        # Enter on next trading day after CMS publication
        mask = pair_r.index > ed
        if mask.sum() < hold_days:
            continue
        entry_idx = pair_r.index[mask][0]
        p = pair_r.index.get_loc(entry_idx)
        ep = min(p + hold_days, len(pair_r))
        seg = pair_r.iloc[p:ep]
        cr = float((1 + seg).prod() - 1)
        pnl.iloc[p:ep] += seg.values[:ep - p]

        # SPY over same window
        sp_mask = spy_r.index >= entry_idx
        sc = None
        if sp_mask.sum() >= hold_days:
            spi = spy_r.index.get_loc(spy_r.index[sp_mask][0])
            sc = float((1 + spy_r.iloc[spi:spi + hold_days]).prod() - 1)

        # HUM single-leg return (short side)
        hum_mask = hum_r.index >= entry_idx
        hum_cr = None
        if hum_mask.sum() >= hold_days:
            hi = hum_r.index.get_loc(hum_r.index[hum_mask][0])
            hum_cr = float((1 + hum_r.iloc[hi:hi + hold_days]).prod() - 1)

        evts.append({
            "cms_date": str(ed.date()),
            "entry_date": str(entry_idx.date()),
            "pair_return": round(cr, 4),
            "hum_return_raw": round(hum_cr, 4) if hum_cr is not None else None,
            "short_hum_return": round(-hum_cr, 4) if hum_cr is not None else None,
            "spy_return": round(sc, 4) if sc is not None else None,
        })

    if not evts:
        return mark_failed(sid, "no valid events after filtering")

    ip = pnl[pnl != 0]
    if len(ip) < 30:
        return mark_failed(sid, f"insufficient active trading days ({len(ip)})")

    m = compute_metrics(ip, benchmark=spy_r,
                        name="CMS MA Rate Below Trend → Short HUM / Long UNH")
    pr = [e["pair_return"] for e in evts]
    save_result(sid, m, extra={
        "rule": "Short HUM, long UNH (equal notional) for 65 trading days when CMS Final MA Rate < consensus medical cost trend by >50 bps",
        "mechanism": "HUM has ~85% MA earnings exposure vs UNH ~50%; when CMS rates undercut cost trends, HUM MLR compression is amplified, compressing margins disproportionately; pair trade isolates MA-concentration risk",
        "source": "CMS Final Rate Announcements (cms.gov); HUM/UNH 10-Q EDGAR; yfinance prices",
        "n_events": len(evts),
        "avg_pair_return": round(float(np.mean(pr)), 4),
        "pair_win_rate": round(float(np.mean([r > 0 for r in pr])), 4),
        "events": evts,
    })
    print(f"Done: {len(evts)} events, avg pair return={np.mean(pr):.2%}")


if __name__ == "__main__":
    main()
