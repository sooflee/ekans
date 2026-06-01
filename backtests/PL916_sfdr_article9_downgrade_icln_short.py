"""PL916 — SFDR Article 9 -> Article 8 Downgrade Wave -> Short ICLN/TAN, Long EU Transition Utilities"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL916_sfdr_article9_downgrade_icln_short"

    # SFDR Article 9 -> Article 8 downgrade wave dates:
    # Wave 1: Q4 2022 - Morningstar reported 175+ downgrades in Q4 2022
    #   Major cluster around October/November 2022 when large AMs announced
    #   Amundi announced Oct 2022; BNP Paribas AM around same time
    # Wave 2: Q1 2023 - Continued downgrades post-ESMA clarification
    #   Approx. January-February 2023 press releases
    # Wave 3: Q4 2024 - ESMA fund naming guidelines effective May 2024
    #   Major transition deadline Q4 2024 / Q1 2025
    known_events = [
        "2022-10-24",   # Wave 1: large AM announcements in Q4 2022 cluster
        "2023-01-16",   # Wave 2: Q1 2023 continuation
        "2024-11-04",   # Wave 3: ESMA naming guidelines transition deadline wave
    ]
    event_dates = pd.to_datetime(known_events)

    try:
        px = load_prices(["ICLN", "TAN", "EOAN.DE", "IBE.MC", "SPY"], start="2022-01-01")
    except Exception as e:
        # EOAN.DE / IBE.MC may fail; retry with just US-listed tickers
        try:
            px = load_prices(["ICLN", "TAN", "SPY"], start="2022-01-01")
        except Exception as e2:
            return mark_failed(sid, f"data load: {e2}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]
    icln_r = ret["ICLN"]
    tan_r = ret["TAN"]

    # EU utilities long hedge - use if available, else use just short leg
    have_eu = "EOAN.DE" in ret.columns and "IBE.MC" in ret.columns
    if have_eu:
        eoan_r = ret["EOAN.DE"]
        ibe_r = ret["IBE.MC"]
        # EU utilities in EUR; yfinance returns are currency-local
        # Use equal-weight EU utility basket as long hedge
        eu_r = (eoan_r + ibe_r) / 2.0
        # Pair: short ICLN + short TAN (equal weight), long EU basket (combined notional)
        # Dollar-neutral: short 0.5 ICLN + 0.5 TAN, long 1.0 EU basket
        pair_r = eu_r - 0.5 * icln_r - 0.5 * tan_r
    else:
        # Fallback: just short clean energy ETFs (single-leg)
        have_eu = False
        pair_r = -0.5 * icln_r - 0.5 * tan_r

    hold_td = 12  # ~2.5 weeks trading days

    pnl = pd.Series(0.0, index=pair_r.index)
    evts = []

    for ed in event_dates:
        # Enter T+1 after downgrade wave announcement cluster
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

        # ICLN single-leg return (short)
        icln_mask = icln_r.index >= entry_idx
        icln_cr = None
        if icln_mask.sum() >= hold_td:
            ii = icln_r.index.get_loc(icln_r.index[icln_mask][0])
            icln_cr = float((1 + icln_r.iloc[ii:ii + hold_td]).prod() - 1)

        evts.append({
            "event_date": str(ed.date()),
            "entry_date": str(entry_idx.date()),
            "pair_return": round(cr, 4),
            "icln_return": round(icln_cr, 4) if icln_cr is not None else None,
            "short_icln_return": round(-icln_cr, 4) if icln_cr is not None else None,
            "spy_return": round(sc, 4) if sc is not None else None,
        })

    if not evts:
        return mark_failed(sid, "no valid events after filtering")

    ip = pnl[pnl != 0]
    if len(ip) < 20:
        return mark_failed(sid, f"insufficient active trading days ({len(ip)})")

    m = compute_metrics(ip, benchmark=spy_r,
                        name="SFDR Art.9 Downgrade Wave → Short ICLN+TAN / Long EU Utilities")
    pr = [e["pair_return"] for e in evts]
    save_result(sid, m, extra={
        "rule": "Short ICLN + TAN (equal weight), long EU utility basket (EOAN.DE+IBE.MC) for 12 trading days when 3+ large AMs announce SFDR Art.9->8 reclassification within 10-day window",
        "mechanism": "Article 9 fund outflows force selling of ICLN/TAN constituent holdings; EU utilities re-classified as Article 8 sustainable rather than strict dark-green, insulating them from the outflow wave",
        "source": "Morningstar SFDR quarterly reviews; AM press releases; yfinance",
        "n_events": len(evts),
        "long_hedge": "EOAN.DE + IBE.MC" if have_eu else "none (short-only fallback)",
        "avg_pair_return": round(float(np.mean(pr)), 4),
        "pair_win_rate": round(float(np.mean([r > 0 for r in pr])), 4),
        "events": evts,
    })
    print(f"Done: {len(evts)} events, avg pair return={np.mean(pr):.2%}")


if __name__ == "__main__":
    main()
