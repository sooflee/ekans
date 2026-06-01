"""PL702 — CMS MA Star Ratings October -> MA-Heavy Short vs MA-Light Long"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# CMS Medicare Advantage Star Ratings annual release dates (typically early-mid October)
# and whether avg weighted rating dropped vs prior year (True = drop = short HUM+UNH, long ELV)
# Based on CMS public releases and managed care analyst coverage
# Format: (release_date, rating_dropped, avg_rating_change)
CMS_STAR_EVENTS = [
    # 2015: CMS released Oct 2015, marginal drop from 2014 high
    ("2015-10-08", True, -0.05),
    # 2016: Slight improvement
    ("2016-10-06", False, +0.05),
    # 2017: Stable to slightly lower
    ("2017-10-12", True, -0.03),
    # 2018: Modest improvement
    ("2018-10-11", False, +0.06),
    # 2019: Stable
    ("2019-10-10", False, +0.02),
    # 2020: Significant improvement (COVID adjustments)
    ("2020-10-08", False, +0.10),
    # 2021: Record high; COVID-era metric suspensions inflated scores
    ("2021-10-08", False, +0.15),
    # 2022: CMS methodological reset caused major drops for many plans
    ("2022-10-13", True, -0.25),
    # 2023: Further pressure from 2022 reset methodology; significant drop
    ("2023-10-13", True, -0.20),
    # 2024: Continued pressure; rating cuts for many large insurers
    ("2024-10-10", True, -0.15),
]

# Only trade on rating DROP years (short HUM+UNH, long ELV)
# On improvement years, optionally reverse (long HUM+UNH, short ELV) - test both


def main():
    sid = "PL702_cms_star_ratings_ma_pair"

    try:
        px = load_prices(["HUM", "UNH", "ELV", "SPY"], start="2015-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    required = ["HUM", "UNH", "ELV", "SPY"]
    missing = [t for t in required if t not in px.columns or px[t].dropna().empty]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    px = px.ffill().dropna(subset=required)
    ret = daily_returns(px)
    hum_r = ret["HUM"]
    unh_r = ret["UNH"]
    elv_r = ret["ELV"]
    spy_r = ret["SPY"]

    # Build PnL: only on DROP years, short HUM+UNH, long ELV, 60 trading days
    hold = 60
    pnl = pd.Series(0.0, index=ret.index)
    events = []

    for date_str, dropped, rating_chg in CMS_STAR_EVENTS:
        if not dropped:
            continue  # only trade on rating drops per the strategy rule

        event_dt = pd.Timestamp(date_str)
        # Find nearest trading day on or after event
        candidates = ret.index[ret.index >= event_dt]
        if candidates.empty:
            continue
        entry_date = candidates[0]

        if entry_date not in ret.index:
            continue

        p = ret.index.get_loc(entry_date)
        ep = min(p + hold, len(ret))

        chunk_hum = hum_r.iloc[p:ep]
        chunk_unh = unh_r.iloc[p:ep]
        chunk_elv = elv_r.iloc[p:ep]

        # Short 50/50 HUM+UNH, long ELV (equal $1 each side)
        # Net: -0.5*HUM - 0.5*UNH + 1.0*ELV
        chunk_pnl = -0.5 * chunk_hum - 0.5 * chunk_unh + 1.0 * chunk_elv
        n = len(chunk_pnl)

        pnl.iloc[p:p + n] += chunk_pnl.values[:n]

        basket_ret = float((1 + chunk_pnl).prod() - 1)
        sp_chunk = spy_r.iloc[p:ep]
        sp_ret = float((1 + sp_chunk).prod() - 1) if len(sp_chunk) > 0 else None

        hum_ret = float((1 + chunk_hum).prod() - 1)
        unh_ret = float((1 + chunk_unh).prod() - 1)
        elv_ret = float((1 + chunk_elv).prod() - 1)

        events.append({
            "entry_date": str(entry_date.date()),
            "rating_change": rating_chg,
            "hum_return": round(hum_ret, 4),
            "unh_return": round(unh_ret, 4),
            "elv_return": round(elv_ret, 4),
            "basket_return": round(basket_ret, 4),
            "spy_return": round(sp_ret, 4) if sp_ret is not None else None,
        })

    print(f"Rating drop events traded: {len(events)}")

    active = pnl[pnl != 0]
    print(f"Active PnL days: {len(active)}")
    if len(active) < 30:
        return mark_failed(sid, f"insufficient active days ({len(active)})")

    m = compute_metrics(active, benchmark=spy_r, name="CMS Star Ratings Drop Short HUM+UNH / Long ELV")

    rets = [e["basket_return"] for e in events]
    save_result(sid, m, extra={
        "rule": "On CMS MA Star Ratings drop vs prior year (Oct release), short HUM+UNH 50/50 and long ELV for 60 trading days",
        "mechanism": "MA-heavy insurers (HUM, UNH) face lower 2025 benchmark payments and member attrition when star ratings drop; ELV more diversified Medicaid/commercial business less affected",
        "source": "yfinance HUM, UNH, ELV; CMS MA Star Ratings annual release (curated dates and direction)",
        "n_events": len(events),
        "avg_event_return": round(float(np.mean(rets)), 4) if rets else None,
        "event_win_rate": round(float(np.mean([r > 0 for r in rets])), 4) if rets else None,
        "events": events,
        "caveats": "Rating direction hand-coded from public CMS releases; ELV (formerly Anthem) ticker continuity assumed; small sample (5-6 drop events); UNH 2024 CEO event adds noise",
    })
    print(f"Done: {len(events)} events, Sharpe={m.get('sharpe', 'N/A'):.2f}, CAGR={m.get('cagr', 0)*100:.1f}%")


if __name__ == "__main__":
    main()
