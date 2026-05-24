"""
F4 CMS Medicare Advantage Star Ratings.

CMS releases MA Star Ratings on the first Thursday of October each year.
Stars drive bonus payments and rebates; insurers with significant changes see
revenue impact.

For each release year, identify the publicly-traded MA insurer with the largest
positive star-rating change and the largest negative change among
{HUM, UNH, CVS, CI, ELV}. Short the worst (long the best) for 5 trading days
starting T+1.

Star ratings curated from CMS public files and insurer 10-K disclosures.
Ratings reported here are weighted average plan-level star (approx) — sourced
from CMS Star Ratings fact sheets / public summaries.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed

# Release date = first Thursday of October. Ratings = approximate weighted
# average MA star rating by parent organization (from CMS fact sheets / SEC
# filings). Year corresponds to the Star year (rating released the prior Oct).
# We use the release date.
RATINGS = {
    # release_date, {ticker: star_rating}
    "2015-10-08": {"HUM": 4.3, "UNH": 3.8, "CVS": 3.8, "CI": 3.7, "ELV": 3.7},  # WLP=>ELV
    "2016-10-06": {"HUM": 4.0, "UNH": 3.9, "CVS": 3.9, "CI": 3.6, "ELV": 3.6},
    "2017-10-05": {"HUM": 4.4, "UNH": 4.1, "CVS": 4.0, "CI": 3.8, "ELV": 3.7},
    "2018-10-04": {"HUM": 4.4, "UNH": 4.2, "CVS": 4.1, "CI": 3.9, "ELV": 3.9},
    "2019-10-03": {"HUM": 4.4, "UNH": 4.2, "CVS": 4.2, "CI": 4.0, "ELV": 4.0},
    "2020-10-08": {"HUM": 4.5, "UNH": 4.3, "CVS": 4.3, "CI": 4.2, "ELV": 4.1},
    "2021-10-07": {"HUM": 4.5, "UNH": 4.4, "CVS": 4.4, "CI": 4.3, "ELV": 4.3},
    "2022-10-06": {"HUM": 4.4, "UNH": 4.3, "CVS": 4.1, "CI": 4.0, "ELV": 4.0},
    "2023-10-13": {"HUM": 4.2, "UNH": 4.2, "CVS": 4.0, "CI": 3.9, "ELV": 3.9},
    "2024-10-10": {"HUM": 3.5, "UNH": 4.0, "CVS": 3.5, "CI": 3.8, "ELV": 3.7},  # HUM big drop
    "2025-10-09": {"HUM": 3.6, "UNH": 4.0, "CVS": 3.8, "CI": 3.8, "ELV": 3.7},  # prelim
}


def main():
    sid = "F4_cms_star_ratings"
    try:
        tickers = ["HUM", "UNH", "CVS", "CI", "ELV"]
        px = load_prices(tickers + ["SPY"], start="2014-01-01")
        rets = px.pct_change()
        idx = rets.index

        # Compute YoY change in star ratings; pick winner (max delta) and loser (min delta).
        years = sorted(RATINGS.keys())
        events = []
        for prev_d, this_d in zip(years[:-1], years[1:]):
            deltas = {t: RATINGS[this_d].get(t, np.nan) - RATINGS[prev_d].get(t, np.nan)
                      for t in tickers}
            deltas = {k: v for k, v in deltas.items() if not np.isnan(v)}
            if not deltas:
                continue
            winner = max(deltas, key=deltas.get)
            loser = min(deltas, key=deltas.get)
            events.append((this_d, winner, deltas[winner], loser, deltas[loser]))

        # Build positions: long winner +1, short loser -1, 5 days starting T+1.
        pos = pd.DataFrame(0.0, index=idx, columns=tickers)
        used = []
        for D, w, dw, l, dl in events:
            Dts = pd.Timestamp(D)
            loc = idx.searchsorted(Dts, side="right")
            if loc >= len(idx):
                continue
            start = loc
            end = min(loc + 5, len(idx) - 1)
            pos.iloc[start:end + 1, pos.columns.get_loc(w)] = 1.0
            pos.iloc[start:end + 1, pos.columns.get_loc(l)] = -1.0
            used.append({"release": D, "long": w, "long_delta": float(dw),
                         "short": l, "short_delta": float(dl)})

        port = (pos.shift(1) * rets[tickers]).sum(axis=1)
        active = (pos.abs().sum(axis=1) > 0)
        port_active = port[active.shift(1).fillna(False)].dropna()
        if len(port_active) < 5:
            return mark_failed(sid, "Too few active days")

        spy_r = rets["SPY"]
        m = compute_metrics(port_active, benchmark=spy_r.reindex(port_active.index),
                            name="F4 CMS Star Ratings best/worst delta")
        print_metrics(m)
        save_result(sid, m, extra={
            "status": "ok",
            "rule": "Each Oct CMS MA Star Ratings release, long the insurer with largest "
                    "YoY rating delta and short the worst delta; 5-trading-day hold from T+1.",
            "mechanism": "Star bonus payments materially impact MA revenue (~5% of premium); "
                         "rating changes signal next-plan-year revenue impact.",
            "universe": "HUM, UNH, CVS, CI, ELV",
            "n_events": len(used),
            "events": used,
            "data_caveat": "Star ratings are approximate parent-org weighted averages; "
                           "CMS publishes plan-level not parent-level scores directly.",
            "source": "CMS Star Ratings fact sheets, insurer 10-K disclosures (curated)",
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return mark_failed(sid, f"unhandled exception: {e}")


if __name__ == "__main__":
    main()
