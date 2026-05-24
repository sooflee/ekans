"""
X7 Royalty/streaming deal announcement drift.

Hypothesis: WPM and FNV stream/royalty acquisitions > $200M tend to be value-
accretive (high IRR on long-life assets). Market under-reacts on day 1, drift
positive over ~45 trading days.

Curated >$200M deal announcement dates from WPM and FNV press releases
(2018-2025). Long the announcer at t+1 close, hold 45 trading days, benchmark
vs SPY.

Sources:
- WPM news: https://www.wheatonpm.com/news/
- FNV news: https://www.franco-nevada.com/news-events/
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, save_result, mark_failed

# Curated > $200M stream/royalty acquisitions from WPM and FNV press releases.
# (ticker, announcement_date, deal_value_usd_m, asset, note)
DEALS = [
    # Wheaton Precious Metals (WPM)
    ("WPM", "2018-10-25",  290, "Stillwater PGM stream"),
    ("WPM", "2020-06-23",  300, "Hudbay Rosemont silver stream upsize"),
    ("WPM", "2021-12-09",  340, "Marmato Caldas gold stream (Aris Mining)"),
    ("WPM", "2022-02-22",  300, "Curipamba silver/gold stream (Adventus/Salazar)"),
    ("WPM", "2023-07-31",  475, "Mineral Park stream restructure + Platreef"),
    ("WPM", "2024-02-20",  350, "Goose project gold stream (B2Gold)"),
    ("WPM", "2024-08-12",  600, "Koné gold stream (Montage Gold)"),
    # Franco-Nevada (FNV)
    ("FNV", "2018-12-14",  300, "Continental Resources royalty"),
    ("FNV", "2019-06-07",  300, "Cobre Panama 1% royalty top-up"),
    ("FNV", "2020-01-27",  300, "Haynesville natgas royalties"),
    ("FNV", "2020-09-07",  538, "Vale debenture exchange"),
    ("FNV", "2021-10-12",  225, "Condestable copper stream (Pan American)"),
    ("FNV", "2022-05-23",  250, "G Mining Tocantinzinho gold stream"),
    ("FNV", "2023-06-26",  340, "Solaris Warintza copper royalty"),
    ("FNV", "2024-03-25",  350, "Greenstone gold stream upsize"),
]


def main():
    sid = "X7_streaming_deal_drift"
    tickers = sorted(set([d[0] for d in DEALS] + ["SPY", "GDX", "GLD"]))
    try:
        px = load_prices(tickers, start="2017-01-01", end="2025-12-31")
    except Exception as e:
        return mark_failed(sid, f"price load failed: {e}")

    if "SPY" not in px.columns:
        return mark_failed(sid, "SPY missing")

    spy = px["SPY"].dropna()
    gdx = px.get("GDX", spy).dropna()
    gld = px.get("GLD", spy).dropna()

    hold = 45
    per_event = []
    pnl_dates = {}
    for tkr, ddate, val_m, asset in DEALS:
        if tkr not in px.columns:
            continue
        s = px[tkr].dropna()
        d = pd.Timestamp(ddate)
        idx_buy = s.index[s.index >= d]
        if len(idx_buy) < 2:
            continue
        t0 = idx_buy[0]
        i0 = s.index.get_loc(t0)
        i_entry = i0 + 1
        i_exit = i_entry + hold
        if i_exit >= len(s):
            continue
        stk_ret = float(s.iloc[i_exit] / s.iloc[i_entry] - 1)
        spy_w = spy.loc[s.index[i_entry]:s.index[i_exit]]
        gdx_w = gdx.loc[s.index[i_entry]:s.index[i_exit]]
        gld_w = gld.loc[s.index[i_entry]:s.index[i_exit]]
        spy_r = float(spy_w.iloc[-1] / spy_w.iloc[0] - 1) if len(spy_w) >= 2 else 0.0
        gdx_r = float(gdx_w.iloc[-1] / gdx_w.iloc[0] - 1) if len(gdx_w) >= 2 else 0.0
        gld_r = float(gld_w.iloc[-1] / gld_w.iloc[0] - 1) if len(gld_w) >= 2 else 0.0
        per_event.append({
            "ticker": tkr, "announce": ddate, "deal_usd_m": val_m, "asset": asset,
            "stk_45d": stk_ret, "spy_45d": spy_r, "excess_vs_spy": stk_ret - spy_r,
            "excess_vs_gdx": stk_ret - gdx_r, "excess_vs_gld": stk_ret - gld_r,
        })

    if len(per_event) < 5:
        return mark_failed(sid, f"too few events ({len(per_event)})",
                           extra={"n_events": len(per_event)})

    def stats(arr):
        a = np.array(arr)
        mean = float(a.mean()); std = float(a.std(ddof=1)) if len(a)>1 else 0.0
        t = mean / (std/np.sqrt(len(a))) if std>0 else 0.0
        return {"n": len(a), "mean": mean, "median": float(np.median(a)),
                "std": std, "t_stat": float(t), "hit_rate": float((a > 0).mean()),
                "min": float(a.min()), "max": float(a.max())}

    result = {
        "name": "X7 streaming deal drift 45d",
        "n_events": len(per_event),
        "abs_45d":          stats([e["stk_45d"] for e in per_event]),
        "excess_vs_spy":    stats([e["excess_vs_spy"] for e in per_event]),
        "excess_vs_gdx":    stats([e["excess_vs_gdx"] for e in per_event]),
        "excess_vs_gld":    stats([e["excess_vs_gld"] for e in per_event]),
    }
    print(f"X7: n={len(per_event)}, abs-mean={result['abs_45d']['mean']*100:+.2f}%, "
          f"vs-SPY mean={result['excess_vs_spy']['mean']*100:+.2f}% t={result['excess_vs_spy']['t_stat']:.2f}, "
          f"vs-GDX={result['excess_vs_gdx']['mean']*100:+.2f}% t={result['excess_vs_gdx']['t_stat']:.2f}")
    save_result(sid, result, extra={
        "status": "ok",
        "rule": "Long announcer (WPM or FNV) at t+1 close after deal announcement > $200M, hold 45 trading days.",
        "mechanism": "Accretive long-life royalty/stream deals; market under-reacts day 1 -> drift up over 45d.",
        "source": "Curated WPM and FNV press releases 2018-2024.",
        "events": per_event,
        "universe": ["WPM", "FNV"],
        "hold_days": hold,
    })


if __name__ == "__main__":
    main()
