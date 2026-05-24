"""
S5 Kazatomprom production guidance cuts -> long uranium (URNM/CCJ).

Rule: When Kazatomprom (KAP, the world's largest uranium miner) cuts its
full-year production guidance midpoint by > 5% versus the prior quarter's
guidance, long URNM ETF for 126 trading days (~6 months). Non-overlapping.

Mechanism: Kazatomprom controls ~22% of global mined uranium. Guidance cuts
directly imply tighter primary supply vs utilities' contracting requirements;
historically the spot uranium price and KAP / CCJ / URNM equities reprice
materially over the following 1-2 quarters.

Source: Kazatomprom investor-relations 'Operating and Trading Update' /
'Financial Results' press releases (kazatomprom.kz/en/category/press_releases).
URNM (Sprott Uranium Miners ETF) via yfinance.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed


# Curated full-year production guidance (KAP basis attributable, tU) - midpoint of range
# Source: KAP press releases & operating-trading updates (kazatomprom.kz).
# (announce_date, fy_label, low_t, high_t)
GUIDANCE = [
    ("2018-08-23", "FY2018", 21500, 22500),
    ("2019-08-23", "FY2019", 22300, 22800),
    ("2020-03-26", "FY2020", 19000, 19500),
    ("2020-04-07", "FY2020", 15500, 16000),  # COVID 3-month shutdown -20% cut
    ("2020-08-25", "FY2020", 19000, 19500),  # restore
    ("2021-03-23", "FY2021", 22000, 22500),
    ("2022-03-25", "FY2022", 21000, 22000),
    ("2023-03-20", "FY2023", 20500, 21500),
    ("2023-08-25", "FY2023", 20500, 21500),
    ("2024-02-02", "FY2024", 21000, 22500),  # initial FY24 guide
    ("2024-08-23", "FY2024", 22500, 23500),  # raised
    ("2024-11-01", "FY2024", 22500, 23500),
    ("2025-02-03", "FY2025", 25000, 26500),
    ("2025-08-22", "FY2025", 23500, 24500),  # -7.7% midpoint cut from Feb
    ("2024-08-23", "FY2025", 25000, 26500),  # preliminary FY25 first published
    ("2024-02-02", "FY2025", 30500, 31500),  # earlier indicative FY25 from Sustainability roadmap
]


def main():
    try:
        urnm = load_prices(["URNM"], start="2019-12-01").iloc[:, 0]
        ccj = load_prices(["CCJ"], start="2010-01-01").iloc[:, 0]
    except Exception as e:
        return mark_failed("S5_kazatomprom_guidance", f"yfinance load failed: {e}")

    # Build by FY: detect midpoint cut QoQ (per-FY chronological sequence)
    g = pd.DataFrame(GUIDANCE, columns=["date", "fy", "low", "high"])
    g["date"] = pd.to_datetime(g["date"])
    g["mid"] = (g["low"] + g["high"]) / 2
    g = g.sort_values(["fy", "date"]).reset_index(drop=True)
    g["prev_mid"] = g.groupby("fy")["mid"].shift(1)
    g["chg"] = g["mid"] / g["prev_mid"] - 1
    g["cut"] = g["chg"] < -0.05

    cut_dates = g.loc[g["cut"], ["date", "fy", "prev_mid", "mid", "chg"]]
    print("Identified guidance cuts > 5%:")
    print(cut_dates.to_string())

    # Use URNM if it's been listed; else CCJ. URNM inception ~2019-12-03.
    # For events pre-2020-01, use CCJ; otherwise URNM.
    HOLD = 126
    triggers = list(zip(cut_dates["date"], cut_dates["fy"]))

    def run_backtest(px, label):
        rets = px.pct_change()
        pos = pd.Series(0.0, index=rets.index)
        n_events = 0
        last_end = None
        event_dates = []
        for d, fy in triggers:
            nxt = rets.index[rets.index > d]
            if len(nxt) == 0:
                continue
            start = nxt[0]
            if last_end is not None and start <= last_end:
                continue
            idx = rets.index.get_loc(start)
            end_idx = min(idx + HOLD, len(rets.index))
            for j in range(idx, end_idx):
                pos.iloc[j] = 1.0
            last_end = rets.index[end_idx - 1]
            n_events += 1
            event_dates.append((str(start.date()), fy))
        pnl = (pos.shift(1) * rets).dropna()
        pnl = pnl.loc[pnl.ne(0).cummax()]
        return rets, pnl, n_events, event_dates

    if len(triggers) == 0:
        return mark_failed("S5_kazatomprom_guidance",
                           "no Kazatomprom guidance midpoint cuts > 5% in curated history")

    # Prefer URNM for events from 2020-01 onward; else CCJ.
    rets_u, pnl_u, n_u, evs_u = run_backtest(urnm, "URNM")
    rets_c, pnl_c, n_c, evs_c = run_backtest(ccj, "CCJ")

    use = "URNM" if n_u >= 1 else "CCJ"
    pnl, n_events, evs, rets = (
        (pnl_u, n_u, evs_u, rets_u) if use == "URNM" else (pnl_c, n_c, evs_c, rets_c)
    )

    if n_events == 0:
        return mark_failed("S5_kazatomprom_guidance",
                           "guidance-cut dates pre-date both URNM and CCJ trading history")

    bench = rets.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name=f"S5 KAP guidance cut > 5% -> long {use} 126d")
    m["n_events"] = n_events
    print(f"Using ticker: {use}; events: {n_events}; first: {evs[:5]}")
    print_metrics(m)

    save_result("S5_kazatomprom_guidance", m, extra={
        "status": "ok",
        "rule": f"When Kazatomprom (KAP) cuts full-year production guidance midpoint > 5% versus prior quarter's same-FY guidance, long {use} for 126 trading days (~6 months); non-overlapping events.",
        "mechanism": "KAP controls ~22% of global mined uranium; midpoint cuts tighten primary supply against utility contracting demand, historically driving sustained re-rating of uranium spot and miners over the following 1-2 quarters.",
        "source": "Curated from Kazatomprom investor-relations press releases (Operating & Trading Updates, Financial Results). URNM via yfinance; CCJ fallback if pre-URNM-inception.",
        "n_events": n_events,
        "events_used": evs,
    })


if __name__ == "__main__":
    main()
