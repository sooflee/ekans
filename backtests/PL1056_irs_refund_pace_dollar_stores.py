"""PL1056_irs_refund_pace_dollar_stores — IRS Daily Treasury Statement Refund Pace YoY
-> Dollar Store Q1 Comps (DG/DLTR/FIVE).

Each year: sum DTS individual income-tax refund daily disbursements Feb 1-28
(truncate at day 28 in leap years for symmetry), compute YoY gap vs prior Feb.
GAP >= +5% -> LONG equal-weight dollar-store basket; GAP <= -5% -> SHORT;
else flat. Hold first trading day of March -> last trading day of May,
beta-hedged with SPY (trailing 252d basket beta as of end of Feb).
Basket membership point-in-time: DLTR (all years), DG from 2010, FIVE from 2013.
"""
import sys
import json
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics, save_result, mark_failed,
                     daily_returns, print_metrics, DATA)

SID = "PL1056_irs_refund_pace_dollar_stores"
API = ("https://api.fiscaldata.treasury.gov/services/api/fiscal_service/"
       "v1/accounting/dts/income_tax_refunds_issued")
THRESH = 0.05
FIRST_SIGNAL_YEAR = 2007   # needs Feb 2006 baseline (DTS starts 2005-10-03)
LAST_SIGNAL_YEAR = 2026


def fetch_feb_individual_refunds():
    """Return {year: cumulative individual refunds ($MM) Feb 1-28}.

    Sums tax_refund_today_amt over all rows whose tax_refund_type contains
    'individual' (case-insensitive), Feb 1-28 of each year (day 29 excluded
    for leap-year symmetry). Cached to data/.
    """
    cache = DATA / "PL1056_dts_feb_individual_refunds.json"
    if cache.exists():
        with open(cache) as f:
            return {int(k): v for k, v in json.load(f).items()}

    import requests
    out = {}
    for yr in range(FIRST_SIGNAL_YEAR - 1, LAST_SIGNAL_YEAR + 1):
        url = (f"{API}?filter=record_date:gte:{yr}-02-01,"
               f"record_date:lte:{yr}-02-28&page[size]=900")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        rows = r.json()["data"]
        total = 0.0
        n = 0
        labels = set()
        for row in rows:
            if "individual" in (row.get("tax_refund_type") or "").lower():
                amt = row.get("tax_refund_today_amt")
                if amt is not None and amt not in ("null", ""):
                    total += float(amt)
                    n += 1
                    labels.add(row["tax_refund_type"])
        out[yr] = total
        print(f"  Feb {yr}: cum individual refunds ${total:,.0f}MM "
              f"({n} rows; labels={sorted(labels)})")
        time.sleep(0.3)

    with open(cache, "w") as f:
        json.dump(out, f, indent=2)
    return out


def main():
    try:
        refunds = fetch_feb_individual_refunds()
    except Exception as e:
        return mark_failed(SID, f"DTS API fetch: {e}")

    try:
        px = load_prices(["DLTR", "DG", "FIVE", "SPY"], start="2004-01-01")
    except Exception as e:
        return mark_failed(SID, f"price load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    events = []
    pnl_parts = []
    pos_parts = []

    for yr in range(FIRST_SIGNAL_YEAR, LAST_SIGNAL_YEAR + 1):
        cur, prev = refunds.get(yr), refunds.get(yr - 1)
        if not cur or not prev:
            continue
        gap = cur / prev - 1.0

        if gap >= THRESH:
            direction = 1
        elif gap <= -THRESH:
            direction = -1
        else:
            direction = 0

        # Point-in-time basket
        basket = ["DLTR"]
        if yr >= 2010:
            basket.append("DG")
        if yr >= 2013:
            basket.append("FIVE")
        basket = [t for t in basket if t in ret.columns]

        # Mar 1 -> last trading day of May window
        idx = spy_r.index
        win = idx[(idx >= f"{yr}-03-01") & (idx <= f"{yr}-05-31")]
        if len(win) < 40:
            continue

        basket_r_all = ret[basket].mean(axis=1)

        # Trailing 252d basket beta vs SPY as of last trading day before window
        hist_end_loc = idx.get_loc(win[0])
        hist = pd.DataFrame({
            "b": basket_r_all.reindex(idx),
            "s": spy_r,
        }).iloc[max(0, hist_end_loc - 252):hist_end_loc].dropna()
        if len(hist) >= 120 and hist["s"].var() > 0:
            beta = float(hist["b"].cov(hist["s"]) / hist["s"].var())
        else:
            beta = 1.0

        # Enter at close of first trading day of March (signal known Feb 28,
        # no lookahead); PnL accrues from the 2nd window day through May close.
        hold = win[1:]
        b_r = basket_r_all.reindex(hold).fillna(0.0)
        s_r = spy_r.reindex(hold).fillna(0.0)
        ev_pnl = direction * (b_r - beta * s_r)

        ev_ret = float((1 + ev_pnl).prod() - 1) if direction != 0 else 0.0
        b_cum = float((1 + b_r).prod() - 1)
        s_cum = float((1 + s_r).prod() - 1)

        events.append({
            "year": yr,
            "feb_cum_refunds_mm": round(cur, 1),
            "prior_feb_cum_refunds_mm": round(prev, 1),
            "yoy_gap": round(gap, 4),
            "direction": direction,
            "basket": basket,
            "beta": round(beta, 3),
            "entry": str(win[0].date()),
            "exit": str(hold[-1].date()),
            "basket_return": round(b_cum, 4),
            "spy_return": round(s_cum, 4),
            "hedged_event_return": round(ev_ret, 4),
        })
        print(f"  {yr}: gap {gap*+100:+6.1f}% dir {direction:+d} "
              f"basket {','.join(basket):<12} beta {beta:.2f} "
              f"basket {b_cum*100:+6.1f}% SPY {s_cum*100:+6.1f}% "
              f"hedged {ev_ret*100:+6.1f}%")

        if direction != 0:
            pnl_parts.append(ev_pnl)
            pos_parts.append(pd.Series(float(direction), index=hold))

    if not pnl_parts:
        return mark_failed(SID, "no triggered years")

    pnl = pd.concat(pnl_parts).sort_index()
    positions = pd.concat(pos_parts).sort_index()

    traded = [e for e in events if e["direction"] != 0]
    ev_rets = [e["hedged_event_return"] for e in traded]
    ev_rets_ex2020 = [e["hedged_event_return"] for e in traded if e["year"] != 2020]

    print(f"\nTriggered years: {len(traded)}/{len(events)}  "
          f"active days: {len(pnl)}")
    print(f"Mean hedged event return: {np.mean(ev_rets)*100:+.2f}%  "
          f"hit rate: {np.mean([r > 0 for r in ev_rets])*100:.0f}%")
    if ev_rets_ex2020 and len(ev_rets_ex2020) < len(ev_rets):
        print(f"Ex-2020: mean {np.mean(ev_rets_ex2020)*100:+.2f}%  "
              f"hit rate: {np.mean([r > 0 for r in ev_rets_ex2020])*100:.0f}%")

    m = compute_metrics(pnl, benchmark=spy_r, positions=positions,
                        name="IRS Feb Refund Pace YoY -> Dollar Stores Mar-May")

    save_result(SID, m, extra={
        "rule": ("Sum DTS individual income-tax refunds Feb 1-28; if cumulative "
                 ">= +5% YoY go LONG equal-weight DLTR/DG/FIVE (point-in-time "
                 "membership) at first March close, <= -5% go SHORT, else flat; "
                 "beta-hedged with SPY (trailing 252d); exit last May close."),
        "mechanism": ("Tax refunds are the largest discretionary cash infusion "
                      "for low-income households; refund pace known by Feb 28 "
                      "leads dollar-store Q1 comps and late-May earnings "
                      "reactions (e.g. 2017 PATH Act delay -> DG/DLTR Q1 comp "
                      "misses)."),
        "source": ("Treasury Fiscal Data API v1/accounting/dts/"
                   "income_tax_refunds_issued; yfinance DLTR/DG/FIVE/SPY"),
        "n_events": len(traded),
        "avg_event_return": round(float(np.mean(ev_rets)), 4),
        "event_hit_rate": round(float(np.mean([r > 0 for r in ev_rets])), 4),
        "avg_event_return_ex2020": (round(float(np.mean(ev_rets_ex2020)), 4)
                                    if ev_rets_ex2020 else None),
        "caveats": ("2020 Mar-May window overlaps COVID crash; DTS refund-type "
                    "labels change across vintages (matched case-insensitive "
                    "'individual'); one observation per year."),
        "events": events,
    }, pnl=pnl)
    print_metrics(m)


if __name__ == "__main__":
    main()
