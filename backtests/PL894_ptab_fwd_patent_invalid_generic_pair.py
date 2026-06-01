"""PL894_ptab_fwd_patent_invalid_generic_pair - USPTO PTAB FWD Invalidates Orange Book Patent -> Long Generic vs Short Brand

Event-study backtest on PTAB Final Written Decisions that invalidate Orange Book
pharma patents for drugs with >$500M US revenue. Long TEVA/AMRX/VTRS vs short
the brand sponsor, hold 90 calendar days (~63 trading days).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# Known PTAB FWD events for major pharma Orange Book patents
# Brand sponsor | FWD date | brand_tickers
EVENTS = [
    # (fwd_date, brand_tickers, drug_name)
    ("2016-12-15", ["AGN"],         "Restasis (cyclosporine)"),   # Allergan Restasis (pre-AbbVie acquisition)
    ("2017-10-05", ["AGN"],         "Restasis follow-up"),
    ("2018-12-19", ["PFE"],         "Lyrica (pregabalin)"),
    ("2019-06-14", ["PFE"],         "Lyrica follow-up"),
    ("2021-11-18", ["ABBV"],        "Humira (adalimumab) formulation"),
    ("2022-08-30", ["BMY", "PFE"],  "Eliquis (apixaban) polymorph"),
    ("2023-03-15", ["BMY", "PFE"],  "Eliquis dosing patents"),
]
HOLD_DAYS = 63  # ~90 calendar days


def main():
    sid = "PL894_ptab_fwd_patent_invalid_generic_pair"
    # Load all tickers
    brand_ticks = ["AGN", "PFE", "ABBV", "BMY"]
    generic_ticks = ["TEVA", "AMRX", "VTRS"]
    try:
        px = load_prices(brand_ticks + generic_ticks + ["SPY"], start="2015-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    ret = daily_returns(px)
    if "SPY" not in ret.columns:
        return mark_failed(sid, "SPY missing")
    spy_r = ret["SPY"]

    pnl_parts = []
    events_detail = []

    for fwd_date, brand_list, drug_name in EVENTS:
        td = pd.Timestamp(fwd_date)
        # Entry: T+1 after FWD date
        mask = ret.index > td
        if mask.sum() < 2:
            continue
        idxs = ret.index[mask]
        entry = idxs[0]
        loc = ret.index.get_loc(entry)
        end = min(loc + HOLD_DAYS, len(ret))
        if end - loc < 20:
            continue

        window = ret.iloc[loc:end]

        # Short brand(s)
        short_legs = []
        for bt in brand_list:
            if bt in window.columns and window[bt].notna().sum() > 10:
                short_legs.append(-window[bt])

        if not short_legs:
            continue

        short_basket = sum(s / len(short_legs) for s in short_legs)

        # Long generics: TEVA always; AMRX if post-May-2018; VTRS if post-Nov-2020
        long_tickers = ["TEVA"]
        if td >= pd.Timestamp("2018-05-01"):
            long_tickers.append("AMRX")
        if td >= pd.Timestamp("2020-11-01"):
            long_tickers.append("VTRS")

        long_legs = []
        for lt in long_tickers:
            if lt in window.columns and window[lt].notna().sum() > 10:
                long_legs.append(window[lt])

        if not long_legs:
            continue

        long_basket = sum(l / len(long_legs) for l in long_legs)

        # PnL = 0.5 long generics + 0.5 short brand (dollar-neutral pair)
        pnl_window = 0.5 * long_basket + 0.5 * short_basket

        pnl_parts.append(pnl_window)
        cumret = float((1 + pnl_window).prod() - 1)
        brand_ret = float((1 + short_basket).prod() - 1)  # short basket return (positive = brands fell)
        gen_ret = float((1 + long_basket).prod() - 1)

        events_detail.append({
            "fwd_date": fwd_date,
            "drug": drug_name,
            "brands": brand_list,
            "generics": long_tickers,
            "entry": str(entry.date()),
            "n_days": len(pnl_window),
            "net_return": round(cumret, 4),
            "brand_short_return": round(brand_ret, 4),
            "generic_long_return": round(gen_ret, 4),
        })

    if not events_detail:
        return mark_failed(sid, "no valid events after data availability check")

    df = pd.concat(pnl_parts, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()

    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")

    m = compute_metrics(pnl, benchmark=spy_r, name="PTAB FWD Generic vs Brand Pair")
    save_result(sid, m, extra={
        "rule": "T+1: long TEVA/AMRX/VTRS equal-weight + short brand sponsor equal-weight, hold 63 trading days (~90 calendar days), on PTAB FWD invalidating Orange Book patent for >$500M drug.",
        "mechanism": "PTAB invalidation removes patent exclusivity, enabling generic entry and compressing brand sponsor revenues. Generic makers benefit from new addressable market.",
        "source": "USPTO PTAB FWD records (petitions.uspto.gov); FDA Orange Book; yfinance prices",
        "n_events": len(events_detail),
        "events": events_detail,
        "avg_net_return": round(float(np.mean([e["net_return"] for e in events_detail])), 4),
        "event_win_rate": round(float(np.mean([e["net_return"] > 0 for e in events_detail])), 4),
    })

    sharpe = m.get("sharpe", 0)
    cagr = m.get("cagr", 0)
    print(f"Done: {len(events_detail)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%, n_days={len(pnl)}")
    for e in events_detail:
        print(f"  {e['fwd_date']} {e['drug']}: net={e['net_return']:+.3f} (brand_short={e['brand_short_return']:+.3f}, gen_long={e['generic_long_return']:+.3f})")


if __name__ == "__main__":
    main()
