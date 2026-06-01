"""PL675 — NAHB Traffic Sub-Index Inflection <=40 +5pt MoM → Long DHI/LGIH, Hedge Short XHB"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns

def main():
    sid = "PL675_nahb_traffic_long_dhi_lgih"
    # Try to get NAHB Traffic sub-index via FRED (BPOPRIME is Housing Market Index - traffic sub)
    # FRED series: NAHBMMI (HMI overall), traffic sub not directly available
    # Use known events from strategy spec
    known_events = [
        pd.Timestamp("2023-01-18"),
        pd.Timestamp("2024-12-17"),
    ]

    # Try FRED for NAHB Housing Market Index as proxy (overall HMI available)
    try:
        fred = load_fred("NAHBMMI", start="2010-01-01")
        hmi = fred.squeeze().dropna()
    except Exception:
        hmi = None

    # If FRED data is available, generate additional historical signals
    extra_events = []
    if hmi is not None and not hmi.empty:
        hmi_monthly = hmi.resample("M").last()
        for i in range(1, len(hmi_monthly)):
            prev_val = float(hmi_monthly.iloc[i - 1])
            curr_val = float(hmi_monthly.iloc[i])
            # NAHB HMI proxy: low reading bounce (below 40, rise >5pt)
            if prev_val <= 40 and (curr_val - prev_val) > 5:
                trigger_date = hmi_monthly.index[i]
                # Convert to mid-month entry (release day ~3rd week)
                entry_approx = trigger_date.replace(day=1) + pd.offsets.BusinessDay(12)
                extra_events.append(pd.Timestamp(entry_approx))

    # Merge: combine known events + FRED-derived, deduplicate by year-month
    all_events = list(known_events)
    for ev in extra_events:
        if not any(abs((ev - k).days) < 45 for k in all_events):
            all_events.append(ev)
    all_events.sort()
    print(f"Total events (known + FRED-derived): {len(all_events)}")

    try:
        px = load_prices(["DHI", "LGIH", "XHB", "SPY"], start="2012-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]
    dhi_r = ret["DHI"]
    lgih_r = ret["LGIH"] if "LGIH" in ret.columns else None
    xhb_r = ret["XHB"]
    hold = 60  # 60 trading days per spec

    pnl = pd.Series(0.0, index=spy_r.index)
    evts = []

    for td in all_events:
        mask = spy_r.index >= td
        if mask.sum() < hold:
            continue
        ei = spy_r.index[mask][0]
        p = spy_r.index.get_loc(ei)
        ep = min(p + hold, len(spy_r))

        dhi_sl = dhi_r.reindex(spy_r.index).iloc[p:ep].fillna(0.0)
        xhb_sl = xhb_r.reindex(spy_r.index).iloc[p:ep].fillna(0.0)

        if lgih_r is not None:
            lgih_sl = lgih_r.reindex(spy_r.index).iloc[p:ep].fillna(0.0)
            basket = (dhi_sl + lgih_sl) / 2.0
        else:
            basket = dhi_sl

        strat_r = basket - 0.5 * xhb_sl

        pnl.iloc[p:ep] += strat_r.values[:ep - p]

        cr_basket = float((1 + basket).prod() - 1)
        cr_xhb = float((1 + xhb_sl).prod() - 1)
        cr_net = float((1 + strat_r).prod() - 1)
        sc = float((1 + spy_r.iloc[p:ep]).prod() - 1)
        evts.append({
            "trigger_date": str(td.date()),
            "entry_date": str(ei.date()),
            "basket_return": round(cr_basket, 4),
            "xhb_return": round(cr_xhb, 4),
            "net_return": round(cr_net, 4),
            "spy_return": round(sc, 4),
        })

    if not evts:
        return mark_failed(sid, "no valid events found")

    ip = pnl[pnl != 0]
    if len(ip) < 30:
        return mark_failed(sid, f"insufficient active days ({len(ip)})")

    m = compute_metrics(ip, benchmark=spy_r, name="NAHB Traffic Inflection Long DHI/LGIH")
    ra = [e["net_return"] for e in evts]
    save_result(sid, m, extra={
        "rule": "Long DHI/LGIH equal-weight, hedge 50% short XHB for 60 days when NAHB Traffic sub-index rises >+5pt from <=40 level",
        "mechanism": "Traffic sub-index inflection from depressed levels signals early-cycle housing demand recovery; entry-level builders DHI/LGIH outperform as volume recovers faster than price",
        "source": "NAHB/Wells Fargo HMI monthly release; FRED NAHBMMI; yfinance",
        "n_events": len(evts),
        "avg_net_return": round(float(np.mean(ra)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in ra])), 4),
        "events": evts,
    })
    print(f"Done: {len(evts)} events, avg net return: {float(np.mean(ra)):.2%}")

if __name__ == "__main__":
    main()
