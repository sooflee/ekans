"""PL661 — Boeing Book-to-Bill <1.0 x2 → Short HEI/TDG/HXL, Hedge Long XLI"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns

def main():
    sid = "PL661_boeing_btb_below_one_short_hei"
    # Known events: Boeing reported BTB<1.0 for 2 consecutive months
    # First business day after the release dates
    known_events = [
        pd.Timestamp("2024-04-09"),
        pd.Timestamp("2024-07-09"),
    ]

    try:
        px = load_prices(["HEI", "TDG", "HXL", "XLI", "SPY"], start="2020-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]
    hold = 60  # 60 trading days per spec

    # Short basket: HEI+TDG+HXL equal-weighted, hedge 50% long XLI
    basket_r = (ret["HEI"] + ret["TDG"] + ret["HXL"]) / 3.0  # short leg
    hedge_r = ret["XLI"]  # long leg hedge

    pnl = pd.Series(0.0, index=spy_r.index)
    evts = []

    for td in known_events:
        # Find next available trading day on or after trigger date
        mask = spy_r.index >= td
        if mask.sum() < hold:
            continue
        ei = spy_r.index[mask][0]
        p = spy_r.index.get_loc(ei)
        ep = min(p + hold, len(spy_r))

        # Strategy: short basket (negative sign) + 0.5 long XLI
        bask_slice = basket_r.reindex(spy_r.index).iloc[p:ep].fillna(0.0)
        xli_slice = hedge_r.reindex(spy_r.index).iloc[p:ep].fillna(0.0)
        strat_r = -bask_slice + 0.5 * xli_slice  # net notional 1.5 gross, 0.5 net

        pnl.iloc[p:ep] += strat_r.values[:ep - p]

        cr_basket = float((1 - basket_r.iloc[p:ep]).prod() - 1)  # short return
        cr_xli = float((1 + xli_slice).prod() - 1)
        cr_net = float((1 + strat_r).prod() - 1)
        sc = float((1 + spy_r.iloc[p:ep]).prod() - 1)
        evts.append({
            "trigger_date": str(td.date()),
            "entry_date": str(ei.date()),
            "basket_short_return": round(cr_basket, 4),
            "xli_hedge_return": round(cr_xli, 4),
            "net_return": round(cr_net, 4),
            "spy_return": round(sc, 4),
        })

    if not evts:
        return mark_failed(sid, "no valid events found")

    ip = pnl[pnl != 0]
    if len(ip) < 20:
        return mark_failed(sid, f"insufficient active days ({len(ip)})")

    m = compute_metrics(ip, benchmark=spy_r, name="Boeing BTB<1.0 Short HEI/TDG/HXL")
    ra = [e["net_return"] for e in evts]
    save_result(sid, m, extra={
        "rule": "Short HEI/TDG/HXL equal-weight, hedge 50% long XLI for 60 days when Boeing reports BTB<1.0 two consecutive months",
        "mechanism": "Boeing order weakness cascades to aerospace suppliers via reduced forward delivery commitments; XLI hedge neutralizes broad industrial beta",
        "source": "Boeing Investor Relations monthly orders/deliveries; yfinance",
        "n_events": len(evts),
        "avg_net_return": round(float(np.mean(ra)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in ra])), 4),
        "events": evts,
    })
    print(f"Done: {len(evts)} events, avg net return: {float(np.mean(ra)):.2%}")

if __name__ == "__main__":
    main()
