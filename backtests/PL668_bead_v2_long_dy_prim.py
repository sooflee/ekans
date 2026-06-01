"""PL668 — NTIA BEAD Vol2 Approvals → Long DY/MTZ/PRIM Hedge Short XLI"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns

def main():
    sid = "PL668_bead_v2_long_dy_prim"
    # NTIA BEAD Vol2 Initial Proposal approvals for 3+ states in 30-day window
    known_events = [
        pd.Timestamp("2024-08-19"),
        pd.Timestamp("2024-12-01"),
    ]

    try:
        px = load_prices(["DY", "MTZ", "PRIM", "XLI", "SPY"], start="2023-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]
    dy_r = ret["DY"]
    mtz_r = ret["MTZ"]
    prim_r = ret["PRIM"]
    xli_r = ret["XLI"]
    hold = 90  # 90 trading days per spec

    pnl = pd.Series(0.0, index=spy_r.index)
    evts = []

    for td in known_events:
        mask = spy_r.index >= td
        if mask.sum() < hold:
            continue
        ei = spy_r.index[mask][0]
        p = spy_r.index.get_loc(ei)
        ep = min(p + hold, len(spy_r))

        dy_sl = dy_r.reindex(spy_r.index).iloc[p:ep].fillna(0.0)
        mtz_sl = mtz_r.reindex(spy_r.index).iloc[p:ep].fillna(0.0)
        prim_sl = prim_r.reindex(spy_r.index).iloc[p:ep].fillna(0.0)
        xli_sl = xli_r.reindex(spy_r.index).iloc[p:ep].fillna(0.0)

        # Long basket (1.0 notional) + short XLI (0.5 notional)
        basket = (dy_sl + mtz_sl + prim_sl) / 3.0
        strat_r = basket - 0.5 * xli_sl

        pnl.iloc[p:ep] += strat_r.values[:ep - p]

        cr_basket = float((1 + basket).prod() - 1)
        cr_xli = float((1 + xli_sl).prod() - 1)
        cr_net = float((1 + strat_r).prod() - 1)
        sc = float((1 + spy_r.iloc[p:ep]).prod() - 1)
        evts.append({
            "trigger_date": str(td.date()),
            "entry_date": str(ei.date()),
            "basket_return": round(cr_basket, 4),
            "xli_return": round(cr_xli, 4),
            "net_return": round(cr_net, 4),
            "spy_return": round(sc, 4),
        })

    if not evts:
        return mark_failed(sid, "no valid events found")

    ip = pnl[pnl != 0]
    if len(ip) < 20:
        return mark_failed(sid, f"insufficient active days ({len(ip)})")

    m = compute_metrics(ip, benchmark=spy_r, name="BEAD Vol2 Approval Long DY/MTZ/PRIM")
    ra = [e["net_return"] for e in evts]
    save_result(sid, m, extra={
        "rule": "Long DY/MTZ/PRIM equal-weight, hedge 50% short XLI for 90 days when NTIA approves BEAD Vol2 for 3+ states in 30-day window",
        "mechanism": "BEAD broadband infrastructure approvals directly fund broadband construction contracts, lifting fiber/telecom infrastructure contractors DY, MTZ, PRIM while reducing pure industrial beta via XLI hedge",
        "source": "NTIA BEAD program press releases; yfinance",
        "n_events": len(evts),
        "avg_net_return": round(float(np.mean(ra)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in ra])), 4),
        "events": evts,
    })
    print(f"Done: {len(evts)} events, avg net return: {float(np.mean(ra)):.2%}")

if __name__ == "__main__":
    main()
