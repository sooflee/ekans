"""PL103 — RFI/GDP Crosses 4.0% from Below → Long Homebuilders 12mo"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns

def main():
    sid = "PL103_rfi_gdp_share_homebuilders"
    try:
        fred = load_fred("A011RE1Q156NBEA", start="1990-01-01")
        data = fred.squeeze()
    except Exception as e:
        return mark_failed(sid, f"FRED: {e}")
    if data.empty:
        return mark_failed(sid, "no data")
    triggers = []
    below_count = 0
    fired = False
    for i in range(len(data)):
        v = float(data.iloc[i])
        if np.isnan(v): continue
        if v < 4.0:
            below_count += 1
            fired = False
        elif v >= 4.0 and below_count >= 8 and not fired:
            triggers.append(data.index[i])
            fired = True
            below_count = 0
        elif v >= 4.0:
            below_count = 0
    print(f"Events: {len(triggers)}")
    if not triggers:
        return mark_failed(sid, "no events")
    try:
        px = load_prices(["LEN", "DHI", "TOL", "SPY"], start="1998-01-01")
    except Exception as e:
        return mark_failed(sid, f"price: {e}")
    ret = daily_returns(px)
    available = [t for t in ["LEN", "DHI", "TOL"] if t in ret.columns]
    if not available:
        return mark_failed(sid, "no tickers")
    basket_r = ret[available].mean(axis=1)
    spy_r = ret["SPY"]
    hold = 252
    pnl = pd.Series(0.0, index=basket_r.index)
    evts = []
    for td in triggers:
        # Entry at start of next quarter
        ed = td + pd.offsets.QuarterBegin(1)
        mask = basket_r.index >= ed
        if mask.sum() < hold: continue
        ei = basket_r.index[mask][0]
        p = basket_r.index.get_loc(ei)
        ep = min(p + hold, len(basket_r))
        er = basket_r.iloc[p:ep]
        cr = float((1 + er).prod() - 1)
        pnl.iloc[p:ep] = er.values[:ep - p]
        sc = None
        if ei in spy_r.index:
            sp = spy_r.index.get_loc(ei)
            sc = float((1 + spy_r.iloc[sp:min(sp + hold, len(spy_r))]).prod() - 1)
        evts.append({"trigger_date": str(td.date()), "rfi_gdp": round(float(data.loc[td]), 2),
                     "basket_return": round(cr, 4), "spy_return": round(sc, 4) if sc else None})
    if not evts:
        return mark_failed(sid, "no valid events")
    ip = pnl[pnl != 0]
    if len(ip) < 30:
        return mark_failed(sid, f"insufficient days ({len(ip)})")
    m = compute_metrics(ip, benchmark=spy_r, name="RFI/GDP Cross → Long Homebuilders")
    ra = [e["basket_return"] for e in evts]
    save_result(sid, m, extra={
        "rule": "Long LEN+DHI+TOL 252d when A011RE1Q156NBEA crosses above 4.0 after 8+ quarters below",
        "source": "FRED A011RE1Q156NBEA; yfinance",
        "n_events": len(evts),
        "avg_event_return": round(float(np.mean(ra)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in ra])), 4),
        "events": evts,
    })
    print(f"Done: {len(evts)} events")

if __name__ == "__main__":
    main()
