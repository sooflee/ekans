"""PL90 — Henry Hub Below $2.50 → Long EQT+AR+RRC 12mo"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns
def main():
    sid = "PL90_henry_hub_below_cash_cost_producers"
    try: fred = load_fred("MHHNGSP", start="1997-01-01"); data = fred.squeeze()
    except Exception as e: return mark_failed(sid, f"FRED: {e}")
    if data.empty: return mark_failed(sid, "no data")
    triggers = []; prev = False
    for i in range(len(data)):
        v = float(data.iloc[i])
        if np.isnan(v): continue
        if v < 2.50 and not prev:
            if not triggers or (data.index[i] - triggers[-1]).days > 365: triggers.append(data.index[i])
            prev = True
        elif v >= 3.0: prev = False
    print(f"Events: {len(triggers)}")
    if not triggers: return mark_failed(sid, "no events")
    try: px = load_prices(["EQT","AR","RRC","SPY"], start="2005-01-01")
    except Exception as e: return mark_failed(sid, f"price: {e}")
    ret = daily_returns(px)
    available = [t for t in ["EQT","AR","RRC"] if t in ret.columns]
    if not available: return mark_failed(sid, "no tickers")
    basket_r = ret[available].mean(axis=1); spy_r = ret["SPY"]; hold = 252
    pnl = pd.Series(0.0, index=basket_r.index); evts = []
    for td in triggers:
        ed = td + pd.offsets.MonthBegin(1); mask = basket_r.index >= ed
        if mask.sum() < hold: continue
        ei = basket_r.index[mask][0]; p = basket_r.index.get_loc(ei); ep = min(p+hold, len(basket_r))
        er = basket_r.iloc[p:ep]; cr = float((1+er).prod()-1); pnl.iloc[p:ep] = er.values[:ep-p]
        sc = None
        if ei in spy_r.index: sp=spy_r.index.get_loc(ei); sc=float((1+spy_r.iloc[sp:min(sp+hold,len(spy_r))]).prod()-1)
        evts.append({"trigger_date":str(td.date()),"hh_price":round(float(data.loc[td]),2),
            "basket_return":round(cr,4),"spy_return":round(sc,4) if sc else None})
    if not evts: return mark_failed(sid, "no valid events")
    ip = pnl[pnl!=0]
    if len(ip)<30: return mark_failed(sid, f"insufficient days ({len(ip)})")
    m = compute_metrics(ip, benchmark=spy_r, name="HH Below Cash Cost → Long NatGas Producers")
    ra = [e["basket_return"] for e in evts]
    save_result(sid, m, extra={"rule":"Long EQT+AR+RRC 252d when MHHNGSP < $2.50",
        "source":"FRED MHHNGSP; yfinance","n_events":len(evts),"avg_event_return":round(float(np.mean(ra)),4),
        "event_win_rate":round(float(np.mean([r>0 for r in ra])),4),"events":evts})
    print(f"Done: {len(evts)} events")
if __name__=="__main__": main()
