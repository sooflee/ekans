"""PL88 — Goods Trade Deficit Widens Beyond -$80B/mo → Long EXPD+CHRW+MATX"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns
def main():
    sid = "PL88_trade_deficit_widening_logistics"
    try: fred = load_fred("BOPGSTB", start="1992-01-01"); data = fred.squeeze()
    except Exception as e: return mark_failed(sid, f"FRED: {e}")
    if data.empty: return mark_failed(sid, "no data")
    # BOPGSTB is in millions; threshold -80000 (=$80B)
    unit = data.min()
    thresh = -80000 if unit < -10000 else (-80 if unit < -100 else -80)
    triggers = []; prev = False
    for i in range(len(data)):
        v = float(data.iloc[i])
        if np.isnan(v): continue
        if v < thresh and not prev:
            if not triggers or (data.index[i] - triggers[-1]).days > 180: triggers.append(data.index[i])
            prev = True
        elif v >= thresh * 0.9: prev = False
    print(f"Events: {len(triggers)} (threshold={thresh})")
    if not triggers: return mark_failed(sid, "no events found (deficit never exceeded $80B)")
    try: px = load_prices(["EXPD","CHRW","MATX","SPY"], start="1993-01-01")
    except Exception as e: return mark_failed(sid, f"price: {e}")
    ret = daily_returns(px)
    available = [t for t in ["EXPD","CHRW","MATX"] if t in ret.columns]
    if not available: return mark_failed(sid, "no tickers")
    basket_r = ret[available].mean(axis=1); spy_r = ret["SPY"]; hold = 126
    pnl = pd.Series(0.0, index=basket_r.index); evts = []
    for td in triggers:
        ed = td + pd.offsets.MonthBegin(1); mask = basket_r.index >= ed
        if mask.sum() < hold: continue
        ei = basket_r.index[mask][0]; p = basket_r.index.get_loc(ei); ep = min(p+hold, len(basket_r))
        er = basket_r.iloc[p:ep]; cr = float((1+er).prod()-1); pnl.iloc[p:ep] = er.values[:ep-p]
        sc = None
        if ei in spy_r.index: sp=spy_r.index.get_loc(ei); sc=float((1+spy_r.iloc[sp:min(sp+hold,len(spy_r))]).prod()-1)
        evts.append({"trigger_date":str(td.date()),"basket_return":round(cr,4),"spy_return":round(sc,4) if sc else None})
    if not evts: return mark_failed(sid, "no valid events")
    ip = pnl[pnl!=0]
    if len(ip)<30: return mark_failed(sid, f"insufficient days ({len(ip)})")
    m = compute_metrics(ip, benchmark=spy_r, name="Trade Deficit → Long Logistics")
    ra = [e["basket_return"] for e in evts]
    save_result(sid, m, extra={"rule":"Long EXPD+CHRW+MATX 126d when BOPGSTB < -$80B",
        "source":"FRED BOPGSTB; yfinance","n_events":len(evts),"avg_event_return":round(float(np.mean(ra)),4),
        "event_win_rate":round(float(np.mean([r>0 for r in ra])),4),"events":evts})
    print(f"Done: {len(evts)} events")
if __name__=="__main__": main()
