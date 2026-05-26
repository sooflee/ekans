"""PL87 — Trade-Weighted Dollar Declines >5% from 6mo Peak → Long HON+CAT+DE"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns
def main():
    sid = "PL87_twd_decline_industrial_exporters"
    try: fred = load_fred("DTWEXBGS", start="2006-01-01"); data = fred.squeeze()
    except Exception as e: return mark_failed(sid, f"FRED: {e}")
    if data.empty: return mark_failed(sid, "no data")
    peak126 = data.rolling(126).max().dropna()
    drawdown = data / peak126 - 1
    triggers = []; prev = False
    for i in range(len(drawdown)):
        v = float(drawdown.iloc[i])
        if np.isnan(v): continue
        if v < -0.05 and not prev:
            if not triggers or (drawdown.index[i] - triggers[-1]).days > 180: triggers.append(drawdown.index[i])
            prev = True
        elif v >= -0.03: prev = False
    print(f"Events: {len(triggers)}")
    if not triggers: return mark_failed(sid, "no events")
    try: px = load_prices(["HON","CAT","DE","SPY"], start="2006-01-01")
    except Exception as e: return mark_failed(sid, f"price: {e}")
    ret = daily_returns(px)
    available = [t for t in ["HON","CAT","DE"] if t in ret.columns]
    if not available: return mark_failed(sid, "no tickers")
    basket_r = ret[available].mean(axis=1); spy_r = ret["SPY"]; hold = 126
    pnl = pd.Series(0.0, index=basket_r.index); evts = []
    for td in triggers:
        mask = basket_r.index >= td
        if mask.sum() < hold: continue
        ei = basket_r.index[mask][0]; p = basket_r.index.get_loc(ei); ep = min(p+hold, len(basket_r))
        er = basket_r.iloc[p:ep]; cr = float((1+er).prod()-1); pnl.iloc[p:ep] = er.values[:ep-p]
        sc = None
        if ei in spy_r.index: sp=spy_r.index.get_loc(ei); sc=float((1+spy_r.iloc[sp:min(sp+hold,len(spy_r))]).prod()-1)
        evts.append({"trigger_date":str(td.date()),"basket_return":round(cr,4),"spy_return":round(sc,4) if sc else None})
    if not evts: return mark_failed(sid, "no valid events")
    ip = pnl[pnl!=0]
    if len(ip)<30: return mark_failed(sid, f"insufficient days ({len(ip)})")
    m = compute_metrics(ip, benchmark=spy_r, name="TWD Decline → Long Exporters")
    ra = [e["basket_return"] for e in evts]
    save_result(sid, m, extra={"rule":"Long HON+CAT+DE 126d when DTWEXBGS drops >5% from 6mo peak",
        "source":"FRED DTWEXBGS; yfinance","n_events":len(evts),"avg_event_return":round(float(np.mean(ra)),4),
        "event_win_rate":round(float(np.mean([r>0 for r in ra])),4),"events":evts})
    print(f"Done: {len(evts)} events")
if __name__=="__main__": main()
