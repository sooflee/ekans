"""PL95 — Savings Rate Declines >4pp from Peak → Long BKNG+MAR+H"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns
def main():
    sid = "PL95_savings_rate_decline_leisure"
    try: fred = load_fred("PSAVERT", start="1990-01-01"); data = fred.squeeze()
    except Exception as e: return mark_failed(sid, f"FRED: {e}")
    if data.empty: return mark_failed(sid, "no data")
    peak12 = data.rolling(12).max().dropna()
    diff = data - peak12; diff = diff.dropna()
    triggers = []; prev = False
    for i in range(len(diff)):
        v = float(diff.iloc[i])
        if np.isnan(v): continue
        if v < -4.0 and not prev:
            if not triggers or (diff.index[i] - triggers[-1]).days > 365: triggers.append(diff.index[i])
            prev = True
        elif v >= -2.0: prev = False
    print(f"Events: {len(triggers)}")
    if not triggers: return mark_failed(sid, "no events")
    try: px = load_prices(["BKNG","MAR","H","SPY"], start="2006-01-01")
    except Exception as e: return mark_failed(sid, f"price: {e}")
    ret = daily_returns(px)
    available = [t for t in ["BKNG","MAR","H"] if t in ret.columns]
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
    m = compute_metrics(ip, benchmark=spy_r, name="Savings Rate Decline → Long Leisure")
    ra = [e["basket_return"] for e in evts]
    save_result(sid, m, extra={"rule":"Long BKNG+MAR+H 126d when PSAVERT drops >4pp from 12mo peak",
        "source":"FRED PSAVERT; yfinance","n_events":len(evts),"avg_event_return":round(float(np.mean(ra)),4),
        "event_win_rate":round(float(np.mean([r>0 for r in ra])),4),"events":evts})
    print(f"Done: {len(evts)} events")
if __name__=="__main__": main()
