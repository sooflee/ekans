"""PL84 — TED Spread Contracts Below 30bps After 50bps+ → Long KRE"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns
def main():
    sid = "PL84_ted_spread_normalization_kre"
    try: fred = load_fred("TEDRATE", start="1986-01-01"); data = fred.squeeze()
    except Exception as e: return mark_failed(sid, f"FRED: {e}")
    if data.empty: return mark_failed(sid, "no data")
    triggers = []; above50_days = 0
    for i in range(1, len(data)):
        v = float(data.iloc[i]); prev = float(data.iloc[i-1])
        if np.isnan(v) or np.isnan(prev): continue
        if v > 0.50: above50_days += 1
        elif v <= 0.30 and prev > 0.30 and above50_days >= 63:
            if not triggers or (data.index[i] - triggers[-1]).days > 365:
                triggers.append(data.index[i])
            above50_days = 0
        elif v <= 0.50: above50_days = 0
    print(f"Events: {len(triggers)}")
    if not triggers: return mark_failed(sid, "no events")
    try: px = load_prices(["KRE","SPY"], start="2006-01-01")
    except Exception as e: return mark_failed(sid, f"price: {e}")
    ret = daily_returns(px); kre_r = ret["KRE"]; spy_r = ret["SPY"]; hold = 126
    pnl = pd.Series(0.0, index=kre_r.index); evts = []
    for td in triggers:
        mask = kre_r.index >= td
        if mask.sum() < hold: continue
        ei = kre_r.index[mask][0]; p = kre_r.index.get_loc(ei); ep = min(p+hold, len(kre_r))
        er = kre_r.iloc[p:ep]; cr = float((1+er).prod()-1); pnl.iloc[p:ep] = er.values[:ep-p]
        sc = None
        if ei in spy_r.index: sp=spy_r.index.get_loc(ei); sc=float((1+spy_r.iloc[sp:min(sp+hold,len(spy_r))]).prod()-1)
        evts.append({"trigger_date":str(td.date()),"kre_return":round(cr,4),"spy_return":round(sc,4) if sc else None})
    if not evts: return mark_failed(sid, "no valid events")
    ip = pnl[pnl!=0]
    if len(ip)<30: return mark_failed(sid, f"insufficient days ({len(ip)})")
    m = compute_metrics(ip, benchmark=spy_r, name="TED Spread Norm → Long KRE")
    ra = [e["kre_return"] for e in evts]
    save_result(sid, m, extra={"rule":"Long KRE 126d when TEDRATE<0.30 after 3+ months >0.50",
        "source":"FRED TEDRATE; yfinance","n_events":len(evts),"avg_event_return":round(float(np.mean(ra)),4),
        "event_win_rate":round(float(np.mean([r>0 for r in ra])),4),"events":evts})
    print(f"Done: {len(evts)} events")
if __name__=="__main__": main()
