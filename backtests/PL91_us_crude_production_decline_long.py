"""PL91 — US Crude Production Declines MoM for 3mo → Long CL=F 6mo"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns
def main():
    sid = "PL91_us_crude_production_decline_long"
    try: fred = load_fred("MCRFPUS1", start="2000-01-01"); data = fred.squeeze()
    except Exception as e: return mark_failed(sid, f"FRED: {e}")
    if data.empty: return mark_failed(sid, "no data")
    mom = data.diff().dropna()
    triggers = []; streak = 0
    for i in range(len(mom)):
        v = float(mom.iloc[i])
        if np.isnan(v): continue
        if v < 0: streak += 1
        else: streak = 0
        if streak == 3:
            if not triggers or (mom.index[i] - triggers[-1]).days > 180: triggers.append(mom.index[i])
    print(f"Events: {len(triggers)}")
    if not triggers: return mark_failed(sid, "no events")
    try: px = load_prices(["CL=F","SPY"], start="2000-01-01")
    except Exception as e: return mark_failed(sid, f"price: {e}")
    ret = daily_returns(px)
    if "CL=F" not in ret.columns: return mark_failed(sid, "CL=F data missing")
    cl_r = ret["CL=F"]; spy_r = ret["SPY"]; hold = 126
    pnl = pd.Series(0.0, index=cl_r.index); evts = []
    for td in triggers:
        ed = td + pd.offsets.MonthBegin(1); mask = cl_r.index >= ed
        if mask.sum() < hold: continue
        ei = cl_r.index[mask][0]; p = cl_r.index.get_loc(ei); ep = min(p+hold, len(cl_r))
        er = cl_r.iloc[p:ep]; cr = float((1+er).prod()-1); pnl.iloc[p:ep] = er.values[:ep-p]
        sc = None
        if ei in spy_r.index: sp=spy_r.index.get_loc(ei); sc=float((1+spy_r.iloc[sp:min(sp+hold,len(spy_r))]).prod()-1)
        evts.append({"trigger_date":str(td.date()),"cl_return":round(cr,4),"spy_return":round(sc,4) if sc else None})
    if not evts: return mark_failed(sid, "no valid events")
    ip = pnl[pnl!=0]
    if len(ip)<30: return mark_failed(sid, f"insufficient days ({len(ip)})")
    m = compute_metrics(ip, benchmark=spy_r, name="US Crude Production Decline → Long CL=F")
    ra = [e["cl_return"] for e in evts]
    save_result(sid, m, extra={"rule":"Long CL=F 126d when MCRFPUS1 declines MoM for 3 consecutive months",
        "source":"FRED MCRFPUS1; yfinance","n_events":len(evts),"avg_event_return":round(float(np.mean(ra)),4),
        "event_win_rate":round(float(np.mean([r>0 for r in ra])),4),"events":evts})
    print(f"Done: {len(evts)} events")
if __name__=="__main__": main()
