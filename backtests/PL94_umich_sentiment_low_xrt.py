"""PL94 — UMich Sentiment Below 55 Then Rises 2mo → Long XRT"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns
def main():
    sid = "PL94_umich_sentiment_low_xrt"
    try: fred = load_fred("UMCSENT", start="1978-01-01"); data = fred.squeeze()
    except Exception as e: return mark_failed(sid, f"FRED: {e}")
    if data.empty: return mark_failed(sid, "no data")
    triggers = []; was_below55 = False
    for i in range(2, len(data)):
        v0=float(data.iloc[i-2]); v1=float(data.iloc[i-1]); v2=float(data.iloc[i])
        if any(np.isnan(x) for x in [v0,v1,v2]): continue
        if v0 < 55: was_below55 = True
        if was_below55 and v1 > v0 and v2 > v1:
            if not triggers or (data.index[i] - triggers[-1]).days > 365: triggers.append(data.index[i])
            was_below55 = False
    print(f"Events: {len(triggers)}")
    if not triggers: return mark_failed(sid, "no events")
    try: px = load_prices(["XRT","SPY"], start="2006-01-01")
    except Exception as e: return mark_failed(sid, f"price: {e}")
    ret = daily_returns(px); xrt_r = ret["XRT"]; spy_r = ret["SPY"]; hold = 126
    pnl = pd.Series(0.0, index=xrt_r.index); evts = []
    for td in triggers:
        ed = td + pd.offsets.MonthBegin(1); mask = xrt_r.index >= ed
        if mask.sum() < hold: continue
        ei = xrt_r.index[mask][0]; p = xrt_r.index.get_loc(ei); ep = min(p+hold, len(xrt_r))
        er = xrt_r.iloc[p:ep]; cr = float((1+er).prod()-1); pnl.iloc[p:ep] = er.values[:ep-p]
        sc = None
        if ei in spy_r.index: sp=spy_r.index.get_loc(ei); sc=float((1+spy_r.iloc[sp:min(sp+hold,len(spy_r))]).prod()-1)
        evts.append({"trigger_date":str(td.date()),"sentiment":round(float(data.loc[td]),1),
            "xrt_return":round(cr,4),"spy_return":round(sc,4) if sc else None})
    if not evts: return mark_failed(sid, "no valid events")
    ip = pnl[pnl!=0]
    if len(ip)<30: return mark_failed(sid, f"insufficient days ({len(ip)})")
    m = compute_metrics(ip, benchmark=spy_r, name="UMich Sentiment Trough → Long XRT")
    ra = [e["xrt_return"] for e in evts]
    save_result(sid, m, extra={"rule":"Long XRT 126d when UMCSENT below 55 then rises 2 consecutive months",
        "source":"FRED UMCSENT; yfinance","n_events":len(evts),"avg_event_return":round(float(np.mean(ra)),4),
        "event_win_rate":round(float(np.mean([r>0 for r in ra])),4),"events":evts})
    print(f"Done: {len(evts)} events")
if __name__=="__main__": main()
