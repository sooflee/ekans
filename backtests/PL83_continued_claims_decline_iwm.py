"""PL83 — Continued Claims Decline 8+ Weeks from Peak → Long IWM"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns
def main():
    sid = "PL83_continued_claims_decline_iwm"
    try: fred = load_fred("CCSA", start="1990-01-01"); data = fred.squeeze()
    except Exception as e: return mark_failed(sid, f"FRED: {e}")
    if data.empty: return mark_failed(sid, "no data")
    peak26 = data.rolling(26).max()
    triggers = []; consec_decline = 0; near_peak_start = False
    for i in range(1, len(data)):
        v = float(data.iloc[i]); prev = float(data.iloc[i-1]); pk = float(peak26.iloc[i])
        if np.isnan(v) or np.isnan(prev) or np.isnan(pk): continue
        if prev >= pk * 0.95: near_peak_start = True
        if near_peak_start and v < prev: consec_decline += 1
        else: consec_decline = 0; near_peak_start = False if v >= pk * 0.95 else near_peak_start
        if consec_decline >= 8:
            if not triggers or (data.index[i] - triggers[-1]).days > 365:
                triggers.append(data.index[i])
            consec_decline = 0; near_peak_start = False
    print(f"Events: {len(triggers)}")
    if not triggers: return mark_failed(sid, "no events")
    try: px = load_prices(["IWM","SPY"], start="2000-01-01")
    except Exception as e: return mark_failed(sid, f"price: {e}")
    ret = daily_returns(px); iwm_r = ret["IWM"]; spy_r = ret["SPY"]; hold = 126
    pnl = pd.Series(0.0, index=iwm_r.index); evts = []
    for td in triggers:
        mask = iwm_r.index >= td
        if mask.sum() < hold: continue
        ei = iwm_r.index[mask][0]; p = iwm_r.index.get_loc(ei); ep = min(p+hold, len(iwm_r))
        er = iwm_r.iloc[p:ep]; cr = float((1+er).prod()-1); pnl.iloc[p:ep] = er.values[:ep-p]
        sc = None
        if ei in spy_r.index: sp=spy_r.index.get_loc(ei); sc=float((1+spy_r.iloc[sp:min(sp+hold,len(spy_r))]).prod()-1)
        evts.append({"trigger_date":str(td.date()),"iwm_return":round(cr,4),"spy_return":round(sc,4) if sc else None})
    if not evts: return mark_failed(sid, "no valid events")
    ip = pnl[pnl!=0]
    if len(ip)<30: return mark_failed(sid, f"insufficient days ({len(ip)})")
    m = compute_metrics(ip, benchmark=spy_r, name="Continued Claims Decline → Long IWM")
    ra = [e["iwm_return"] for e in evts]
    save_result(sid, m, extra={"rule":"Long IWM 126d when CCSA declines 8+ weeks from near-peak",
        "source":"FRED CCSA; yfinance","n_events":len(evts),"avg_event_return":round(float(np.mean(ra)),4),
        "event_win_rate":round(float(np.mean([r>0 for r in ra])),4),"events":evts})
    print(f"Done: {len(evts)} events")
if __name__=="__main__": main()
