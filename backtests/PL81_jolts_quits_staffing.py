"""PL81 — JOLTS Quits Rate Rise >+0.3pp from Trough → Long RHI"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns
def main():
    sid = "PL81_jolts_quits_staffing"
    try:
        fred = load_fred("JTSQUR", start="2000-01-01"); data = fred.squeeze()
    except Exception as e: return mark_failed(sid, f"FRED: {e}")
    if data.empty: return mark_failed(sid, "no data")
    low6 = data.rolling(6).min(); diff = data - low6; diff = diff.dropna()
    triggers = []
    for i in range(len(diff)):
        if diff.iloc[i] > 0.3:
            if not triggers or (diff.index[i] - triggers[-1]).days > 180: triggers.append(diff.index[i])
    print(f"Events: {len(triggers)}")
    if not triggers: return mark_failed(sid, "no events")
    try: px = load_prices(["RHI", "SPY"], start="2000-01-01")
    except Exception as e: return mark_failed(sid, f"price: {e}")
    ret = daily_returns(px); rhi_r = ret["RHI"]; spy_r = ret["SPY"]; hold = 126
    pnl = pd.Series(0.0, index=rhi_r.index); evts = []
    for td in triggers:
        ed = td + pd.offsets.MonthBegin(1); mask = rhi_r.index >= ed
        if mask.sum() < hold: continue
        ei = rhi_r.index[mask][0]; p = rhi_r.index.get_loc(ei); ep = min(p+hold, len(rhi_r))
        er = rhi_r.iloc[p:ep]; cr = float((1+er).prod()-1); pnl.iloc[p:ep] = er.values[:ep-p]
        sc = None
        if ei in spy_r.index: sp=spy_r.index.get_loc(ei); sc=float((1+spy_r.iloc[sp:min(sp+hold,len(spy_r))]).prod()-1)
        evts.append({"trigger_date":str(td.date()),"rhi_return":round(cr,4),"spy_return":round(sc,4) if sc else None})
    if not evts: return mark_failed(sid, "no valid events")
    ip = pnl[pnl!=0]
    if len(ip)<30: return mark_failed(sid, f"insufficient days ({len(ip)})")
    m = compute_metrics(ip, benchmark=spy_r, name="JOLTS Quits → Long RHI")
    ra = [e["rhi_return"] for e in evts]
    save_result(sid, m, extra={"rule":"Long RHI 126d when JTSQUR > 6mo low + 0.3pp","source":"FRED JTSQUR; yfinance",
        "n_events":len(evts),"avg_event_return":round(float(np.mean(ra)),4),
        "event_win_rate":round(float(np.mean([r>0 for r in ra])),4),"events":evts})
    print(f"Done: {len(evts)} events")
if __name__=="__main__": main()
