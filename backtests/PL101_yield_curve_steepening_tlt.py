"""PL101 — 10Y-3M Un-Inverts After Deep Inversion → Long TLT 12mo"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns
def main():
    sid = "PL101_yield_curve_steepening_tlt"
    try: fred = load_fred("T10Y3M", start="1982-01-01"); data = fred.squeeze()
    except Exception as e: return mark_failed(sid, f"FRED: {e}")
    if data.empty: return mark_failed(sid, "no data")
    triggers = []; deep_inv_days = 0
    for i in range(1, len(data)):
        v = float(data.iloc[i]); prev = float(data.iloc[i-1])
        if np.isnan(v) or np.isnan(prev): continue
        if v < -1.0: deep_inv_days += 1
        elif v >= 0 and prev < 0 and deep_inv_days >= 126:
            if not triggers or (data.index[i] - triggers[-1]).days > 365: triggers.append(data.index[i])
            deep_inv_days = 0
        elif v >= -1.0 and v < 0: pass  # still inverted but not deep
        else: deep_inv_days = 0
    print(f"Events: {len(triggers)}")
    if not triggers: return mark_failed(sid, "no events")
    try: px = load_prices(["TLT","SPY"], start="2002-01-01")
    except Exception as e: return mark_failed(sid, f"price: {e}")
    ret = daily_returns(px); tlt_r = ret["TLT"]; spy_r = ret["SPY"]; hold = 252
    pnl = pd.Series(0.0, index=tlt_r.index); evts = []
    for td in triggers:
        mask = tlt_r.index >= td
        if mask.sum() < hold: continue
        ei = tlt_r.index[mask][0]; p = tlt_r.index.get_loc(ei); ep = min(p+hold, len(tlt_r))
        er = tlt_r.iloc[p:ep]; cr = float((1+er).prod()-1); pnl.iloc[p:ep] = er.values[:ep-p]
        sc = None
        if ei in spy_r.index: sp=spy_r.index.get_loc(ei); sc=float((1+spy_r.iloc[sp:min(sp+hold,len(spy_r))]).prod()-1)
        evts.append({"trigger_date":str(td.date()),"tlt_return":round(cr,4),"spy_return":round(sc,4) if sc else None})
    if not evts: return mark_failed(sid, "no valid events")
    ip = pnl[pnl!=0]
    if len(ip)<30: return mark_failed(sid, f"insufficient days ({len(ip)})")
    m = compute_metrics(ip, benchmark=spy_r, name="10Y-3M Un-Inversion → Long TLT")
    ra = [e["tlt_return"] for e in evts]
    save_result(sid, m, extra={"rule":"Long TLT 252d when T10Y3M crosses above 0 after 6+ months below -1.0",
        "source":"FRED T10Y3M; yfinance","n_events":len(evts),"avg_event_return":round(float(np.mean(ra)),4),
        "event_win_rate":round(float(np.mean([r>0 for r in ra])),4),"events":evts})
    print(f"Done: {len(evts)} events")
if __name__=="__main__": main()
