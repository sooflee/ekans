"""PL96 — Retail Sales ex-Auto MoM > +0.5% for 2mo After Flat → Long XLY"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns
def main():
    sid = "PL96_retail_sales_ex_auto_xly"
    try: fred = load_fred("RSFSXMV", start="1992-01-01"); data = fred.squeeze()
    except Exception as e: return mark_failed(sid, f"FRED: {e}")
    if data.empty: return mark_failed(sid, "no data")
    mom = data.pct_change().dropna()
    triggers = []; flat_count = 0
    for i in range(1, len(mom)):
        v = float(mom.iloc[i]); prev = float(mom.iloc[i-1])
        if np.isnan(v) or np.isnan(prev): continue
        if v <= 0: flat_count += 1
        elif v > 0.005 and prev > 0.005 and flat_count >= 3:
            if not triggers or (mom.index[i] - triggers[-1]).days > 180: triggers.append(mom.index[i])
            flat_count = 0
        else: flat_count = 0 if v > 0 else flat_count
    print(f"Events: {len(triggers)}")
    if not triggers: return mark_failed(sid, "no events")
    try: px = load_prices(["XLY","SPY"], start="1998-01-01")
    except Exception as e: return mark_failed(sid, f"price: {e}")
    ret = daily_returns(px); xly_r = ret["XLY"]; spy_r = ret["SPY"]; hold = 63
    pnl = pd.Series(0.0, index=xly_r.index); evts = []
    for td in triggers:
        ed = td + pd.offsets.MonthBegin(1); mask = xly_r.index >= ed
        if mask.sum() < hold: continue
        ei = xly_r.index[mask][0]; p = xly_r.index.get_loc(ei); ep = min(p+hold, len(xly_r))
        er = xly_r.iloc[p:ep]; cr = float((1+er).prod()-1); pnl.iloc[p:ep] = er.values[:ep-p]
        sc = None
        if ei in spy_r.index: sp=spy_r.index.get_loc(ei); sc=float((1+spy_r.iloc[sp:min(sp+hold,len(spy_r))]).prod()-1)
        evts.append({"trigger_date":str(td.date()),"xly_return":round(cr,4),"spy_return":round(sc,4) if sc else None})
    if not evts: return mark_failed(sid, "no valid events")
    ip = pnl[pnl!=0]
    if len(ip)<30: return mark_failed(sid, f"insufficient days ({len(ip)})")
    m = compute_metrics(ip, benchmark=spy_r, name="Retail Sales ex-Auto Turn → Long XLY")
    ra = [e["xly_return"] for e in evts]
    save_result(sid, m, extra={"rule":"Long XLY 63d when RSFSXMV MoM>+0.5% for 2mo after 3+ flat months",
        "source":"FRED RSFSXMV; yfinance","n_events":len(evts),"avg_event_return":round(float(np.mean(ra)),4),
        "event_win_rate":round(float(np.mean([r>0 for r in ra])),4),"events":evts})
    print(f"Done: {len(evts)} events")
if __name__=="__main__": main()
