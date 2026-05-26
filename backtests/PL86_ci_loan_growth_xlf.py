"""PL86 — C&I Loan Growth YoY Positive After 6mo Decline → Long XLF"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns
def main():
    sid = "PL86_ci_loan_growth_xlf"
    try: fred = load_fred("BUSLOANS", start="1990-01-01"); data = fred.squeeze()
    except Exception as e: return mark_failed(sid, f"FRED: {e}")
    if data.empty: return mark_failed(sid, "no data")
    monthly = data.resample("M").last().dropna()
    yoy = monthly.pct_change(12).dropna()
    triggers = []; neg_count = 0; fired = False
    for i in range(len(yoy)):
        v = float(yoy.iloc[i])
        if np.isnan(v): continue
        if v < 0: neg_count += 1; fired = False
        elif v >= 0 and neg_count >= 6 and not fired:
            triggers.append(yoy.index[i]); fired = True; neg_count = 0
        else: neg_count = 0
    print(f"Events: {len(triggers)}")
    if not triggers: return mark_failed(sid, "no events")
    try: px = load_prices(["XLF","SPY"], start="1998-01-01")
    except Exception as e: return mark_failed(sid, f"price: {e}")
    ret = daily_returns(px); xlf_r = ret["XLF"]; spy_r = ret["SPY"]; hold = 252
    pnl = pd.Series(0.0, index=xlf_r.index); evts = []
    for td in triggers:
        ed = td + pd.offsets.MonthBegin(1); mask = xlf_r.index >= ed
        if mask.sum() < hold: continue
        ei = xlf_r.index[mask][0]; p = xlf_r.index.get_loc(ei); ep = min(p+hold, len(xlf_r))
        er = xlf_r.iloc[p:ep]; cr = float((1+er).prod()-1); pnl.iloc[p:ep] = er.values[:ep-p]
        sc = None
        if ei in spy_r.index: sp=spy_r.index.get_loc(ei); sc=float((1+spy_r.iloc[sp:min(sp+hold,len(spy_r))]).prod()-1)
        evts.append({"trigger_date":str(td.date()),"xlf_return":round(cr,4),"spy_return":round(sc,4) if sc else None})
    if not evts: return mark_failed(sid, "no valid events")
    ip = pnl[pnl!=0]
    if len(ip)<30: return mark_failed(sid, f"insufficient days ({len(ip)})")
    m = compute_metrics(ip, benchmark=spy_r, name="C&I Loan Growth → Long XLF")
    ra = [e["xlf_return"] for e in evts]
    save_result(sid, m, extra={"rule":"Long XLF 252d when BUSLOANS YoY turns positive after 6+ months negative",
        "source":"FRED BUSLOANS; yfinance","n_events":len(evts),"avg_event_return":round(float(np.mean(ra)),4),
        "event_win_rate":round(float(np.mean([r>0 for r in ra])),4),"events":evts})
    print(f"Done: {len(evts)} events")
if __name__=="__main__": main()
