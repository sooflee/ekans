"""PL82 — Avg Weekly Hours Mfg Crosses 41.0 from Below 40.5 → Long XLI"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns
def main():
    sid = "PL82_mfg_weekly_hours_cyclicals"
    try: fred = load_fred("AWHMAN", start="1990-01-01"); data = fred.squeeze()
    except Exception as e: return mark_failed(sid, f"FRED: {e}")
    if data.empty: return mark_failed(sid, "no data")
    triggers = []; below_count = 0; fired = False
    for i in range(len(data)):
        v = float(data.iloc[i])
        if np.isnan(v): continue
        if v < 40.5: below_count += 1; fired = False
        elif v >= 41.0 and below_count >= 4 and not fired:
            triggers.append(data.index[i]); fired = True; below_count = 0
        elif v >= 40.5: pass
    print(f"Events: {len(triggers)}")
    if not triggers: return mark_failed(sid, "no events")
    try: px = load_prices(["XLI","SPY"], start="1998-01-01")
    except Exception as e: return mark_failed(sid, f"price: {e}")
    ret = daily_returns(px); xli_r = ret["XLI"]; spy_r = ret["SPY"]; hold = 126
    pnl = pd.Series(0.0, index=xli_r.index); evts = []
    for td in triggers:
        ed = td + pd.offsets.MonthBegin(1); mask = xli_r.index >= ed
        if mask.sum() < hold: continue
        ei = xli_r.index[mask][0]; p = xli_r.index.get_loc(ei); ep = min(p+hold, len(xli_r))
        er = xli_r.iloc[p:ep]; cr = float((1+er).prod()-1); pnl.iloc[p:ep] = er.values[:ep-p]
        sc = None
        if ei in spy_r.index: sp=spy_r.index.get_loc(ei); sc=float((1+spy_r.iloc[sp:min(sp+hold,len(spy_r))]).prod()-1)
        evts.append({"trigger_date":str(td.date()),"hours":round(float(data.loc[td]),1),"xli_return":round(cr,4),"spy_return":round(sc,4) if sc else None})
    if not evts: return mark_failed(sid, "no valid events")
    ip = pnl[pnl!=0]
    if len(ip)<30: return mark_failed(sid, f"insufficient days ({len(ip)})")
    m = compute_metrics(ip, benchmark=spy_r, name="Mfg Hours → Long XLI")
    ra = [e["xli_return"] for e in evts]
    save_result(sid, m, extra={"rule":"Long XLI 126d when AWHMAN crosses 41.0 after 4+ months below 40.5",
        "source":"FRED AWHMAN; yfinance","n_events":len(evts),"avg_event_return":round(float(np.mean(ra)),4),
        "event_win_rate":round(float(np.mean([r>0 for r in ra])),4),"events":evts})
    print(f"Done: {len(evts)} events")
if __name__=="__main__": main()
