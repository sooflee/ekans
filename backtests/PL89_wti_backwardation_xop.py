"""PL89 — WTI Backwardation (spot > 126d MA + $3) → Long XOP 63d"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns
def main():
    sid = "PL89_wti_backwardation_xop"
    try: fred = load_fred("DCOILWTICO", start="2005-01-01"); wti = fred.squeeze()
    except Exception as e: return mark_failed(sid, f"FRED: {e}")
    if wti.empty: return mark_failed(sid, "no data")
    ma126 = wti.rolling(126).mean().dropna()
    spread = wti - ma126; spread = spread.dropna()
    triggers = []; prev = False
    for i in range(len(spread)):
        v = float(spread.iloc[i])
        if np.isnan(v): continue
        if v > 3.0 and not prev:
            if not triggers or (spread.index[i] - triggers[-1]).days > 90: triggers.append(spread.index[i])
            prev = True
        elif v <= 1.0: prev = False
    print(f"Events: {len(triggers)}")
    if not triggers: return mark_failed(sid, "no events")
    try: px = load_prices(["XOP","SPY"], start="2006-01-01")
    except Exception as e: return mark_failed(sid, f"price: {e}")
    ret = daily_returns(px); xop_r = ret["XOP"]; spy_r = ret["SPY"]; hold = 63
    pnl = pd.Series(0.0, index=xop_r.index); evts = []
    for td in triggers:
        mask = xop_r.index >= td
        if mask.sum() < hold: continue
        ei = xop_r.index[mask][0]; p = xop_r.index.get_loc(ei); ep = min(p+hold, len(xop_r))
        er = xop_r.iloc[p:ep]; cr = float((1+er).prod()-1); pnl.iloc[p:ep] = er.values[:ep-p]
        sc = None
        if ei in spy_r.index: sp=spy_r.index.get_loc(ei); sc=float((1+spy_r.iloc[sp:min(sp+hold,len(spy_r))]).prod()-1)
        evts.append({"trigger_date":str(td.date()),"wti_spread":round(float(spread.loc[td]),2),
            "xop_return":round(cr,4),"spy_return":round(sc,4) if sc else None})
    if not evts: return mark_failed(sid, "no valid events")
    ip = pnl[pnl!=0]
    if len(ip)<30: return mark_failed(sid, f"insufficient days ({len(ip)})")
    m = compute_metrics(ip, benchmark=spy_r, name="WTI Backwardation → Long XOP")
    ra = [e["xop_return"] for e in evts]
    save_result(sid, m, extra={"rule":"Long XOP 63d when WTI > 126d MA + $3",
        "source":"FRED DCOILWTICO; yfinance","n_events":len(evts),"avg_event_return":round(float(np.mean(ra)),4),
        "event_win_rate":round(float(np.mean([r>0 for r in ra])),4),"events":evts})
    print(f"Done: {len(evts)} events")
if __name__=="__main__": main()
