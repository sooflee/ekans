"""PL99 — 5Y Breakeven Drops Below 2.0% After 6mo Above 2.3% → Long TIP"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns
def main():
    sid = "PL99_breakeven_below_target_tips"
    try: fred = load_fred("T5YIE", start="2003-01-01"); data = fred.squeeze()
    except Exception as e: return mark_failed(sid, f"FRED: {e}")
    if data.empty: return mark_failed(sid, "no data")
    triggers = []; above23_days = 0
    for i in range(1, len(data)):
        v = float(data.iloc[i]); prev = float(data.iloc[i-1])
        if np.isnan(v) or np.isnan(prev): continue
        if v > 2.3: above23_days += 1
        elif v <= 2.0 and prev > 2.0 and above23_days >= 126:
            if not triggers or (data.index[i] - triggers[-1]).days > 365: triggers.append(data.index[i])
            above23_days = 0
        elif v <= 2.3: above23_days = 0
    print(f"Events: {len(triggers)}")
    if not triggers: return mark_failed(sid, "no events")
    try: px = load_prices(["TIP","IEF","SPY"], start="2003-01-01")
    except Exception as e: return mark_failed(sid, f"price: {e}")
    ret = daily_returns(px); tip_r = ret["TIP"]; spy_r = ret["SPY"]; hold = 252
    pnl = pd.Series(0.0, index=tip_r.index); evts = []
    for td in triggers:
        mask = tip_r.index >= td
        if mask.sum() < hold: continue
        ei = tip_r.index[mask][0]; p = tip_r.index.get_loc(ei); ep = min(p+hold, len(tip_r))
        er = tip_r.iloc[p:ep]; cr = float((1+er).prod()-1); pnl.iloc[p:ep] = er.values[:ep-p]
        sc = None
        if ei in spy_r.index: sp=spy_r.index.get_loc(ei); sc=float((1+spy_r.iloc[sp:min(sp+hold,len(spy_r))]).prod()-1)
        evts.append({"trigger_date":str(td.date()),"breakeven":round(float(data.loc[td]),2),
            "tip_return":round(cr,4),"spy_return":round(sc,4) if sc else None})
    if not evts: return mark_failed(sid, "no valid events")
    ip = pnl[pnl!=0]
    if len(ip)<30: return mark_failed(sid, f"insufficient days ({len(ip)})")
    m = compute_metrics(ip, benchmark=spy_r, name="Breakeven Below Target → Long TIP")
    ra = [e["tip_return"] for e in evts]
    save_result(sid, m, extra={"rule":"Long TIP 252d when T5YIE drops below 2.0% after 6+ months above 2.3%",
        "source":"FRED T5YIE; yfinance","n_events":len(evts),"avg_event_return":round(float(np.mean(ra)),4),
        "event_win_rate":round(float(np.mean([r>0 for r in ra])),4),"events":evts})
    print(f"Done: {len(evts)} events")
if __name__=="__main__": main()
