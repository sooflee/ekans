"""PL100 — Real Fed Funds Rate Turns Negative → Long QQQ 12mo"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns
def main():
    sid = "PL100_negative_real_rate_qqq"
    try:
        dff = load_fred("DFF", start="1998-01-01").squeeze()
        pce = load_fred("PCEPILFE", start="1998-01-01").squeeze()
    except Exception as e: return mark_failed(sid, f"FRED: {e}")
    if dff.empty or pce.empty: return mark_failed(sid, "no data")
    # Forward-fill PCE to daily
    pce_daily = pce.resample("D").ffill()
    idx = dff.index.intersection(pce_daily.index)
    real_rate = dff.loc[idx] - pce_daily.loc[idx]
    real_rate = real_rate.dropna()
    triggers = []; pos_days = 0
    for i in range(1, len(real_rate)):
        v = float(real_rate.iloc[i]); prev = float(real_rate.iloc[i-1])
        if np.isnan(v) or np.isnan(prev): continue
        if v > 0: pos_days += 1
        elif v <= 0 and prev > 0 and pos_days >= 126:
            if not triggers or (real_rate.index[i] - triggers[-1]).days > 365: triggers.append(real_rate.index[i])
            pos_days = 0
        elif v <= 0: pos_days = 0
    print(f"Events: {len(triggers)}")
    if not triggers: return mark_failed(sid, "no events")
    try: px = load_prices(["QQQ","SPY"], start="1999-01-01")
    except Exception as e: return mark_failed(sid, f"price: {e}")
    ret = daily_returns(px); qqq_r = ret["QQQ"]; spy_r = ret["SPY"]; hold = 252
    pnl = pd.Series(0.0, index=qqq_r.index); evts = []
    for td in triggers:
        mask = qqq_r.index >= td
        if mask.sum() < hold: continue
        ei = qqq_r.index[mask][0]; p = qqq_r.index.get_loc(ei); ep = min(p+hold, len(qqq_r))
        er = qqq_r.iloc[p:ep]; cr = float((1+er).prod()-1); pnl.iloc[p:ep] = er.values[:ep-p]
        sc = None
        if ei in spy_r.index: sp=spy_r.index.get_loc(ei); sc=float((1+spy_r.iloc[sp:min(sp+hold,len(spy_r))]).prod()-1)
        evts.append({"trigger_date":str(td.date()),"qqq_return":round(cr,4),"spy_return":round(sc,4) if sc else None})
    if not evts: return mark_failed(sid, "no valid events")
    ip = pnl[pnl!=0]
    if len(ip)<30: return mark_failed(sid, f"insufficient days ({len(ip)})")
    m = compute_metrics(ip, benchmark=spy_r, name="Negative Real Rate → Long QQQ")
    ra = [e["qqq_return"] for e in evts]
    save_result(sid, m, extra={"rule":"Long QQQ 252d when DFF-PCEPILFE turns negative after 6+ months positive",
        "source":"FRED DFF, PCEPILFE; yfinance","n_events":len(evts),"avg_event_return":round(float(np.mean(ra)),4),
        "event_win_rate":round(float(np.mean([r>0 for r in ra])),4),"events":evts})
    print(f"Done: {len(evts)} events")
if __name__=="__main__": main()
