"""PL85 — Reserve Balances Surge > +20% in 3mo → Long SPY"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns
def main():
    sid = "PL85_excess_reserves_surge_spy"
    try: fred = load_fred("WRESBAL", start="2002-01-01"); data = fred.squeeze()
    except Exception as e: return mark_failed(sid, f"FRED: {e}")
    if data.empty: return mark_failed(sid, "no data")
    chg13 = data.pct_change(13).dropna()
    triggers = []; prev_above = False
    for i in range(len(chg13)):
        v = float(chg13.iloc[i])
        if np.isnan(v): continue
        if v > 0.20 and not prev_above:
            if not triggers or (chg13.index[i] - triggers[-1]).days > 180: triggers.append(chg13.index[i])
            prev_above = True
        elif v <= 0.20: prev_above = False
    print(f"Events: {len(triggers)}")
    if not triggers: return mark_failed(sid, "no events")
    try: px = load_prices("SPY", start="2002-01-01")
    except Exception as e: return mark_failed(sid, f"price: {e}")
    ret = daily_returns(px)
    if isinstance(ret, pd.DataFrame): spy_r = ret.iloc[:,0]
    else: spy_r = ret
    hold = 126; pnl = pd.Series(0.0, index=spy_r.index); evts = []
    for td in triggers:
        mask = spy_r.index >= td
        if mask.sum() < hold: continue
        ei = spy_r.index[mask][0]; p = spy_r.index.get_loc(ei); ep = min(p+hold, len(spy_r))
        er = spy_r.iloc[p:ep]; cr = float((1+er).prod()-1); pnl.iloc[p:ep] = er.values[:ep-p]
        evts.append({"trigger_date":str(td.date()),"spy_return":round(cr,4)})
    if not evts: return mark_failed(sid, "no valid events")
    ip = pnl[pnl!=0]
    if len(ip)<30: return mark_failed(sid, f"insufficient days ({len(ip)})")
    m = compute_metrics(ip, benchmark=spy_r, name="Reserves Surge → Long SPY")
    ra = [e["spy_return"] for e in evts]
    save_result(sid, m, extra={"rule":"Long SPY 126d when WRESBAL 13-week change > +20%",
        "source":"FRED WRESBAL; yfinance","n_events":len(evts),"avg_event_return":round(float(np.mean(ra)),4),
        "event_win_rate":round(float(np.mean([r>0 for r in ra])),4),"events":evts})
    print(f"Done: {len(evts)} events")
if __name__=="__main__": main()
