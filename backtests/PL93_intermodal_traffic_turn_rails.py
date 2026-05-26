"""PL93 — Railroad Intermodal Traffic YoY Positive After 6mo Decline → Long UNP+CSX"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns
def main():
    sid = "PL93_intermodal_traffic_turn_rails"
    try: fred = load_fred("RAILFRTINTERAM", start="2000-01-01"); data = fred.squeeze()
    except Exception as e: return mark_failed(sid, f"FRED: {e}")
    if data.empty: return mark_failed(sid, "no data")
    yoy = data.pct_change(12).dropna()
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
    try: px = load_prices(["UNP","CSX","SPY"], start="2000-01-01")
    except Exception as e: return mark_failed(sid, f"price: {e}")
    ret = daily_returns(px)
    available = [t for t in ["UNP","CSX"] if t in ret.columns]
    if not available: return mark_failed(sid, "no tickers")
    basket_r = ret[available].mean(axis=1); spy_r = ret["SPY"]; hold = 126
    pnl = pd.Series(0.0, index=basket_r.index); evts = []
    for td in triggers:
        ed = td + pd.offsets.MonthBegin(1); mask = basket_r.index >= ed
        if mask.sum() < hold: continue
        ei = basket_r.index[mask][0]; p = basket_r.index.get_loc(ei); ep = min(p+hold, len(basket_r))
        er = basket_r.iloc[p:ep]; cr = float((1+er).prod()-1); pnl.iloc[p:ep] = er.values[:ep-p]
        sc = None
        if ei in spy_r.index: sp=spy_r.index.get_loc(ei); sc=float((1+spy_r.iloc[sp:min(sp+hold,len(spy_r))]).prod()-1)
        evts.append({"trigger_date":str(td.date()),"basket_return":round(cr,4),"spy_return":round(sc,4) if sc else None})
    if not evts: return mark_failed(sid, "no valid events")
    ip = pnl[pnl!=0]
    if len(ip)<30: return mark_failed(sid, f"insufficient days ({len(ip)})")
    m = compute_metrics(ip, benchmark=spy_r, name="Intermodal Traffic Turn → Long UNP+CSX")
    ra = [e["basket_return"] for e in evts]
    save_result(sid, m, extra={"rule":"Long UNP+CSX 126d when RAILFRTINTERAM YoY turns positive after 6+ months negative",
        "source":"FRED RAILFRTINTERAM; yfinance","n_events":len(evts),"avg_event_return":round(float(np.mean(ra)),4),
        "event_win_rate":round(float(np.mean([r>0 for r in ra])),4),"events":evts})
    print(f"Done: {len(evts)} events")
if __name__=="__main__": main()
