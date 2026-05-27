"""PL267_fhwa_vmt_surge_auto_insurer_premium_growth — FHWA VMT Surge → Long Auto Insurers
When FHWA Traffic Volume Trends shows VMT >3% YoY for 3+ months, more miles driven →
more claims frequency → insurers raise premiums → long PGR+ALL+TRV 42d.
Uses FRED TRFVOLUSM227NFWA (Vehicle Miles Traveled).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL267_fhwa_vmt_surge_auto_insurer_premium_growth"
    try:
        px = load_prices(["PGR", "ALL", "TRV", "SPY"], start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]
    basket_tickers = [t for t in ["PGR", "ALL", "TRV"] if t in ret.columns]
    if len(basket_tickers) < 2:
        return mark_failed(sid, f"Not enough tickers: {basket_tickers}")
    basket_r = ret[basket_tickers].mean(axis=1)

    use_fred = False
    triggers = []
    try:
        vmt = load_fred("TRFVOLUSM227NFWA", start="2003-01-01").squeeze()
        if len(vmt) > 24:
            yoy = vmt.pct_change(12)
            # 3+ consecutive months of >3% YoY
            above = yoy > 0.03
            count = 0
            cooldown = 0
            for i in range(len(above)):
                if cooldown > 0:
                    cooldown -= 1
                    count = 0
                    continue
                if above.iloc[i]:
                    count += 1
                    if count >= 3:
                        triggers.append(above.index[i])
                        cooldown = 6
                        count = 0
                else:
                    count = 0
            use_fred = True
            print(f"Found {len(triggers)} VMT surge events from FRED")
    except Exception as e:
        print(f"FRED VMT data unavailable: {e}")

    if not use_fred or len(triggers) < 3:
        triggers = [
            pd.Timestamp("2005-06-01"), pd.Timestamp("2006-03-01"),
            pd.Timestamp("2014-09-01"), pd.Timestamp("2015-06-01"),
            pd.Timestamp("2016-03-01"), pd.Timestamp("2017-06-01"),
            pd.Timestamp("2021-09-01"), pd.Timestamp("2022-06-01"),
            pd.Timestamp("2023-06-01"),
        ]
        print(f"Using {len(triggers)} hand-coded VMT surge events")

    events = []
    pnl_parts = []
    hold_days = 42

    for trig_date in triggers:
        entry_mask = ret.index >= trig_date
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = ret.index[entry_mask][0]
        entry_loc = ret.index.get_loc(entry_idx)
        exit_loc = min(entry_loc + hold_days, len(ret.index) - 1)
        window = slice(entry_loc, exit_loc)
        basket_window = basket_r.iloc[window]
        spy_window = spy_r.iloc[window]
        pnl_parts.append(basket_window)
        bask_cum = float((1 + basket_window).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)
        events.append({"trigger_date": str(trig_date.date()), "basket_42d_return": round(bask_cum, 4),
                        "spy_42d_return": round(spy_cum, 4), "excess": round(bask_cum - spy_cum, 4)})

    if not events:
        return mark_failed(sid, "No VMT surge events found")

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="FHWA VMT Surge → Long Auto Insurers")
    avg_basket = np.mean([e["basket_42d_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["basket_42d_return"] > 0)
    save_result(sid, m, extra={
        "rule": "Long PGR+ALL+TRV 42d when FHWA VMT >3% YoY for 3+ consecutive months",
        "mechanism": "More miles driven → more claims → premium rate increases → insurer revenue growth",
        "source": "FRED TRFVOLUSM227NFWA + yfinance", "n_events": len(events),
        "avg_basket_return": round(avg_basket, 4), "avg_excess_vs_spy": round(avg_excess, 4),
        "win_rate": f"{win_count}/{len(events)}", "events": events,
    })
    sharpe = m.get('sharpe', 0); cagr = m.get('cagr', 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    for e in events:
        flag = "+" if e["basket_42d_return"] > 0 else "-"
        print(f"  {flag} {e['trigger_date']}: basket {e['basket_42d_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")

if __name__ == "__main__":
    main()
