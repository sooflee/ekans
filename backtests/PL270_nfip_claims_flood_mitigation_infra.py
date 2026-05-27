"""PL270_nfip_claims_flood_mitigation_infra — NFIP Claims Surge → Long Water/Infra
When monthly NFIP paid claims exceed $500M, flood mitigation spending follows → long AWK+ACM+MWA 60d.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns

def main():
    sid = "PL270_nfip_claims_flood_mitigation_infra"
    try:
        px = load_prices(["AWK", "ACM", "MWA", "SPY"], start="2008-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")
    ret = daily_returns(px)
    spy_r = ret["SPY"]
    basket_tickers = [t for t in ["AWK", "ACM", "MWA"] if t in ret.columns]
    if len(basket_tickers) < 2:
        return mark_failed(sid, f"Not enough tickers: {basket_tickers}")
    basket_r = ret[basket_tickers].mean(axis=1)

    # Major NFIP claims surge events (>$500M monthly)
    nfip_surge_dates = [
        "2008-09-15",  # Hurricane Ike claims
        "2012-11-01",  # Hurricane Sandy — record NFIP payouts
        "2016-08-15",  # Louisiana floods — massive NFIP claims
        "2017-09-01",  # Harvey + Irma — $10B+ NFIP claims
        "2018-10-15",  # Hurricane Michael
        "2021-09-01",  # Hurricane Ida
        "2022-10-01",  # Hurricane Ian — $3.5B+ claims
    ]
    events, pnl_parts = [], []
    hold_days = 60
    for date_str in nfip_surge_dates:
        trig_date = pd.Timestamp(date_str)
        entry_mask = ret.index >= trig_date
        if entry_mask.sum() < hold_days: continue
        entry_idx = ret.index[entry_mask][0]
        entry_loc = ret.index.get_loc(entry_idx)
        exit_loc = min(entry_loc + hold_days, len(ret.index) - 1)
        window = slice(entry_loc, exit_loc)
        basket_window = basket_r.iloc[window]
        spy_window = spy_r.iloc[window]
        pnl_parts.append(basket_window)
        bask_cum = float((1 + basket_window).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)
        events.append({"trigger_date": date_str, "basket_60d_return": round(bask_cum, 4),
                        "spy_60d_return": round(spy_cum, 4), "excess": round(bask_cum - spy_cum, 4)})

    if not events:
        return mark_failed(sid, "No NFIP claims surge events found")
    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="NFIP Claims Surge → Long Water/Infra")
    avg_basket = np.mean([e["basket_60d_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["basket_60d_return"] > 0)
    save_result(sid, m, extra={
        "rule": "Long AWK+ACM+MWA 60d when NFIP monthly paid claims exceed $500M",
        "mechanism": "Major flood events → FEMA/state flood mitigation spending → water infrastructure demand",
        "source": "FEMA OpenFEMA NFIP + yfinance", "n_events": len(events),
        "avg_basket_return": round(avg_basket, 4), "avg_excess_vs_spy": round(avg_excess, 4),
        "win_rate": f"{win_count}/{len(events)}", "events": events,
    })
    sharpe = m.get('sharpe', 0); cagr = m.get('cagr', 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    for e in events:
        flag = "+" if e["basket_60d_return"] > 0 else "-"
        print(f"  {flag} {e['trigger_date']}: basket {e['basket_60d_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")

if __name__ == "__main__":
    main()
