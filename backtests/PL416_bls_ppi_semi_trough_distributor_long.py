"""PL416_bls_ppi_semi_trough_distributor_long — BLS PPI Semiconductor Trough -> Long Chip Distributors
When FRED PPI semiconductor YoY deflation exceeds -5% then starts recovering (turns less negative),
long ARW+AVT for 42 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL416_bls_ppi_semi_trough_distributor_long"
    try:
        px = load_prices(["ARW", "AVT", "SPY"], start="2000-01-01")
    except Exception as e:
        return mark_failed(sid, f"price data load: {e}")

    # Load PPI semiconductor from FRED
    ppi = None
    for series in ["PCU334413334413", "PCU3344133344131", "PPIIDC"]:
        try:
            ppi = load_fred(series, start="1998-01-01").squeeze()
            if ppi.dropna().empty:
                ppi = None
                continue
            break
        except Exception:
            continue
    if ppi is None:
        return mark_failed(sid, "Could not load FRED PPI semiconductor (PCU334413334413)")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    basket_tickers = [t for t in ["ARW", "AVT"] if t in ret.columns]
    if not basket_tickers:
        return mark_failed(sid, "No chip distributor tickers available")
    basket_r = ret[basket_tickers].mean(axis=1)

    # Monthly PPI, YoY
    ppi_m = ppi.resample("M").last().dropna()
    ppi_yoy = ppi_m.pct_change(12)

    # Find trough-to-recovery: YoY was < -5% and now improving (less negative by > 2pp)
    triggers = []
    last_trigger = None
    trough = 0.0

    for i in range(13, len(ppi_yoy)):
        dt = ppi_yoy.index[i]
        val = float(ppi_yoy.iloc[i])
        prev = float(ppi_yoy.iloc[i-1])
        if np.isnan(val) or np.isnan(prev):
            continue

        if val < -0.05:
            trough = min(trough, val)
        elif trough < -0.05 and val > trough + 0.02:
            # Recovery from deep deflation
            if last_trigger is None or (dt - last_trigger).days >= 90:
                triggers.append(dt)
                last_trigger = dt
                trough = 0.0
        else:
            if val > 0:
                trough = 0.0

    if not triggers:
        return mark_failed(sid, "No PPI semiconductor trough-recovery events found")

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

        events.append({
            "trigger_date": str(trig_date.date()),
            "ppi_yoy": round(float(ppi_yoy.loc[trig_date]), 4),
            "basket_42d_return": round(bask_cum, 4),
            "spy_42d_return": round(spy_cum, 4),
            "excess": round(bask_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No tradeable PPI semi trough events")

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="PPI Semi Trough Recovery -> Long ARW+AVT")

    avg_basket = np.mean([e["basket_42d_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["basket_42d_return"] > 0)

    save_result(sid, m, extra={
        "rule": "FRED PPI semiconductor YoY deflation > -5% then recovers by > 2pp -> long ARW+AVT 42d",
        "mechanism": "PPI deflation trough signals end of inventory destocking; chip distributors benefit from restocking cycle",
        "source": "FRED PCU334413334413 + yfinance",
        "n_events": len(events),
        "avg_basket_return": round(avg_basket, 4),
        "avg_excess_vs_spy": round(avg_excess, 4),
        "win_rate": f"{win_count}/{len(events)}",
        "events": events,
    })
    sharpe = m.get('sharpe', 0)
    cagr = m.get('cagr', 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    for e in events:
        flag = "+" if e["basket_42d_return"] > 0 else "-"
        print(f"  {flag} {e['trigger_date']}: ppi_yoy={e['ppi_yoy']*100:.1f}%, basket {e['basket_42d_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} -- Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
