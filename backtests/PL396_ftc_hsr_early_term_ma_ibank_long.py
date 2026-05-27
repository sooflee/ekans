"""PL396_ftc_hsr_early_term_ma_ibank_long — FTC HSR Early Termination Surge -> Long M&A-Levered Investment Banks
Proxy: when IAI (broker-dealer ETF) outperforms SPY by > 5% over 20 trading days,
long EVR+PJT (or EVR+LAZ pre-2015) for 42 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL396_ftc_hsr_early_term_ma_ibank_long"
    try:
        px = load_prices(["EVR", "PJT", "LAZ", "IAI", "SPY"], start="2006-01-01")
    except Exception as e:
        return mark_failed(sid, f"price data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    if "IAI" not in ret.columns:
        return mark_failed(sid, "IAI (broker-dealer ETF) data not available")

    # Compute 20-day rolling excess return of IAI vs SPY
    iai_r = ret["IAI"]
    iai_cum20 = (1 + iai_r).rolling(20).apply(lambda x: x.prod() - 1, raw=True)
    spy_cum20 = (1 + spy_r).rolling(20).apply(lambda x: x.prod() - 1, raw=True)
    excess_20d = iai_cum20 - spy_cum20

    # Find dates where excess crosses +5%
    triggers = []
    last_trigger = None
    for i in range(1, len(excess_20d)):
        dt = excess_20d.index[i]
        val = float(excess_20d.iloc[i])
        prev = float(excess_20d.iloc[i-1])
        if np.isnan(val) or np.isnan(prev):
            continue
        # Cross above 5% threshold
        if val > 0.05 and prev <= 0.05:
            if last_trigger is None or (dt - last_trigger).days >= 90:
                triggers.append(dt)
                last_trigger = dt

    if not triggers:
        return mark_failed(sid, "No IAI outperformance > 5% over 20d events found")

    events = []
    pnl_parts = []
    hold_days = 42

    for trig_date in triggers:
        # Pick basket based on date (PJT available from Oct 2015)
        if trig_date >= pd.Timestamp("2015-11-01"):
            tickers = [t for t in ["EVR", "PJT"] if t in ret.columns]
        else:
            tickers = [t for t in ["EVR", "LAZ"] if t in ret.columns]
        if not tickers:
            continue
        basket_r = ret[tickers].mean(axis=1)

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
            "iai_excess_20d": round(float(excess_20d.loc[trig_date]), 4),
            "basket_42d_return": round(bask_cum, 4),
            "spy_42d_return": round(spy_cum, 4),
            "excess": round(bask_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No tradeable M&A broker-dealer events")

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="IAI Outperformance -> Long EVR+PJT (M&A Boutiques)")

    avg_basket = np.mean([e["basket_42d_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["basket_42d_return"] > 0)

    save_result(sid, m, extra={
        "rule": "IAI (broker-dealer ETF) outperforms SPY by > 5% over 20d -> long EVR+PJT/LAZ 42 days",
        "mechanism": "Broker-dealer momentum signals M&A cycle turning; boutique advisors have highest operating leverage to deal volume",
        "source": "FTC HSR (proxy via IAI momentum) + yfinance",
        "n_events": len(events),
        "avg_basket_return": round(avg_basket, 4),
        "avg_excess_vs_spy": round(avg_excess, 4),
        "win_rate": f"{win_count}/{len(events)}",
        "events": events,
    })
    sharpe = m.get('sharpe', 0)
    cagr = m.get('cagr', 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    print(f"  Avg basket: {avg_basket*100:.1f}%  Avg excess: {avg_excess*100:.1f}%  Win: {win_count}/{len(events)}")
    for e in events:
        flag = "+" if e["basket_42d_return"] > 0 else "-"
        print(f"  {flag} {e['trigger_date']}: IAI_ex={e['iai_excess_20d']*100:.1f}%, basket {e['basket_42d_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} -- Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
