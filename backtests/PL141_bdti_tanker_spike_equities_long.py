"""PL141_bdti_tanker_spike_equities_long — Tanker Stock 2-Week Momentum Spike >15% → Long 20d
When equal-weight FRO+STNG+INSW basket 10-day return exceeds +15%, long 20 more days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL141_bdti_tanker_spike_equities_long"
    try:
        px = load_prices(["FRO", "STNG", "INSW", "SPY"], start="2001-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    basket_tickers = [t for t in ["FRO", "STNG", "INSW"] if t in ret.columns]
    if not basket_tickers:
        return mark_failed(sid, "No tanker tickers available")
    basket_r = ret[basket_tickers].mean(axis=1)

    # Compute rolling 10-day return
    basket_px = (1 + basket_r).cumprod()
    roll_10d = basket_px / basket_px.shift(10) - 1

    # Find signals where 10-day return > 15%
    triggers = []
    cooldown = 0
    for i in range(10, len(roll_10d)):
        if cooldown > 0:
            cooldown -= 1
            continue
        if float(roll_10d.iloc[i]) > 0.15:
            triggers.append(roll_10d.index[i])
            cooldown = 30  # avoid overlapping

    events = []
    pnl_parts = []
    hold_days = 20

    for trig_date in triggers:
        entry_loc = ret.index.get_loc(trig_date)
        if entry_loc + hold_days >= len(ret.index):
            continue
        window = slice(entry_loc + 1, entry_loc + 1 + hold_days)
        basket_window = basket_r.iloc[window]
        spy_window = spy_r.iloc[window]
        pnl_parts.append(basket_window)

        bask_cum = float((1 + basket_window).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)

        events.append({
            "trigger_date": str(trig_date.date()),
            "momentum_10d": round(float(roll_10d.loc[trig_date]), 4),
            "basket_20d_return": round(bask_cum, 4),
            "spy_20d_return": round(spy_cum, 4),
            "excess": round(bask_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No tanker momentum spike events found")

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="Tanker Momentum Spike → Long 20d")

    avg_basket = np.mean([e["basket_20d_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["basket_20d_return"] > 0)

    save_result(sid, m, extra={
        "rule": "Equal-weight FRO+STNG+INSW 10d return > +15% → long basket 20 more days",
        "mechanism": "Tanker rate spikes visible in equity momentum tend to persist as charters cascade",
        "source": "yfinance (FRO, STNG, INSW)",
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
        flag = "+" if e["basket_20d_return"] > 0 else "-"
        print(f"  {flag} {e['trigger_date']} (mom={e['momentum_10d']*100:.1f}%): basket {e['basket_20d_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
