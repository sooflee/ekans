"""PL38_rrp_drain_equity_liquidity — Fed ON RRP Drain Below $500B → Long SPY
When RRPONTSYD crosses below $500B after being above $1T in prior 12mo: long SPY 252 days.
Also test: monthly RRP change as continuous signal.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL38_rrp_drain_equity_liquidity"
    try:
        fred = load_fred("RRPONTSYD", start="2013-01-01")
        rrp = fred.squeeze()
    except Exception as e:
        return mark_failed(sid, f"FRED data load: {e}")

    if rrp.empty:
        return mark_failed(sid, "RRPONTSYD data empty")

    # Convert to billions
    rrp_b = rrp / 1e3 if rrp.max() > 1e6 else rrp  # already in millions on FRED? Check units
    # FRED RRPONTSYD is in millions of dollars
    if rrp.max() > 1e9:
        rrp_b = rrp / 1e9
    elif rrp.max() > 1e6:
        rrp_b = rrp / 1e3
    else:
        rrp_b = rrp  # already in billions

    print(f"RRP range: {rrp_b.min():.1f}B to {rrp_b.max():.1f}B")

    # Event-based: cross below $500B after being above $1T in prior 252 business days
    trigger_dates = []
    for i in range(252, len(rrp_b)):
        if rrp_b.iloc[i] < 500 and rrp_b.iloc[i-1] >= 500:
            # Check if was above 1T in prior 12 months
            lookback = rrp_b.iloc[max(0, i-252):i]
            if lookback.max() > 1000:
                trigger_dates.append(rrp_b.index[i])

    print(f"RRP cross below $500B events (after >$1T): {len(trigger_dates)}")
    for d in trigger_dates:
        print(f"  {d.date()}: RRP = {rrp_b.loc[d]:.1f}B")

    # Also test continuous signal: SPY returns when RRP is draining >$100B/month
    rrp_monthly = rrp_b.resample("M").last().dropna()
    rrp_monthly_chg = rrp_monthly.diff()
    draining = rrp_monthly_chg < -100  # draining >$100B/month

    try:
        px = load_prices("SPY", start="2013-01-01")
        spy = px.squeeze() if isinstance(px, pd.DataFrame) else px
    except Exception as e:
        return mark_failed(sid, f"equity data load: {e}")

    spy_ret = daily_returns(px)
    if isinstance(spy_ret, pd.DataFrame):
        spy_ret = spy_ret.iloc[:, 0]

    hold_days = 252
    pnl_series = pd.Series(0.0, index=spy_ret.index)
    event_results = []

    # Event-based trades
    for td in trigger_dates:
        entry_mask = spy_ret.index >= td
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = spy_ret.index[entry_mask][0]
        pos = spy_ret.index.get_loc(entry_idx)
        end_pos = min(pos + hold_days, len(spy_ret))
        event_rets = spy_ret.iloc[pos:end_pos]
        cumret = float((1 + event_rets).prod() - 1)
        pnl_series.iloc[pos:end_pos] = event_rets.values[:end_pos - pos]
        event_results.append({
            "trigger_date": str(td.date()),
            "rrp_b": round(float(rrp_b.loc[td]), 1),
            "spy_12m_return": round(cumret, 4),
        })

    # If no threshold events, use continuous draining signal
    if len(event_results) == 0:
        print("No threshold events — using continuous draining signal (RRP declining >$100B/month)")
        for date, is_drain in draining.items():
            if not is_drain:
                continue
            entry_date = date + pd.offsets.MonthBegin(1)
            entry_mask = spy_ret.index >= entry_date
            if entry_mask.sum() < 126:
                continue
            entry_idx = spy_ret.index[entry_mask][0]
            pos = spy_ret.index.get_loc(entry_idx)
            end_pos = min(pos + 126, len(spy_ret))
            event_rets = spy_ret.iloc[pos:end_pos]
            cumret = float((1 + event_rets).prod() - 1)
            pnl_series.iloc[pos:end_pos] = event_rets.values[:end_pos - pos]
            event_results.append({
                "trigger_date": str(date.date()),
                "rrp_monthly_chg_b": round(float(rrp_monthly_chg.loc[date]), 1),
                "spy_6m_return": round(cumret, 4),
            })

    if len(event_results) == 0:
        return mark_failed(sid, "no RRP drain events found")

    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) < 30:
        return mark_failed(sid, f"insufficient in-position days ({len(in_pos)})")

    m = compute_metrics(in_pos, benchmark=spy_ret, name="RRP Drain → Long SPY")
    key = "spy_12m_return" if "spy_12m_return" in event_results[0] else "spy_6m_return"
    rets_arr = [e[key] for e in event_results]
    save_result(sid, m, extra={
        "rule": "Long SPY when ON RRP drains below $500B (after being >$1T) for 252 days, or when draining >$100B/month",
        "mechanism": "RRP drain → liquidity flowing back into banking system and risk assets",
        "source": "FRED RRPONTSYD; yfinance",
        "n_events": len(event_results),
        "avg_event_return": round(float(np.mean(rets_arr)), 4),
        "events": event_results,
    })
    print(f"Done: {len(event_results)} events, avg return={np.mean(rets_arr)*100:.2f}%")


if __name__ == "__main__":
    main()
