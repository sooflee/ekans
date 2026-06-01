"""PL1026_mba_purchase_collapse_short_xhb — MBA Purchase Application Collapse (High-Rate Proxy) → Short XHB"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL1026_mba_purchase_collapse_short_xhb"

    try:
        fred = load_fred(["MORTGAGE30US", "HSN1F"], start="2000-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED data load: {e}")

    try:
        px = load_prices(["XHB", "DHI", "LEN", "SPY"], start="2001-01-01")
    except Exception as e:
        return mark_failed(sid, f"price data load: {e}")

    if "XHB" not in px.columns or "SPY" not in px.columns:
        return mark_failed(sid, "XHB or SPY not in price data")

    ret = daily_returns(px)
    spy_r = ret["SPY"]
    xhb_r = ret["XHB"]

    # Resample FRED weekly rate to daily (forward fill)
    rate = fred["MORTGAGE30US"].dropna().resample("D").ffill().reindex(xhb_r.index).ffill()
    # HSN1F is monthly, forward fill to daily
    hsn = fred["HSN1F"].dropna().resample("D").ffill().reindex(xhb_r.index).ffill()

    # Signal conditions:
    # 1. 30yr rate >= 6.5% for 4+ consecutive weeks (20 trading days)
    rate_high = (rate >= 6.5).astype(int)
    rate_consec = rate_high.rolling(20).sum() >= 20

    # 2. HSN1F 3-month rolling decline >= 5%: current < 3 months ago by >=5%
    hsn_3m_ago = hsn.shift(63)  # ~63 trading days = 3 months
    hsn_decline = (hsn < hsn_3m_ago * 0.95)  # >= 5% decline

    signal = rate_consec & hsn_decline

    signal_starts = signal.astype(int).diff().fillna(0) == 1
    signal_dates = signal.index[signal_starts]
    print(f"30yr rate >=6.5% for 20d: {rate_consec.sum()} days")
    print(f"HSN1F 3m decline >=5%: {hsn_decline.sum()} days")
    print(f"Combined signal days: {signal.sum()}")
    print(f"Signal episodes: {len(signal_dates)}")
    for d in signal_dates:
        print(f"  {d.date()}: rate={rate.get(d, np.nan):.2f}%, HSN={hsn.get(d, np.nan):.0f}K")

    hold_days = 42  # trading days
    take_profit = -0.15  # XHB drops 15%
    stop_loss = 0.08     # XHB rises 8%
    cooldown = 60        # calendar days

    pnl = pd.Series(0.0, index=ret.index)
    event_records = []
    last_exit = None

    for entry_date in signal_dates:
        if last_exit is not None and (entry_date - last_exit).days < cooldown:
            continue

        future_idx = xhb_r.index[xhb_r.index >= entry_date]
        if len(future_idx) < 5:
            continue

        entry_idx = future_idx[0]
        entry_loc = xhb_r.index.get_loc(entry_idx)
        exit_loc = min(entry_loc + hold_days, len(xhb_r))

        trade_window = xhb_r.iloc[entry_loc + 1:exit_loc]
        if len(trade_window) == 0:
            continue

        exit_reason = "time"
        active_dates = []
        cum_xhb = 1.0

        for td, r in trade_window.items():
            cum_xhb *= (1 + r)
            active_dates.append(td)

            # Take profit: XHB drops 15% (we win)
            if cum_xhb - 1 <= take_profit:
                exit_reason = "take_profit"
                break
            # Stop loss: XHB rises 8% (we lose)
            if cum_xhb - 1 >= stop_loss:
                exit_reason = "stop_loss"
                break

        if not active_dates:
            continue

        last_exit = active_dates[-1]

        xhb_cum = float((1 + xhb_r.loc[active_dates]).prod() - 1)
        spy_cum = float((1 + spy_r.reindex(active_dates).fillna(0)).prod() - 1)
        trade_pnl = -xhb_cum  # short XHB

        event_records.append({
            "entry": str(entry_idx.date()),
            "exit": str(active_dates[-1].date()),
            "n_days": len(active_dates),
            "rate_at_entry": round(float(rate.get(entry_idx, np.nan)), 2),
            "hsn_at_entry": round(float(hsn.get(entry_idx, np.nan)), 0),
            "xhb_cum": round(xhb_cum, 4),
            "trade_pnl": round(trade_pnl, 4),
            "spy_cum": round(spy_cum, 4),
            "exit_reason": exit_reason,
        })

        for td in active_dates:
            if td in pnl.index:
                pnl[td] -= xhb_r.get(td, 0.0)

    print(f"\nTrade records ({len(event_records)}):")
    for e in event_records:
        print(f"  {e['entry']} -> {e['exit']} ({e['n_days']}d, {e['exit_reason']}, rate={e['rate_at_entry']}%): "
              f"XHB={e['xhb_cum']:.2%}, pnl={e['trade_pnl']:.2%}, SPY={e['spy_cum']:.2%}")

    active_pnl = pnl[pnl != 0]
    print(f"Active pnl days: {len(active_pnl)}")

    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active pnl days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="MBA Purchase Collapse Short XHB")

    save_result(sid, m, extra={
        "rule": "Short XHB when MORTGAGE30US >=6.5% for 20+ consecutive days AND HSN1F 3m decline >=5%. Hold 42 days, take profit at -15%, stop at +8%.",
        "mechanism": "High mortgage rates crush housing affordability; falling new home sales confirm demand destruction. Homebuilder stocks reprice lower as forward earnings estimates decline.",
        "source": "FRED MORTGAGE30US (weekly 30yr fixed); FRED HSN1F (monthly new home sales); XHB via yfinance.",
        "n_events": len(event_records),
        "events": event_records,
    })


if __name__ == "__main__":
    main()
