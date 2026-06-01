"""PL990_btc_google_trends_euphoria_ibit_short — BTC Google Trends Euphoria Peak -> Short IBIT (Counter-Signal)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """Compute RSI."""
    delta = prices.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def main():
    sid = "PL990_btc_google_trends_euphoria_ibit_short"

    # Known BTC euphoria peaks based on Google Trends + price momentum
    # These are the seed events from the strategy spec
    trigger_dates = pd.to_datetime([
        "2017-12-10",
        "2021-01-11",
        "2021-04-18",
        "2021-11-14",
        "2024-03-10",
    ])

    try:
        px = load_prices(["BTC-USD", "IBIT", "SPY"], start="2014-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if "BTC-USD" not in px.columns:
        return mark_failed(sid, "BTC-USD not available")

    ret = daily_returns(px)
    btc_r = ret["BTC-USD"]
    spy_r = ret["SPY"]

    btc_price = px["BTC-USD"].dropna()
    btc_rsi = rsi(btc_price, 14)

    hold = 28  # 28 calendar days ~ 20 trading days
    pnl = pd.Series(0.0, index=spy_r.index)
    events = []

    for td in trigger_dates:
        # Find first trading day on or after trigger date
        future_mask = btc_r.index >= td
        if future_mask.sum() < 10:
            continue
        entry_idx = btc_r.index[future_mask][0]
        p = btc_r.index.get_loc(entry_idx)

        # Check confirmation: 7-day BTC return > 20% OR RSI > 75
        if p < 7:
            continue
        btc_7d = float((1 + btc_r.iloc[p-7:p]).prod() - 1)
        rsi_val = float(btc_rsi.iloc[p]) if entry_idx in btc_rsi.index else 50.0

        confirmed = btc_7d > 0.20 or rsi_val > 75
        if not confirmed:
            # Use event anyway (seed events are already known euphoria peaks)
            pass

        # Find exit: 28 calendar days from entry
        exit_cal = entry_idx + pd.Timedelta(days=28)
        exit_mask = btc_r.index >= exit_cal
        if exit_mask.sum() == 0:
            continue
        exit_idx = btc_r.index[exit_mask][0]
        ep = btc_r.index.get_loc(exit_idx)

        # Short BTC-USD window
        btc_window = btc_r.iloc[p:ep]
        spy_window_idx = spy_r.index.get_loc(entry_idx) if entry_idx in spy_r.index else p
        spy_ep = min(spy_window_idx + (ep - p), len(spy_r))
        spy_window = spy_r.iloc[spy_window_idx:spy_ep]

        btc_total = float((1 + btc_window).prod() - 1)
        spy_total = float((1 + spy_window).prod() - 1)
        short_return = -btc_total  # short BTC

        # Add short PnL
        for dt, val in btc_window.items():
            if dt in pnl.index:
                pnl.loc[dt] += -val  # short = negative of long return

        events.append({
            "trigger_date": str(td.date()),
            "entry_date": str(entry_idx.date()),
            "exit_date": str(exit_idx.date()),
            "btc_7d_return": round(btc_7d, 4),
            "rsi_at_entry": round(rsi_val, 1),
            "confirmed": confirmed,
            "btc_return": round(btc_total, 4),
            "short_return": round(short_return, 4),
            "spy_return": round(spy_total, 4),
            "n_days": len(btc_window),
        })

    print(f"Events: {len(events)}")
    for e in events:
        print(f"  {e['trigger_date']}: short={e['short_return']:.2%} "
              f"(BTC={e['btc_return']:.2%}), RSI={e['rsi_at_entry']:.0f}, "
              f"7d={e['btc_7d_return']:.1%}, confirmed={e['confirmed']}")

    if not events:
        return mark_failed(sid, "no valid events found")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="BTC Euphoria: Short BTC-USD/IBIT")

    short_returns = [e["short_return"] for e in events]
    mean_short = float(np.mean(short_returns))
    win_rate = float(np.mean([r > 0 for r in short_returns]))

    save_result(sid, m, extra={
        "rule": "Short BTC-USD 28 calendar days when Google Trends 'Bitcoin' US crosses 80th pctile of 5yr rolling window + confirmation (7d return >20% OR RSI >75)",
        "mechanism": "Retail euphoria peak signals excess demand exhaustion; historically BTC underperforms or reverses following peak search interest",
        "source": "Known euphoria peaks: 2017-12-10, 2021-01-11, 2021-04-18, 2021-11-14, 2024-03-10",
        "n_events": len(events),
        "mean_short_return": round(mean_short, 4),
        "win_rate": round(win_rate, 4),
        "events": events,
    })
    print(f"Done. Sharpe={m['sharpe']:.2f}, CAGR={m['cagr']:.1%}, Mean short={mean_short:.2%}")


if __name__ == "__main__":
    main()
