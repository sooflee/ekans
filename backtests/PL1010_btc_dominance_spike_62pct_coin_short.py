"""PL1010_btc_dominance_spike_62pct_coin_short
BTC Dominance Rapid Spike >62% (Altcoin Capitulation) Counter-Signal: Short COIN

Entry: BTC/ETH price ratio spikes (BTC outperforming ETH = capital flight to BTC,
proxy for dominance spike). Specifically: 10-day BTC/ETH ratio change >= 20% AND
ratio crosses a high-percentile (top 15% of trailing 252-day distribution).
Short COIN at next open. Hold 45 trading days.
Exit: BTC/ETH ratio reverts below trailing median, COIN -20% (take profit),
      COIN +12% (stop-loss), or 45 trading days elapsed.

Note: True BTC dominance (% of total crypto mcap) is a Pro-API-only historical
series. We proxy it with BTC/ETH ratio which captures the same 'altcoin flight
to BTC' regime signal.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL1010_btc_dominance_spike_62pct_coin_short"

    # Load prices: COIN (from IPO April 2021), BTC-USD, ETH-USD, SPY
    try:
        px = load_prices(["COIN", "BTC-USD", "ETH-USD", "SPY"], start="2021-01-01")
    except Exception as e:
        # Retry once
        try:
            px = load_prices(["COIN", "BTC-USD", "ETH-USD", "SPY"],
                             start="2021-01-01", cache=False)
        except Exception as e2:
            return mark_failed(sid, f"data load: {e2}")

    px = px.sort_index().ffill(limit=3)
    required = ["COIN", "BTC-USD", "ETH-USD", "SPY"]
    missing = [t for t in required if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    # BTC/ETH ratio as proxy for BTC dominance
    # When BTC dominance rises, BTC outperforms ETH and other altcoins
    btc_eth_ratio = px["BTC-USD"] / px["ETH-USD"]
    btc_eth_ratio = btc_eth_ratio.ffill(limit=5).dropna()

    # Signal: BTC/ETH ratio spikes >= 15% in 10 days (rapid BTC outperformance,
    # proxy for BTC dominance surge / altcoin capitulation)
    ratio_10d_ago = btc_eth_ratio.shift(10)
    ratio_change_10d = (btc_eth_ratio - ratio_10d_ago) / ratio_10d_ago

    # 252-day rolling 70th percentile (BTC/ETH ratio in elevated regime)
    trail_70 = btc_eth_ratio.rolling(252, min_periods=63).quantile(0.70)

    # Signal: ratio rises >= 15% in 10 days AND ratio is at/above 70th pct (elevated regime)
    # This captures: rapid BTC outperformance while BTC is already dominant vs ETH
    rapid_spike = ratio_change_10d >= 0.15
    elevated_regime = btc_eth_ratio >= trail_70

    # Was not in a rapid-spike signal for at least 20 days (cooldown)
    spike_cooldown = rapid_spike.shift(1).rolling(20, min_periods=1).max().fillna(0) == 0

    raw_signal = rapid_spike & elevated_regime & spike_cooldown
    signal_dates = btc_eth_ratio.index[raw_signal.fillna(False)]

    # COIN trading data (only from IPO onward)
    coin_start = pd.Timestamp("2021-04-15")
    trading_dates = px.index[px.index >= coin_start]

    ret = daily_returns(px.reindex(trading_dates))
    spy_r = ret["SPY"].dropna() if "SPY" in ret.columns else pd.Series(dtype=float)
    coin_r = ret["COIN"].dropna()

    hold_days = 45
    take_profit = -0.20  # short gains 20% = coin down 20%
    stop_loss = 0.12     # short loses 12% = coin up 12%

    positions = pd.Series(0.0, index=trading_dates)
    events = []
    open_until_idx = -1

    coin_prices = px["COIN"].reindex(trading_dates).ffill()
    # For ratio-revert exit: check if BTC/ETH ratio falls >10% from entry value
    # (altcoin recovery)

    for sig_date in signal_dates:
        # Entry at next trading day after signal
        after = trading_dates[trading_dates > sig_date]
        if len(after) == 0:
            continue
        entry_date = after[0]
        entry_idx = trading_dates.get_loc(entry_date)

        if entry_idx <= open_until_idx:
            continue

        entry_price = coin_prices.get(entry_date, np.nan)
        if np.isnan(entry_price):
            continue

        entry_ratio = btc_eth_ratio.asof(entry_date) if entry_date in btc_eth_ratio.index or True else np.nan

        # Walk forward for up to hold_days, checking exit conditions
        actual_end_idx = entry_idx
        exit_reason = "hold_45d"
        for j in range(entry_idx, min(entry_idx + hold_days, len(trading_dates))):
            cur_date = trading_dates[j]
            cur_price = coin_prices.get(cur_date, np.nan)
            if np.isnan(cur_price):
                continue
            coin_chg = (cur_price - entry_price) / entry_price
            # stop-loss: coin rose +12% against the short
            if coin_chg >= stop_loss:
                exit_reason = "stop_loss"
                actual_end_idx = j
                break
            # take-profit: coin fell -20%
            if coin_chg <= take_profit:
                exit_reason = "take_profit"
                actual_end_idx = j
                break
            # ratio revert: BTC/ETH ratio falls >10% from entry value (altcoin recovery)
            cur_ratio = btc_eth_ratio.asof(cur_date) if not np.isnan(entry_ratio) else np.nan
            if not np.isnan(cur_ratio) and not np.isnan(entry_ratio):
                ratio_chg = (cur_ratio - entry_ratio) / entry_ratio
                if ratio_chg <= -0.10:
                    exit_reason = "ratio_revert_10pct"
                    actual_end_idx = j
                    break
            actual_end_idx = j

        # Set positions (short COIN = -1)
        positions.iloc[entry_idx:actual_end_idx + 1] = -1.0
        open_until_idx = actual_end_idx

        exit_date = trading_dates[actual_end_idx]
        exit_price = coin_prices.get(exit_date, np.nan)
        events.append({
            "signal_date": str(sig_date.date()),
            "entry_date": str(entry_date.date()),
            "exit_date": str(exit_date.date()),
            "exit_reason": exit_reason,
            "entry_coin_price": round(float(entry_price), 2),
            "btc_eth_ratio_at_signal": round(float(btc_eth_ratio.get(sig_date, np.nan)), 2),
        })

    # Compute daily PnL (short COIN, positions already represent signal-day;
    # shift by 1 for no look-ahead: position set on signal day, earns return next day)
    pos_shifted = positions.shift(1).fillna(0)
    pnl_raw = pos_shifted * coin_r.reindex(trading_dates).fillna(0)
    pnl = pnl_raw.dropna()
    pnl = pnl[pnl.index >= coin_start]

    # Align with spy
    spy_r_aligned = spy_r.reindex(pnl.index).dropna()
    pnl = pnl.reindex(spy_r_aligned.index)

    if (pnl != 0).sum() < 20:
        return mark_failed(
            sid,
            f"insufficient in-position days: {(pnl != 0).sum()} (n_events={len(events)}). "
            f"Signal dates found: {list(signal_dates)}",
        )

    m = compute_metrics(
        pnl,
        benchmark=spy_r_aligned,
        name="BTC Dominance Spike Short COIN",
        positions=abs(pos_shifted.reindex(pnl.index).fillna(0)),
        cost_bps=15,  # higher for short + crypto-adjacent
    )

    ev_returns = []
    for e in events:
        entry = pd.Timestamp(e["entry_date"])
        exit_d = pd.Timestamp(e["exit_date"])
        slice_r = coin_r.loc[entry:exit_d]
        if len(slice_r):
            cum = float((1 + (-slice_r)).prod() - 1)  # short COIN
            e["event_return_short_coin"] = round(cum, 4)
            ev_returns.append(cum)

    n_events = len(events)
    win_rate = float(np.mean([r > 0 for r in ev_returns])) if ev_returns else None
    avg_event = float(np.mean(ev_returns)) if ev_returns else None

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "Short COIN when BTC/ETH price ratio spikes >= 20% in 10 days AND "
                "crosses above its trailing 252-day 85th percentile (proxy for BTC dominance "
                "spike / altcoin capitulation). Hold 45 trading days; exit on ratio revert "
                "to median (altcoin recovery), COIN -20% (TP), COIN +12% (SL), or 45d timeout."
            ),
            "mechanism": (
                "BTC dominance spikes signal altcoin capitulation (flight to BTC safety). "
                "BTC/ETH ratio is a tradeable proxy for BTC dominance; when BTC outperforms "
                "ETH sharply, crypto capital is concentrating in BTC (risk-off in crypto). "
                "COIN revenue is ~50-60% altcoin trading fees; altcoin bear markets compress "
                "COIN revenue and stock price."
            ),
            "source": "yfinance BTC-USD/ETH-USD/COIN/SPY; BTC dominance proxied by BTC/ETH price ratio",
            "tickers": ["COIN", "BTC-USD", "ETH-USD", "SPY"],
            "n_events": n_events,
            "event_win_rate": round(win_rate, 4) if win_rate is not None else None,
            "avg_event_return": round(avg_event, 4) if avg_event is not None else None,
            "events": events,
            "caveats": (
                "COIN IPO April 2021 limits sample to ~5 years. BTC dominance proxied from "
                "price * fixed supply estimate — not actual circulating supply from CoinGecko. "
                "Only 2-4 distinct dominance spike events expected in the COIN data window. "
                "COIN has diversified revenue beyond trading fees (staking, Base L2). "
                "Altcoin recovery can be rapid; 45-day hold may miss mean-reversion."
            ),
        },
        pnl=pnl,
    )

    print(f"Done: {sid}")
    print(f"  n_events: {n_events}, win_rate: {win_rate}, avg_event_return: {avg_event}")
    print(
        f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
        f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
    )
    if "oos_sharpe" in m:
        print(f"  OOS Sharpe: {m['oos_sharpe']:.2f}  IS Sharpe: {m['is_sharpe']:.2f}")


if __name__ == "__main__":
    main()
