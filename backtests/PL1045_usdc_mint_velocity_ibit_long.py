"""PL1045 — USDC On-Chain Mint Velocity Surge → Stablecoin Dry-Powder → Long IBIT"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import urllib.request
import json as json_mod
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def fetch_usdc_market_cap(days=365):
    """Fetch USDC market cap data from CoinGecko (free tier, max 365 days)."""
    url = f"https://api.coingecko.com/api/v3/coins/usd-coin/market_chart?vs_currency=usd&days={days}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json_mod.loads(resp.read())
        if "error" in data:
            return None, str(data["error"])
        market_caps = data.get("market_caps", [])
        if not market_caps:
            return None, "empty market_caps"
        # Convert to daily series
        df = pd.DataFrame(market_caps, columns=["timestamp_ms", "market_cap"])
        df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms").dt.normalize()
        df = df.set_index("date").sort_index()
        # Deduplicate (sometimes CoinGecko returns hourly for recent days)
        df = df["market_cap"].resample("D").last().dropna()
        return df, None
    except Exception as e:
        return None, str(e)


def main():
    sid = "PL1045_usdc_mint_velocity_ibit_long"

    # Fetch USDC supply (as market cap proxy) from CoinGecko
    usdc_supply, err = fetch_usdc_market_cap(days=365)
    if usdc_supply is None or err is not None:
        return mark_failed(sid, f"USDC data fetch failed: {err}")

    print(f"USDC data: {len(usdc_supply)} days, from {usdc_supply.index[0].date()} to {usdc_supply.index[-1].date()}")
    print(f"USDC range: {usdc_supply.min()/1e9:.1f}B - {usdc_supply.max()/1e9:.1f}B")

    # Resample to weekly (Monday) to reduce noise
    usdc_weekly = usdc_supply.resample("W-MON").last().dropna()

    # Weekly change
    weekly_change = usdc_weekly.diff()
    # 3-week rolling average of weekly change
    rolling_3w_avg = weekly_change.rolling(window=3, min_periods=3).mean()

    # Signal: current week's supply increase > 3-week rolling avg * 1.20
    # Only trigger on positive supply increases
    signal_mask = (
        (weekly_change > 0) &
        (rolling_3w_avg > 0) &
        (weekly_change > rolling_3w_avg * 1.20)
    )

    signal_dates = usdc_weekly.index[signal_mask].tolist()
    print(f"Raw signals: {[str(d.date()) for d in signal_dates]}")

    if not signal_dates:
        return mark_failed(sid, "no USDC mint velocity acceleration signals found in 365-day window")

    # Load prices
    try:
        px = load_prices(["IBIT", "BTC-USD", "SPY"], start="2024-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    ret = daily_returns(px)
    # Use IBIT if available (launched Jan 2024), fall back to BTC-USD
    if "IBIT" in ret.columns and ret["IBIT"].dropna().shape[0] > 100:
        vehicle_r = ret["IBIT"]
        vehicle_name = "IBIT"
    else:
        vehicle_r = ret["BTC-USD"]
        vehicle_name = "BTC-USD"
    spy_r = ret["SPY"]

    # 50-day SMA filter
    vehicle_px = px[vehicle_name].dropna() if vehicle_name in px else px["IBIT"].dropna()
    sma50 = vehicle_px.rolling(50).mean()

    hold_td = 15  # ~3 weeks of trading days
    pnl = pd.Series(0.0, index=vehicle_r.index)
    events = []
    last_exit_date = None
    COOLDOWN_WEEKS = 3

    for sig_date in signal_dates:
        # Skip if still in cooldown
        if last_exit_date is not None and (sig_date - last_exit_date).days < COOLDOWN_WEEKS * 7:
            continue

        # Entry on Monday open of following week (next Monday after signal)
        next_monday = sig_date + pd.DateOffset(weeks=1)
        future_mask = vehicle_r.index >= next_monday
        if future_mask.sum() < 5:
            continue
        entry_idx = vehicle_r.index[future_mask][0]
        entry_pos = vehicle_r.index.get_loc(entry_idx)

        # Check 50-day SMA filter
        if entry_idx in sma50.index:
            sma_val = float(sma50.loc[entry_idx]) if not np.isnan(float(sma50.loc[entry_idx])) else None
            price_val = float(vehicle_px.loc[entry_idx]) if entry_idx in vehicle_px.index else None
            if sma_val and price_val and price_val < sma_val:
                print(f"Skipping {sig_date.date()} — {vehicle_name} below 50-day SMA")
                continue

        exit_pos = min(entry_pos + hold_td, len(vehicle_r))
        window = vehicle_r.iloc[entry_pos:exit_pos]

        # Stop-loss: vehicle drops >8% from entry
        cum = 1.0
        actual_exit = exit_pos
        stop_type = None
        for j, r in enumerate(window):
            cum *= (1 + r)
            if cum < 0.92:
                actual_exit = entry_pos + j + 1
                stop_type = "stop_loss"
                break

        window_actual = vehicle_r.iloc[entry_pos:actual_exit]
        cum_ret = float((1 + window_actual).prod() - 1)

        spy_future = spy_r.index >= next_monday
        spy_cum = None
        if spy_future.sum() >= 5:
            sp_entry = spy_r.index[spy_future][0]
            sp_pos = spy_r.index.get_loc(sp_entry)
            sp_win = spy_r.iloc[sp_pos: sp_pos + len(window_actual)]
            spy_cum = float((1 + sp_win).prod() - 1)

        pnl.iloc[entry_pos:actual_exit] = window_actual.values[: actual_exit - entry_pos]
        last_exit_date = vehicle_r.index[actual_exit - 1] if actual_exit > entry_pos else entry_idx

        events.append({
            "signal_date": str(sig_date.date()),
            "entry_date": str(entry_idx.date()),
            "usdc_weekly_change_bn": round(float(weekly_change.loc[sig_date]) / 1e9, 2),
            "vehicle_return": round(cum_ret, 4),
            "spy_return": round(spy_cum, 4) if spy_cum is not None else None,
            "exit_type": stop_type or "time_stop",
        })

    print(f"Events after filtering: {events}")

    if not events:
        return mark_failed(sid, "no valid events after entry filters (SMA/cooldown)")

    active = pnl[pnl != 0]
    print(f"Active trading days: {len(active)}")

    if len(active) < 15:
        return mark_failed(sid, f"insufficient active trading days ({len(active)}) — {len(events)} event(s) in 365-day window")

    m = compute_metrics(active, benchmark=spy_r, name=f"USDC Mint Velocity → Long {vehicle_name}")
    veh_rets = [e["vehicle_return"] for e in events]
    save_result(sid, m, extra={
        "rule": f"Long {vehicle_name} for ~3 weeks when weekly USDC supply increase > 3-week rolling avg * 1.20 and {vehicle_name} above 50-day SMA",
        "mechanism": "USDC mint velocity acceleration signals fresh capital entering crypto ecosystem (dry-powder inflow), typically preceding BTC/IBIT price appreciation within 1-3 weeks",
        "source": "CoinGecko USDC market cap (365-day free tier); yfinance IBIT, BTC-USD, SPY",
        "n_events": len(events),
        "avg_event_return": round(float(np.mean(veh_rets)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in veh_rets])), 4),
        "events": events,
        "caveat": "CoinGecko free tier limits to 365 days of history; full backtest requires paid plan or alternative USDC supply data source",
    })
    print(f"Done: {len(events)} events, Sharpe={m.get('sharpe', 'N/A')}")


if __name__ == "__main__":
    main()
