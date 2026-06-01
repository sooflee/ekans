"""PL1044_ibit_ap_creation_mstr_leveraged_lead — Spot BTC ETF AP Net Creation Surge → MSTR Leveraged NAV Lead"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL1044_ibit_ap_creation_mstr_leveraged_lead"

    # IBIT launched Jan 11, 2024 - we use volume as proxy for AP creation activity
    # High volume relative to recent baseline signals AP creations/redemptions
    try:
        px = load_prices(["MSTR", "BTC-USD", "IBIT", "SPY"], start="2024-01-01")
    except Exception as e:
        try:
            px = load_prices(["MSTR", "BTC-USD", "IBIT", "SPY"], start="2024-01-01")
        except Exception as e2:
            return mark_failed(sid, f"price load: {e2}")

    if px.empty or len(px) < 30:
        return mark_failed(sid, "insufficient price data (IBIT window too short)")

    # We need volume data for IBIT - use yfinance directly
    import yfinance as yf
    ibit_data = yf.download("IBIT", start="2024-01-01", auto_adjust=True, progress=False)
    if ibit_data.empty:
        return mark_failed(sid, "IBIT data not available")

    # Use IBIT Volume as proxy for AP creation activity
    # High volume → more AP creation/redemption → inflows → BTC demand → MSTR leads
    ibit_vol = ibit_data["Volume"].squeeze()
    if hasattr(ibit_vol, 'columns'):
        ibit_vol = ibit_vol.iloc[:, 0]

    ibit_vol = ibit_vol.dropna()
    if len(ibit_vol) < 30:
        return mark_failed(sid, "insufficient IBIT volume data")

    # Compute rolling stats for volume signal
    roll_mean = ibit_vol.rolling(20).mean()
    roll_std = ibit_vol.rolling(20).std()

    # Signal: volume > mean + 0.5*std for 3 consecutive days
    vol_high = ibit_vol > (roll_mean + 0.5 * roll_std)

    # Find 3 consecutive days of high volume
    consec = vol_high.rolling(3).sum()
    raw_signals = consec >= 3

    # Get signal dates
    signal_dates = raw_signals[raw_signals].index.tolist()

    # Deduplicate: only fire at start of each streak
    triggers = []
    last_trigger = None
    for dt in signal_dates:
        if last_trigger is None or (dt - last_trigger).days > 12:
            triggers.append(dt)
            last_trigger = dt

    print(f"Signal events: {len(triggers)}")
    if not triggers:
        return mark_failed(sid, "no signal events found")

    ret = daily_returns(px)
    mstr_r = ret["MSTR"]
    btc_r = ret["BTC-USD"]
    spy_r = ret["SPY"]

    # BTC 20-day SMA filter
    btc_close = px["BTC-USD"]
    btc_sma20 = btc_close.rolling(20).mean()
    btc_above_sma = btc_close > btc_sma20

    hold_max = 10  # 10 trading days max hold
    pnl = pd.Series(0.0, index=mstr_r.index)
    events = []
    last_exit_date = None

    for td in triggers:
        # BTC filter
        if td not in btc_above_sma.index or not btc_above_sma.loc[td]:
            continue

        # Entry on close of signal day
        mask = mstr_r.index >= td
        if mask.sum() < 2:
            continue

        entry_idx = mstr_r.index[mask][0]
        p = mstr_r.index.get_loc(entry_idx)

        if p + 1 >= len(mstr_r):
            continue

        exit_p = min(p + hold_max, len(mstr_r))

        # MSTR/BTC pair trade: long MSTR, short BTC-USD (equal notional)
        mstr_cum = 0.0
        btc_cum = 0.0
        actual_exit = exit_p
        pair_pnl_series = []

        for j in range(p, exit_p):
            mr = mstr_r.iloc[j]
            br = btc_r.iloc[j] if mstr_r.index[j] in btc_r.index else 0.0

            # Pair PnL: long MSTR + short BTC
            daily_pair = mr - br
            pair_pnl_series.append(daily_pair)

            mstr_cum += mr
            btc_cum += br

            # Stop: MSTR/BTC ratio drops >5% (pair trade goes negative by >5%)
            pair_ret_so_far = mstr_cum - btc_cum
            if pair_ret_so_far < -0.05:
                actual_exit = j + 1
                break

            # Exit: IBIT volume signal drops below mean for 2 consecutive days
            if j >= p + 2:
                vol_check_date = mstr_r.index[j]
                if vol_check_date in vol_high.index:
                    two_day_vol = vol_high.iloc[max(0, vol_high.index.get_loc(vol_check_date)-1):
                                                vol_high.index.get_loc(vol_check_date)+1]
                    if len(two_day_vol) >= 2 and not two_day_vol.any():
                        actual_exit = j + 1
                        break

        seg_len = actual_exit - p
        if seg_len == 0 or not pair_pnl_series:
            continue

        for j, daily_val in enumerate(pair_pnl_series[:seg_len]):
            pnl.iloc[p + j] = daily_val

        last_exit_date = mstr_r.index[actual_exit - 1]

        pair_total = float(sum(pair_pnl_series[:seg_len]))
        spy_seg = spy_r.iloc[p:actual_exit]
        spy_ret = float((1 + spy_seg).prod() - 1) if len(spy_seg) > 0 else None

        events.append({
            "trigger_date": str(td.date()),
            "entry_date": str(entry_idx.date()),
            "hold_days": seg_len,
            "pair_pnl": round(pair_total, 4),
            "spy_return": round(spy_ret, 4) if spy_ret is not None else None,
        })

    print(f"Events executed: {len(events)}")
    if not events:
        return mark_failed(sid, "no executable events after filters")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active trading days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="IBIT AP Creation Surge → Long MSTR/Short BTC")
    returns_list = [e["pair_pnl"] for e in events]
    save_result(sid, m, extra={
        "rule": "Long MSTR / Short BTC-USD 10d when IBIT volume > 20d mean+0.5std for 3 consecutive days, BTC above 20d SMA",
        "mechanism": "IBIT AP creation surge signals institutional inflow demand; MSTR leveraged BTC exposure leads BTC spot on upside",
        "source": "yfinance IBIT/MSTR/BTC-USD/SPY; volume proxy for AP creations",
        "n_events": len(events),
        "avg_event_return": round(float(np.mean(returns_list)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in returns_list])), 4),
        "events": events,
        "implementation_note": "True shares-outstanding data not available via free API; IBIT volume used as AP creation proxy",
    })
    print(f"Done: {len(events)} events, metrics saved.")


if __name__ == "__main__":
    main()
