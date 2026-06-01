"""PL773_tsm_skew_compression_short
TSM Complacency Regime (6m >+30% + Low VIX) -> Short TSM Counter-Signal

Entry: TSM 6-month return > +30%, TSM RSI(14) > 70, VIX < 14.
Short TSM for up to 30 trading days. Exit on RSI < 50, VIX > 20,
-8% target, +4% stop, or 30-day time stop.
Max 1 trade per quarter (63-day cooldown).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def main():
    sid = "PL773_tsm_skew_compression_short"

    try:
        px = load_prices(["TSM", "SPY"], start="2010-01-01")
    except Exception as e:
        try:
            px = load_prices(["TSM", "SPY"], start="2010-01-01")
        except Exception as e2:
            return mark_failed(sid, f"equity data load: {e2}")

    try:
        px_vix = load_prices(["^VIX"], start="2010-01-01")
    except Exception as e:
        try:
            px_vix = load_prices(["^VIX"], start="2010-01-01")
        except Exception as e2:
            return mark_failed(sid, f"VIX data load: {e2}")

    for t in ["TSM", "SPY"]:
        if t not in px.columns:
            return mark_failed(sid, f"missing ticker: {t}")

    if "^VIX" not in px_vix.columns:
        return mark_failed(sid, "missing ^VIX")

    px = px.join(px_vix, how="outer").sort_index().ffill(limit=5)

    ret = daily_returns(px)
    tsm_ret = ret["TSM"]
    spy_r = ret["SPY"].dropna()

    tsm_px = px["TSM"]
    vix = px["^VIX"]

    # Compute signals
    tsm_6m = tsm_px.pct_change(126)  # ~6 months trading days
    tsm_rsi = rsi(tsm_px, 14)

    # Entry: TSM 6m > 30%, RSI(14) > 70, VIX < 14
    entry_signal = (tsm_6m > 0.30) & (tsm_rsi > 70) & (vix < 14)

    max_hold = 30  # trading days
    cooldown = 63  # 1 quarter

    positions = pd.Series(0.0, index=ret.index)
    idx_list = list(ret.index)

    in_pos = False
    hold_count = 0
    entry_price = None
    last_entry_i = -cooldown - 1

    for i, d in enumerate(idx_list):
        tsm_p = tsm_px.get(d, np.nan) if d in tsm_px.index else np.nan
        rsi_v = tsm_rsi.get(d, np.nan) if d in tsm_rsi.index else np.nan
        vix_v = vix.get(d, np.nan) if d in vix.index else np.nan

        if in_pos:
            hold_count += 1
            cum_ret = (tsm_p / entry_price - 1) if (entry_price and not np.isnan(tsm_p)) else 0
            # Exit conditions
            exit_cond = (
                (not np.isnan(rsi_v) and rsi_v < 50) or   # RSI exhaustion
                (not np.isnan(vix_v) and vix_v > 20) or   # VIX spike
                cum_ret <= -0.08 or                         # profit target (short, so -8% = gain)
                cum_ret >= 0.04 or                          # stop-loss
                hold_count >= max_hold
            )
            if exit_cond:
                in_pos = False
                hold_count = 0
            else:
                positions.iloc[i] = -1.0  # short TSM

        if not in_pos and (i - last_entry_i) > cooldown:
            sig = entry_signal.get(d, False) if d in entry_signal.index else False
            if sig and not np.isnan(tsm_p):
                in_pos = True
                hold_count = 0
                entry_price = tsm_p
                last_entry_i = i

    # Short PnL: position=-1 applied to next day's return
    pnl = positions.shift(1).fillna(0) * tsm_ret.reindex(ret.index).fillna(0)

    spy_aligned = spy_r.dropna()
    pnl_aligned = pnl.reindex(spy_aligned.index).fillna(0)

    n_signals = int(entry_signal.fillna(False).sum())
    active_days = int((positions != 0).sum())

    m = compute_metrics(pnl_aligned, benchmark=spy_aligned, name="TSM Complacency Short Counter-Signal")
    m["n_events"] = n_signals

    save_result(sid, m, extra={
        "rule": "Short TSM when 6m return >30%, RSI(14) >70, VIX <14. Exit RSI <50, VIX >20, -8% target, +4% stop, or 30-day max. 63-day cooldown.",
        "mechanism": "Counter-signal to long-semis stack. Complacency regime (vol compression + overbought tech) precedes mean reversion. TSM concentration in Taiwan adds geopolitical tail risk.",
        "source": "TSM/SPY/^VIX yfinance. Counter to PL023/PL042/PL107 long-semis signals.",
        "status": "ok",
        "n_trigger_signals": n_signals,
        "active_days": active_days,
    }, pnl=pnl_aligned)

    print(f"\n=== {sid} ===")
    print(f"Trigger signals: {n_signals}, Active days: {active_days}")
    print(f"Sharpe: {m.get('sharpe', 'N/A'):.3f}  CAGR: {m.get('cagr', 0):.2%}  MaxDD: {m.get('max_dd', 0):.2%}  t-stat: {m.get('t_stat', 0):.3f}")
    if 'oos_sharpe' in m:
        print(f"OOS Sharpe: {m['oos_sharpe']:.3f}")


if __name__ == "__main__":
    main()
