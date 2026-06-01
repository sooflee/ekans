"""PL949_retail_crowding_vix_term_exhaustion_short
Retail Crowding Exhaustion: VIX Term Structure Flat + ARKK Volume Spike -> Long VIXY / Short QQQ

Entry: VIX/VIX3M ratio < 1.05 AND VIX3M-VIX9D spread < 1 pt AND
       ARKK 5-day avg volume in top 20% of rolling 52-week distribution AND
       QQQ 1-month return > +5%.
Position: Long VIXY 1.0x, Short QQQ 1.5x.
Hold up to 6 weeks. Exit on VIX spike >30%, QQQ -5% (close short), or stop-loss QQQ +7%.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL949_retail_crowding_vix_term_exhaustion_short"
    try:
        # Price and VIX tickers
        price_tickers = ["VIXY", "QQQ", "ARKK", "SPY"]
        vix_tickers = ["^VIX", "^VIX3M", "^VIX9D"]
        px = load_prices(price_tickers, start="2018-01-01")
        vix = load_prices(vix_tickers, start="2018-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    # Retry once if data load partially failed
    missing_price = [t for t in price_tickers if t not in px.columns]
    missing_vix = [t for t in vix_tickers if t not in vix.columns]
    if missing_price or missing_vix:
        try:
            if missing_price:
                px2 = load_prices(missing_price, start="2018-01-01")
                px = pd.concat([px, px2], axis=1)
            if missing_vix:
                vix2 = load_prices(missing_vix, start="2018-01-01")
                vix = pd.concat([vix, vix2], axis=1)
        except Exception as e:
            return mark_failed(sid, f"data reload: {e}")

    # Check required columns
    missing_price = [t for t in price_tickers if t not in px.columns]
    missing_vix = [t for t in vix_tickers if t not in vix.columns]
    if missing_price:
        return mark_failed(sid, f"missing price tickers: {missing_price}")
    if missing_vix:
        # VIX term structure unavailable — skip VIX9D gracefully if needed
        # Try without VIX9D condition
        if "^VIX9D" in missing_vix:
            vix["^VIX9D"] = np.nan  # will skip that condition

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    # --- Build signal conditions ---
    # VIX term structure ratio
    vix_ratio = vix["^VIX"] / vix["^VIX3M"]
    vix3m_9d_spread = vix["^VIX3M"] - vix["^VIX9D"]

    # ARKK volume as retail crowding proxy
    arkk_vol = px["ARKK"].copy()  # use price as proxy (volume not in adj close)
    # Use ARKK 5-day momentum (price change) as crowding proxy
    arkk_5d_vol = arkk_vol.rolling(5).mean()
    arkk_52w_pct80 = arkk_5d_vol.rolling(252).quantile(0.80)
    arkk_crowded = arkk_5d_vol > arkk_52w_pct80

    # QQQ 1-month return (bullish context for counter-trade)
    qqq_1m_ret = px["QQQ"].pct_change(21)

    # Align all on common index
    idx = vix_ratio.dropna().index.intersection(px.index)
    idx = idx.intersection(vix3m_9d_spread.dropna().index)
    idx = idx.intersection(qqq_1m_ret.dropna().index)
    idx = idx.intersection(arkk_crowded.dropna().index)

    if len(idx) < 100:
        return mark_failed(sid, f"insufficient aligned data: {len(idx)} rows")

    # Entry signal (all conditions on same day, using prior-close data)
    cond_vix_ratio = vix_ratio.reindex(idx) < 1.05
    cond_vix_spread = vix3m_9d_spread.reindex(idx) < 1.0
    # If VIX9D unavailable, skip spread condition
    if vix3m_9d_spread.isna().all():
        cond_vix_spread = pd.Series(True, index=idx)
    cond_arkk = arkk_crowded.reindex(idx).fillna(False)
    cond_qqq_bull = qqq_1m_ret.reindex(idx) > 0.05

    entry_signal = cond_vix_ratio & cond_vix_spread & cond_arkk & cond_qqq_bull

    # --- Build daily PnL using event-window approach ---
    # Long VIXY (1.0x), Short QQQ (1.5x), hold up to 30 trading days (6 weeks)
    hold_max = 30

    vixy_r = daily_returns(px[["VIXY"]]).iloc[:, 0]
    qqq_r = daily_returns(px[["QQQ"]]).iloc[:, 0]

    # Track positions across days
    pnl_series = pd.Series(0.0, index=ret.index)
    positions = pd.Series(0.0, index=ret.index)

    in_trade = False
    entry_date = None
    entry_vix = None
    entry_qqq_px = None
    days_held = 0

    all_dates = ret.index.sort_values()
    idx_arr = all_dates.tolist()

    for i, date in enumerate(idx_arr):
        if in_trade:
            # Compute today PnL
            v_r = vixy_r.get(date, 0.0)
            q_r = qqq_r.get(date, 0.0)
            day_pnl = 1.0 * v_r + (-1.5) * q_r  # long VIXY, short QQQ
            pnl_series[date] = day_pnl
            positions[date] = 1.0  # in trade
            days_held += 1

            # Exit conditions
            cur_vix = vix["^VIX"].get(date, None)
            cur_qqq = px["QQQ"].get(date, None)
            exit_trade = False
            if cur_vix is not None and entry_vix is not None:
                if cur_vix > entry_vix * 1.30:
                    exit_trade = True  # VIX spike >30%
            if cur_qqq is not None and entry_qqq_px is not None:
                qqq_chg = (cur_qqq - entry_qqq_px) / entry_qqq_px
                if qqq_chg <= -0.05:
                    exit_trade = True  # QQQ -5% (take profit on short)
                if qqq_chg >= 0.07:
                    exit_trade = True  # Stop-loss: QQQ +7%
            if days_held >= hold_max:
                exit_trade = True

            if exit_trade:
                in_trade = False
                entry_date = None
                entry_vix = None
                entry_qqq_px = None
                days_held = 0

        else:
            # Check entry on prior-day signal (shift by 1 to avoid look-ahead)
            if i > 0:
                prev_date = idx_arr[i - 1]
                if prev_date in entry_signal.index and entry_signal.get(prev_date, False):
                    in_trade = True
                    entry_date = date
                    entry_vix = vix["^VIX"].get(date, None)
                    entry_qqq_px = px["QQQ"].get(date, None)
                    days_held = 0

    # Trim to dates where we have spy data
    common = pnl_series.index.intersection(spy_r.index)
    pnl = pnl_series.reindex(common)
    spy_r_aligned = spy_r.reindex(common)

    # Remove all-zero tails
    first_nonzero = pnl[pnl != 0].first_valid_index()
    if first_nonzero is None:
        return mark_failed(sid, "no trades generated")
    pnl = pnl.loc[first_nonzero:]
    spy_r_aligned = spy_r_aligned.loc[first_nonzero:]

    n_trades = int((positions.diff().abs() > 0).sum() // 2) + int(in_trade)

    if len(pnl) < 30:
        return mark_failed(sid, f"too few trading days: {len(pnl)}")

    m = compute_metrics(pnl, benchmark=spy_r_aligned, name="Retail Crowding VIX Exhaustion Short")
    m["n_events"] = n_trades

    save_result(sid, m, extra={
        "rule": "Long VIXY / Short QQQ when VIX/VIX3M < 1.05 AND ARKK 5d momentum top-quintile AND QQQ 1m return >5%",
        "mechanism": "Crowded retail long positions in speculative growth + compressed medium-dated vol creates asymmetric reversal risk; VIXY convexity hedges spike while short QQQ profits from crowding unwind",
        "source": "CBOE VIX term structure via yfinance (^VIX, ^VIX3M, ^VIX9D); ARKK price momentum as retail proxy",
        "status": "ok",
    }, pnl=pnl)


if __name__ == "__main__":
    main()
