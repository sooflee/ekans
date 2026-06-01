"""PL803_argentina_risk_relief_ewz_brazil_beta
Argentine Risk-Relief Rally (ARGT +20% in 60d) -> Brazil EM Beta Lift (EWZ / ITUB)

Rolling signal: when ARGT 60-calendar-day return first crosses +20% after being
below that threshold, enter long EWZ for 45 calendar days. Checks that EWZ is
not in a deep drawdown at entry. Covers 2011-present (ARGT inception).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL803_argentina_risk_relief_ewz_brazil_beta"
    tickers = ["ARGT", "EWZ", "SPY"]

    try:
        px = load_prices(tickers, start="2011-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()
    ewz_r = ret["EWZ"].fillna(0)

    argt_px = px["ARGT"]
    ewz_px = px["EWZ"]

    # ARGT 60-trading-day return (use 60 trading days ≈ 84 calendar days, close enough)
    argt_60d_ret = argt_px.pct_change(60)

    # EWZ 60d rolling max for drawdown filter
    ewz_60d_high = ewz_px.rolling(60).max()
    ewz_dd_from_high = (ewz_px - ewz_60d_high) / ewz_60d_high

    # Signal: ARGT 60d return crosses +20% from below
    prev_argt = argt_60d_ret.shift(1)
    signal = (prev_argt <= 0.20) & (argt_60d_ret > 0.20) & (ewz_dd_from_high > -0.15)

    signal_dates = argt_60d_ret.index[signal.fillna(False)]

    hold_calendar_days = 45
    trailing_stop_pct = -0.10
    min_gap_days = 60  # avoid overlapping / rapid re-entry

    positions = pd.Series(0.0, index=ret.index)
    last_exit_date = pd.Timestamp("2000-01-01")

    for sig_dt in signal_dates:
        if (sig_dt - last_exit_date).days < min_gap_days:
            continue

        # Enter next trading day
        future = ret.index[ret.index > sig_dt]
        if len(future) == 0:
            continue
        entry_date = future[0]
        entry_idx = ret.index.get_loc(entry_date)

        # Hard exit: 45 calendar days from entry
        hard_exit_dt = entry_date + pd.Timedelta(days=hold_calendar_days)
        hard_exit_candidates = ret.index[ret.index >= hard_exit_dt]
        if len(hard_exit_candidates) == 0:
            hard_exit_date = ret.index[-1]
        else:
            hard_exit_date = hard_exit_candidates[0]

        # Trailing stop: track EWZ from entry
        entry_price = ewz_px.asof(entry_date)
        peak_price = entry_price

        exit_date = hard_exit_date
        for dt in ret.index[entry_idx:]:
            if dt > hard_exit_date:
                break
            cur_price = ewz_px.asof(dt)
            peak_price = max(peak_price, cur_price)
            dd = (cur_price - peak_price) / peak_price if peak_price > 0 else 0
            if dd < trailing_stop_pct:
                exit_date = dt
                break

        hold_mask = (ret.index >= entry_date) & (ret.index <= exit_date)
        positions.loc[ret.index[hold_mask]] = 1.0
        last_exit_date = exit_date

    pnl = positions.shift(1).fillna(0) * ewz_r
    pnl = pnl.reindex(spy_r.index).fillna(0)

    m = compute_metrics(pnl, benchmark=spy_r, name="Argentina Risk-Relief EWZ Long")
    save_result(sid, m, extra={
        "rule": "When ARGT 60d return crosses +20% from below (and EWZ not in >15% drawdown), enter long EWZ for 45 days or until 10% trailing stop. Min 60d between signals.",
        "mechanism": "Argentine risk-relief rallies (Milei election, Macri election, IMF tranche) reduce EM contagion fears. Brazil (EWZ) rallies as the largest EM neighbor and main Mercosur partner, with direct financial/trade linkages.",
        "source": "ARGT (Global X MSCI Argentina ETF) price as proxy for Argentine credit risk. Key events: 2015-12 Macri election, 2023-11 Milei election.",
    })


if __name__ == "__main__":
    main()
