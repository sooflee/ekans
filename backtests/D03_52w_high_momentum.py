"""
D03 52-week-high momentum (George & Hwang 2004).

For our basket of ~75 large-cap US stocks, monthly:
  signal_i,t = price_i,t / max(price_i over trailing 252 trading days)
Rank cross-sectionally each month-end. Long top decile (closest to 52w high),
short bottom decile (furthest). Equal-weight, hold 1 month, then rebalance.

Universe-shortcut: fixed, survivorship-biased basket of ~75 large-caps from
_universe.py — NOT CRSP point-in-time. Results overstate realisable returns.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from _universe import load_universe_prices, UNIVERSE
from harness import daily_returns, compute_metrics, print_metrics, save_result


def main():
    px = load_universe_prices()
    spy = px["SPY"]
    bench = spy.pct_change().dropna()

    eq_px = px[[c for c in px.columns if c in UNIVERSE]].copy()

    # 252-day trailing max
    trail_max = eq_px.rolling(252, min_periods=200).max()
    score = eq_px / trail_max  # in (0, 1], closer to 1 = closer to 52w high

    # Resample to month-end signals
    me_score = score.resample("ME").last()
    rets = eq_px.pct_change()
    me_rets = (1 + rets).resample("ME").prod() - 1  # forward simple monthly ret

    # Build cross-sectional ranks each month (require min 30 valid stocks)
    def decile_positions(row):
        s = row.dropna()
        if len(s) < 30:
            return pd.Series(0.0, index=row.index)
        r = s.rank(pct=True)
        pos = pd.Series(0.0, index=row.index)
        pos.loc[r.index[r >= 0.9]] = 1.0
        pos.loc[r.index[r <= 0.1]] = -1.0
        # equal-weight within each leg
        long_n = (pos == 1).sum()
        short_n = (pos == -1).sum()
        if long_n > 0:
            pos[pos == 1] = 1.0 / long_n
        if short_n > 0:
            pos[pos == -1] = -1.0 / short_n
        return pos

    positions = me_score.apply(decile_positions, axis=1)

    # Use NEXT month's return for each position (no look-ahead).
    pos_shift = positions.shift(1)
    active = (pos_shift.abs().sum(axis=1) > 0)
    pnl_m = (pos_shift * me_rets).sum(axis=1, min_count=1).where(active).dropna()

    # Convert monthly pnl to a pseudo-daily-equivalent series for metric consistency
    # by simply scaling so compute_metrics' ann factor works. Easier: compute metrics
    # at monthly frequency manually.
    if len(pnl_m) < 12:
        from harness import mark_failed
        mark_failed("D03_52w_high_momentum", "not enough months")
        return

    eq = (1 + pnl_m).cumprod()
    years = len(pnl_m) / 12.0
    cagr = eq.iloc[-1] ** (1 / years) - 1
    vol = pnl_m.std() * np.sqrt(12)
    sharpe = pnl_m.mean() / pnl_m.std() * np.sqrt(12) if pnl_m.std() > 0 else 0
    dd = (eq / eq.cummax() - 1)
    max_dd = float(dd.min())
    hit = float((pnl_m > 0).mean())
    t_stat = pnl_m.mean() / (pnl_m.std() / np.sqrt(len(pnl_m))) if pnl_m.std() > 0 else 0
    bench_m = (1 + bench).resample("ME").prod() - 1
    bench_m = bench_m.reindex(pnl_m.index)

    metrics = {
        "name": "D03 52w-high momentum (monthly)",
        "start": str(pnl_m.index[0].date()),
        "end": str(pnl_m.index[-1].date()),
        "n_months": int(len(pnl_m)),
        "n_days": int(len(pnl_m)),  # for harness compatibility
        "cagr": float(cagr),
        "ann_vol": float(vol),
        "sharpe": float(sharpe),
        "max_dd": float(max_dd),
        "calmar": float(cagr / abs(max_dd)) if max_dd < 0 else None,
        "hit_rate": hit,
        "t_stat": float(t_stat),
        "bench_cagr": float((1 + bench_m).prod() ** (1 / years) - 1),
        "bench_sharpe": float(bench_m.mean() / bench_m.std() * np.sqrt(12)),
    }
    metrics["excess_cagr"] = metrics["cagr"] - metrics["bench_cagr"]
    print_metrics(metrics)
    save_result("D03_52w_high_momentum", metrics, extra={
        "status": "ok",
        "rule": "Monthly: rank price/252d-max cross-sectionally; long top decile, short bottom; equal-weight.",
        "universe": f"Fixed basket of {len(UNIVERSE)} large-cap US tickers (see _universe.py).",
        "source": "George & Hwang 2004 'The 52-Week High and Momentum Investing'.",
        "shortcut_note": "Static survivorship-biased basket; no CRSP. Monthly rebalance, no costs.",
        "frequency": "monthly",
    })


if __name__ == "__main__":
    main()
