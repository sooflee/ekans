"""
D05 MAX / lottery effect (Bali, Cakici & Whitelaw 2011).

For each stock at month-end, MAX(5) = sum of the 5 highest daily returns in the
prior month. Cross-sectionally rank; LONG bottom decile (low MAX = boring),
SHORT top decile (high MAX = lottery-like). Equal-weight, monthly rebalance.

Universe-shortcut: fixed ~75 large-cap US basket (see _universe.py).
This understates the MAX effect — it's strongest in small caps. Use as
demonstration only.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from _universe import load_universe_prices, UNIVERSE
from harness import save_result, print_metrics


def main():
    px = load_universe_prices()
    eq_px = px[[c for c in px.columns if c in UNIVERSE]].copy()
    rets = eq_px.pct_change()

    # For each month, compute sum of top-5 daily returns per stock
    def max5(g):
        # g: DataFrame of daily returns within one calendar month
        return g.apply(lambda col: col.dropna().nlargest(5).sum() if col.notna().sum() >= 15 else np.nan)

    monthly_max5 = rets.groupby(pd.Grouper(freq="ME")).apply(max5)
    # Sometimes the result has a MultiIndex; coerce
    if isinstance(monthly_max5.index, pd.MultiIndex):
        monthly_max5 = monthly_max5.reset_index(level=1, drop=True)

    # Monthly forward returns
    me_rets = (1 + rets).resample("ME").prod() - 1

    positions = pd.DataFrame(0.0, index=monthly_max5.index, columns=monthly_max5.columns)
    for d, row in monthly_max5.iterrows():
        s = row.dropna()
        if len(s) < 30:
            continue
        r = s.rank(pct=True)
        top = r.index[r >= 0.9]
        bot = r.index[r <= 0.1]
        # LONG bottom (low MAX), SHORT top (high MAX)
        if len(bot) > 0:
            positions.loc[d, bot] = 1.0 / len(bot)
        if len(top) > 0:
            positions.loc[d, top] = -1.0 / len(top)

    pos_shift = positions.shift(1)
    active = (pos_shift.abs().sum(axis=1) > 0)
    pnl_m = (pos_shift * me_rets).sum(axis=1, min_count=1).where(active).dropna()

    eq = (1 + pnl_m).cumprod()
    years = len(pnl_m) / 12.0
    cagr = eq.iloc[-1] ** (1 / years) - 1
    vol = pnl_m.std() * np.sqrt(12)
    sharpe = pnl_m.mean() / pnl_m.std() * np.sqrt(12) if pnl_m.std() > 0 else 0
    dd = (eq / eq.cummax() - 1)
    max_dd = float(dd.min())
    hit = float((pnl_m > 0).mean())
    t_stat = pnl_m.mean() / (pnl_m.std() / np.sqrt(len(pnl_m))) if pnl_m.std() > 0 else 0
    spy_m = (1 + px["SPY"].pct_change()).resample("ME").prod() - 1
    bench_m = spy_m.reindex(pnl_m.index).dropna()

    metrics = {
        "name": "D05 MAX(5) lottery effect (long low-MAX, short high-MAX)",
        "start": str(pnl_m.index[0].date()),
        "end": str(pnl_m.index[-1].date()),
        "n_months": int(len(pnl_m)),
        "n_days": int(len(pnl_m)),
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
    save_result("D05_max_lottery", metrics, extra={
        "status": "ok",
        "rule": "Monthly: MAX(5)=sum of 5 largest daily returns last month. Long bottom decile, short top decile, equal-weight.",
        "universe": f"Fixed basket of {len(UNIVERSE)} large-cap US tickers.",
        "source": "Bali, Cakici & Whitelaw 2011 'Maxing Out'.",
        "shortcut_note": "Static survivorship-biased basket; MAX effect strongest in small caps so this UNDER-states the premium.",
        "frequency": "monthly",
    })


if __name__ == "__main__":
    main()
