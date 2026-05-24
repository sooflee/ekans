"""
D17 Negative coskewness premium (Harvey & Siddique 2000).

For each stock at each month-end, compute coskewness with SPY over the prior 252
trading days:
    coskew_i = E[(r_i - mu_i) * (r_m - mu_m)^2]  /  (sigma_i * sigma_m^2)
Rank cross-sectionally. LONG top quintile of NEGATIVE coskewness (i.e. names
with the MOST-negative coskew — they crash hardest when the market crashes;
investors demand a premium to hold them).
SHORT the bottom quintile of negative coskew (= top of raw coskew).

So if we define `negcosk = -coskew`, we long top quintile by negcosk and short
bottom. Monthly rebalance, equal-weight.

Universe-shortcut: fixed ~75 large-cap basket.
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
    spy_r = px["SPY"].pct_change()

    WIN = 252

    # Vectorized rolling coskew:
    #   numer = mean( (r_i - mean_i) * (r_m - mean_m)^2 )
    #   denom = std_i * var_m
    mu_m = spy_r.rolling(WIN).mean()
    var_m = spy_r.rolling(WIN).var()
    sig_m = np.sqrt(var_m)
    dev_m_sq = (spy_r - mu_m) ** 2

    coskew = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)
    for tkr in rets.columns:
        r = rets[tkr]
        mu_i = r.rolling(WIN).mean()
        sig_i = r.rolling(WIN).std()
        # E[(r_i - mu_i) * (r_m - mu_m)^2]
        prod = (r - mu_i) * dev_m_sq
        numer = prod.rolling(WIN).mean()
        denom = sig_i * var_m
        coskew[tkr] = numer / denom

    # Signal is NEGATIVE coskew: we LONG most-negative coskew (top quintile by -coskew)
    negcosk = -coskew

    me_signal = negcosk.resample("ME").last()
    me_rets = (1 + rets).resample("ME").prod() - 1

    positions = pd.DataFrame(0.0, index=me_signal.index, columns=me_signal.columns)
    for d, row in me_signal.iterrows():
        s = row.dropna()
        if len(s) < 25:
            continue
        r = s.rank(pct=True)
        # quintiles
        top = r.index[r >= 0.8]  # most-negative coskew  — LONG
        bot = r.index[r <= 0.2]  # least-negative coskew — SHORT
        if len(top) > 0:
            positions.loc[d, top] = 1.0 / len(top)
        if len(bot) > 0:
            positions.loc[d, bot] = -1.0 / len(bot)

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
    spy_m = (1 + spy_r).resample("ME").prod() - 1
    bench_m = spy_m.reindex(pnl_m.index).dropna()

    metrics = {
        "name": "D17 Negative coskewness premium",
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
    save_result("D17_coskewness", metrics, extra={
        "status": "ok",
        "rule": "Monthly: 252d coskew vs SPY. Long top quintile of negative coskew (crash-coexposed), short bottom.",
        "universe": f"Fixed basket of {len(UNIVERSE)} large-cap US tickers.",
        "source": "Harvey & Siddique 2000 'Conditional Skewness in Asset Pricing Tests'.",
        "shortcut_note": "Static survivorship basket; no costs; raw spread (no risk-norm).",
        "frequency": "monthly",
    })


if __name__ == "__main__":
    main()
