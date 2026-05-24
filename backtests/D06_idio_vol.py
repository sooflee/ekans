"""
D06 Idiosyncratic volatility puzzle (Ang, Hodrick, Xing & Zhang 2006).

For each stock, monthly:
  - regress daily returns on SPY over the prior 60 trading days
  - residual std (idio vol) is the signal
Long bottom decile (low idio vol), short top decile (high idio vol),
equal-weight, monthly rebalance.

Universe-shortcut: fixed ~75 large-cap basket (see _universe.py). The original
paper used CRSP and the effect is largely small-cap. This is a demonstration.
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

    WIN = 60
    # Compute rolling beta and idio vol via:
    # idio_var = var(r) - beta^2 * var(spy);   beta = cov(r,spy)/var(spy)
    # then residual_vol_ann ≈ sqrt(idio_var) * sqrt(252)
    var_m = spy_r.rolling(WIN).var()
    idio_vol = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)
    for tkr in rets.columns:
        r = rets[tkr]
        cov_rm = r.rolling(WIN).cov(spy_r)
        beta = cov_rm / var_m
        var_r = r.rolling(WIN).var()
        idio_var = var_r - beta**2 * var_m
        idio_var = idio_var.where(idio_var > 0)
        idio_vol[tkr] = np.sqrt(idio_var * 252)

    # Sample at month-end
    me_idio = idio_vol.resample("ME").last()
    me_rets = (1 + rets).resample("ME").prod() - 1

    positions = pd.DataFrame(0.0, index=me_idio.index, columns=me_idio.columns)
    for d, row in me_idio.iterrows():
        s = row.dropna()
        if len(s) < 30:
            continue
        r = s.rank(pct=True)
        top = r.index[r >= 0.9]
        bot = r.index[r <= 0.1]
        # LONG low idio vol, SHORT high idio vol
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
    spy_m = (1 + spy_r).resample("ME").prod() - 1
    bench_m = spy_m.reindex(pnl_m.index).dropna()

    metrics = {
        "name": "D06 Idiosyncratic vol puzzle (long low IV, short high IV)",
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
    save_result("D06_idio_vol", metrics, extra={
        "status": "ok",
        "rule": "Monthly: residual vol from 60d regression of r_i on SPY. Long bottom decile, short top decile.",
        "universe": f"Fixed basket of {len(UNIVERSE)} large-cap US tickers.",
        "source": "Ang, Hodrick, Xing & Zhang 2006.",
        "shortcut_note": "Static survivorship-biased basket; effect is largest in small caps and absent in our basket.",
        "frequency": "monthly",
    })


if __name__ == "__main__":
    main()
