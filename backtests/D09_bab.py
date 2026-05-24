"""
D09 Betting Against Beta (Frazzini & Pedersen 2014).

For each stock, 60-day beta vs SPY = cov(r_i, r_spy)/var(r_spy).
Monthly: long bottom QUINTILE (low beta), short top quintile (high beta),
equal-weight within each leg. We SKIP the FP leverage normalization — present
the raw decile spread, per task spec.

Universe-shortcut: fixed ~75 large-cap basket. The basket has limited
cross-sectional beta dispersion vs CRSP universe; effect will be muted.
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
    var_m = spy_r.rolling(WIN).var()
    beta = pd.DataFrame(index=rets.index, columns=rets.columns, dtype=float)
    for tkr in rets.columns:
        cov_rm = rets[tkr].rolling(WIN).cov(spy_r)
        beta[tkr] = cov_rm / var_m

    me_beta = beta.resample("ME").last()
    me_rets = (1 + rets).resample("ME").prod() - 1

    positions = pd.DataFrame(0.0, index=me_beta.index, columns=me_beta.columns)
    for d, row in me_beta.iterrows():
        s = row.dropna()
        if len(s) < 25:
            continue
        r = s.rank(pct=True)
        # quintiles
        top = r.index[r >= 0.8]   # high beta — SHORT
        bot = r.index[r <= 0.2]   # low beta  — LONG
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
        "name": "D09 Betting Against Beta (raw L-S quintile spread, no FP leverage scaling)",
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
    save_result("D09_bab", metrics, extra={
        "status": "ok",
        "rule": "Monthly: 60d beta vs SPY. Long bottom-quintile, short top-quintile, equal-weight. No FP leverage scaling.",
        "universe": f"Fixed basket of {len(UNIVERSE)} large-cap US tickers.",
        "source": "Frazzini & Pedersen 2014.",
        "shortcut_note": "Static survivorship basket; limited cross-sectional beta dispersion vs CRSP. Raw L-S only (no leverage normalization).",
        "frequency": "monthly",
    })


if __name__ == "__main__":
    main()
