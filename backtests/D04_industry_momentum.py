"""
D04 Industry momentum (Moskowitz & Grinblatt 1999).

Use Ken French 49-industry daily portfolio returns. Monthly:
  rank industries by trailing 6-month return (skip most-recent month is
  classic, but for simplicity we use the full 6m formation including last).
Long top-5 industries, short bottom-5; equal-weight within each leg.
Hold for 1 month. Rebalance monthly.

Universe-shortcut: Ken French value-weighted industry portfolios — already
look-ahead-free and CRSP-grade. No basket bias here.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pandas_datareader.data as web

from harness import compute_metrics, print_metrics, save_result, mark_failed


def main():
    try:
        ind = web.DataReader("49_Industry_Portfolios_daily",
                             "famafrench", start="1990-01-01")[0] / 100.0
        ff = web.DataReader("F-F_Research_Data_Factors_daily",
                            "famafrench", start="1990-01-01")[0] / 100.0
    except Exception as e:
        mark_failed("D04_industry_momentum", f"FF data fetch failed: {e}")
        return

    ind.index = pd.to_datetime(ind.index)
    ff.index = pd.to_datetime(ff.index)
    # Some industries have -99.99 sentinels for missing — already /100 = -0.9999.
    # Replace anything <= -0.99 with NaN.
    ind = ind.where(ind > -0.99)

    # Monthly compounded industry returns
    mret = (1 + ind).resample("ME").prod() - 1
    mkt_m = (1 + ff["Mkt-RF"]).resample("ME").prod() - 1

    # 6-month trailing return as ranking score, set at end of month T-1
    score = (1 + mret).rolling(6).apply(np.prod, raw=True) - 1

    N_LONG = 5
    N_SHORT = 5

    positions = pd.DataFrame(0.0, index=score.index, columns=score.columns)
    for d, row in score.iterrows():
        s = row.dropna()
        if len(s) < (N_LONG + N_SHORT):
            continue
        top = s.nlargest(N_LONG).index
        bot = s.nsmallest(N_SHORT).index
        positions.loc[d, top] = 1.0 / N_LONG
        positions.loc[d, bot] = -1.0 / N_SHORT

    # Apply positions at end-of-T to T+1's monthly return
    pos_shift = positions.shift(1)
    active = (pos_shift.abs().sum(axis=1) > 0)
    pnl_m = (pos_shift * mret).sum(axis=1, min_count=1).where(active).dropna()

    eq = (1 + pnl_m).cumprod()
    years = len(pnl_m) / 12.0
    cagr = eq.iloc[-1] ** (1 / years) - 1
    vol = pnl_m.std() * np.sqrt(12)
    sharpe = pnl_m.mean() / pnl_m.std() * np.sqrt(12) if pnl_m.std() > 0 else 0
    dd = (eq / eq.cummax() - 1)
    max_dd = float(dd.min())
    hit = float((pnl_m > 0).mean())
    t_stat = pnl_m.mean() / (pnl_m.std() / np.sqrt(len(pnl_m))) if pnl_m.std() > 0 else 0
    bench_m = mkt_m.reindex(pnl_m.index).dropna()

    metrics = {
        "name": "D04 Industry momentum (49 inds, top5 - bot5, 6m formation)",
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
    save_result("D04_industry_momentum", metrics, extra={
        "status": "ok",
        "rule": "Rank Ken French 49 industries by 6m return; long top5, short bot5, equal-weight, monthly.",
        "universe": "Ken French 49 industry value-weighted daily portfolios.",
        "source": "Moskowitz & Grinblatt 1999.",
        "shortcut_note": "No look-ahead, no costs. Industry portfolios already free of single-name survivorship.",
        "frequency": "monthly",
    })


if __name__ == "__main__":
    main()
