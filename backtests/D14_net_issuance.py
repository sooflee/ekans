"""
D14 Net stock issuance (Pontiff & Woodgate 2008; Daniel & Titman 2006).

Monthly: rank firms by 12-month % change in shares outstanding.
  Long bottom decile (net repurchasers / share-count shrinkers)
  Short top decile (net issuers).
Equal-weight, monthly rebalance.

Universe-shortcut: ~75 large-cap basket. yfinance get_shares_full only goes
back to ~2015-10 for most names, so this is a ~9-10y backtest (not 20y).
Static survivorship-biased basket.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import warnings
warnings.filterwarnings("ignore")

import hashlib
import numpy as np
import pandas as pd
import yfinance as yf

from _universe import load_universe_prices, UNIVERSE
from harness import save_result, print_metrics, mark_failed

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def fetch_shares_panel():
    """Return DataFrame (date x ticker) of shares outstanding, ffill'd to daily."""
    cache_fp = DATA / "shares_panel_universe_v1.parquet"
    if cache_fp.exists():
        return pd.read_parquet(cache_fp)
    frames = []
    for t in UNIVERSE:
        try:
            s = yf.Ticker(t).get_shares_full(start="2005-01-01")
            if s is None or len(s) == 0:
                continue
            s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
            # drop duplicate dates, keep last
            s = s[~s.index.duplicated(keep="last")]
            s.name = t
            frames.append(s)
        except Exception as e:
            print(f"shares fetch err {t}: {e}")
    if not frames:
        return pd.DataFrame()
    panel = pd.concat(frames, axis=1).sort_index()
    panel.to_parquet(cache_fp)
    return panel


def main():
    px = load_universe_prices()
    eq_px = px[[c for c in px.columns if c in UNIVERSE]].copy()
    rets = eq_px.pct_change()
    spy_r = px["SPY"].pct_change()

    panel = fetch_shares_panel()
    if panel.empty:
        mark_failed("D14_net_issuance", "no shares data available from yfinance")
        return

    # Daily-aligned shares: reindex to price index then ffill
    daily_idx = eq_px.index
    sh_daily = panel.reindex(daily_idx, method="ffill")

    # Keep only universe tickers present in panel
    cols = [c for c in eq_px.columns if c in sh_daily.columns]
    sh_daily = sh_daily[cols]

    # 12-month (252-trading-day) pct change in shares
    issuance = sh_daily.pct_change(252)

    me_issuance = issuance.resample("ME").last()
    me_rets = (1 + rets[cols]).resample("ME").prod() - 1

    positions = pd.DataFrame(0.0, index=me_issuance.index, columns=me_issuance.columns)
    for d, row in me_issuance.iterrows():
        s = row.dropna()
        if len(s) < 25:
            continue
        r = s.rank(pct=True)
        top = r.index[r >= 0.9]
        bot = r.index[r <= 0.1]
        # LONG bottom (shrinkers / buybackers), SHORT top (issuers)
        if len(bot) > 0:
            positions.loc[d, bot] = 1.0 / len(bot)
        if len(top) > 0:
            positions.loc[d, top] = -1.0 / len(top)

    pos_shift = positions.shift(1)
    # Only count months where we actually have a non-zero position
    active = (pos_shift.abs().sum(axis=1) > 0)
    pnl_m = (pos_shift * me_rets).sum(axis=1, min_count=1)
    pnl_m = pnl_m.where(active).dropna()

    if len(pnl_m) < 12:
        mark_failed("D14_net_issuance",
                    f"only {len(pnl_m)} months of usable signal")
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
    spy_m = (1 + spy_r).resample("ME").prod() - 1
    bench_m = spy_m.reindex(pnl_m.index).dropna()

    metrics = {
        "name": "D14 Net stock issuance (long buybackers, short issuers)",
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
    save_result("D14_net_issuance", metrics, extra={
        "status": "ok",
        "rule": "Monthly: 12m pct change in shares outstanding. Long bottom decile (buybackers), short top decile (issuers).",
        "universe": f"Subset of {len(cols)} of {len(UNIVERSE)} large-caps where yfinance shares history was available.",
        "source": "Pontiff & Woodgate 2008; Daniel & Titman 2006.",
        "shortcut_note": "yfinance get_shares_full coverage starts ~2015-10, so backtest is ~9-10y. Static survivorship basket.",
        "frequency": "monthly",
        "n_tickers": len(cols),
    })


if __name__ == "__main__":
    main()
