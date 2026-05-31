"""
PL601 — Platinum/Palladium Ratio < 0.50 -> Long PL=F / Short PA=F Spread

Entry: PL/PA ratio < 0.50 AND at least 35% below trailing 2520-day (10y) rolling median.
Exit: earliest of (i) 80 trading days, (ii) ratio recovers to 10y rolling median,
      (iii) trailing 5-day ratio rises +20% above entry ratio.
PnL: notional-matched long PL=F / short PA=F = PL_ret - PA_ret while open.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics, save_result,
                     mark_failed, daily_returns)


SIGNAL_ID = "PL601_pt_pd_ratio_long_pl_short_pa"
NAME = "PL/PA Ratio <0.50 + 35% below 10y Median -> Long PL=F / Short PA=F"


def main():
    sid = SIGNAL_ID
    try:
        px = load_prices(["PL=F", "PA=F", "SPY"], start="2000-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if "PL=F" not in px.columns or "PA=F" not in px.columns or "SPY" not in px.columns:
        return mark_failed(sid, f"missing tickers; got {list(px.columns)}")

    px = px[["PL=F", "PA=F", "SPY"]].dropna(how="all").sort_index()

    # forward-fill small gaps (futures sometimes miss a day)
    px["PL=F"] = px["PL=F"].ffill(limit=5)
    px["PA=F"] = px["PA=F"].ffill(limit=5)

    px = px.dropna(subset=["PL=F", "PA=F"])
    if len(px) < 1000:
        return mark_failed(sid, f"insufficient PL/PA history: {len(px)} days")

    rets = daily_returns(px)
    pl_r = rets["PL=F"]
    pa_r = rets["PA=F"]
    spy_r = rets["SPY"]

    ratio = (px["PL=F"] / px["PA=F"]).dropna()
    # 10-year rolling median (~2520 trading days). Use min_periods=504 (~2y) so
    # we can start trading after a couple of years of history rather than wait
    # the full decade.
    roll_med = ratio.rolling(2520, min_periods=504).median()

    # Entry: ratio<0.50 AND ratio at least 35% below rolling median
    # (i.e. ratio <= roll_med * 0.65). Need both criteria.
    cond_abs = ratio < 0.50
    cond_rel = ratio <= 0.65 * roll_med
    entry_signal = (cond_abs & cond_rel).fillna(False)

    # Build positions Series (1 = in spread, 0 = flat). Iterate the index.
    idx = ratio.index
    positions = pd.Series(0, index=idx, dtype=float)
    in_pos = False
    entry_i = -1
    entry_ratio = np.nan
    entry_date = None

    # short rolling for exit condition (iii)
    ratio_5d = ratio.rolling(5).mean()

    for i, dt in enumerate(idx):
        if not in_pos:
            if entry_signal.iloc[i]:
                in_pos = True
                entry_i = i
                entry_ratio = ratio.iloc[i]
                entry_date = dt
                positions.iloc[i] = 1.0
            # else stay flat
        else:
            # already in position — mark held
            positions.iloc[i] = 1.0
            held = i - entry_i
            cur_ratio = ratio.iloc[i]
            cur_med = roll_med.iloc[i]
            cur_5d = ratio_5d.iloc[i]
            exit_now = False
            # (i) 80 trading days
            if held >= 80:
                exit_now = True
            # (ii) ratio recovers to median
            elif pd.notna(cur_med) and cur_ratio >= cur_med:
                exit_now = True
            # (iii) trailing 5-day ratio rises +20% above entry ratio
            elif pd.notna(cur_5d) and pd.notna(entry_ratio) and entry_ratio > 0 \
                 and cur_5d >= entry_ratio * 1.20:
                exit_now = True

            if exit_now:
                in_pos = False
                entry_i = -1
                entry_ratio = np.nan
                entry_date = None
                # position recorded for today (still held on exit day); next day flat

    # PnL: apply yesterday's position to today's spread return (no look-ahead)
    spread_r = (pl_r - pa_r).reindex(idx).fillna(0.0)
    pnl = positions.shift(1).fillna(0.0) * spread_r
    pnl = pnl.dropna()

    if len(pnl) < 100 or positions.sum() < 30:
        return mark_failed(
            sid,
            f"insufficient signal coverage: {int(positions.sum())} days in position",
            extra={
                "rule": "Long PL=F / short PA=F when PL/PA<0.50 AND <=65% of 10y median; exit on +80d, median revert, or +20% 5d ratio",
                "mechanism": "Platinum supply deficits historically coincide with deep PL/PA ratio lows; mean reversion in the spread",
                "source": "PL601 idea catalog",
            },
        )

    m = compute_metrics(
        pnl,
        benchmark=spy_r.reindex(pnl.index),
        name=NAME,
        positions=positions,
        cost_bps=10,
    )

    # diagnostics about entries
    pos_shift = positions.diff().fillna(positions.iloc[0])
    entries = (pos_shift > 0).sum()
    exits = (pos_shift < 0).sum()
    m["n_entries"] = int(entries)
    m["n_exits"] = int(exits)
    m["pct_in_market"] = float(positions.mean())
    m["status"] = "ok"

    save_result(
        sid,
        m,
        extra={
            "rule": "Enter long PL=F / short PA=F when PL/PA ratio < 0.50 AND at least 35% below 10y rolling median. Exit at the earliest of: 80 trading days, ratio recovers to 10y median, or trailing 5-day ratio rises +20% above entry ratio.",
            "mechanism": "Platinum market supply deficits (WPIC) historically coincide with multi-decade lows in the PL/PA ratio. The ratio threshold acts as a price-only proxy for the deficit regime, and the spread mean-reverts as deficit-driven platinum demand catches up.",
            "source": "PL601 — Platinum/Palladium Ratio idea catalog; WPIC Quarterly Bulletin as conceptual anchor (pure-price implementation)",
            "caveats": "PA=F history thin pre-1986; ratio threshold (<0.50) only triggers during rare deep-deficit regimes; exit on median-revert assumes the rolling-median anchor is stable (it shifts as deficit regimes persist).",
        },
        pnl=pnl,
    )
    print(f"Saved {sid}: Sharpe={m.get('sharpe'):.2f}  CAGR={m.get('cagr',0)*100:.2f}%  n_entries={entries}")


if __name__ == "__main__":
    main()
