"""
PL620 — OECD Crude Build + WTI Below Dividend-Breakeven Proxy -> Short XOM/CVX

Signal:
  - WCESTUS1 (weekly US crude stocks ex-SPR) had positive WoW change in >=12 of last 16 weeks
  - DCOILWTICO 20-day MA < $55/bbl
Position: short equal-weight (0.5 XOM + 0.5 CVX)
Exit (earliest of):
  - 30 trading days
  - basket cumulative return <= -8% (profit on short)
  - DCOILWTICO 20-day MA > $60 (regime exit)
  - basket cumulative return >= +6% (stop)

Counter-signal against long_energy / long_SPY.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


SIGNAL_ID = "PL620_crude_inv_wti_low_short_xom_cvx"
NAME = "WCESTUS1 12/16 Builds + WTI20<$55 -> Short XOM/CVX (30d)"


def main():
    sid = SIGNAL_ID
    try:
        px = load_prices(["XOM", "CVX", "SPY"], start="2000-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load (prices): {e}")

    for t in ["XOM", "CVX", "SPY"]:
        if t not in px.columns:
            return mark_failed(sid, f"missing ticker {t}")

    try:
        fr = load_fred(["WCESTUS1", "DCOILWTICO"], start="2000-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load (FRED): {e}")

    if "WCESTUS1" not in fr.columns or "DCOILWTICO" not in fr.columns:
        return mark_failed(sid, f"missing FRED series; got {list(fr.columns)}")

    px = px.sort_index().ffill(limit=3)
    ret = daily_returns(px)
    spy_r = ret["SPY"]
    basket_r = (0.5 * ret["XOM"] + 0.5 * ret["CVX"]).dropna()
    trading_idx = ret.index

    weekly_inv = fr["WCESTUS1"].dropna()
    wow = weekly_inv.diff()
    wow_pos = (wow > 0).astype(int)
    pos_count_16 = wow_pos.rolling(16, min_periods=8).sum()
    # forward fill to daily
    pos_count_daily = pos_count_16.reindex(trading_idx, method="ffill")

    wti = fr["DCOILWTICO"].dropna()
    wti_daily = wti.reindex(trading_idx, method="ffill")
    wti_20 = wti_daily.rolling(20, min_periods=5).mean()

    cond_builds = pos_count_daily >= 12
    cond_wti = wti_20 < 55.0
    entry_signal = (cond_builds & cond_wti).fillna(False)

    hold_days = 30
    take_profit = -0.08  # short profit: basket falls >=8%
    stop_loss = 0.06     # short stop: basket up 6%

    positions = pd.Series(0.0, index=trading_idx)
    events = []
    last_exit_loc = -1

    for i, dt in enumerate(trading_idx):
        if i <= last_exit_loc:
            continue
        if not entry_signal.iloc[i]:
            continue
        entry_loc = i + 1
        if entry_loc >= len(trading_idx) - 1:
            continue
        if pd.isna(px["XOM"].iloc[entry_loc]) or pd.isna(px["CVX"].iloc[entry_loc]):
            continue
        end_loc = min(entry_loc + hold_days, len(trading_idx) - 1)
        bw = basket_r.reindex(trading_idx).iloc[entry_loc:end_loc + 1].fillna(0.0)
        cumret = (1 + bw).cumprod() - 1
        exit_loc = end_loc
        exit_reason = "max_hold"
        for k, val in enumerate(cumret.values):
            cur_i = entry_loc + k
            if pd.notna(val) and val <= take_profit:
                exit_loc = cur_i
                exit_reason = "profit_target"
                break
            if pd.notna(val) and val >= stop_loss:
                exit_loc = cur_i
                exit_reason = "stop_loss"
                break
            if cur_i < len(wti_20) and pd.notna(wti_20.iloc[cur_i]) and wti_20.iloc[cur_i] > 60.0:
                exit_loc = cur_i
                exit_reason = "regime_exit"
                break
        positions.iloc[entry_loc:exit_loc + 1] = -1.0
        last_exit_loc = exit_loc
        events.append({
            "trigger_date": str(dt.date()),
            "entry_date": str(trading_idx[entry_loc].date()),
            "exit_date": str(trading_idx[exit_loc].date()),
            "exit_reason": exit_reason,
            "basket_return": float(cumret.iloc[min(exit_loc - entry_loc, len(cumret) - 1)]),
            "wti_20d_at_trigger": float(wti_20.iloc[i]) if pd.notna(wti_20.iloc[i]) else None,
            "pos_count_16w": int(pos_count_daily.iloc[i]) if pd.notna(pos_count_daily.iloc[i]) else None,
        })

    if not events:
        return mark_failed(
            sid,
            "no joint WCESTUS1-builds + WTI<55 events fired",
            extra={
                "rule": "Short XOM/CVX 30d on WCESTUS1 12/16 builds + WTI20<$55",
                "mechanism": "Sustained crude builds below dividend breakeven squeeze majors",
                "source": "PL620 catalog",
            },
        )

    pnl = positions.shift(1).fillna(0.0) * basket_r.reindex(positions.index).fillna(0.0)
    pnl = pnl.dropna()
    first_entry = pd.Timestamp(events[0]["entry_date"])
    pnl = pnl.loc[pnl.index >= first_entry]
    active_pos = positions.loc[positions.index >= first_entry]

    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient PnL: {len(pnl)} days")

    m = compute_metrics(
        pnl,
        benchmark=spy_r.reindex(pnl.index),
        name=NAME,
        positions=active_pos.abs(),
        cost_bps=10,
    )
    m["n_events"] = len(events)
    m["pct_in_market"] = float(active_pos.abs().mean())
    m["events"] = events[:20]
    m["status"] = "ok"

    save_result(
        sid,
        m,
        extra={
            "rule": "Short equal-weight (0.5*XOM + 0.5*CVX) basket at next-day close when (a) WCESTUS1 (US crude stocks ex-SPR) had positive WoW change in >=12 of last 16 weekly prints AND (b) DCOILWTICO 20-day MA < $55/bbl. Hold 30 trading days; exit on -8% (profit), +6% (stop), or DCOILWTICO 20-day MA > $60 (regime exit).",
            "mechanism": "Sustained US crude inventory builds (US dominates OECD swing) + WTI below the XOM/CVX dividend-plus-base-capex breakeven of ~$55/bbl (per 2024 corporate guidance) signals a structural earnings squeeze on the integrated majors. Counter-signal against long_energy / long_SPY defaults that assume mean-reversion in oil-related equities.",
            "source": "PL620 idea catalog; FRED WCESTUS1, DCOILWTICO; XOM/CVX 2024 dividend-breakeven guidance",
            "caveats": "US-only WCESTUS1 substitutes for OECD-wide inventories. $55 threshold is corporate-guidance approximation. Counter-signal against widely-held long energy positioning.",
        },
        pnl=pnl,
    )
    print(f"Saved {sid}: n_events={len(events)}  Sharpe={m.get('sharpe',0):.2f}  CAGR={m.get('cagr',0)*100:.2f}%")


if __name__ == "__main__":
    main()
