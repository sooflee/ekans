"""PL723_scfi_wci_div_short_zim
SCFI Europe vs Drewry WCI Divergence -> Short ZIM (Counter)

Original spec: When 4-week SCFI Europe % change minus 4-week Drewry WCI % change
> 15pp, short ZIM for 30 trading days.

Implementation note: SCFI (Shanghai Containerized Freight Index) Europe leg and
Drewry WCI are proprietary subscription datasets not available in this environment.
We use a price-based proxy: ZIM 4-week excess return vs MATX (Matson Inc, a proxy
for the U.S. container shipping sector). When ZIM's 20-day excess return over MATX
crosses above 2-sigma of its trailing-252d distribution (ZIM outperforming sector
=> early-cycle hype peak), short ZIM for 30 trading days.

ZIM is far more volatile than MATX (spot-rate exposed vs contract-heavy U.S. routes)
so the divergence captures the same mechanism: ZIM rallies on SCFI spike optimism,
but if WCI doesn't confirm, the rally is fragile and ZIM reverts.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


HOLD_DAYS = 30
LOOKBACK_SIGNAL = 20      # ~4 weeks of trading days
LOOKBACK_BASELINE = 252   # 1y trailing mean/std for z-score
ZSCORE_THRESHOLD = 2.0    # signal when z-score > 2.0
MIN_COOLDOWN_DAYS = 45    # avoid re-triggering during hold


def main():
    sid = "PL723_scfi_wci_div_short_zim"
    tickers = ["ZIM", "MATX", "SPY"]

    try:
        # ZIM IPO Jan 28 2021; MATX long history
        px = load_prices(tickers, start="2021-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    zim_r  = ret["ZIM"].dropna()
    matx_r = ret["MATX"].dropna()
    spy_r  = ret["SPY"].dropna()

    # Align all series
    common = zim_r.index.intersection(matx_r.index).intersection(spy_r.index)
    zim_r  = zim_r.reindex(common)
    matx_r = matx_r.reindex(common)
    spy_r  = spy_r.reindex(common)

    # 20-day excess return: ZIM vs MATX (proxy for SCFI Europe vs WCI divergence)
    zim_px  = (1 + zim_r).cumprod()
    matx_px = (1 + matx_r).cumprod()

    zim_20d  = zim_px  / zim_px.shift(LOOKBACK_SIGNAL) - 1
    matx_20d = matx_px / matx_px.shift(LOOKBACK_SIGNAL) - 1
    excess   = zim_20d - matx_20d

    # Rolling z-score of excess return vs trailing 252d mean/std
    roll_mean = excess.rolling(LOOKBACK_BASELINE).mean()
    roll_std  = excess.rolling(LOOKBACK_BASELINE).std()
    z_score   = (excess - roll_mean) / roll_std.clip(lower=1e-6)

    # Trigger: z-score crosses from below to above ZSCORE_THRESHOLD (ZIM hype peak)
    z_prev    = z_score.shift(1)
    raw_signal = (z_prev <= ZSCORE_THRESHOLD) & (z_score > ZSCORE_THRESHOLD)

    # Apply cooldown
    triggers = []
    last_trigger_idx = -999
    for i, (dt, is_sig) in enumerate(raw_signal.items()):
        if is_sig and not pd.isna(is_sig):
            if (i - last_trigger_idx) >= MIN_COOLDOWN_DAYS:
                triggers.append(dt)
                last_trigger_idx = i

    print(f"Signal triggers: {len(triggers)}")
    if len(triggers) < 3:
        return mark_failed(sid, f"too few signal triggers: {len(triggers)} (need >= 3)")

    # Build daily PnL series: short ZIM on next session after trigger, hold HOLD_DAYS
    pnl       = pd.Series(0.0, index=common)
    positions = pd.Series(0.0, index=common)

    event_log = []
    for trig_dt in triggers:
        future = common[common > trig_dt]
        if len(future) == 0:
            continue
        entry_dt  = future[0]
        entry_idx = common.get_loc(entry_dt)
        exit_idx  = min(entry_idx + HOLD_DAYS, len(common))
        exit_dt   = common[exit_idx - 1] if exit_idx > entry_idx else entry_dt

        slice_zim = zim_r.iloc[entry_idx:exit_idx]
        slice_spy = spy_r.iloc[entry_idx:exit_idx]
        gross_ev  = float((1 + (-slice_zim)).prod() - 1)
        spy_ev    = float((1 + slice_spy).prod() - 1)

        event_log.append({
            "trigger_date":     str(trig_dt.date()),
            "z_score":          round(float(z_score.loc[trig_dt]), 3),
            "entry_date":       str(entry_dt.date()),
            "exit_date":        str(exit_dt.date()),
            "n_hold_days":      int(exit_idx - entry_idx),
            "short_zim_return": round(gross_ev, 4),
            "spy_return":       round(spy_ev, 4),
            "excess_vs_spy":    round(gross_ev - spy_ev, 4),
        })

        # Fill PnL (short ZIM = -1 * ZIM return); no double-stacking
        for j in range(entry_idx, exit_idx):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = -1.0
                pnl.iloc[j]       = -zim_r.iloc[j]

    n_events = len([e for e in event_log if e.get("entry_date")])
    if n_events == 0:
        return mark_failed(sid, "no valid events with entry_date")

    held_pnl = pnl[positions != 0]
    held_spy = spy_r.reindex(held_pnl.index).dropna()

    if len(held_pnl) < 30:
        return mark_failed(sid, f"insufficient held days: {len(held_pnl)}")

    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name="SCFI/WCI Proxy Short ZIM (held-days only)",
    )

    event_rets = [e["short_zim_return"] for e in event_log if e.get("short_zim_return") is not None]
    summary = {
        "n_events":            n_events,
        "avg_short_return":    round(float(np.mean(event_rets)), 4),
        "median_short_return": round(float(np.median(event_rets)), 4),
        "win_rate":            round(float(np.mean([r > 0 for r in event_rets])), 4),
        "best":                round(float(np.max(event_rets)), 4),
        "worst":               round(float(np.min(event_rets)), 4),
    }

    save_result(
        sid, m,
        extra={
            "status": "ok",
            "rule": (
                "Compute ZIM 20-day excess return vs MATX (U.S. container shipping proxy). "
                f"When rolling z-score crosses above +{ZSCORE_THRESHOLD:.0f}-sigma vs trailing "
                f"252-day baseline (ZIM hype peak), short ZIM for {HOLD_DAYS} trading days."
            ),
            "mechanism": (
                "ZIM (spot-rate exposed, primarily SCFI-linked routes) tends to rally "
                "strongly when SCFI Europe spot rates spike, while MATX (U.S. routes, "
                "contract-heavy) lags. This divergence proxy captures the same signal as "
                "SCFI vs WCI: ZIM has priced in the spot surge but WCI/contract rates don't "
                "confirm, making the rally unsustainable. Short ZIM captures mean-reversion "
                "within the 30-day window."
            ),
            "source": (
                "yfinance ZIM, MATX, SPY. Original spec uses hand-coded SCFI Europe and "
                "Drewry WCI data (proprietary); price proxy used here as a feasibility "
                "approximation."
            ),
            "caveats": (
                "MATX routes are Hawaii/Alaska/Guam — not a perfect sector proxy for "
                "transpacific/Europe container rates. ZIM history starts Jan 2021 so the "
                "sample period is limited. ZIM's extreme volatility (spot-rate driven) "
                "means this short can be painful during sustained freight market rallies."
            ),
            "tickers": tickers,
            "hold_days": HOLD_DAYS,
            "zscore_threshold": ZSCORE_THRESHOLD,
            "lookback_baseline_days": LOOKBACK_BASELINE,
            "n_events": n_events,
            "events": event_log,
            "summary": summary,
        },
        pnl=pnl[positions != 0],
    )

    print(f"Done: {sid}")
    print(f"  n_events: {n_events}")
    print(f"  summary:  {summary}")
    if "error" not in m:
        print(
            f"  Sharpe: {m.get('sharpe', float('nan')):.2f}  "
            f"CAGR: {m.get('cagr', float('nan'))*100:.2f}%  "
            f"MaxDD: {m.get('max_dd', float('nan'))*100:.2f}%  "
            f"t-stat: {m.get('t_stat', float('nan')):.2f}"
        )


if __name__ == "__main__":
    main()
