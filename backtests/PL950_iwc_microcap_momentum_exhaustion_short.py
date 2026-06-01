"""PL950_iwc_microcap_momentum_exhaustion_short — Retail-Driven Microcap Momentum Exhaustion: Short IWC / Long IEF"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL950_iwc_microcap_momentum_exhaustion_short"

    try:
        px = load_prices(["IWC", "IEF", "SPY"], start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    for t in ["IWC", "IEF", "SPY"]:
        if t not in px.columns or px[t].dropna().shape[0] < 100:
            return mark_failed(sid, f"{t} data unavailable or insufficient")

    spy_r = daily_returns(px[["SPY"]]).iloc[:, 0].dropna()
    iwc_r = daily_returns(px[["IWC"]]).iloc[:, 0].dropna()
    ief_r = daily_returns(px[["IEF"]]).iloc[:, 0].dropna()

    common_idx = spy_r.index.intersection(iwc_r.index).intersection(ief_r.index)
    spy_r = spy_r.reindex(common_idx)
    iwc_r = iwc_r.reindex(common_idx)
    ief_r = ief_r.reindex(common_idx)

    # Compute 60-day rolling returns (using price levels)
    iwc_px = px["IWC"].reindex(common_idx).ffill()
    spy_px = px["SPY"].reindex(common_idx).ffill()

    iwc_60d = iwc_px.pct_change(60)
    spy_60d = spy_px.pct_change(60)

    # Signal conditions:
    # (1) IWC 60-day return > 15%
    # (2) IWC 60-day return - SPY 60-day return > 10pp
    cond1 = iwc_60d > 0.15
    cond2 = (iwc_60d - spy_60d) > 0.10

    # Both conditions hold for 2 consecutive trading days
    both = cond1 & cond2
    signal_raw = both & both.shift(1)

    # Pair return: short IWC (60%) + long IEF (40%)
    pair_r = -0.6 * iwc_r + 0.4 * ief_r

    # Detect entry dates with 30-day cooldown
    signal_dates = []
    last_entry = None
    cooldown = 30

    for dt in signal_raw[signal_raw].index:
        if last_entry is None or (dt - last_entry).days >= cooldown * 1.4:  # approx calendar days
            signal_dates.append(dt)
            last_entry = dt

    print(f"Signal dates: {len(signal_dates)}")
    for sd in signal_dates:
        mom = iwc_60d.get(sd, float('nan'))
        excess = (iwc_60d - spy_60d).get(sd, float('nan'))
        print(f"  {sd.date()}: IWC 60d={mom*100:.1f}%, excess={excess*100:.1f}pp")

    if not signal_dates:
        return mark_failed(sid, "no valid signal dates found (IWC momentum threshold not reached)")

    hold_days = 40
    pnl = pd.Series(0.0, index=common_idx)
    events = []

    for entry_date in signal_dates:
        if entry_date not in common_idx:
            continue
        entry_idx = common_idx.get_loc(entry_date)
        exit_idx = min(entry_idx + hold_days, len(common_idx))

        pair_slice = pair_r.iloc[entry_idx:exit_idx]
        spy_slice = spy_r.iloc[entry_idx:exit_idx]
        iwc_slice = iwc_r.iloc[entry_idx:exit_idx]

        # Exit conditions:
        # (a) IWC 60-day return drops below +5% (momentum exhausted)
        iwc_60d_during = iwc_60d.reindex(pair_slice.index).ffill()
        momentum_exhausted = iwc_60d_during < 0.05

        # (b) Stop-loss: IWC short leg rises >15% from entry (cumulative)
        iwc_cum = iwc_r.iloc[entry_idx:exit_idx].cumsum()
        stop_hit = iwc_cum > 0.15

        # Take the earliest exit signal
        any_exit = momentum_exhausted | stop_hit
        if any_exit.any():
            exit_point = any_exit.idxmax()
            pair_slice = pair_slice.loc[:exit_point]
            spy_slice = spy_r.reindex(pair_slice.index).fillna(0)
            iwc_slice = iwc_r.reindex(pair_slice.index).fillna(0)

        actual_exit_idx = entry_idx + len(pair_slice)

        # Check for overlaps
        target_slice = pnl.iloc[entry_idx:actual_exit_idx]
        if (target_slice != 0).sum() > 0:
            print(f"  Skipping {entry_date.date()} — overlaps")
            continue

        pnl.iloc[entry_idx:actual_exit_idx] = pair_slice.values

        cum_pair_total = float((1 + pair_slice).prod() - 1)
        cum_iwc_total = float((1 + iwc_slice).prod() - 1)
        cum_spy_total = float((1 + spy_slice).prod() - 1)
        stop_triggered = bool(stop_hit.reindex(pair_slice.index).fillna(False).any())
        mom_exit = bool(momentum_exhausted.reindex(pair_slice.index).fillna(False).any())

        events.append({
            "entry_date": str(entry_date.date()),
            "hold_days": len(pair_slice),
            "pair_return": round(cum_pair_total, 4),
            "iwc_return": round(cum_iwc_total, 4),
            "spy_return": round(cum_spy_total, 4),
            "alpha": round(cum_pair_total - cum_spy_total, 4),
            "stop_hit": stop_triggered,
            "momentum_exhausted_exit": mom_exit,
        })

    if not events:
        return mark_failed(sid, "no events with valid PnL computed")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 30:
        return mark_failed(sid, f"insufficient active trading days ({len(active_pnl)})")

    print(f"Events: {len(events)}, active days: {len(active_pnl)}")
    for ev in events:
        print(f"  {ev['entry_date']}: pair={ev['pair_return']*100:.1f}%, IWC={ev['iwc_return']*100:.1f}%, alpha={ev['alpha']*100:.1f}%, hold={ev['hold_days']}d")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="Microcap Momentum Exhaustion → Short IWC / Long IEF")
    m["n_events"] = len(events)

    avg_alpha = float(np.mean([e["alpha"] for e in events]))
    win_rate = float(np.mean([1 if e["pair_return"] > 0 else 0 for e in events]))

    save_result(sid, m, extra={
        "rule": "Short IWC (60%) + Long IEF (40%) when IWC 60-day return > 15% AND IWC excess vs SPY > 10pp for 2 consecutive days. Hold up to 40 trading days. Exit on momentum normalization (<5%) or stop-loss (IWC +15%).",
        "mechanism": "Microcap ETF momentum overshoot driven by retail flows exhausts marginal buyers. Mean reversion to fundamentals occurs when retail positioning peaks. IEF long leg provides carry and low-correlation offset.",
        "source": "yfinance (IWC, IEF, SPY)",
        "n_events": len(events),
        "avg_event_alpha": round(avg_alpha, 4),
        "event_win_rate": round(win_rate, 4),
        "events": events,
    })
    print(f"Done: Sharpe={m.get('sharpe', 'N/A'):.2f}, CAGR={m.get('cagr', 'N/A')*100:.1f}%, MaxDD={m.get('max_dd', 'N/A')*100:.1f}%")


if __name__ == "__main__":
    main()
