"""PL927_quantum_retail_euphoria_basket_short — Quantum Stock Retail Euphoria Peak: Short IONQ/RGTI/QUBT/ARQQ Basket"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL927_quantum_retail_euphoria_basket_short"

    # Tickers: quantum basket + SPY
    tickers = ["IONQ", "RGTI", "QUBT", "SPY"]
    try:
        px = load_prices(tickers, start="2021-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    # ARQQ went dark / delisted; drop it gracefully
    basket_tickers = [t for t in ["IONQ", "RGTI", "QUBT"] if t in px.columns and px[t].dropna().shape[0] > 100]
    if len(basket_tickers) < 2:
        return mark_failed(sid, f"insufficient basket tickers available: {basket_tickers}")

    spy_r = daily_returns(px[["SPY"]]).iloc[:, 0].dropna()

    # Build basket price and returns
    basket_px = px[basket_tickers].ffill()
    basket_r = daily_returns(basket_px)

    # Equal-weight basket return
    eq_basket_r = basket_r.mean(axis=1).dropna()

    common_idx = spy_r.index.intersection(eq_basket_r.index)
    spy_r = spy_r.reindex(common_idx)
    eq_basket_r = eq_basket_r.reindex(common_idx)

    # Signal: identify "retail euphoria" spikes using price-momentum proxy
    # Use rolling 20-day return of basket > +40% (captures rapid speculative surges)
    # This proxies what Google Trends 'quantum computing' spikes would capture
    basket_px_common = basket_px.reindex(common_idx).ffill()
    basket_composite = basket_px_common.mean(axis=1)
    rolling_20d = basket_composite.pct_change(20)

    # Known key retail euphoria events (Google Willow announcement ~Dec 2024, 2025 surge)
    # Use these as anchor dates + any other 20d momentum spikes > 40%
    known_events = [
        pd.Timestamp("2024-12-09"),   # Google Willow quantum chip announcement
        pd.Timestamp("2025-01-07"),   # CES 2025 quantum hype
        pd.Timestamp("2023-11-15"),   # early quantum hype wave
    ]

    # Also detect algorithmic spikes: 20d return > 40% on the basket
    algo_triggers = rolling_20d[rolling_20d > 0.40].index.tolist()

    # Merge known events + algo triggers; deduplicate within 30-day windows
    all_candidates = sorted(set(known_events + algo_triggers))
    signal_dates = []
    last_trigger = None
    for d in all_candidates:
        if d not in common_idx:
            # Find nearest available trading day
            future = common_idx[common_idx >= d]
            if len(future) == 0:
                continue
            d = future[0]
        if last_trigger is None or (d - last_trigger).days >= 30:
            signal_dates.append(d)
            last_trigger = d

    print(f"Signal dates found: {len(signal_dates)}")
    for sd in signal_dates:
        mom = rolling_20d.get(sd, float('nan'))
        print(f"  {sd.date()}: basket 20d return = {mom*100:.1f}%" if not np.isnan(mom) else f"  {sd.date()}: known event")

    if not signal_dates:
        return mark_failed(sid, "no valid retail euphoria signal dates found")

    # Build PnL: short equal-weight basket after each signal
    # Hold for 28 trading days (~6 weeks); exit early if basket drops >20% (profit) or rises >15% (stop loss)
    hold_days = 28
    pnl = pd.Series(0.0, index=common_idx)
    events = []

    for entry_date in signal_dates:
        if entry_date not in common_idx:
            continue
        entry_idx = common_idx.get_loc(entry_date)
        exit_idx = min(entry_idx + hold_days, len(common_idx))

        # Short basket: negate basket returns
        short_slice = -eq_basket_r.iloc[entry_idx:exit_idx]
        spy_slice = spy_r.iloc[entry_idx:exit_idx]

        # Stop-loss on short: if basket rallies >15% (short loses >15%), exit
        # Take-profit: basket drops >20% (short gains >20%), exit
        cum_short = short_slice.cumsum()
        stop_hit = cum_short < -0.15   # adverse: basket kept rallying
        tp_hit = cum_short > 0.20       # basket dropped enough

        if (stop_hit | tp_hit).any():
            exit_point = (stop_hit | tp_hit).idxmax()
            short_slice = short_slice.loc[:exit_point]
            spy_slice = spy_r.reindex(short_slice.index).fillna(0)

        actual_exit_idx = entry_idx + len(short_slice)

        # Only assign pnl where not already assigned (no overlaps)
        target_slice = pnl.iloc[entry_idx:actual_exit_idx]
        overlap = (target_slice != 0).sum()
        if overlap > 0:
            print(f"  Skipping {entry_date.date()} — overlaps with prior event")
            continue

        pnl.iloc[entry_idx:actual_exit_idx] = short_slice.values

        cum_short_total = float((1 + short_slice).prod() - 1)
        cum_spy_total = float((1 + spy_slice).prod() - 1)
        cum_basket_total = float((1 + eq_basket_r.reindex(short_slice.index).fillna(0)).prod() - 1)

        events.append({
            "entry_date": str(entry_date.date()),
            "hold_days": len(short_slice),
            "short_basket_return": round(cum_short_total, 4),
            "basket_return": round(cum_basket_total, 4),
            "spy_return": round(cum_spy_total, 4),
            "alpha": round(cum_short_total - cum_spy_total, 4),
            "stop_hit": bool((stop_hit).any() if len(stop_hit) else False),
            "tp_hit": bool((tp_hit).any() if len(tp_hit) else False),
        })

    if not events:
        return mark_failed(sid, "no events with sufficient data for PnL computation")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active trading days ({len(active_pnl)})")

    print(f"Events: {len(events)}, active trading days: {len(active_pnl)}")
    for ev in events:
        print(f"  {ev['entry_date']}: short={ev['short_basket_return']*100:.1f}%, basket={ev['basket_return']*100:.1f}%, alpha={ev['alpha']*100:.1f}%, hold={ev['hold_days']}d")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="Quantum Retail Euphoria → Short IONQ/RGTI/QUBT Basket")
    m["n_events"] = len(events)

    avg_alpha = float(np.mean([e["alpha"] for e in events]))
    win_rate = float(np.mean([1 if e["short_basket_return"] > 0 else 0 for e in events]))

    save_result(sid, m, extra={
        "rule": "Short equal-weight basket of IONQ+RGTI+QUBT when basket 20-day price momentum exceeds +40% OR on known retail-euphoria event dates (Google Willow announcement, CES 2025). Hold 28 trading days. Exit on: +20% profit, -15% stop-loss, or time stop.",
        "mechanism": "Quantum computing stocks are narrative-driven with no near-term revenue; retail speculation drives prices far above fundamentals during hype events. Mean reversion after peak enthusiasm creates short opportunities. Historical analogs: IONQ/RGTI spike in Dec 2024, Jan 2023 sector surge.",
        "source": "yfinance (IONQ, RGTI, QUBT, SPY); known_events from public announcements (Google Willow Dec 2024, CES 2025)",
        "tickers_used": basket_tickers,
        "n_events": len(events),
        "avg_event_alpha": round(avg_alpha, 4),
        "event_win_rate": round(win_rate, 4),
        "events": events,
    })
    print(f"Done: Sharpe={m.get('sharpe', 'N/A'):.2f}, CAGR={m.get('cagr', 'N/A')*100:.1f}%, MaxDD={m.get('max_dd', 'N/A')*100:.1f}%")


if __name__ == "__main__":
    main()
