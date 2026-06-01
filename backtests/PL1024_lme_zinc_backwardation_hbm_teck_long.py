"""PL1024_lme_zinc_backwardation_hbm_teck_long — LME Zinc Backwardation (Smelter Cut Signal) — Long HBM / TECK"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL1024_lme_zinc_backwardation_hbm_teck_long"

    # Load FRED monthly zinc price as backwardation proxy
    try:
        zinc_raw = load_fred("PZINCUSDM", start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED load: {e}")

    zinc = zinc_raw.squeeze().dropna()
    if zinc.empty:
        return mark_failed(sid, "PZINCUSDM data empty")

    # Load prices
    try:
        px = load_prices(["HBM", "TECK", "SPY"], start="2010-01-01")
    except Exception as e:
        try:
            px = load_prices(["HBM", "TECK", "SPY"], start="2010-01-01")
        except Exception as e2:
            return mark_failed(sid, f"price load: {e2}")

    if "HBM" not in px.columns or "TECK" not in px.columns:
        return mark_failed(sid, f"Missing tickers in price data: {list(px.columns)}")

    ret = daily_returns(px)
    hbm_r = ret["HBM"]
    teck_r = ret["TECK"]
    spy_r = ret["SPY"]

    # Find months where zinc >10% MoM increase (backwardation proxy)
    zinc_mom = zinc.pct_change()
    signal_months = zinc_mom[zinc_mom > 0.10].index
    print(f"Months with >10% MoM zinc jump: {len(signal_months)}")

    # For each signal month, enter on first trading day of following month
    # Hold 12 weeks (~84 calendar days)
    hold_days = 84
    pnl = pd.Series(0.0, index=spy_r.index)
    events = []
    # Track occupied dates to avoid overlap
    occupied = set()

    for sig_dt in signal_months:
        # Entry: first trading day of next month
        next_month = sig_dt + pd.offsets.MonthBegin(1)
        entry_mask = hbm_r.index >= next_month
        if not entry_mask.any():
            continue
        entry_idx = hbm_r.index[entry_mask][0]

        # Exit: 84 calendar days from entry
        exit_dt = entry_idx + pd.Timedelta(days=hold_days)
        exit_mask = (hbm_r.index > entry_idx) & (hbm_r.index <= exit_dt)
        if not exit_mask.any():
            continue

        window_hbm = hbm_r[exit_mask]
        window_teck = teck_r.reindex(window_hbm.index).fillna(0.0)
        window_spy = spy_r.reindex(window_hbm.index).fillna(0.0)

        # Equal-weight HBM + TECK long
        daily_pos = 0.5 * window_hbm + 0.5 * window_teck

        # Add to PnL, skip days already covered (overlap prevention)
        for idx, val in daily_pos.items():
            if idx not in occupied:
                pnl[idx] += val
                occupied.add(idx)

        total_hbm = float((1 + window_hbm).prod() - 1)
        total_teck = float((1 + window_teck).prod() - 1)
        total_spy = float((1 + window_spy).prod() - 1)
        combined = 0.5 * total_hbm + 0.5 * total_teck
        zinc_val = float(zinc.loc[sig_dt]) if sig_dt in zinc.index else None
        zinc_pct = float(zinc_mom.loc[sig_dt]) if sig_dt in zinc_mom.index else None

        print(f"  Signal {sig_dt.date()} (zinc +{zinc_pct:.1%}): "
              f"entry={entry_idx.date()}, HBM={total_hbm:.3f}, TECK={total_teck:.3f}, "
              f"comb={combined:.3f}, SPY={total_spy:.3f}")

        events.append({
            "signal_month": str(sig_dt.date()),
            "entry_date": str(entry_idx.date()),
            "exit_date": str(exit_dt.date()),
            "n_days_held": len(window_hbm),
            "zinc_mom_pct": round(zinc_pct, 4) if zinc_pct is not None else None,
            "hbm_return": round(total_hbm, 4),
            "teck_return": round(total_teck, 4),
            "combined_return": round(combined, 4),
            "spy_return": round(total_spy, 4),
            "excess_return": round(combined - total_spy, 4),
        })

    if not events:
        return mark_failed(sid, "no qualifying zinc signal events")

    active_pnl = pnl[pnl != 0.0]
    print(f"\nEvents: {len(events)}, Active days: {len(active_pnl)}")

    if len(active_pnl) < 30:
        return mark_failed(sid, f"insufficient active days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="Zinc Backwardation → Long HBM+TECK")
    avg_excess = float(np.mean([e["excess_return"] for e in events]))
    win_rate = float(np.mean([e["combined_return"] > e["spy_return"] for e in events]))

    save_result(sid, m, extra={
        "rule": "Long HBM + TECK equal-weight on first trading day of month following >10% MoM zinc price jump (FRED PZINCUSDM); hold 12 weeks",
        "mechanism": "LME zinc cash backwardation signals physical tightness (smelter cuts, low inventories); zinc-producing miners (HBM, TECK) benefit from spot price premium and improved concentrate margins",
        "source": "FRED PZINCUSDM; yfinance HBM, TECK, SPY",
        "n_events": len(events),
        "avg_excess_return": round(avg_excess, 4),
        "win_rate_vs_spy": round(win_rate, 4),
        "events": events,
    })
    print(f"Done: {len(events)} events")


if __name__ == "__main__":
    main()
