"""PL1023_cftc_cot_gold_mm_extreme_long_fade_short_gld — CFTC COT Gold Extreme Long Positioning → Short GLD Sentiment Fade"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL1023_cftc_cot_gold_mm_extreme_long_fade_short_gld"

    # Load CFTC COT gold data (commercial net — negative = commercials short = speculators/MM very long)
    try:
        cot = pd.read_parquet("data/cftc_disagg_GOLD.parquet")
    except Exception as e:
        return mark_failed(sid, f"CFTC data load: {e}")

    # Spec net ≈ -(commercial net); most negative commercial = most extreme spec long
    # Use commercial net z-score: very negative z-score → spec net very positive → fade gold
    window = 156  # 3-year rolling window (as per strategy)
    cot["roll_mean"] = cot["net"].rolling(window).mean()
    cot["roll_std"] = cot["net"].rolling(window).std()
    cot["z_score"] = (cot["net"] - cot["roll_mean"]) / cot["roll_std"]
    # Commercial net z < -2.0 means specs are >2σ extreme long → fade long = short GLD
    # Also apply absolute threshold: commercial net < -250K (robust level)
    cot["signal"] = (cot["z_score"] < -2.0) & (cot["net"] < -250000)

    # Check signal
    signal_weeks = cot[cot["signal"] & cot["z_score"].notna()]
    print(f"Signal weeks (extreme spec long, n={len(signal_weeks)})")
    print(cot[cot["signal"]].head(10)[["net", "z_score"]].to_string())

    if len(signal_weeks) < 3:
        return mark_failed(sid, f"too few signal weeks: {len(signal_weeks)}")

    try:
        px = load_prices(["GLD", "SPY"], start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"price data load: {e}")

    if "GLD" not in px.columns:
        return mark_failed(sid, "GLD not in price data")

    ret = daily_returns(px)
    spy_r = ret["SPY"]
    gld_r = ret["GLD"]

    # CFTC report is for prior Tuesday positions, published following Friday
    # Entry: 2 trading days after Friday publication (Monday/Tuesday of next week)
    # We'll approximate: use the cot.index date + 7 days as approximate entry signal
    hold_calendar = 56  # 8 weeks
    stop_pct = 0.08  # stop if GLD gains 8%
    revert_thresh = -200000  # commercial net above this (less short) = MM reversion signal

    pnl = pd.Series(0.0, index=ret.index)
    event_records = []

    # De-duplicate: minimum 3-week (21 day) gap between entries
    last_entry = None

    for dt, row in cot.iterrows():
        if not row["signal"]:
            continue
        if pd.isna(row["z_score"]):
            continue

        # Entry: ~2 trading days after CFTC Friday publication
        entry_date = dt + pd.Timedelta(days=5)  # ~following Friday + 2 days ~ Wednesday
        if last_entry is not None and (entry_date - last_entry).days < 21:
            continue  # too close to prior trade

        # Find actual trading day at or after entry_date
        future_idx = gld_r.index[gld_r.index >= entry_date]
        if len(future_idx) < 3:
            continue
        entry_idx = future_idx[0]

        if last_entry is not None and (entry_idx - last_entry).days < 21:
            continue

        # Exit conditions
        exit_date = entry_idx + pd.Timedelta(days=hold_calendar)
        trade_dates = gld_r.index[(gld_r.index > entry_idx) & (gld_r.index <= exit_date)]
        if len(trade_dates) == 0:
            continue

        exit_reason = "time"
        active_dates = []
        cum_gld = 1.0

        for td in trade_dates:
            r = gld_r.get(td, 0.0)
            cum_gld *= (1 + r)
            active_dates.append(td)

            # Stop: GLD gains 8% (adverse for short)
            if cum_gld - 1 >= stop_pct:
                exit_reason = "stop_loss"
                break

            # Check COT reversion: find the most recent COT report <= td
            recent_cot = cot[cot.index <= td]
            if len(recent_cot) > 0:
                latest_comm_net = recent_cot.iloc[-1]["net"]
                if latest_comm_net > revert_thresh:  # less negative = spec longs unwinding
                    exit_reason = "cot_reversion"
                    break

        if not active_dates:
            continue

        last_entry = entry_idx
        gld_cum = float((1 + gld_r.loc[active_dates]).prod() - 1)
        spy_cum = float((1 + spy_r.reindex(active_dates).fillna(0)).prod() - 1)
        trade_pnl = -gld_cum  # short GLD

        event_records.append({
            "entry": str(entry_idx.date()),
            "exit": str(active_dates[-1].date()),
            "n_days": len(active_dates),
            "comm_net": int(row["net"]),
            "z_score": round(float(row["z_score"]), 2),
            "gld_cum": round(gld_cum, 4),
            "trade_pnl": round(trade_pnl, 4),
            "spy_cum": round(spy_cum, 4),
            "exit_reason": exit_reason,
        })

        for td in active_dates:
            if td in pnl.index:
                pnl[td] -= gld_r.get(td, 0.0)  # short GLD = negative return

    print(f"\nTrade records ({len(event_records)}):")
    for e in event_records:
        print(f"  {e['entry']} -> {e['exit']} ({e['n_days']}d, {e['exit_reason']}, z={e['z_score']:.1f}): GLD={e['gld_cum']:.2%}, pnl={e['trade_pnl']:.2%}")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active pnl days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="CFTC COT Gold Extreme Long -> Short GLD")

    save_result(sid, m, extra={
        "rule": "Short GLD when CFTC commercial gold net < -250K AND z-score of commercial net < -2.0 (3yr rolling). Cover at COT reversion above -200K, 56 calendar days, or GLD +8% stop.",
        "mechanism": "When speculative/managed-money net longs in COMEX gold are 2+ std devs above their historical mean, sentiment is over-extended; mean-reversion in positioning leads to gold price correction.",
        "source": "CFTC Disaggregated COT gold (data/cftc_disagg_GOLD.parquet); GLD via yfinance.",
        "n_events": len(event_records),
        "events": event_records,
    })


if __name__ == "__main__":
    main()
