"""
AF-3: Container Freight Spike → Retail Margin Compression
When ZIM 60-day return >+80% (freight rate proxy spike): short {TGT, DG, FIVE} 60 days later, hold 30 days.
Also hand-coded major freight spike events as alternative.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from backtests.harness import load_prices, daily_returns, compute_metrics, save_result, mark_failed, print_metrics

SIGNAL_ID = "AF-3"

# Hand-coded freight spike events (SCFI peaks)
# Entry is 60 days AFTER the spike peak — allows time for margin impact to show
FREIGHT_SPIKES = [
    # (spike_peak_date, description)
    ("2021-09-15", "COVID supply chain crisis — SCFI near all-time high"),
    ("2022-01-15", "Continued from 2021 — sustained high rates into Q1 2022"),
    ("2024-01-20", "Red Sea / Houthi rerouting — SCFI spiked sharply"),
]

SHORT_BASKET = ["TGT", "DG", "FIVE"]
DELAY_DAYS = 60  # wait 60 days after spike before shorting
HOLD_DAYS = 30


def run():
    print(f"Freight spike events: {len(FREIGHT_SPIKES)}")
    for d, desc in FREIGHT_SPIKES:
        print(f"  {d}: {desc}")

    # Load prices
    all_tickers = SHORT_BASKET + ["ZIM", "SPY"]
    prices = load_prices(all_tickers, start="2020-01-01")
    rets = daily_returns(prices)

    # Also check ZIM-based signal as validation
    if "ZIM" in prices.columns:
        zim = prices["ZIM"].dropna()
        zim_60d_ret = zim.pct_change(60)
        zim_triggers = zim_60d_ret[zim_60d_ret > 0.80]
        print(f"\nZIM 60d >+80% days: {len(zim_triggers)}")
        if len(zim_triggers) > 0:
            # Show first occurrence in each cluster (at least 30 days apart)
            clusters = []
            last_date = None
            for d in zim_triggers.index:
                if last_date is None or (d - last_date).days > 60:
                    clusters.append(d)
                    last_date = d
            print(f"  Cluster starts: {[str(c.date()) for c in clusters]}")

    event_results = []
    all_pnl = []

    for spike_date_str, desc in FREIGHT_SPIKES:
        spike_date = pd.Timestamp(spike_date_str)
        # Entry is DELAY_DAYS after spike
        entry_target = spike_date + pd.Timedelta(days=DELAY_DAYS)
        valid_dates = rets.index[rets.index >= entry_target]
        if len(valid_dates) < HOLD_DAYS:
            continue
        start_idx = valid_dates[0]
        start_loc = rets.index.get_loc(start_idx)
        end_loc = min(start_loc + HOLD_DAYS, len(rets) - 1)

        period_rets = rets.iloc[start_loc:end_loc + 1]
        available = [t for t in SHORT_BASKET if t in period_rets.columns and period_rets[t].notna().sum() > 0]
        if not available:
            continue

        # SHORT position → negate returns
        basket_daily = -period_rets[available].mean(axis=1)
        spy_daily = period_rets["SPY"] if "SPY" in period_rets.columns else pd.Series(0, index=period_rets.index)

        total_ret = (1 + basket_daily).prod() - 1
        spy_ret = (1 + spy_daily).prod() - 1

        # Per-leg returns (for the longs these are the raw unreversed returns of the shorted tickers)
        leg_rets = {}
        for t in available:
            leg_rets[t] = float((1 + period_rets[t]).prod() - 1)

        event_results.append({
            "spike_date": spike_date_str,
            "description": desc,
            "entry": str(start_idx.date()),
            "exit": str(rets.index[end_loc].date()),
            "short_basket_return": float(total_ret),  # positive = short was profitable
            "spy_return": float(spy_ret),
            "excess_vs_spy": float(total_ret - spy_ret),
            "per_leg_returns": leg_rets,
        })

        all_pnl.append(basket_daily)

    if not all_pnl:
        mark_failed(SIGNAL_ID, "No valid events after delay period", extra={
            "rule": "Freight spike → short discount retail 60d later, hold 30d",
            "mechanism": "Higher shipping costs compress margins for importers (discount retail) with 1-2 quarter lag",
            "source": "SCFI data / ZIM as proxy (hand-coded events)",
        })
        return

    combined_pnl = pd.concat(all_pnl)
    combined_pnl = combined_pnl[~combined_pnl.index.duplicated(keep='first')]
    combined_pnl = combined_pnl.sort_index()

    spy_rets = rets["SPY"].reindex(combined_pnl.index)

    metrics = compute_metrics(combined_pnl, benchmark=spy_rets, name="AF-3 Freight Spike → Short Retail")
    print_metrics(metrics)

    print("\nPer-event results:")
    for er in event_results:
        print(f"  Spike {er['spike_date']} → entry {er['entry']}: short-basket {er['short_basket_return']*100:.1f}%, SPY {er['spy_return']*100:.1f}%")
        for t, r in er["per_leg_returns"].items():
            print(f"    {t}: {r*100:.1f}% (shorted)")

    extra = {
        "status": "ok",
        "n_events": len(event_results),
        "events": event_results,
        "rule": "Container freight spike (ZIM 60d ret >80% or SCFI peak) → short {TGT,DG,FIVE} 60d later, hold 30d",
        "mechanism": "Higher shipping costs compress margins for high-import-exposure discount retailers with 1-2 quarter lag",
        "source": "SCFI peaks (hand-coded) + ZIM as freight rate proxy",
    }

    save_result(SIGNAL_ID, metrics, extra=extra)
    print(f"\nSaved → results/{SIGNAL_ID}.json")


if __name__ == "__main__":
    run()
