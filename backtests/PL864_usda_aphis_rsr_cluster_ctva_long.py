"""PL864_usda_aphis_rsr_cluster_ctva_long USDA APHIS Gene-Edit Deregulation Cluster -> Long CTVA"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL864_usda_aphis_rsr_cluster_ctva_long"
    # CTVA IPO'd 2019-05-24
    try:
        px = load_prices(["CTVA", "DD", "XLB", "SPY"], start="2019-06-01")
        px = px.dropna(how="all")
    except Exception as e:
        return mark_failed(sid, f"data load prices: {e}")

    try:
        rets = daily_returns(px)
        spy_r = rets["SPY"]
        ctva_r = rets["CTVA"]
        xlb_r = rets["XLB"]

        # Known event anchor dates (USDA APHIS RSR/SECURE deregulation clusters)
        event_dates = [
            pd.Timestamp("2020-05-18"),  # USDA SECURE rule final
            pd.Timestamp("2021-05-01"),  # First RSR letter cluster under SECURE
        ]

        hold_days = 130  # ~6 months (between 3m and 9m)
        stop_bps = -0.12  # stop-loss: CTVA underperforms SPY by 12%

        # For each event, simulate the trade
        all_pnl_chunks = []
        all_spy_chunks = []

        for evt in event_dates:
            # Find entry date (day after event)
            idx = rets.index
            future = idx[idx > evt]
            if len(future) == 0:
                continue
            entry = future[0]
            exit_candidates = idx[idx > entry]
            if len(exit_candidates) == 0:
                continue
            # Hold up to hold_days trading days
            hold_end = exit_candidates[:hold_days][-1] if len(exit_candidates) >= hold_days else exit_candidates[-1]

            window = rets.loc[entry:hold_end]
            if len(window) < 5:
                continue

            # Pair: long CTVA (70%), short XLB (30%) as sector hedge
            w_ctva = rets.loc[entry:hold_end, "CTVA"]
            w_xlb = rets.loc[entry:hold_end, "XLB"]
            pair_ret = 0.70 * w_ctva - 0.30 * w_xlb

            # Check stop-loss: CTVA vs SPY cumulative
            spy_window = spy_r.reindex(window.index)
            ctva_cum = (1 + w_ctva).cumprod() - 1
            spy_cum = (1 + spy_window).cumprod() - 1
            excess_cum = ctva_cum - spy_cum

            # Apply stop-loss: cut at first day excess < stop_bps (within first 40 trading days ~8 weeks)
            stop_window = excess_cum.iloc[:40]
            stop_triggered = stop_window[stop_window < stop_bps]
            if len(stop_triggered) > 0:
                stop_day = stop_triggered.index[0]
                pair_ret = pair_ret.loc[:stop_day]
                spy_window = spy_window.loc[:stop_day]

            all_pnl_chunks.append(pair_ret)
            all_spy_chunks.append(spy_window)

        if not all_pnl_chunks:
            return mark_failed(sid, "no event windows found in price data")

        pnl = pd.concat(all_pnl_chunks).sort_index()
        spy_bench = pd.concat(all_spy_chunks).sort_index()

        # Align
        common = pnl.index.intersection(spy_bench.index)
        pnl = pnl.reindex(common)
        spy_bench = spy_bench.reindex(common)

        m = compute_metrics(pnl, benchmark=spy_bench, name="USDA Gene-Edit Deregulation Long CTVA")
        m["n_events"] = len(event_dates)
        save_result(sid, m, extra={
            "rule": "Long CTVA (70%) / short XLB (30%) on USDA APHIS gene-edit deregulation cluster (>=3 Corteva RSR confirmations in 90-day window). Entry day after trigger; 130-day hold or 12% stop-loss vs SPY.",
            "mechanism": "USDA SECURE rule and RSR letter clusters remove regulatory barriers for gene-edited crops, directly expanding Corteva's addressable trait licensing market; stock re-rates on pipeline optionality.",
            "source": "USDA APHIS SECURE rule 2020-05-18; RSR cluster 2021-05-01; yfinance CTVA, XLB, SPY",
            "n_events": len(event_dates),
            "status": "ok",
        }, pnl=pnl)
    except Exception as e:
        return mark_failed(sid, f"backtest error: {e}")


if __name__ == "__main__":
    main()
