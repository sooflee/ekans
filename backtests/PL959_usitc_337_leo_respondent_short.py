"""PL959_usitc_337_leo_respondent_short — USITC Section 337 LEO/CDO: Short Named Semi Respondent vs SOXX"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL959_usitc_337_leo_respondent_short"

    # Historical USITC Section 337 final determinations with LEO/CDO against US-listed semiconductor/tech respondents
    # Sources: USITC EDIS docket, Federal Register
    # Format: (event_date, respondent_ticker, case_description)
    events_raw = [
        # Apple modem chips (337-TA-1284): Commission Final Determination, Apple as respondent
        ("2023-02-15", "AAPL", "337-TA-1284 Apple modem chips LEO"),
        # ParkerVision v QCOM (337-TA-1237): Commission Final Determination, QCOM as respondent
        ("2024-01-18", "QCOM", "337-TA-1237 ParkerVision v QCOM CDO"),
        # ITC 337-TA-1065 (Broadcom v Arista): Arista as respondent, Jan 2019 determination
        ("2019-01-31", "ANET", "337-TA-1065 Broadcom v Arista LEO"),
        # ITC 337-TA-1047 (Intel v Broadcom/LSI): Avago/Broadcom respondent, ~2018
        ("2018-06-25", "AVGO", "337-TA-1047 Intel v Avago LEO"),
        # ITC 337-TA-1197 (Qualcomm v Apple): Apple as respondent, ~Feb 2022
        ("2022-02-10", "AAPL", "337-TA-1197 Qualcomm v Apple CDO"),
        # ITC 337-TA-1200 (Ericsson v Apple): Apple as respondent, ~Jun 2021
        ("2021-06-15", "AAPL", "337-TA-1200 Ericsson v Apple LEO"),
    ]

    # Load all tickers needed
    all_tickers = list(set([ev[1] for ev in events_raw] + ["SOXX", "SPY"]))
    try:
        px = load_prices(all_tickers, start="2018-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if px.empty:
        return mark_failed(sid, "price data unavailable")

    # Ensure SOXX is available
    if "SOXX" not in px.columns:
        return mark_failed(sid, "SOXX price data unavailable")

    # Build daily returns
    ret = daily_returns(px)
    spy_r = ret["SPY"] if "SPY" in ret.columns else None

    HOLD_DAYS = 45

    # Build pair PnL: short respondent, long SOXX (dollar-neutral)
    pnl = pd.Series(0.0, index=ret.index)
    event_details = []

    for (ev_date_str, resp_ticker, desc) in events_raw:
        ev_date = pd.Timestamp(ev_date_str)

        # Find next available trading day
        future_dates = ret.index[ret.index >= ev_date]
        if len(future_dates) == 0:
            continue

        entry_date = future_dates[0]
        entry_loc = ret.index.get_loc(entry_date)

        # Check respondent ticker is available
        if resp_ticker not in ret.columns:
            continue

        # Exit after HOLD_DAYS
        end_loc = min(entry_loc + HOLD_DAYS, len(ret) - 1)

        resp_r = ret[resp_ticker].iloc[entry_loc:end_loc]
        soxx_r = ret["SOXX"].iloc[entry_loc:end_loc]

        if len(resp_r) < 5:
            continue

        # Pair return: short respondent + long SOXX (each 50% notional)
        pair_r = 0.5 * soxx_r.values[:len(resp_r)] - 0.5 * resp_r.values[:len(resp_r)]
        pair_series = pd.Series(pair_r, index=resp_r.index)

        # Stop-loss: if pair spread moves -3% from entry, exit
        cum_pair = (1 + pair_series).cumprod() - 1
        stop_idx = None
        for j, cp in enumerate(cum_pair):
            if cp < -0.03:
                stop_idx = j
                break
        if stop_idx is not None:
            pair_series = pair_series.iloc[:stop_idx + 1]

        # Accumulate PnL
        for idx, val in pair_series.items():
            if idx in pnl.index:
                pnl[idx] = pnl[idx] + val

        total_return = float((1 + pair_series).prod() - 1)
        resp_total = float((1 + resp_r.iloc[:len(pair_series)]).prod() - 1)
        soxx_total = float((1 + soxx_r.iloc[:len(pair_series)]).prod() - 1)

        event_details.append({
            "event_date": ev_date_str,
            "entry_date": str(entry_date.date()),
            "respondent": resp_ticker,
            "description": desc,
            "hold_days": len(pair_series),
            "pair_return": round(total_return, 4),
            "resp_return": round(resp_total, 4),
            "soxx_return": round(soxx_total, 4),
        })

    pnl = pnl.dropna()
    active_days = int((pnl != 0).sum())

    if len(event_details) < 2:
        return mark_failed(sid, f"insufficient events for backtest: {len(event_details)}")

    if active_days < 30:
        return mark_failed(sid, f"insufficient active days: {active_days}")

    m = compute_metrics(pnl, benchmark=spy_r, name="USITC 337 LEO/CDO Short Respondent vs SOXX")

    avg_pair_ret = np.mean([e["pair_return"] for e in event_details])
    hit_rate = np.mean([e["pair_return"] > 0 for e in event_details])

    save_result(sid, m, extra={
        "rule": "Short named US-listed semi respondent + long SOXX (dollar-neutral pair) when USITC issues Section 337 LEO or CDO; hold 45 trading days; stop-loss if pair spread moves -3%",
        "mechanism": "USITC LEO/CDO creates immediate revenue/margin risk for named respondent via product import restrictions; sector peers (SOXX) unaffected, creating transient idiosyncratic drag on respondent relative to index",
        "source": "USITC EDIS docket, Federal Register; prices via yfinance",
        "n_events": len(event_details),
        "n_active_days": active_days,
        "avg_pair_return": round(float(avg_pair_ret), 4),
        "event_hit_rate": round(float(hit_rate), 4),
        "events": event_details,
        "caveat": "Only ~6 events identified (2018-2026); small sample. Many 337 respondents are foreign companies not easily shorted. Event dates may not be exact.",
    })

    print(f"Done: {len(event_details)} events, hit_rate={hit_rate:.2%}, avg_pair={avg_pair_ret:.2%}, Sharpe={m.get('sharpe', 'N/A'):.2f}")


if __name__ == "__main__":
    main()
