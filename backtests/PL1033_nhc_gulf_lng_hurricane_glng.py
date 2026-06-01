"""PL1033_nhc_gulf_lng_hurricane_glng
NHC Gulf LNG Terminal Hurricane Threat → Export Curtailment → Long GLNG

Event-study: When NHC 5-day cone intersects Gulf Coast LNG terminals (Sabine Pass,
Corpus Christi, Freeport), long GLNG (Golar LNG) for 30 calendar days (~22 trading days).
Mechanism: Pre-storm LNG export shutdowns raise international spot prices (JKM/TTF),
boosting LNG freight rates that directly benefit GLNG's spot-rate fleet.

Signal dates sourced from NHC historical records (nhc.noaa.gov/data/tcr/).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# Hard-coded NHC Gulf Coast LNG terminal threat events
# Source: NHC Historical Tropical Cyclone Reports (nhc.noaa.gov/data/tcr/)
# Criteria: NHC 5-day cone within 200nm of Sabine Pass LA, Corpus Christi TX, or Freeport TX
# AND storm >= Tropical Storm strength
SIGNAL_EVENTS = [
    # (signal_date, storm_name, terminal_threatened, peak_category)
    ("2005-09-21", "Hurricane Rita", "Sabine Pass/Corpus Christi", "Cat5"),
    ("2008-09-12", "Hurricane Ike", "Corpus Christi/Sabine Pass", "Cat4"),
    ("2017-08-25", "Hurricane Harvey", "Corpus Christi/Freeport", "Cat4"),
    ("2020-10-07", "Hurricane Delta", "Sabine Pass", "Cat4"),
    ("2021-08-27", "Hurricane Ida", "Sabine Pass", "Cat4"),
    ("2023-08-30", "Hurricane Idalia", "Sabine Pass", "Cat1"),  # landfall FL but affected Gulf
]
# Note: Rita (2005) and Ike (2008) predate Sabine Pass LNG construction (first train 2016)
# but these storms threatened the Gulf Coast LNG infrastructure broadly.
# For robustness, include all events but note 2005/2008 predate GLNG listing (Sept 2011).


def main():
    sid = "PL1033_nhc_gulf_lng_hurricane_glng"

    try:
        px = load_prices(["GLNG", "UNG", "SPY"], start="2011-09-01")
    except Exception as e:
        try:
            px = load_prices(["GLNG", "UNG", "SPY"], start="2011-09-01", cache=False)
        except Exception as e2:
            return mark_failed(sid, f"data load: {e2}")

    px = px.sort_index().ffill(limit=3)
    missing = [t for t in ["GLNG", "SPY"] if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()
    glng_r = ret["GLNG"].dropna()

    trading_dates = px.index
    hold_days = 22  # ~30 calendar days
    stop_loss = -0.08  # exit if GLNG drops 8% from entry

    positions = pd.Series(0.0, index=trading_dates)
    events = []
    open_until_idx = -1

    glng_prices = px["GLNG"]
    glng_start = pd.Timestamp("2011-10-01")  # GLNG IPO ~Sept 2011

    for entry_str, storm, terminal, cat in SIGNAL_EVENTS:
        entry_date = pd.Timestamp(entry_str)

        # Skip if before GLNG has trading data
        if entry_date < glng_start:
            continue

        # Find first trading day on or after signal date
        after = trading_dates[trading_dates >= entry_date]
        if len(after) == 0:
            continue
        entry_date_actual = after[0]
        entry_idx = trading_dates.get_loc(entry_date_actual)

        if entry_idx <= open_until_idx:
            continue

        entry_price = glng_prices.get(entry_date_actual, np.nan)
        if np.isnan(entry_price) or entry_price <= 0:
            continue

        # Walk forward checking stop-loss
        actual_end_idx = entry_idx
        exit_reason = "hold_30d"
        for j in range(entry_idx, min(entry_idx + hold_days, len(trading_dates))):
            cur_date = trading_dates[j]
            cur_price = glng_prices.get(cur_date, np.nan)
            if np.isnan(cur_price):
                actual_end_idx = j
                continue
            glng_chg = (cur_price - entry_price) / entry_price
            if glng_chg <= stop_loss:
                exit_reason = "glng_drop_8pct_sl"
                actual_end_idx = j
                break
            actual_end_idx = j

        positions.iloc[entry_idx:actual_end_idx + 1] = 1.0
        open_until_idx = actual_end_idx

        exit_price = glng_prices.get(trading_dates[actual_end_idx], np.nan)
        event_ret = (exit_price - entry_price) / entry_price if not np.isnan(exit_price) else np.nan
        events.append({
            "signal_date": entry_str,
            "entry_date": str(entry_date_actual.date()),
            "exit_date": str(trading_dates[actual_end_idx].date()),
            "exit_reason": exit_reason,
            "storm": storm,
            "terminal": terminal,
            "peak_cat": cat,
            "entry_glng_price": round(float(entry_price), 2),
            "event_return": round(float(event_ret), 4) if not np.isnan(event_ret) else None,
        })

    # PnL: long GLNG, shift positions for no look-ahead
    pos_shifted = positions.shift(1).fillna(0)
    glng_r_aligned = glng_r.reindex(trading_dates).fillna(0)
    pnl_raw = pos_shifted * glng_r_aligned

    pnl = pnl_raw.reindex(spy_r.index).dropna()

    if (pnl != 0).sum() < 20:
        return mark_failed(
            sid,
            f"insufficient in-position days: {(pnl != 0).sum()} (n_events={len(events)}). "
            f"Events: {[e['entry_date'] for e in events]}",
        )

    m = compute_metrics(
        pnl,
        benchmark=spy_r,
        name="NHC Gulf LNG Hurricane Long GLNG",
        positions=pos_shifted.reindex(pnl.index).fillna(0),
        cost_bps=15,  # higher cost for less liquid
    )

    ev_returns = [e["event_return"] for e in events if e["event_return"] is not None]
    n_events = len(events)
    win_rate = float(np.mean([r > 0 for r in ev_returns])) if ev_returns else None
    avg_event = float(np.mean(ev_returns)) if ev_returns else None

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "Long GLNG when NHC 5-day cone intersects Gulf Coast LNG terminals "
                "(Sabine Pass LA, Corpus Christi TX, Freeport TX) during hurricane season. "
                "Storm must be >= Tropical Storm strength. Hold ~22 trading days (~30 calendar). "
                "Stop-loss: GLNG drops 8% from entry."
            ),
            "mechanism": (
                "Pre-storm LNG terminal shutdowns (Sabine Pass, Corpus Christi, Freeport) "
                "reduce US LNG export capacity. International buyers bid up JKM (Asia spot) "
                "and TTF (Europe spot) to source alternative LNG supply, raising LNG tanker "
                "spot freight rates. GLNG (Golar LNG) operates spot-rate-linked FSRU/LNG "
                "vessels; higher freight rates directly benefit GLNG revenue."
            ),
            "source": (
                "NHC historical TCR reports (nhc.noaa.gov/data/tcr/) for signal dates; "
                "yfinance GLNG/UNG/SPY prices"
            ),
            "tickers": ["GLNG", "UNG", "SPY"],
            "n_events": n_events,
            "event_win_rate": round(win_rate, 4) if win_rate is not None else None,
            "avg_event_return": round(avg_event, 4) if avg_event is not None else None,
            "events": events,
            "caveats": (
                "Only 3-4 clear events in GLNG data window (2011-present); "
                "insufficient for statistically robust conclusions. "
                "NHC cone dates hard-coded from public records. "
                "GLNG stock is highly volatile (beta ~1.5-2x); freight rate mechanism "
                "is real but GLNG equity response is noisy. "
                "Storm paths near but not directly hitting terminals may not trigger "
                "actual export curtailment. Henry Hub (UNG) may simultaneously spike "
                "from reduced domestic gas demand, partially offsetting LNG arbitrage."
            ),
        },
        pnl=pnl,
    )

    print(f"Done: {sid}")
    print(f"  n_events: {n_events}, win_rate: {win_rate}, avg_event_return: {avg_event}")
    print(
        f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
        f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
    )
    if "oos_sharpe" in m:
        print(f"  OOS Sharpe: {m['oos_sharpe']:.2f}  IS Sharpe: {m['is_sharpe']:.2f}")


if __name__ == "__main__":
    main()
