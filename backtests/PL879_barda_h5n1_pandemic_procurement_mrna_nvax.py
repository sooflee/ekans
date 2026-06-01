"""PL879 — BARDA H5N1 Pre-Pandemic Stockpile Award -> Long MRNA/NVAX"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics, save_result, mark_failed, daily_returns)


def main():
    sid = "PL879_barda_h5n1_pandemic_procurement_mrna_nvax"

    # Known pandemic procurement trigger events (from strategy spec):
    # 1. 2009-04-26: USDA confirmed H1N1 mammalian spread — NVAX-only (MRNA not listed)
    # 2. 2020-05-15: OWS announcement — MRNA+NVAX basket (MRNA listed Dec 2018)
    # 3. 2024-06-01: BARDA H5N1 contract modification (partial trigger — no H2H)
    #
    # Use NVAX proxy for 2009; full basket for 2020+
    HOLD = 20  # trading days

    # Load prices with start before all events
    try:
        px_full = load_prices(["MRNA", "NVAX", "BNTX", "SPY"], start="2019-01-01")
        spy_r = daily_returns(px_full[["SPY"]]).iloc[:, 0]
        mrna_r = daily_returns(px_full[["MRNA"]]).iloc[:, 0]
        nvax_r = daily_returns(px_full[["NVAX"]]).iloc[:, 0]
    except Exception as e:
        try:
            px_full = load_prices(["MRNA", "NVAX", "BNTX", "SPY"], start="2019-01-01", cache=False)
            spy_r = daily_returns(px_full[["SPY"]]).iloc[:, 0]
            mrna_r = daily_returns(px_full[["MRNA"]]).iloc[:, 0]
            nvax_r = daily_returns(px_full[["NVAX"]]).iloc[:, 0]
        except Exception as e2:
            return mark_failed(sid, f"price load: {e2}")

    # For 2009 event, use NVAX standalone (MRNA not available)
    try:
        px_nvax = load_prices(["NVAX", "SPY"], start="2009-01-01")
        nvax_r_old = daily_returns(px_nvax[["NVAX"]]).iloc[:, 0]
        spy_r_old = daily_returns(px_nvax[["SPY"]]).iloc[:, 0]
    except Exception as e:
        try:
            px_nvax = load_prices(["NVAX", "SPY"], start="2009-01-01", cache=False)
            nvax_r_old = daily_returns(px_nvax[["NVAX"]]).iloc[:, 0]
            spy_r_old = daily_returns(px_nvax[["SPY"]]).iloc[:, 0]
        except Exception as e2:
            return mark_failed(sid, f"NVAX price load: {e2}")

    # Basket return: 60% MRNA + 40% NVAX
    basket_r = 0.6 * mrna_r + 0.4 * nvax_r

    events = []
    # Build combined PnL across all events
    # Event 1: 2009-04-26 (NVAX only)
    for sig_date_str, ret_series, spy_ser, label in [
        ("2009-04-26", nvax_r_old, spy_r_old, "NVAX-only (H1N1 2009)"),
        ("2020-05-15", basket_r, spy_r, "MRNA/NVAX basket (OWS 2020)"),
        ("2024-06-01", basket_r, spy_r, "MRNA/NVAX basket (H5N1 2024)"),
    ]:
        sig_date = pd.Timestamp(sig_date_str)
        future_idx = ret_series.index[ret_series.index >= sig_date]
        if len(future_idx) < HOLD + 1:
            print(f"  Skipping {sig_date_str}: insufficient data ahead")
            continue
        entry_idx = future_idx[0]
        pos = ret_series.index.get_loc(entry_idx)
        end_pos = min(pos + HOLD, len(ret_series))

        basket_window = ret_series.iloc[pos:end_pos]
        spy_window = spy_ser.reindex(basket_window.index).fillna(0)

        # Long basket vs SPY (market-neutral)
        pair_r = basket_window - spy_window

        basket_cum = float((1 + basket_window).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)
        pair_cum = float((1 + pair_r).prod() - 1)

        events.append({
            "signal_date": sig_date_str,
            "entry_date": str(entry_idx.date()),
            "label": label,
            "hold_days": end_pos - pos,
            "basket_return_20d": round(basket_cum, 4),
            "spy_return_20d": round(spy_cum, 4),
            "pair_return_20d": round(pair_cum, 4),
            "daily_returns": pair_r.values.tolist(),
        })
        print(f"  {sig_date_str} ({label}): basket={basket_cum:.1%}, spy={spy_cum:.1%}, pair={pair_cum:.1%}")

    if not events:
        return mark_failed(sid, "no valid events after price join")

    # Build combined PnL series with a proper DatetimeIndex
    # Collect all (date, pair_return) pairs to create a real time series
    all_dates = []
    all_rets = []
    for e in events:
        sig_date = pd.Timestamp(e["signal_date"])
        ret_series_map = {
            "2009-04-26": nvax_r_old,
            "2020-05-15": basket_r,
            "2024-06-01": basket_r,
        }
        ret_ser = ret_series_map[e["signal_date"]]
        spy_ser_map = {
            "2009-04-26": spy_r_old,
            "2020-05-15": spy_r,
            "2024-06-01": spy_r,
        }
        spy_ser = spy_ser_map[e["signal_date"]]
        future_idx = ret_ser.index[ret_ser.index >= sig_date]
        entry_idx = future_idx[0]
        pos = ret_ser.index.get_loc(entry_idx)
        end_pos = min(pos + HOLD, len(ret_ser))
        basket_window = ret_ser.iloc[pos:end_pos]
        spy_window = spy_ser.reindex(basket_window.index).fillna(0)
        pair_r = basket_window - spy_window
        all_dates.extend(pair_r.index.tolist())
        all_rets.extend(pair_r.values.tolist())

    combined_pnl = pd.Series(all_rets, index=pd.DatetimeIndex(all_dates)).sort_index()

    # Clean up event data (remove raw daily returns for storage)
    for e in events:
        del e["daily_returns"]

    if len(combined_pnl) < 20:
        return mark_failed(sid, f"insufficient combined days: {len(combined_pnl)}")

    m = compute_metrics(combined_pnl, name="Pandemic Procurement → Long MRNA/NVAX vs SPY")
    win_rates = [1 if e["pair_return_20d"] > 0 else 0 for e in events]

    save_result(sid, m, extra={
        "rule": "Long MRNA (60%) + NVAX (40%) vs SPY for 20 trading days when APHIS confirms mammalian H5N1 transmission AND BARDA posts emergency RFP on SAM.gov",
        "mechanism": "BARDA procurement signal front-runs vaccine contract award; MRNA/NVAX bid up on expected emergency contract revenue",
        "source": "Known BARDA/OWS events 2009/2020/2024; yfinance MRNA/NVAX/SPY",
        "n_events": len(events),
        "event_win_rate": round(float(np.mean(win_rates)), 4),
        "events": events,
        "caveats": "Only 3 analog events (2009 H1N1, 2020 OWS, 2024 H5N1); 2024 did NOT trigger (no H2H confirmed); combined PnL is concatenated not time-series; MRNA only listed 2018",
    })
    print(f"Done: {len(events)} events, Sharpe={m.get('sharpe', 'N/A'):.3f}, CAGR={m.get('cagr', 0)*100:.1f}%")


if __name__ == "__main__":
    main()
