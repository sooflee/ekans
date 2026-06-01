"""PL791 — Section 232 HRC Tariff Spike + Weak Demand -> Short WHR/AOS/PCAR Basket"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL791_hrc_tariff_spike_no_demand_whr_aos_short"

    # Load HRC futures and equity prices
    try:
        px = load_prices(["WHR", "AOS", "PCAR", "SPY"], start="2017-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    try:
        hrc_px = load_prices(["HRC=F"], start="2017-01-01")
    except Exception as e:
        return mark_failed(sid, f"HRC=F load: {e}")

    # Load FRED AMTMNO (monthly manufacturers new orders)
    try:
        amtmno = load_fred("AMTMNO", start="2017-01-01").squeeze()
    except Exception as e:
        return mark_failed(sid, f"FRED AMTMNO load: {e}")

    if px.empty or hrc_px.empty or amtmno.empty:
        return mark_failed(sid, "empty data")

    # Compute returns
    ret = daily_returns(px)
    spy_r = ret["SPY"]

    # Build short basket: equal-weight WHR + AOS + PCAR
    basket_tickers = [t for t in ["WHR", "AOS", "PCAR"] if t in ret.columns]
    if len(basket_tickers) < 2:
        return mark_failed(sid, f"insufficient basket tickers: {basket_tickers}")
    basket_r = ret[basket_tickers].mean(axis=1)

    # HRC 5-day rolling return
    hrc_close = hrc_px["HRC=F"]
    hrc_roll5 = hrc_close.pct_change(5)

    # AMTMNO MoM change (monthly -> reindex to daily, forward-fill with 1-month lag)
    amtmno_mom = amtmno.pct_change(1)
    # Lag by one month to avoid look-ahead (use prior month reading)
    amtmno_daily = amtmno_mom.reindex(px.index, method="ffill").shift(21)  # ~1 month lag

    # Align HRC signal to daily equity index
    hrc_signal_daily = hrc_roll5.reindex(px.index).fillna(method="ffill")

    # Event-based approach: use known tariff escalation dates
    known_events = [
        pd.Timestamp("2018-03-23"),  # First Section 232 steel tariffs
        pd.Timestamp("2018-06-01"),  # EU/Canada/Mexico tariffs
        pd.Timestamp("2025-03-12"),  # IEEPA 25% steel/aluminum restoration
    ]

    # Also detect signal-driven events: HRC +5% in 5 days AND AMTMNO MoM < 0
    triggered_dates = []
    for i in range(len(hrc_signal_daily)):
        d = hrc_signal_daily.index[i]
        hrc_val = hrc_signal_daily.iloc[i]
        amtmno_val = amtmno_daily.iloc[i] if d in amtmno_daily.index else np.nan
        if (not np.isnan(hrc_val) and hrc_val > 0.05 and
                not np.isnan(amtmno_val) and amtmno_val < 0):
            triggered_dates.append(d)

    # Merge known events + signal-triggered events, deduplicate (90-day exclusion)
    all_triggers = sorted(set(known_events) | set(triggered_dates))
    deduped_triggers = []
    last_trigger = None
    for td in all_triggers:
        if last_trigger is None or (td - last_trigger).days >= 90:
            deduped_triggers.append(td)
            last_trigger = td

    print(f"Triggers (deduped): {len(deduped_triggers)}")

    if not deduped_triggers:
        return mark_failed(sid, "no trigger events found")

    # Build PnL: short basket for up to 8 weeks (40 trading days), exit on stops
    hold_days = 40  # 8 weeks
    pnl = pd.Series(0.0, index=basket_r.index)
    events = []

    for trigger_date in deduped_triggers:
        # Find nearest trading date on or after trigger
        future_mask = basket_r.index >= trigger_date
        if not future_mask.any():
            continue
        entry_date = basket_r.index[future_mask][0]
        entry_idx = basket_r.index.get_loc(entry_date)
        end_idx = min(entry_idx + hold_days, len(basket_r))

        # Short = negative of basket returns
        period_ret = -basket_r.iloc[entry_idx:end_idx]
        period_spy = spy_r.iloc[entry_idx:end_idx]

        # Compute cumulative for stop-loss check
        basket_cum = (1 + basket_r.iloc[entry_idx:end_idx]).cumprod() - 1
        hrc_entry = hrc_close.reindex([entry_date]).iloc[0] if entry_date in hrc_close.index else np.nan

        # Apply stop: if basket up >10% (short squeeze), cover early
        exit_idx = end_idx
        for j in range(len(basket_cum)):
            if basket_cum.iloc[j] > 0.10:  # basket up 10% -> short squeeze
                exit_idx = entry_idx + j + 1
                break
        # HRC stop: HRC falls >8% from entry -> cost pressure easing -> cover
        if not np.isnan(hrc_entry):
            hrc_future = hrc_close.iloc[entry_idx:end_idx]
            for j, (d, v) in enumerate(hrc_future.items()):
                if (hrc_entry - v) / hrc_entry > 0.08:
                    if entry_idx + j + 1 < exit_idx:
                        exit_idx = entry_idx + j + 1
                    break

        actual_ret = period_ret.iloc[:exit_idx - entry_idx]
        cum_ret = float((1 + actual_ret).prod() - 1)
        spy_cum = float((1 + period_spy.iloc[:exit_idx - entry_idx]).prod() - 1)

        pnl.iloc[entry_idx:exit_idx] = period_ret.values[:exit_idx - entry_idx]
        events.append({
            "trigger_date": str(trigger_date.date()),
            "entry_date": str(entry_date.date()),
            "basket_return_short": round(cum_ret, 4),
            "spy_return": round(spy_cum, 4),
            "days_held": exit_idx - entry_idx,
        })

    active = pnl[pnl != 0]
    print(f"Events: {len(events)}, Active PnL days: {len(active)}")

    if len(active) < 30:
        return mark_failed(sid, f"insufficient active days: {len(active)}")

    m = compute_metrics(active, benchmark=spy_r, name="HRC Tariff Spike -> Short WHR/AOS/PCAR")
    save_result(sid, m, extra={
        "rule": "Short equal-weight WHR+AOS+PCAR when HRC=F +5% in 5 days + AMTMNO MoM negative; cover at 8 weeks, HRC -8%, or basket +10%",
        "mechanism": "Section 232 tariff-driven steel cost spike with no demand offset forces margin compression in steel-intensive consumer durables and industrials",
        "source": "yfinance HRC=F, FRED AMTMNO, Section 232 tariff dates",
        "n_events": len(events),
        "events": events,
        "status": "ok",
    })
    print(f"Saved result. Events: {len(events)}")


if __name__ == "__main__":
    main()
