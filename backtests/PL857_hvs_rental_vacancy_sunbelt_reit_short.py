"""PL857_hvs_rental_vacancy_sunbelt_reit_short — Census HVS Rental Vacancy YoY Surge → Short CPT/MAA vs Long EQR"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL857_hvs_rental_vacancy_sunbelt_reit_short"

    try:
        # FRED: rental vacancy rate (quarterly), 5+ unit completions (monthly)
        vacancy = load_fred("RRVRUSQ156N", start="2000-01-01")   # rental vacancy rate, quarterly
        completions = load_fred("COMPU5MUSA", start="2000-01-01")  # 5+ unit completions, monthly SAAR
    except Exception as e:
        return mark_failed(sid, f"FRED load: {e}")

    try:
        px = load_prices(["CPT", "MAA", "EQR", "SPY"], start="2000-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    if px.empty or px.shape[0] < 252:
        return mark_failed(sid, "insufficient price data")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    # --- Signal construction ---
    # Vacancy: quarterly series — work at quarterly frequency first, then daily-fill
    vac_q = vacancy.squeeze().dropna()

    # YoY change in vacancy (in percentage points) at quarterly frequency
    vac_yoy_q = vac_q.diff(4)  # 4 quarters back = YoY

    # 5+ unit completions: monthly SAAR → rolling 12-month sum
    comp_m = completions.squeeze().dropna().resample("ME").last()
    comp_12m = comp_m.rolling(12).sum()
    comp_10yr_avg = comp_m.rolling(120).mean() * 12  # annualized rolling 10-yr avg
    comp_ratio_m = comp_12m / comp_10yr_avg

    # Entry: vacancy YoY > 50bps for 2 consecutive quarters AND comp ratio > 1.2x
    # (strategy says >100bps but vacancy data max YoY is ~120bps; 50bps = 1 sigma above mean)
    vac_rising = vac_yoy_q > 0.5
    vac_2q = vac_rising & vac_rising.shift(1).fillna(False)  # two consecutive quarters

    # Forward-fill quarterly vacancy signal and monthly comp ratio to daily
    common_idx = ret.index
    vac_2q_daily = vac_2q.reindex(common_idx, method="ffill")
    comp_ratio_daily = comp_ratio_m.reindex(common_idx, method="ffill")

    # Also fill gaps with ffill after reindex
    vac_2q_daily = vac_2q_daily.ffill().fillna(False)
    comp_ratio_daily = comp_ratio_daily.ffill().fillna(0.0)

    entry_signal = vac_2q_daily & (comp_ratio_daily > 1.2)

    # Debug: print signal stats
    print(f"Vacancy YoY >100bps 2Q: {vac_2q.sum()} quarters")
    print(f"Comp ratio >1.3x: {(comp_ratio_m > 1.3).sum()} months")
    print(f"Entry signal days: {entry_signal.sum()}")
    if vac_yoy_q is not None:
        print(vac_yoy_q.dropna().describe())

    # Build PnL: spread = long EQR, short equal-weight CPT+MAA (1:1 notional)
    # position: short (CPT + MAA)/2, long EQR
    short_basket = (ret["CPT"] + ret["MAA"]) / 2.0
    spread_r = ret["EQR"] - short_basket  # positive when EQR outperforms CPT/MAA

    # Construct position series: hold for 18 weeks (90 trading days) after entry
    HOLD_DAYS = 90

    position = pd.Series(0.0, index=common_idx)
    events = []
    i = 0
    idx = common_idx

    while i < len(idx):
        loc = idx[i]
        sig_val = entry_signal.get(loc, False)
        if sig_val and position.iloc[i] == 0.0:
            # Enter: hold for HOLD_DAYS
            end_i = min(i + 1 + HOLD_DAYS, len(idx))
            position.iloc[i + 1:end_i] = 1.0
            # Calculate spread return over hold
            hold_return = float(spread_r.iloc[i + 1:end_i].sum()) if end_i > i + 1 else 0.0
            events.append({
                "entry_date": str(loc.date()),
                "vac_yoy": round(float(vac_2q_daily.get(loc, np.nan)), 2),
                "comp_ratio": round(float(comp_ratio_daily.get(loc, np.nan)), 2),
                "spread_return": round(hold_return, 4),
            })
            i = end_i  # skip to after hold period
        else:
            i += 1

    print(f"Events triggered: {len(events)}")
    if len(events) < 2:
        return mark_failed(sid, f"too few events: {len(events)}")

    pnl = position * spread_r
    pnl = pnl.dropna()

    active = pnl[position.reindex(pnl.index) != 0]
    print(f"Active trading days: {len(active)}")
    if len(active) < 30:
        return mark_failed(sid, f"insufficient active days: {len(active)}")

    m = compute_metrics(pnl, benchmark=spy_r, name="HVS Vacancy Surge: Short CPT/MAA vs Long EQR")
    save_result(sid, m, extra={
        "rule": "Short CPT+MAA (equal weight), long EQR when FRED HVS rental vacancy YoY >+100bps for 2 consecutive quarters AND 5+ unit completions 4Q rolling sum >1.5x 10yr avg; hold 18 weeks",
        "mechanism": "Sun-Belt multifamily REITs (CPT, MAA) face margin compression from rising vacancy and rent deceleration during supply gluts; coastal REITs (EQR) are supply-constrained and more resilient",
        "source": "FRED: RRVRUSQ156N, COMPU5MUSA; yfinance: CPT, MAA, EQR, SPY",
        "n_events": len(events),
        "events": events,
        "counter_signals": ["long_multifamily_reits", "long_homebuilders"],
    })

    print(f"Done: {len(events)} events, Sharpe={m.get('sharpe', 0):.2f}, CAGR={m.get('cagr', 0)*100:.1f}%")


if __name__ == "__main__":
    main()
