"""PL739_bnef_asp_poly_short_tan_run
BNEF Module ASP Floor + Poly Collapse -> Short TAN/RUN (Counter)

When BNEF monthly module ASP falls to within 5% of the marginal-cost floor AND
polysilicon spot drops >=20% YoY in the same month, short equal-weight TAN + RUN
for 45 trading days. Counter-signal to long_solar.

Hand-coded monthly BNEF module ASP + PV-InfoLink polysilicon spot 2018-2025.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# Hand-coded monthly BNEF utility-scale module ASP (USD/W, approx)
# Source: BNEF Solar Module Price Index / PV Magazine spot reports
BNEF_ASP = {
    "2018-01": 0.35, "2018-02": 0.34, "2018-03": 0.34, "2018-04": 0.33,
    "2018-05": 0.32, "2018-06": 0.32, "2018-07": 0.31, "2018-08": 0.31,
    "2018-09": 0.30, "2018-10": 0.28, "2018-11": 0.26, "2018-12": 0.25,
    "2019-01": 0.24, "2019-02": 0.24, "2019-03": 0.23, "2019-04": 0.23,
    "2019-05": 0.22, "2019-06": 0.22, "2019-07": 0.22, "2019-08": 0.21,
    "2019-09": 0.21, "2019-10": 0.21, "2019-11": 0.21, "2019-12": 0.21,
    "2020-01": 0.21, "2020-02": 0.21, "2020-03": 0.20, "2020-04": 0.19,
    "2020-05": 0.19, "2020-06": 0.19, "2020-07": 0.19, "2020-08": 0.18,
    "2020-09": 0.19, "2020-10": 0.20, "2020-11": 0.21, "2020-12": 0.22,
    "2021-01": 0.22, "2021-02": 0.22, "2021-03": 0.23, "2021-04": 0.23,
    "2021-05": 0.24, "2021-06": 0.25, "2021-07": 0.25, "2021-08": 0.25,
    "2021-09": 0.25, "2021-10": 0.26, "2021-11": 0.26, "2021-12": 0.27,
    "2022-01": 0.27, "2022-02": 0.27, "2022-03": 0.27, "2022-04": 0.27,
    "2022-05": 0.26, "2022-06": 0.26, "2022-07": 0.25, "2022-08": 0.24,
    "2022-09": 0.24, "2022-10": 0.24, "2022-11": 0.22, "2022-12": 0.21,
    "2023-01": 0.20, "2023-02": 0.19, "2023-03": 0.18, "2023-04": 0.17,
    "2023-05": 0.16, "2023-06": 0.15, "2023-07": 0.14, "2023-08": 0.13,
    "2023-09": 0.13, "2023-10": 0.13, "2023-11": 0.12, "2023-12": 0.12,
    "2024-01": 0.12, "2024-02": 0.12, "2024-03": 0.11, "2024-04": 0.11,
    "2024-05": 0.11, "2024-06": 0.10, "2024-07": 0.10, "2024-08": 0.10,
    "2024-09": 0.10, "2024-10": 0.10, "2024-11": 0.10, "2024-12": 0.10,
    "2025-01": 0.10, "2025-02": 0.10, "2025-03": 0.10, "2025-04": 0.10,
    "2025-05": 0.10,
}

# Hand-coded monthly PV-InfoLink polysilicon spot price (USD/kg, approx)
# Source: PV-InfoLink weekly spot reports (monthly average)
POLY_SPOT = {
    "2018-01": 17.0, "2018-02": 17.2, "2018-03": 17.1, "2018-04": 15.5,
    "2018-05": 14.0, "2018-06": 12.5, "2018-07": 11.5, "2018-08": 11.0,
    "2018-09": 10.5, "2018-10": 10.0, "2018-11": 9.5,  "2018-12": 9.0,
    "2019-01": 8.5,  "2019-02": 8.2,  "2019-03": 8.0,  "2019-04": 7.8,
    "2019-05": 7.5,  "2019-06": 7.2,  "2019-07": 7.0,  "2019-08": 7.0,
    "2019-09": 7.2,  "2019-10": 7.5,  "2019-11": 7.8,  "2019-12": 8.0,
    "2020-01": 8.0,  "2020-02": 7.5,  "2020-03": 7.0,  "2020-04": 6.8,
    "2020-05": 7.0,  "2020-06": 7.5,  "2020-07": 8.0,  "2020-08": 9.0,
    "2020-09": 9.5,  "2020-10": 10.0, "2020-11": 10.5, "2020-12": 11.0,
    "2021-01": 12.0, "2021-02": 12.5, "2021-03": 13.0, "2021-04": 14.0,
    "2021-05": 16.0, "2021-06": 18.0, "2021-07": 20.0, "2021-08": 22.0,
    "2021-09": 24.0, "2021-10": 26.0, "2021-11": 28.0, "2021-12": 30.0,
    "2022-01": 32.0, "2022-02": 30.0, "2022-03": 28.0, "2022-04": 26.0,
    "2022-05": 24.0, "2022-06": 23.0, "2022-07": 22.0, "2022-08": 24.0,
    "2022-09": 28.0, "2022-10": 30.0, "2022-11": 28.0, "2022-12": 25.0,
    "2023-01": 22.0, "2023-02": 18.0, "2023-03": 15.0, "2023-04": 14.0,
    "2023-05": 12.0, "2023-06": 9.5,  "2023-07": 8.0,  "2023-08": 7.5,
    "2023-09": 7.0,  "2023-10": 6.5,  "2023-11": 6.0,  "2023-12": 6.0,
    "2024-01": 6.0,  "2024-02": 5.8,  "2024-03": 5.5,  "2024-04": 5.2,
    "2024-05": 5.0,  "2024-06": 5.0,  "2024-07": 5.2,  "2024-08": 5.5,
    "2024-09": 5.5,  "2024-10": 5.5,  "2024-11": 5.8,  "2024-12": 6.0,
    "2025-01": 6.0,  "2025-02": 6.0,  "2025-03": 5.8,  "2025-04": 5.5,
    "2025-05": 5.5,
}

# Marginal cost floor for crystalline silicon modules (USD/W)
# ~$0.10/W is widely cited as the Chinese cost floor for utility-scale c-Si
# (post-2023 market realities); prior years had higher cost floors.
# We use a time-varying floor: ~$0.20/W (2018-2020), ~$0.17/W (2021-2022), $0.10/W (2023+)
COST_FLOOR = {
    "2018": 0.20, "2019": 0.18, "2020": 0.17, "2021": 0.17,
    "2022": 0.16, "2023": 0.12, "2024": 0.10, "2025": 0.10,
}


def main():
    sid = "PL739_bnef_asp_poly_short_tan_run"
    tickers = ["TAN", "RUN", "SPY"]

    try:
        px = load_prices(tickers, start="2018-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)

    # Check TAN exists (RUN IPO'd May 2020; handle missing)
    if "TAN" not in px.columns:
        return mark_failed(sid, "TAN not available in yfinance")
    if "SPY" not in px.columns:
        return mark_failed(sid, "SPY not available in yfinance")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    # Build monthly Series from hand-coded dicts
    asp_series = pd.Series({
        pd.Timestamp(k + "-01"): v for k, v in BNEF_ASP.items()
    })
    poly_series = pd.Series({
        pd.Timestamp(k + "-01"): v for k, v in POLY_SPOT.items()
    })

    # Compute YoY change for polysilicon
    poly_yoy = poly_series.pct_change(12)  # rolling 12-month YoY

    # Determine signal months: ASP within 5% of cost floor AND poly YoY <= -20%
    signal_months = []
    for ts in asp_series.index:
        year_str = str(ts.year)
        floor = COST_FLOOR.get(year_str, 0.10)
        asp = asp_series.loc[ts]
        # within 5% of floor: asp <= floor * 1.05
        asp_near_floor = asp <= floor * 1.05
        # poly YoY drop >= 20%
        if ts in poly_yoy.index:
            pyoy = poly_yoy.loc[ts]
            poly_collapse = (not np.isnan(pyoy)) and (pyoy <= -0.20)
        else:
            poly_collapse = False
        if asp_near_floor and poly_collapse:
            signal_months.append(ts)

    if not signal_months:
        return mark_failed(sid, "No signal months found with hand-coded data")

    # For each signal month, enter SHORT at the first trading day of the
    # following month, hold 45 trading days.
    hold_days = 45

    # Build equal-weight short basket: TAN + RUN (if available)
    # Apply position to NEXT day's return (no look-ahead)
    idx_list = list(ret.index)
    n = len(idx_list)

    daily_pnl = pd.Series(0.0, index=ret.index)
    positions = pd.Series(0.0, index=ret.index)

    # Compute basket return: mean of available short legs (negate for short)
    # If RUN is missing for a day, use only TAN
    basket_legs = []
    for t in ["TAN", "RUN"]:
        if t in ret.columns:
            basket_legs.append(t)

    events = []
    active_until_idx = -1  # track when current position ends

    for signal_ts in signal_months:
        # Entry: first trading day of the month following the signal month
        entry_month_start = signal_ts + pd.offsets.MonthBegin(1)
        # Find the first trading day >= entry_month_start
        entry_candidates = [d for d in idx_list if d >= entry_month_start]
        if not entry_candidates:
            continue
        entry_date = entry_candidates[0]
        entry_idx = idx_list.index(entry_date)

        # Respect no-overlap: if previous position still running, skip
        if entry_idx <= active_until_idx:
            continue

        end_idx = min(entry_idx + hold_days, n)
        active_until_idx = end_idx - 1

        event_pnl = []
        for j in range(entry_idx, end_idx):
            d = idx_list[j]
            leg_rets = []
            for t in basket_legs:
                r = ret[t].get(d, np.nan)
                if not np.isnan(r):
                    leg_rets.append(r)
            if leg_rets:
                basket_r = np.mean(leg_rets)
                short_r = -basket_r  # short
                daily_pnl.iloc[j] = short_r
                positions.iloc[j] = -1.0
                event_pnl.append(short_r)

        if event_pnl:
            cum_event = float((1 + pd.Series(event_pnl)).prod() - 1)
            events.append({
                "signal_month": str(signal_ts.date()),
                "entry_date": str(entry_date.date()),
                "exit_date": str(idx_list[min(end_idx - 1, n - 1)].date()),
                "event_return": round(cum_event, 4),
            })

    pnl = daily_pnl.dropna()

    n_events = len(events)
    if n_events < 2:
        return mark_failed(sid, f"too few events: {n_events}")

    m = compute_metrics(
        pnl,
        benchmark=spy_r,
        name="BNEF ASP Floor + Poly Collapse Short TAN/RUN",
        positions=positions.reindex(pnl.index).fillna(0),
        cost_bps=10,
    )
    m["n_events"] = n_events

    ev_returns = [e["event_return"] for e in events]
    win_rate = float(np.mean([r > 0 for r in ev_returns])) if ev_returns else None
    avg_event = float(np.mean(ev_returns)) if ev_returns else None

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "When BNEF monthly module ASP falls to within 5% of marginal-cost floor "
                "AND polysilicon spot drops >=20% YoY in same month, short equal-weight "
                "TAN + RUN for 45 trading days."
            ),
            "mechanism": (
                "When module ASPs compress to marginal cost while polysilicon simultaneously "
                "crashes, the solar supply chain faces margin destruction. Pure-play solar "
                "downstream installers and panel ETFs reprice to reflect reduced revenue "
                "visibility and stranded capacity. The counter-directional trade to long_solar "
                "captures the mean-reversion from equity premiums awarded during the "
                "cost-decline euphoria phase."
            ),
            "source": "BNEF Solar Module Price Index; PV-InfoLink polysilicon spot (hand-coded 2018-2025)",
            "tickers": basket_legs,
            "n_events": n_events,
            "event_win_rate": round(win_rate, 4) if win_rate is not None else None,
            "avg_event_return": round(avg_event, 4) if avg_event is not None else None,
            "events": events,
            "caveats": (
                "Hand-coded BNEF/PV-InfoLink data introduces approximation error. "
                "RUN IPO was May 2020, reducing early-period basket to TAN only. "
                "Solar ETFs amplify macro beta; short losses can be large in "
                "policy-driven rallies (IRA 2022). Limited to ~7 years of data."
            ),
        },
        pnl=pnl,
    )

    print(f"Done: {sid}")
    print(f"  n_events={n_events}, win_rate={win_rate}, avg_event_return={avg_event}")
    print(
        f"  Sharpe={m.get('sharpe', 0):.2f}  CAGR={m.get('cagr', 0)*100:.2f}%  "
        f"MaxDD={m.get('max_dd', 0)*100:.2f}%  t-stat={m.get('t_stat', 0):.2f}"
    )
    if "oos_sharpe" in m:
        print(f"  OOS Sharpe={m['oos_sharpe']:.2f}")


if __name__ == "__main__":
    main()
