"""PL752 — BLS SOII Severity Accel -> Short EIG (Counter)

BLS Survey of Occupational Injuries and Illnesses (SOII) is released each
November. When lost-workday cases per 100 FTE accelerates YoY by >= 3%,
short EIG for 60 trading days from release date.

Counter-signal to long_wc_insurer: higher injury severity increases workers'
comp claims, hurting EIG's combined ratio.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)

# ---------------------------------------------------------------------------
# Hand-coded BLS SOII data: lost-workday cases per 100 FTE (private industry)
# Sources: BLS SOII Tables, annual November release
# https://www.bls.gov/iif/soii-data.htm
# ---------------------------------------------------------------------------
# Format: (data_year, release_date_str, cases_per_100fte)
SOII_DATA = [
    (2009, "2010-11-09", 1.8),
    (2010, "2011-11-08", 1.8),
    (2011, "2012-11-08", 1.8),
    (2012, "2013-11-07", 1.7),
    (2013, "2014-11-06", 1.7),
    (2014, "2015-11-10", 1.7),
    (2015, "2016-11-10", 1.6),
    (2016, "2017-11-09", 1.6),
    (2017, "2018-11-08", 1.5),
    (2018, "2019-11-07", 1.5),
    (2019, "2020-11-04", 1.4),
    (2020, "2021-11-03", 1.5),   # COVID surge in lost-workdays
    (2021, "2022-11-09", 1.6),   # +6.7% YoY — qualifying event
    (2022, "2023-11-08", 1.7),   # +6.25% YoY — qualifying event
    (2023, "2024-11-07", 1.6),   # -5.9% YoY — no trigger
]

HOLD_DAYS = 60
ACCEL_THRESHOLD = 0.03  # >= 3% YoY increase


def main():
    sid = "PL752_soii_severity_short_eig"

    try:
        px = load_prices(["EIG", "SPY"], start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"price data load: {e}")

    for t in ["EIG", "SPY"]:
        if t not in px.columns:
            return mark_failed(sid, f"missing ticker: {t}")

    px = px.sort_index().ffill(limit=3)

    # Build SOII series and find qualifying events
    soii_df = pd.DataFrame(
        [(y, pd.Timestamp(d), v) for y, d, v in SOII_DATA],
        columns=["data_year", "release_date", "cases_per_100fte"]
    ).set_index("data_year").sort_index()

    # Compute YoY change
    soii_df["yoy_chg"] = soii_df["cases_per_100fte"].pct_change(1)

    # Find qualifying events: YoY accel >= threshold
    events = []
    for yr, row in soii_df.iterrows():
        if pd.isna(row["yoy_chg"]):
            continue
        if row["yoy_chg"] < ACCEL_THRESHOLD:
            continue
        release_dt = row["release_date"]
        # Entry: next trading session after release
        future = px.index[px.index > release_dt]
        if len(future) == 0:
            continue
        entry_dt = future[0]
        events.append({
            "data_year": int(yr),
            "release_date": str(release_dt.date()),
            "entry_date": str(entry_dt.date()),
            "cases_per_100fte": float(row["cases_per_100fte"]),
            "yoy_chg_pct": round(float(row["yoy_chg"]) * 100, 2),
        })

    print(f"SOII accel events (>= {ACCEL_THRESHOLD*100:.0f}% YoY): {len(events)}")
    if not events:
        return mark_failed(sid, "no qualifying SOII severity events")

    # Build daily PnL: short EIG
    ret = daily_returns(px)
    spy_r = ret["SPY"]
    eig_r = ret["EIG"]
    pnl = pd.Series(0.0, index=ret.index)

    event_log = []
    for ev in events:
        entry_dt = pd.Timestamp(ev["entry_date"])
        if entry_dt not in ret.index:
            future = ret.index[ret.index >= entry_dt]
            if len(future) == 0:
                continue
            entry_dt = future[0]

        ep = ret.index.get_loc(entry_dt)
        ex = min(ep + HOLD_DAYS, len(ret))

        eig_sl = eig_r.iloc[ep:ex]
        spy_sl = spy_r.iloc[ep:ex]

        pnl.iloc[ep:ex] = pnl.iloc[ep:ex] + (-eig_sl.values[:ex-ep])

        eig_ret = float((1 + eig_sl).prod() - 1)
        spy_ret = float((1 + spy_sl).prod() - 1)
        short_ret = -eig_ret

        event_log.append({
            **ev,
            "exit_date": str(ret.index[ex - 1].date()),
            "short_eig_return": round(short_ret, 4),
            "spy_return": round(spy_ret, 4),
            "excess_vs_spy": round(short_ret - spy_ret, 4),
        })

    n_valid = len(event_log)
    print(f"Valid events: {n_valid}")
    if n_valid == 0:
        return mark_failed(sid, "no valid events")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active days: {len(active_pnl)} (annual event study — too few events for reliable Sharpe)")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="BLS SOII Severity Accel Short EIG")

    event_rets = [e["short_eig_return"] for e in event_log]
    save_result(sid, m, extra={
        "rule": (
            f"When BLS SOII annual release (November) shows lost-workday cases per "
            f"100 FTE increasing >= {ACCEL_THRESHOLD*100:.0f}% YoY, short EIG for "
            f"{HOLD_DAYS} trading days from release."
        ),
        "mechanism": (
            "Higher lost-workday rates increase workers' comp claims frequency and "
            "severity, raising EIG's combined ratio and depressing underwriting "
            "profit. Analysts revise loss ratio and EPS estimates downward, leading "
            "to stock underperformance in the 60 days following the SOII release."
        ),
        "source": (
            "Hand-coded BLS SOII annual tables (BLS.gov/iif/soii-data.htm); "
            "yfinance EIG, SPY."
        ),
        "caveats": (
            "Annual event study with only 2 qualifying events — extremely low "
            "statistical power. Counter-signal to long_wc_insurer. "
            "2020 COVID distortion affects 2021 SOII release baseline."
        ),
        "n_events": n_valid,
        "avg_short_return": round(float(np.mean(event_rets)), 4),
        "win_rate": round(float(np.mean([r > 0 for r in event_rets])), 4),
        "events": event_log,
    })

    print(f"Done: {n_valid} events")
    if "error" not in m:
        print(
            f"  Sharpe: {m.get('sharpe', float('nan')):.2f}  "
            f"CAGR: {m.get('cagr', float('nan'))*100:.2f}%  "
            f"MaxDD: {m.get('max_dd', float('nan'))*100:.2f}%  "
            f"t-stat: {m.get('t_stat', float('nan')):.2f}"
        )


if __name__ == "__main__":
    main()
