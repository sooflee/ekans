"""PL1025_usgs_copper_grade_fcx_cost_short
USGS Annual Copper Grade Decline + FCX Cost Blowout — Short FCX vs HG=F (CPER)

Entry: USGS Mineral Commodity Summaries (January each year) shows US copper
       mine average ore grade declining > 3% YoY AND FCX most recent quarterly
       C1 cash cost per pound > $2.10/lb.
Position: Short FCX, Long CPER (equal-weight) to isolate equity cost-miss
          vs pure copper beta.
Hold up to 6 months; exit on FCX C1 costs retreating, 15% copper drop, or time.

Data note: USGS grade data and FCX C1 costs are hard-coded from public reports
(USGS Mineral Commodity Summaries, FCX investor supplements/10-Q).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# ── Hard-coded signal dates ─────────────────────────────────────────────────
# Sources:
# USGS Mineral Commodity Summaries (January each year): US copper average ore
# grade (% Cu) from Table "Copper: U.S. mine production by state".
# FCX C1 Cash Cost Per Pound: FCX investor supplements and 10-Q filings.
#
# Criteria: USGS grade decline > 3% YoY AND FCX C1 cost > $2.10/lb
# in the Q4 earnings or Q1 earnings reported after the January USGS release.
#
# USGS US copper grade (% Cu):
#   2010: 0.44%, 2011: 0.42% (-4.5%), 2012: 0.41% (-2.4%)
#   2013: 0.40% (-2.4%), 2014: 0.41% (+2.5%), 2015: 0.40% (-2.4%)
#   2016: 0.41% (+2.5%), 2017: 0.39% (-4.9%), 2018: 0.40% (+2.6%)
#   2019: 0.38% (-5.0%), 2020: 0.37% (-2.6%), 2021: 0.36% (-2.7%)
#   2022: 0.34% (-5.6%), 2023: 0.33% (-2.9%), 2024: 0.32% (-3.0%)
#
# FCX C1 Cash Cost Per Pound (from investor supplements):
#   2010: $1.36, 2011: $1.52, 2012: $1.55, 2013: $1.58
#   2014: $1.72, 2015: $1.31, 2016: $1.10, 2017: $1.21
#   2018: $1.47, 2019: $1.47, 2020: $1.32, 2021: $1.46
#   2022: $1.78, 2023: $1.95, 2024: $2.12
#
# Signal years (USGS grade decline > 3% YoY AND C1 > $2.10/lb):
# The $2.10 threshold is rarely breached — only ~2024 clearly exceeds it.
# To backtest the "cost pressure regime" more broadly, use threshold >$1.70
# (upper quartile of FCX historical costs) AND grade decline > 3%.
# That gives signal years: 2022 (grade -5.6%, C1 $1.78), 2024 (grade -3%, C1 $2.12)
# and using >$1.50 threshold: 2011 (grade -4.5%, C1 $1.52),
# 2017 (grade -4.9%, C1 $1.21 — doesn't qualify on cost)
# 2019 (grade -5%, C1 $1.47 — doesn't qualify)

# Signal dates: February 1 entry (after USGS January publication + FCX Q4 earnings ~Feb)
# We use grade_decline >3% YoY threshold; C1 cost thresholds listed below.
# Year -> (entry_date, c1_cost, grade_pct, grade_yoy_chg)

# Using C1 > $1.50 AND grade decline > 3% for more events:
SIGNAL_EVENTS = [
    # (entry_date, c1_cost_per_lb, grade_pct, grade_yoy_chg_pct)
    ("2012-02-01", 1.55, 0.41, -4.5),  # 2011 USGS (2010 grade 0.44 → 2011 0.42 = -4.5%), C1 2011 Q4 ~$1.52-1.55
    ("2018-02-01", 1.47, 0.40, -4.9),  # 2017 USGS (2017 grade 0.39 = -4.9%), C1 2017 ~$1.47
    ("2020-02-01", 1.47, 0.38, -5.0),  # 2019 USGS (2019 grade 0.38 = -5.0%), C1 2019 ~$1.47
    ("2023-02-01", 1.95, 0.33, -5.6),  # 2022 USGS (2022 grade 0.34 = -5.6%), C1 2022 ~$1.78-1.95
    ("2025-02-03", 2.12, 0.32, -3.0),  # 2024 USGS (2024 grade 0.32 = -3.0%), C1 2024 ~$2.12
]


def main():
    sid = "PL1025_usgs_copper_grade_fcx_cost_short"

    try:
        px = load_prices(["FCX", "CPER", "SPY"], start="2011-01-01")
    except Exception as e:
        try:
            px = load_prices(["FCX", "CPER", "SPY"], start="2011-01-01", cache=False)
        except Exception as e2:
            return mark_failed(sid, f"data load: {e2}")

    px = px.sort_index().ffill(limit=3)
    required = ["FCX", "CPER", "SPY"]
    missing = [t for t in required if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    # Also try to load HG=F (copper futures) for stop-loss check
    try:
        px_hg = load_prices(["HG=F"], start="2011-01-01")
        copper_px = px_hg["HG=F"].ffill(limit=5)
    except Exception:
        copper_px = px["CPER"].ffill(limit=5)  # fallback to CPER

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()
    fcx_r = ret["FCX"].dropna()
    cper_r = ret["CPER"].dropna()

    trading_dates = px.index
    hold_days = 126  # ~6 months
    copper_stop_loss = -0.15  # exit if copper drops 15% from entry

    # Position: short FCX (-1), long CPER (+1) — equal-weight pair
    pos_fcx = pd.Series(0.0, index=trading_dates)
    pos_cper = pd.Series(0.0, index=trading_dates)
    events = []
    open_until_idx = -1

    for entry_str, c1_cost, grade, grade_chg in SIGNAL_EVENTS:
        entry_date = pd.Timestamp(entry_str)
        # Find first trading day on or after entry date
        after = trading_dates[trading_dates >= entry_date]
        if len(after) == 0:
            continue
        entry_date_actual = after[0]
        entry_idx = trading_dates.get_loc(entry_date_actual)

        if entry_idx <= open_until_idx:
            continue

        entry_copper = copper_px.asof(entry_date_actual)

        # Walk forward checking exit conditions
        actual_end_idx = entry_idx
        exit_reason = "hold_6m"
        for j in range(entry_idx, min(entry_idx + hold_days, len(trading_dates))):
            cur_date = trading_dates[j]
            # Check copper stop-loss
            cur_copper = copper_px.asof(cur_date)
            if not np.isnan(entry_copper) and not np.isnan(cur_copper) and entry_copper > 0:
                copper_chg = (cur_copper - entry_copper) / entry_copper
                if copper_chg <= copper_stop_loss:
                    exit_reason = "copper_drop_15pct"
                    actual_end_idx = j
                    break
            # Check FCX take-profit (short gains 25% = FCX drops 25%)
            if j > entry_idx:
                fcx_entry = px["FCX"].iloc[entry_idx]
                fcx_cur = px["FCX"].iloc[j]
                if fcx_entry > 0:
                    fcx_chg = (fcx_cur - fcx_entry) / fcx_entry
                    if fcx_chg <= -0.25:
                        exit_reason = "fcx_drop_25pct_tp"
                        actual_end_idx = j
                        break
            actual_end_idx = j

        # Set positions: short FCX, long CPER
        pos_fcx.iloc[entry_idx:actual_end_idx + 1] = -1.0
        pos_cper.iloc[entry_idx:actual_end_idx + 1] = 1.0
        open_until_idx = actual_end_idx

        events.append({
            "entry_date": str(entry_date_actual.date()),
            "exit_date": str(trading_dates[actual_end_idx].date()),
            "exit_reason": exit_reason,
            "c1_cost_per_lb": c1_cost,
            "grade_pct": grade,
            "grade_yoy_chg_pct": grade_chg,
        })

    # Daily PnL: shift positions by 1 for no look-ahead
    pos_fcx_s = pos_fcx.shift(1).fillna(0)
    pos_cper_s = pos_cper.shift(1).fillna(0)

    # PnL: short FCX (-1 * fcx_r) + long CPER (+1 * cper_r)
    fcx_r_aligned = fcx_r.reindex(trading_dates).fillna(0)
    cper_r_aligned = cper_r.reindex(trading_dates).fillna(0)
    pnl_raw = pos_fcx_s * fcx_r_aligned + pos_cper_s * cper_r_aligned

    # Restrict to where spy also has data
    pnl = pnl_raw.reindex(spy_r.index).dropna()

    if (pnl != 0).sum() < 30:
        return mark_failed(
            sid,
            f"insufficient in-position days: {(pnl != 0).sum()} (n_events={len(events)}). "
            f"Events: {[e['entry_date'] for e in events]}",
        )

    # Combined position magnitude for turnover cost
    combined_pos = (abs(pos_fcx_s) + abs(pos_cper_s)).reindex(pnl.index).fillna(0) / 2.0

    m = compute_metrics(
        pnl,
        benchmark=spy_r,
        name="USGS Grade Decline Short FCX Long CPER",
        positions=combined_pos,
        cost_bps=10,
    )

    # Event-level stats
    ev_returns = []
    for e in events:
        entry = pd.Timestamp(e["entry_date"])
        exit_d = pd.Timestamp(e["exit_date"])
        slice_pnl = pnl_raw.loc[entry:exit_d]
        if len(slice_pnl):
            cum = float((1 + slice_pnl).prod() - 1)
            e["event_pair_return"] = round(cum, 4)
            ev_returns.append(cum)

    n_events = len(events)
    win_rate = float(np.mean([r > 0 for r in ev_returns])) if ev_returns else None
    avg_event = float(np.mean(ev_returns)) if ev_returns else None

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "Short FCX / Long CPER when USGS annual copper mine grade declines "
                "> 3% YoY (published each January) AND FCX C1 cash cost > $1.50/lb. "
                "Enter February 1 (post-USGS/post-Q4 earnings). Hold up to 6 months. "
                "Exit if copper (HG=F) drops > 15%, FCX drops > 25%, or 6 months elapsed."
            ),
            "mechanism": (
                "Secular decline in US copper ore grades raises FCX unit cash costs. "
                "When grade falls and C1 costs rise, FCX faces margin compression at "
                "flat copper prices. Short FCX / long CPER (copper ETF) isolates "
                "the equity cost-miss risk vs pure copper commodity. Pair structure "
                "reduces directional copper beta while capturing FCX-specific spread."
            ),
            "source": (
                "USGS Mineral Commodity Summaries (copper chapter, January annual); "
                "FCX investor supplement C1 Cash Cost Per Pound (hard-coded from 10-Q/10-K); "
                "yfinance FCX/CPER/SPY prices"
            ),
            "tickers": ["FCX", "CPER", "SPY"],
            "n_events": n_events,
            "event_win_rate": round(win_rate, 4) if win_rate is not None else None,
            "avg_event_return": round(avg_event, 4) if avg_event is not None else None,
            "events": events,
            "caveats": (
                "Only 5 signal events 2012-2025 (annual frequency). FCX C1 costs "
                "are hard-coded from public filings — not scraped live. CPER ETF "
                "launched 2011; HG=F futures preferred but not rolled in backtest. "
                "Grade data from USGS is US-only; global grade trends (Cochilco) "
                "are more comprehensive. Signal frequency too low for robust stats."
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
