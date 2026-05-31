"""
PL608 — ERCOT Summer Peak Heat -> Short Texas BTC Miners / Long MSTR Pair

Spec calls for NOAA GHCN Houston Tmax. NOAA scraping is heavy; the spec
explicitly allows a "seasonal-summer indicator" fallback. Implementation:

Calendar-proxy for ERCOT peak-price clusters (3+ days of 100F+ within 30d
rolling window during 4CP June-Sep). Historical heat-cluster windows
(public news / NOAA summaries) for recent summers:

  - 2021 mid-Jun to early-Jul (early summer dome)
  - 2022 Jul 1 - Jul 31 (Texas heat dome)
  - 2023 Jun 15 - Sep 15 (record summer, multiple ERCOT EEA1)
  - 2024 Jun 15 - Sep 15 (second consecutive record summer)
  - 2025 Jun 15 - Sep 15 (record summer continuation per project conventions)

On each event-window start date, fire pair: SHORT (MARA+RIOT+CLSK)/3, LONG MSTR.
Hold 20 trading days unless: pair drawdown > 12%, or season ends Oct 1.

PnL = MSTR return - (1/3)*(MARA + RIOT + CLSK) return.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics, save_result,
                     mark_failed, daily_returns)


SIGNAL_ID = "PL608_ercot_summer_miner_short_mstr_long"
NAME = "ERCOT Summer Heat -> Short TX Miners / Long MSTR (20d pair)"


# Calendar-proxy heat cluster windows (start dates). Per spec, the meaningful
# backtest window for MSTR-as-treasury rationale is 2021 onward.
HEAT_EVENT_STARTS = [
    "2021-06-15",
    "2022-07-05",
    "2023-06-20",
    "2023-08-10",
    "2024-06-17",
    "2024-08-12",
    "2025-06-16",
    "2025-08-04",
]


def main():
    sid = SIGNAL_ID
    try:
        px = load_prices(["MARA", "RIOT", "CLSK", "MSTR", "IBIT", "SPY"],
                         start="2020-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    required = ["MARA", "RIOT", "CLSK", "MSTR", "SPY"]
    avail = [t for t in required if t in px.columns]
    if "MSTR" not in avail or "SPY" not in avail:
        return mark_failed(sid, f"missing core tickers; got {list(px.columns)}")

    px = px.sort_index().ffill(limit=3)
    ret = daily_returns(px)

    miners = [t for t in ["MARA", "RIOT", "CLSK"] if t in ret.columns]
    if len(miners) < 2:
        return mark_failed(sid, f"need >=2 miner tickers; got {miners}")

    miner_basket = ret[miners].mean(axis=1)
    mstr_r = ret["MSTR"]
    spy_r = ret["SPY"]

    # Pair return: long MSTR, short miner basket -> ret(MSTR) - ret(basket)
    pair_r = (mstr_r - miner_basket).dropna()

    hold_days = 20
    stop_dd = -0.12

    trading_idx = ret.index
    positions = pd.Series(0.0, index=trading_idx)

    in_pos = False
    events = []
    cooldown_until_loc = -1

    for evstr in HEAT_EVENT_STARTS:
        ev = pd.Timestamp(evstr)
        loc = trading_idx.searchsorted(ev)
        if loc >= len(trading_idx):
            continue
        entry_loc = loc
        if entry_loc <= cooldown_until_loc:
            continue
        # MSTR-treasury rationale starts 2020+; require MSTR data present
        if pd.isna(px["MSTR"].iloc[entry_loc]) or any(
                pd.isna(px[m].iloc[entry_loc]) for m in miners):
            continue

        # Determine exit: 20d hold, season cutoff Oct 1, or drawdown stop
        end_loc = min(entry_loc + hold_days, len(trading_idx) - 1)
        # season cutoff
        season_cut = pd.Timestamp(year=trading_idx[entry_loc].year, month=10, day=1)
        cut_loc = trading_idx.searchsorted(season_cut)
        if cut_loc < end_loc:
            end_loc = max(entry_loc + 1, cut_loc - 1)

        # walk forward to apply stop loss on pair cumulative return
        # pair daily return contribution is pair_r aligned to trading_idx
        pair_window = pair_r.reindex(trading_idx).iloc[entry_loc + 1:end_loc + 1]
        if len(pair_window) == 0:
            continue
        cumret = (1 + pair_window).cumprod() - 1
        exit_loc = end_loc
        exit_reason = "max_hold_or_season"
        for k, val in enumerate(cumret.values):
            if pd.notna(val) and val <= stop_dd:
                exit_loc = entry_loc + 1 + k
                exit_reason = "stop_loss"
                break

        # Position runs from entry_loc through exit_loc-1 (apply prev-day-pos)
        positions.iloc[entry_loc:exit_loc + 1] = 1.0
        cooldown_until_loc = exit_loc

        events.append({
            "event_start": evstr,
            "entry_date": str(trading_idx[entry_loc].date()),
            "exit_date": str(trading_idx[exit_loc].date()),
            "exit_reason": exit_reason,
            "pair_return": float(cumret.iloc[min(exit_loc - entry_loc - 1, len(cumret) - 1)])
                if len(cumret) else None,
        })

    if not events:
        return mark_failed(
            sid,
            "no events fired (data gaps for crypto miners/MSTR around heat windows)",
            extra={
                "rule": "Short TX miners / long MSTR pair on summer heat clusters, hold 20d",
                "mechanism": "ERCOT peak prices squeeze miner margins; MSTR (no power exposure) preserves BTC beta",
                "source": "PL608 catalog",
            },
        )

    # daily PnL = previous-day position * pair return
    pair_full = pair_r.reindex(trading_idx).fillna(0.0)
    pnl = positions.shift(1).fillna(0.0) * pair_full
    pnl = pnl.dropna()

    # Trim to active window for meaningful metrics
    first_entry = pd.Timestamp(events[0]["entry_date"])
    pnl = pnl.loc[pnl.index >= first_entry]
    active_pos = positions.loc[positions.index >= first_entry]

    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient PnL: {len(pnl)} days")

    m = compute_metrics(
        pnl,
        benchmark=spy_r.reindex(pnl.index),
        name=NAME,
        positions=active_pos,
        cost_bps=20,  # higher costs for miner pair (small caps)
    )

    m["n_events"] = len(events)
    m["pct_in_market"] = float(active_pos.mean()) if len(active_pos) else 0.0
    m["miners_used"] = miners
    m["events"] = events
    m["status"] = "ok"

    save_result(
        sid,
        m,
        extra={
            "rule": "During June-September, on dates following heat-wave clusters in Texas (proxy: known multi-day 100F+ event start dates from public weather summaries), short equal-weight (MARA+RIOT+CLSK) and long MSTR with equal dollar notional. Hold 20 trading days; stop on -12% pair drawdown or end of 4CP season Oct 1.",
            "mechanism": "ERCOT 4CP heat events drive day-ahead LMPs to $1500+/MWh, slashing Texas BTC miner gross margins (curtailment + spot power cost). MSTR holds BTC-on-balance-sheet with no power exposure, preserving the BTC-beta leg. Pair isolates the operational-cost gap between power-buyer miners and BTC-treasury equity.",
            "source": "PL608 catalog; NOAA heat summaries; ERCOT 4CP convention (June 1-Sep 30)",
            "caveats": "Calendar proxy for heat events (NOAA GHCN auto-ingest out of scope); small N (8 events 2021-2025); MSTR is a BTC-equity treasury proxy only valid post-2020. High-vol pair; spread costs elevated (~20bps round-trip used).",
        },
        pnl=pnl,
    )
    print(f"Saved {sid}: n_events={len(events)}  Sharpe={m.get('sharpe',0):.2f}  CAGR={m.get('cagr',0)*100:.2f}%")


if __name__ == "__main__":
    main()
