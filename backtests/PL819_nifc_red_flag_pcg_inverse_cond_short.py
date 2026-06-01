"""PL819_nifc_red_flag_pcg_inverse_cond_short
NIFC Preparedness Level 4/5 + NWS Red Flag Cluster -> Short PCG Inverse-Condemnation Re-Pricing

Event-study backtest using known California fire season escalation dates:
  - 2017-10-08: North Bay fires trigger date
  - 2018-11-08: Camp Fire pre-announcement (NOTE: drove PCG bankruptcy Jan 2019)
  - 2019-10-26: Kincade Fire PSPS
  - 2020-09-07: Glass Fire precursor
  - 2025-01-07: Eaton/Palisades fires

On each trigger date, short PCG for 25 trading days.
Also compute paired PCG/EIX short/long for sector-neutral version.
Hard stop: cover if PCG closes up >12% from entry.
No-press rule: if PCG gaps down >20% on any single day, cover at open of that day.

PCG bankruptcy context: filed Jan 2019, emerged Jun 2020. The 2018-11-08 event
is valid but flagged as extreme outlier. Pre/post-emergence regimes differ due
to AB 1054 wildfire fund.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)

# Known event trigger dates (5th consecutive Red Flag day seed dates)
KNOWN_EVENTS = [
    "2017-10-08",  # North Bay fires
    "2018-11-08",  # Camp Fire (extreme event - drove bankruptcy)
    "2019-10-26",  # Kincade Fire PSPS
    "2020-09-07",  # Glass Fire precursor
    "2025-01-07",  # Eaton/Palisades fires
]

HOLD_DAYS = 25  # 5 calendar weeks
HARD_STOP_RALLY = 0.12   # cover if PCG closes up >12% from entry (short thesis invalid)
GAP_DOWN_COVER = 0.20    # cover on open if PCG gaps down >20% (no-press rule)


def run_event_study(events, px, ret, hold_days=25):
    """Run event study: short PCG on each event date."""
    idx = ret.index
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)
    event_log = []

    pcg_ret = ret["PCG"].fillna(0.0)
    spy_ret = ret["SPY"].fillna(0.0)
    pcg_px = px["PCG"]

    for ev_date in events:
        rel = pd.Timestamp(ev_date)
        future_sessions = idx[idx > rel]
        if len(future_sessions) == 0:
            event_log.append({
                "event_date": ev_date,
                "status": "no_data_after_event",
            })
            continue

        entry_dt = future_sessions[0]
        entry_pos = idx.get_loc(entry_dt)
        entry_price = float(pcg_px.iloc[entry_pos]) if entry_pos < len(pcg_px) else np.nan

        if np.isnan(entry_price) or entry_price <= 0:
            event_log.append({
                "event_date": ev_date,
                "entry_date": str(entry_dt.date()),
                "status": "invalid_entry_price",
            })
            continue

        exit_pos = min(entry_pos + hold_days, len(idx))
        exit_reason = "scheduled_25d"
        actual_exit_pos = exit_pos

        # Walk through holding window applying stops
        for j in range(entry_pos, exit_pos):
            day_close = float(pcg_px.iloc[j])
            day_ret = float(pcg_ret.iloc[j])

            # Hard stop: if PCG closes up >12% from entry (short thesis broken)
            cum_from_entry = (day_close / entry_price) - 1.0
            if cum_from_entry > HARD_STOP_RALLY:
                actual_exit_pos = j + 1  # exit today's close
                exit_reason = "hard_stop_rally"
                # Include today's return
                pnl.iloc[j] -= day_ret
                positions.iloc[j] = 1.0
                break

            # No-press rule: if PCG drops >20% in one day (ignition event priced)
            # cover on gap-down open => skip today's return, exit here
            if day_ret < -GAP_DOWN_COVER:
                actual_exit_pos = j  # do NOT capture today's return
                exit_reason = "gap_down_cover"
                break

            # Normal holding day: short PCG => pnl is negative of PCG return
            pnl.iloc[j] -= day_ret
            positions.iloc[j] = 1.0
        else:
            actual_exit_pos = exit_pos

        exit_dt = idx[actual_exit_pos - 1] if actual_exit_pos > entry_pos else entry_dt
        exit_price = float(pcg_px.iloc[actual_exit_pos - 1]) if actual_exit_pos > entry_pos else entry_price

        gross_pcg_short = -(exit_price / entry_price - 1.0)

        # PCG bankruptcy flag
        is_bankruptcy_period = (
            pd.Timestamp(ev_date) >= pd.Timestamp("2019-01-01") and
            pd.Timestamp(ev_date) <= pd.Timestamp("2020-06-30")
        )
        is_camp_fire = ev_date == "2018-11-08"

        event_log.append({
            "event_date": ev_date,
            "entry_date": str(entry_dt.date()),
            "entry_price": round(entry_price, 2),
            "exit_date": str(exit_dt.date()),
            "exit_price": round(exit_price, 2),
            "exit_reason": exit_reason,
            "n_hold_days": int(actual_exit_pos - entry_pos),
            "gross_pcg_short_return": round(gross_pcg_short, 4),
            "is_camp_fire_extreme": is_camp_fire,
            "is_bankruptcy_period": is_bankruptcy_period,
        })

    return pnl, positions, event_log


def run_paired_study(events, px, ret, hold_days=25):
    """Run paired trade: short PCG / long EIX (sector-neutral)."""
    idx = ret.index
    pnl_paired = pd.Series(0.0, index=idx)
    positions_paired = pd.Series(0.0, index=idx)

    pcg_ret = ret["PCG"].fillna(0.0)
    eix_ret = ret["EIX"].fillna(0.0)
    pcg_px = px["PCG"]

    for ev_date in events:
        rel = pd.Timestamp(ev_date)
        future_sessions = idx[idx > rel]
        if len(future_sessions) == 0:
            continue

        entry_dt = future_sessions[0]
        entry_pos = idx.get_loc(entry_dt)
        entry_price = float(pcg_px.iloc[entry_pos]) if entry_pos < len(pcg_px) else np.nan

        if np.isnan(entry_price) or entry_price <= 0:
            continue

        exit_pos = min(entry_pos + hold_days, len(idx))

        for j in range(entry_pos, exit_pos):
            if positions_paired.iloc[j] == 0.0:
                # Pair spread: -PCG + EIX (dollar neutral)
                spread = -pcg_ret.iloc[j] + eix_ret.iloc[j]
                pnl_paired.iloc[j] = spread
                positions_paired.iloc[j] = 1.0

    return pnl_paired, positions_paired


def main():
    sid = "PL819_nifc_red_flag_pcg_inverse_cond_short"
    tickers = ["PCG", "EIX", "SRE", "SPY"]

    try:
        px = load_prices(tickers, start="2016-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)

    # Check available tickers (PCG mandatory; EIX for paired trade)
    missing = [t for t in ["PCG", "SPY"] if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing mandatory tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    # ---- Primary: Short PCG event study ----
    pnl, positions, event_log = run_event_study(KNOWN_EVENTS, px, ret, HOLD_DAYS)

    n_events = sum(1 for e in event_log if e.get("entry_date") and e.get("gross_pcg_short_return") is not None)
    if n_events == 0:
        return mark_failed(sid, "no valid events with entry data")

    # Held-day PnL
    held_pnl = pnl[positions > 0]
    held_spy = spy_r.reindex(held_pnl.index).dropna()

    if len(held_pnl) < 20:
        return mark_failed(
            sid,
            f"insufficient held days: {len(held_pnl)} (n_events={n_events})",
            extra={"events": event_log},
        )

    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name="Short PCG on NIFC/Red Flag Trigger (held-days)",
        positions=positions.reindex(held_pnl.index).fillna(0),
        cost_bps=10,
    )

    # ---- Secondary: PCG/EIX paired trade ----
    paired_pnl, paired_positions = pd.Series(0.0, index=ret.index), pd.Series(0.0, index=ret.index)
    if "EIX" in px.columns:
        paired_pnl, paired_positions = run_paired_study(KNOWN_EVENTS, px, ret, HOLD_DAYS)
        held_paired = paired_pnl[paired_positions > 0]
        if len(held_paired) >= 20:
            m_paired = compute_metrics(
                held_paired,
                benchmark=held_spy.reindex(held_paired.index).dropna(),
                name="PCG short / EIX long paired (held-days)",
                positions=paired_positions.reindex(held_paired.index).fillna(0),
                cost_bps=10,
            )
        else:
            m_paired = {"error": "insufficient days for paired metric"}
    else:
        m_paired = {"error": "EIX not available"}

    # ---- Event-level stats ----
    event_returns = [e["gross_pcg_short_return"] for e in event_log
                     if e.get("gross_pcg_short_return") is not None]
    win_rate = float(np.mean([r > 0 for r in event_returns])) if event_returns else None
    avg_ret = float(np.mean(event_returns)) if event_returns else None
    median_ret = float(np.median(event_returns)) if event_returns else None

    # Without Camp Fire outlier
    non_extreme_returns = [
        e["gross_pcg_short_return"] for e in event_log
        if e.get("gross_pcg_short_return") is not None and not e.get("is_camp_fire_extreme")
    ]
    win_rate_no_camp = float(np.mean([r > 0 for r in non_extreme_returns])) if non_extreme_returns else None
    avg_ret_no_camp = float(np.mean(non_extreme_returns)) if non_extreme_returns else None

    summary = {
        "n_events": n_events,
        "avg_gross_short_return": round(avg_ret, 4) if avg_ret is not None else None,
        "median_gross_short_return": round(median_ret, 4) if median_ret is not None else None,
        "win_rate": round(win_rate, 4) if win_rate is not None else None,
        "avg_gross_excl_camp_fire": round(avg_ret_no_camp, 4) if avg_ret_no_camp is not None else None,
        "win_rate_excl_camp_fire": round(win_rate_no_camp, 4) if win_rate_no_camp is not None else None,
    }

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "When NIFC National Preparedness Level reaches 4 or 5 AND NWS SPC issues "
                "Red Flag Warnings covering Northern California (PCG service territory) for "
                ">=5 consecutive days, short PCG at close on day 5. Hold 25 trading days. "
                "Hard stop: cover if PCG closes up >12% from entry. No-press: cover on open "
                "if PCG gaps down >20% on a single day. Pair option: short PCG / long EIX "
                "for sector-neutral version."
            ),
            "mechanism": (
                "PG&E (PCG) faces inverse-condemnation liability under California law: "
                "any PCG equipment that ignites a fire triggers full liability regardless "
                "of negligence. High NIFC preparedness + Red Flag cluster signals elevated "
                "ignition risk in PCG territory, prompting insurance market re-pricing of "
                "PCG's uninsured wildfire liability tail. Pre-AB 1054 (pre-2020), this "
                "re-pricing drove dramatic equity declines (bankruptcy after Camp Fire). "
                "Post-emergence under AB 1054 wildfire fund, liability regime is partially "
                "backstopped, but equity still reacts to PSPS orders and ignition events."
            ),
            "source": (
                "NIFC preparedness levels from predictiveservices.nifc.gov; NWS SPC Fire "
                "Weather Outlook archive. Event dates manually seeded per strategy spec. "
                "Prices via yfinance (auto_adjust=True). PCG bankruptcy note: filed Jan 2019, "
                "emerged Jun 2020; 2018-11-08 Camp Fire event is flagged as extreme."
            ),
            "tickers": tickers,
            "hold_days": HOLD_DAYS,
            "known_events": KNOWN_EVENTS,
            "events": event_log,
            "summary": summary,
            "paired_trade_metrics": {
                k: v for k, v in m_paired.items()
                if k in ("sharpe", "cagr", "max_dd", "t_stat", "net_sharpe", "net_cagr", "error")
            },
            "regime_note": (
                "Pre-Jun 2020 (pre-bankruptcy emergence): PCG under original equity "
                "structure with unlimited inverse-condemnation liability. "
                "Post-Jun 2020: reorganized equity under AB 1054 wildfire fund backstop. "
                "Structural break exists; post-2020 returns likely more muted."
            ),
            "caveats": (
                "Tiny sample size (n=5 events). 2018-11-08 Camp Fire is an extreme outlier "
                "that drove actual bankruptcy; its inclusion significantly inflates mean return. "
                "PCG bankruptcy/restructuring (Jan 2019 - Jun 2020) affects price discovery. "
                "Event dates are manually seeded with potential look-ahead bias (we know these "
                "were the major events after the fact). NIFC preparedness levels are not fully "
                "automatable in real-time for live trading. Low sample makes statistical "
                "significance unreliable; treat as directional evidence only."
            ),
        },
        pnl=held_pnl,
    )

    print(f"Done: {sid}")
    print(f"  n_events: {n_events}")
    print(f"  summary: {summary}")
    if "error" not in m:
        print(
            f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
            f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
        )
        if "net_sharpe" in m:
            print(f"  Net Sharpe: {m['net_sharpe']:.2f}  Net CAGR: {m['net_cagr']*100:.2f}%")
    print(f"  Paired PCG/EIX metrics: {m_paired.get('sharpe', 'N/A')} sharpe")


if __name__ == "__main__":
    main()
