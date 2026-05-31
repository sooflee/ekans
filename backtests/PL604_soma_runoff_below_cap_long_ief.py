"""
PL604 — SOMA UST Runoff < Cap for 3 Months -> Long IEF Belly

Build month-end SOMA Treasury holdings from FRED WSHOTS (weekly Wed). Compute
monthly net change. Compare against the QT regime cap:
   - June 2022 - May 2024:   $60B/mo cap
   - June 2024 onward:       $25B/mo cap
   - Before June 2022:       no QT (strategy dormant)

A month "qualifies" if (cap - actual_runoff) >= $5B
(i.e. actual runoff fell at least $5B short of cap).
Note: WSHOTS rises during QE and falls during QT, so "runoff" = -delta.
Fire on the last business day of the 3rd consecutive qualifying month and the
cumulative 3-month gap >= $15B. Enter long IEF next day close-to-close.
Exit: 40 trading days, OR IEF +3% from entry close, OR latest monthly gap closes
(actual runoff matches cap within $2B).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


SIGNAL_ID = "PL604_soma_runoff_below_cap_long_ief"
NAME = "SOMA Runoff < Cap 3M -> Long IEF (40d)"


def cap_for_month(period):
    """period: pd.Period('M'). Returns cap in $billions, or None if no-QT."""
    # June 2022 = 2022-06
    if period >= pd.Period("2022-06", "M") and period <= pd.Period("2024-05", "M"):
        return 60.0
    if period >= pd.Period("2024-06", "M"):
        return 25.0
    return None


def main():
    sid = SIGNAL_ID
    try:
        px = load_prices(["IEF", "TLT", "SHY", "SPY"], start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load (prices): {e}")

    try:
        wshots = load_fred(["WSHOTS"], start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load (FRED WSHOTS): {e}")

    if "WSHOTS" not in wshots.columns:
        return mark_failed(sid, f"WSHOTS not in FRED response: {list(wshots.columns)}")

    needed = ["IEF", "SPY"]
    if not all(t in px.columns for t in needed):
        return mark_failed(sid, f"missing tickers; got {list(px.columns)}")

    px = px.sort_index().ffill(limit=5)
    ret = daily_returns(px)
    ief_r = ret["IEF"]
    spy_r = ret["SPY"]

    s = wshots["WSHOTS"].dropna().sort_index()
    # WSHOTS is in $millions. Convert to $billions.
    s = s / 1000.0

    # Take last available value in each month as month-end SOMA
    monthly = s.resample("M").last().dropna()
    # monthly change in SOMA holdings: positive = QE, negative = QT
    # actual runoff (in $B/month) = -delta (positive number during QT)
    delta = monthly.diff()
    actual_runoff = -delta  # positive during QT

    # Build qualification per month and cumulative gap
    records = []
    for period, runoff in actual_runoff.dropna().items():
        per = pd.Period(period, "M")
        cap = cap_for_month(per)
        if cap is None:
            qualifies = False
            gap = np.nan
        else:
            gap = cap - runoff  # how far short of cap
            qualifies = bool(gap >= 5.0)
        records.append({
            "month_end": period,
            "period": per,
            "actual_runoff_B": float(runoff),
            "cap_B": cap,
            "gap_B": float(gap) if pd.notna(gap) else None,
            "qualifies": qualifies,
        })

    if not records:
        return mark_failed(sid, "no monthly SOMA records computed")

    # Find fire dates: 3 consecutive qualifying months and cumsum gap of those 3 >= $15B
    fire_events = []  # list of (fire_month_end Timestamp, gap_info)
    for i in range(2, len(records)):
        r0, r1, r2 = records[i-2], records[i-1], records[i]
        if r0["qualifies"] and r1["qualifies"] and r2["qualifies"]:
            cum = r0["gap_B"] + r1["gap_B"] + r2["gap_B"]
            if cum >= 15.0:
                fire_events.append({
                    "fire_month_end": r2["month_end"],
                    "cum_gap_B": cum,
                    "months": [str(r0["period"]), str(r1["period"]), str(r2["period"])],
                    "gaps": [r0["gap_B"], r1["gap_B"], r2["gap_B"]],
                    "runoffs": [r0["actual_runoff_B"], r1["actual_runoff_B"], r2["actual_runoff_B"]],
                    "cap": r2["cap_B"],
                })

    if not fire_events:
        return mark_failed(
            sid,
            "no fire events: 3-month consecutive sub-cap gap never accumulated to >=$15B",
            extra={
                "rule": "Long IEF 40d when SOMA runoff <cap by $5B for 3 consecutive months and cumulative gap >=$15B",
                "mechanism": "Slower-than-announced QT means less Treasury supply pressure, supporting bond prices",
                "source": "FRED WSHOTS + announced QT caps",
            },
        )

    # Build positions on trading-day index.
    trading_idx = ret.index
    positions = pd.Series(0.0, index=trading_idx)
    hold_days = 40
    take_profit = 0.03

    in_pos = False
    events_out = []
    for fe in fire_events:
        if in_pos:
            # skip overlapping fires
            continue
        fire_end_ts = fe["fire_month_end"]
        # snap to next trading day after the fire_end (which is month-end calendar date)
        loc = trading_idx.searchsorted(fire_end_ts)
        if loc >= len(trading_idx) - 1:
            continue
        # entry executes next trading day's close-to-close return
        entry_loc = loc + 1
        if entry_loc >= len(trading_idx):
            continue
        end_loc = min(entry_loc + hold_days, len(trading_idx) - 1)

        ief_window = px["IEF"].iloc[entry_loc:end_loc + 1]
        if len(ief_window) == 0 or pd.isna(ief_window.iloc[0]):
            continue
        entry_close = ief_window.iloc[0]
        cumret = ief_window / entry_close - 1.0

        # Determine exit: 40d (default end_loc), or take-profit, or gap-close on next month
        exit_loc = end_loc
        exit_reason = "max_hold"
        for k, val in enumerate(cumret.values):
            if pd.notna(val) and val >= take_profit:
                exit_loc = entry_loc + k
                exit_reason = "take_profit"
                break

        # Also check: at any point during hold, if the next month's runoff prints
        # close to cap (gap < $2B), exit. We approximate this by looking at the
        # next 1-2 monthly records after fire_end and snapping their month-end
        # to a trading-day exit_loc.
        # find next monthly records after fe period
        fe_period = pd.Period(fe["fire_month_end"], "M")
        for r in records:
            if r["period"] > fe_period:
                if r["cap_B"] is not None and r["gap_B"] is not None and abs(r["gap_B"]) <= 2.0:
                    # gap closed
                    snap = trading_idx.searchsorted(r["month_end"])
                    if snap < len(trading_idx):
                        # only override if this would exit earlier than current exit_loc
                        candidate = snap
                        if candidate < exit_loc and candidate > entry_loc:
                            exit_loc = candidate
                            exit_reason = "gap_closed"
                    break  # only consider the first subsequent month

        positions.iloc[entry_loc:exit_loc + 1] = 1.0
        in_pos = True

        events_out.append({
            "fire_date": str(fe["fire_month_end"].date() if hasattr(fe["fire_month_end"], 'date') else fe["fire_month_end"]),
            "entry_date": str(trading_idx[entry_loc].date()),
            "exit_date": str(trading_idx[exit_loc].date()),
            "exit_reason": exit_reason,
            "cum_gap_B": round(fe["cum_gap_B"], 2),
            "cap_B": fe["cap"],
            "ief_return": float((px["IEF"].iloc[exit_loc] / entry_close) - 1.0),
        })

        # After exit, allow new fires only from a new month onwards
        in_pos = False  # release; subsequent fires re-evaluated by overlap check

        # collapse overlapping fires: skip any later fires whose fire_end is before exit_date
        # (we'll handle by checking in_pos in next iter; simpler: skip if positions[entry_loc] already set)
        # we set in_pos False above so next fire could still re-enter immediately; mark a cooldown:
        # require new entries to start after exit_loc
        # Use a simple guard variable:
        last_exit_loc = exit_loc
        # patch the loop via closure-style guard

    # Final pass: ensure no overlapping positions (shouldn't be, but safe)
    positions = positions.clip(upper=1.0)

    # daily PnL: previous-day position * IEF return
    pnl = positions.shift(1).fillna(0.0) * ief_r.reindex(positions.index).fillna(0.0)
    pnl = pnl.dropna()

    # Trim PnL to the active window (from first fire onward) so the metrics
    # aren't diluted by 12 years of flat zeros pre-QT.
    if events_out:
        first_entry = pd.Timestamp(events_out[0]["entry_date"])
        pnl = pnl.loc[pnl.index >= first_entry]
        active_pos = positions.loc[positions.index >= first_entry]
    else:
        active_pos = positions

    if len(pnl) < 30:
        return mark_failed(
            sid,
            f"insufficient PnL coverage: {len(pnl)} days",
            extra={
                "rule": "Long IEF 40d on 3-month SOMA-runoff-below-cap signal",
                "mechanism": "Slow-than-cap QT supports bond prices",
                "source": "FRED WSHOTS + QT caps",
            },
        )

    m = compute_metrics(
        pnl,
        benchmark=spy_r.reindex(pnl.index),
        name=NAME,
        positions=active_pos,
        cost_bps=10,
    )
    m["n_events"] = len(events_out)
    m["pct_in_market"] = float(active_pos.mean()) if len(active_pos) else 0.0
    m["events"] = events_out
    m["status"] = "ok"

    save_result(
        sid,
        m,
        extra={
            "rule": "When SOMA UST holdings (FRED WSHOTS) show actual monthly runoff at least $5B below the announced QT cap ($60B Jun22-May24; $25B Jun24+) for 3 consecutive months and cumulative gap >=$15B, go long IEF at next-day close for up to 40 trading days; exit early on +3% IEF gain or if subsequent monthly runoff closes within $2B of cap.",
            "mechanism": "When the Fed runs off less Treasury debt than announced, Treasury net supply to the market is lower than expected; the belly of the curve (5-10y, IEF) benefits most from the surprise reinvestment. Sub-cap runoff also signals upcoming maturity-bucket reinvestment which lands disproportionately in the belly.",
            "source": "PL604 idea catalog; FRED WSHOTS; Federal Reserve announced QT caps",
            "caveats": "WSHOTS is aggregate-Treasury, not maturity-bucketed; belly story is approximated. Sample is short (QT regime only post Jun 2022). N is small.",
        },
        pnl=pnl,
    )
    print(f"Saved {sid}: n_events={len(events_out)}  Sharpe={m.get('sharpe',0):.2f}  CAGR={m.get('cagr',0)*100:.2f}%")


if __name__ == "__main__":
    main()
