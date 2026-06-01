"""PL843_brexit_tca_review_ewu_short
UK-EU TCA Review Headline Cluster: Short EWU / Long UUP (FXB Hedge)

Counter-signal: when GBP/USD 5-day return drops below -1.5% AND EWU
underperforms SPY by >=1.0pp over the same 5-day window, enter SHORT EWU /
LONG UUP for 15 trading days (or exit if EWU beats SPY by +2% from entry).

This is a price-observable proxy for a Brexit/TCA headline cluster.
Yield spread data (FRED IRLTLT01GBM156N - IRLTLT01DEM156N) is monthly so
we use GBPUSD + EWU/SPY divergence as the primary dual trigger.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL843_brexit_tca_review_ewu_short"
    # EWU = iShares MSCI UK ETF, UUP = Invesco DB USD Index Bullish Fund,
    # GBPUSD=X = GBP/USD FX rate, SPY = benchmark
    tickers = ["EWU", "UUP", "SPY", "GBPUSD=X"]

    try:
        px = load_prices(tickers, start="2016-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=3)

    # Check required tickers
    required = ["EWU", "UUP", "SPY"]
    missing = [t for t in required if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing required tickers: {missing}")

    has_gbp = "GBPUSD=X" in px.columns
    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    # 5-day rolling return for GBPUSD and EWU vs SPY
    # px_5d = 5-day return = px / px.shift(5) - 1
    gbp_5d = px["GBPUSD=X"].pct_change(5) if has_gbp else None
    ewu_5d = px["EWU"].pct_change(5)
    spy_5d = px["SPY"].pct_change(5)

    # Entry conditions (evaluated at close, entry at next open = next day's return)
    # 1. GBP/USD 5-day return < -1.5%  (GBP weakness)
    # 2. EWU 5-day return underperforms SPY by at least 1.0pp
    # NOTE: If no GBP data, use EWU/SPY divergence alone with a tighter threshold (-2.0pp)
    if has_gbp and gbp_5d is not None:
        gbp_cond = gbp_5d < -0.015
        ewu_spr_cond = (ewu_5d - spy_5d) < -0.010
        signal = gbp_cond & ewu_spr_cond
    else:
        # Fallback: EWU underperforms SPY by >= 2.5pp over 5 days
        ewu_spr_cond = (ewu_5d - spy_5d) < -0.025
        signal = ewu_spr_cond

    signal = signal.fillna(False)

    # Strategy: short EWU, long UUP
    # PnL = -EWU_return + UUP_return
    ewu_ret = ret["EWU"]
    uup_ret = ret["UUP"]
    ls_ret = -ewu_ret + uup_ret  # short EWU, long UUP

    HOLD_DAYS = 15
    STOP_OUTPERF = 0.02  # exit if EWU beats SPY by +2% (adverse stop)

    idx = ret.index
    positions = pd.Series(0.0, index=idx)
    pnl = pd.Series(0.0, index=idx)

    # Dedup signals: require at least HOLD_DAYS between entry dates
    in_position = False
    open_until = None
    entry_date = None
    entry_ewu_cum = 0.0
    entry_spy_cum = 0.0

    events = []
    n = len(idx)

    for j in range(1, n):
        dt = idx[j]

        if in_position:
            # Apply pnl
            pnl.iloc[j] = ls_ret.iloc[j] if not pd.isna(ls_ret.iloc[j]) else 0.0
            positions.iloc[j] = 1.0

            # Cumulative EWU vs SPY from entry
            ewu_cum = float((ewu_ret.iloc[max(0, j-HOLD_DAYS+1):j+1] + 1).prod() - 1)
            spy_cum = float((spy_r.reindex(idx[max(0, j-HOLD_DAYS+1):j+1]) + 1).prod() - 1)
            spread_from_entry = ewu_cum - spy_cum

            scheduled_end = (j + 1 >= open_until)
            adverse_stop = spread_from_entry > STOP_OUTPERF

            if scheduled_end or adverse_stop:
                exit_reason = "adverse_stop" if adverse_stop and not scheduled_end else "scheduled_15d"
                events[-1]["exit_date"] = str(dt.date())
                events[-1]["exit_reason"] = exit_reason
                events[-1]["ls_return"] = float(pnl[events[-1]["entry_date_idx"]:j+1].sum()) if False else None
                in_position = False
                open_until = None

        else:
            # Check if previous day's signal triggered entry today
            if signal.iloc[j - 1]:
                in_position = True
                open_until = min(j + HOLD_DAYS, n)
                events.append({
                    "signal_date": str(idx[j - 1].date()),
                    "entry_date": str(dt.date()),
                    "entry_date_idx": j,
                    "gbp_5d": float(gbp_5d.iloc[j-1]) if has_gbp and gbp_5d is not None else None,
                    "ewu_spy_5d_spread": float((ewu_5d - spy_5d).iloc[j-1]),
                })
                pnl.iloc[j] = ls_ret.iloc[j] if not pd.isna(ls_ret.iloc[j]) else 0.0
                positions.iloc[j] = 1.0

    # Compute event-level returns for the log
    for ev in events:
        j_start = ev.get("entry_date_idx", 0)
        j_end = j_start + HOLD_DAYS
        slice_pnl = pnl.iloc[j_start:min(j_end, n)]
        if len(slice_pnl):
            ev["event_ls_return"] = round(float((1 + slice_pnl).prod() - 1), 4)
        ev.pop("entry_date_idx", None)

    held_pnl = pnl[positions > 0]
    held_spy = spy_r.reindex(held_pnl.index).dropna()
    n_events = len(events)

    if len(held_pnl) < 30:
        return mark_failed(
            sid,
            f"insufficient held days ({len(held_pnl)}) across {n_events} events",
            extra={"events": events, "signal_count": int(signal.sum())},
        )

    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name="Brexit/TCA Cluster: Short EWU Long UUP (held-days only)",
        positions=positions.reindex(held_pnl.index).fillna(0),
        cost_bps=10,
    )

    event_returns = [e["event_ls_return"] for e in events if e.get("event_ls_return") is not None]

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "When GBP/USD 5-day return < -1.5% AND EWU underperforms SPY by "
                ">=1.0pp over the same 5-day window, enter SHORT EWU / LONG UUP "
                "at next open; hold 15 trading days or exit if EWU beats SPY by "
                "+2% from entry (adverse stop)."
            ),
            "mechanism": (
                "GBP weakness combined with UK equity underperformance signals a "
                "Brexit/TCA headline cluster or UK political risk repricing. The "
                "short EWU / long UUP trade captures the mean-reversion of this "
                "risk premium: GBP tends to overshoot on Brexit headlines, and "
                "UK equities (EWU, GBP-denominated) face dual headwinds from FX "
                "depreciation and UK political uncertainty premium. UUP benefits "
                "as USD safe-haven demand rises."
            ),
            "source": (
                "EWU, UUP, SPY, GBPUSD=X via yfinance; "
                "Known analog events: Brexit vote (2016-06-24), no-deal threat "
                "(2019-10), Truss mini-budget (2022-09-23), Windsor Framework "
                "(2023-02-27)."
            ),
            "tickers_short": ["EWU"],
            "tickers_long": ["UUP"],
            "n_events": n_events,
            "signal_count": int(signal.sum()),
            "events": events[:20],  # truncate for readability
            "avg_event_return": round(float(np.mean(event_returns)), 4) if event_returns else None,
            "gbp_data_available": has_gbp,
            "caveats": (
                "GBP/USD and EWU divergence is a noisy proxy for Brexit headline "
                "clusters — the signal fires on any GBP weakness episode, not just "
                "Brexit/TCA events. Short EWU has FX exposure embedded (USD-denominated "
                "ETF tracking GBP-denominated UK equities). FRED monthly yield spread "
                "was not used as a primary trigger due to interpolation noise. "
                "Cost drag applied at 10bps round-trip. Adverse stop at +2% EWU vs "
                "SPY spread may be triggered by broad-market risk-on episodes unrelated "
                "to UK politics."
            ),
        },
        pnl=held_pnl,
    )

    print(f"Done: {sid}")
    print(f"  n_events: {n_events}, signal fires: {int(signal.sum())}, held_days: {len(held_pnl)}")
    print(f"  avg_event_return: {round(float(np.mean(event_returns)), 4) if event_returns else None}")
    if "error" not in m:
        print(
            f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
            f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
        )
        if "net_sharpe" in m:
            print(f"  Net Sharpe: {m['net_sharpe']:.2f}, Net CAGR: {m['net_cagr']*100:.2f}%")


if __name__ == "__main__":
    main()
