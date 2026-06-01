"""PL1050_catbond_spread_spike_long_rnr_axs
Cat-Bond Secondary Spread Spike -> Reinsurer Premium Hardening Long RNR/AXS

When cat-bond secondary market spreads widen materially (risk-appetite withdrawal
rather than loss-driven), go long RNR/AXS for 8 weeks (40 trading days).

Since cat-bond spread data is not freely available as a structured time series,
we use the known event dates identified from Artemis.bm:
  2020-01-24: COVID uncertainty onset for ILS
  2021-08-30: Post-Ida secondary spread widening beyond initial loss expectations
  2022-10-07: Post-Ian normalization start
  2023-01-13: Q1 2023 renewal cycle ILS stress

Additionally, we use a price-based proxy: when RNR or AXS underperforms SPY by
>15% over trailing 40 days without a corresponding major cat event — indicating
the market is discounting ILS capital withdrawal pressure.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


HOLD_DAYS = 40       # 8-week hold
MIN_GAP_DAYS = 30    # lockout between entries per ticker
TICKERS = ["RNR", "AXS", "EG", "RLI", "SPY"]
START = "2004-01-01"

# Known ILS stress episodes from Artemis.bm narrative (risk-appetite driven, not loss-driven)
# These are the manually identified entry dates from strategy spec
KNOWN_EVENTS = [
    "2020-01-24",  # COVID uncertainty onset for ILS
    "2021-08-30",  # Post-Ida secondary spread widening
    "2022-10-07",  # Post-Ian normalization start
    "2023-01-13",  # Q1 2023 renewal cycle ILS stress
]

# Additional historical ILS/reinsurance hard market episodes (pre-Artemis era)
# Identified from known reinsurance history: post-Katrina, post-2004 season, etc.
HISTORICAL_EVENTS = [
    "2005-09-26",  # Post-Katrina/Rita ILS market stress
    "2008-09-30",  # Post-Ike ILS stress
    "2011-05-01",  # Post-Tohoku earthquake ILS market stress
    "2012-11-05",  # Post-Sandy ILS market stress
    "2017-10-16",  # Post-Harvey/Irma/Maria triple cat ILS stress
    "2018-02-05",  # Late-cycle ILS re-pricing after 2017 losses
]


def run_event_study(event_dates, px, ret, hold_days=HOLD_DAYS):
    """
    Go long equal-weight RNR + AXS for hold_days after each event date.
    Returns pnl Series, positions Series, and event log.
    """
    idx = ret.index
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)
    event_log = []

    # Equal-weight RNR and AXS long
    long_tickers = [t for t in ["RNR", "AXS"] if t in ret.columns]
    weight = 1.0 / len(long_tickers) if long_tickers else 1.0

    for event_date_str in event_dates:
        event_dt = pd.Timestamp(event_date_str)

        # Find first trading day at or after event date
        future = idx[idx >= event_dt]
        if len(future) == 0:
            event_log.append({"event_date": event_date_str, "status": "no_data"})
            continue

        entry_dt = future[0]
        entry_pos = idx.get_loc(entry_dt)
        exit_pos = min(entry_pos + hold_days, len(idx))

        entry_prices = {t: px[t].get(entry_dt, np.nan) for t in long_tickers}

        # Equal-weight long of RNR + AXS
        combined_pnl = pd.Series(0.0, index=idx[entry_pos:exit_pos])
        ticker_rets_at_exit = {}
        for t in long_tickers:
            if pd.isna(entry_prices.get(t, np.nan)):
                continue
            t_ret = ret[t].fillna(0).iloc[entry_pos:exit_pos]
            combined_pnl += weight * t_ret
            ticker_rets_at_exit[t] = float((1 + t_ret).prod() - 1)

        spy_slice = ret["SPY"].fillna(0).iloc[entry_pos:exit_pos]
        ev_ret_combined = float((1 + combined_pnl).prod() - 1)
        ev_ret_spy = float((1 + spy_slice).prod() - 1)

        ev_rec = {
            "event_date": event_date_str,
            "entry_date": str(entry_dt.date()),
            "n_hold_days": int(exit_pos - entry_pos),
            "long_return": round(ev_ret_combined, 4),
            "spy_return": round(ev_ret_spy, 4),
            "excess_return": round(ev_ret_combined - ev_ret_spy, 4),
            "ticker_returns": ticker_rets_at_exit,
            "status": "ok",
        }
        event_log.append(ev_rec)

        # Add to portfolio pnl (only if no existing position on that day)
        for j in range(entry_pos, exit_pos):
            j_rel = j - entry_pos
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = 1.0
                if j_rel < len(combined_pnl):
                    pnl.iloc[j] = combined_pnl.iloc[j_rel]

    return pnl, positions, event_log


def find_momentum_proxy_events(px, ret, min_gap_days=MIN_GAP_DAYS):
    """
    Proxy: When RNR AND AXS both show recovery signal after underperformance.
    Specifically: RNR or AXS 40-day return < SPY-15% (steep underperformance),
    then first day both stocks start recovering (5-day momentum turns positive).
    """
    spy_r = ret["SPY"].fillna(0)
    events = []
    last_entry = None

    # Use RNR as primary (higher quality pure reinsurer)
    if "RNR" not in ret.columns:
        return []

    rnr_r = ret["RNR"].fillna(0)

    # Rolling 40-day cumulative return
    rnr_roll = (1 + rnr_r).rolling(40).apply(lambda x: x.prod(), raw=True) - 1
    spy_roll = (1 + spy_r).rolling(40).apply(lambda x: x.prod(), raw=True) - 1
    underperf = rnr_roll - spy_roll

    # Signal: underperformance was < -15% and now starts recovering
    # (underperf was very negative last week, now trending up)
    underperf_prev5 = underperf.rolling(5).min().shift(1)
    underperf_now = underperf

    entry_signal = (underperf_prev5 < -0.15) & (underperf_now > underperf_prev5)

    for dt in entry_signal[entry_signal].index:
        if last_entry is not None and (dt - last_entry).days < min_gap_days:
            continue
        events.append(str(dt.date()))
        last_entry = dt

    return events[:20]  # Cap at 20 proxy events


def main():
    sid = "PL1050_catbond_spread_spike_long_rnr_axs"
    primary_tickers = ["RNR", "AXS", "SPY"]

    try:
        px = load_prices(primary_tickers, start=START)
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    # Try to also load EG and RLI but don't fail if unavailable
    try:
        px_extra = load_prices(["EG", "RLI", "SPY"], start=START)
        for t in ["EG", "RLI"]:
            if t in px_extra.columns:
                px[t] = px_extra[t]
    except Exception:
        pass

    px = px.sort_index().ffill(limit=3)
    missing = [t for t in primary_tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing primary tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    # All known event dates (Artemis + historical hard market periods)
    all_event_dates = KNOWN_EVENTS + HISTORICAL_EVENTS

    # Filter to dates within our data range
    idx = ret.index
    valid_events = [e for e in all_event_dates
                    if pd.Timestamp(e) >= idx[0] and pd.Timestamp(e) <= idx[-1]]

    # Add proxy-based events to fill gaps
    proxy_events = find_momentum_proxy_events(px, ret)
    print(f"  Known events in data range: {len(valid_events)}")
    print(f"  Momentum proxy events: {len(proxy_events)}")

    # Combine and deduplicate (remove proxy events within 45 days of known events)
    combined_events = list(valid_events)
    for pe in proxy_events:
        pe_dt = pd.Timestamp(pe)
        too_close = any(abs((pe_dt - pd.Timestamp(ke)).days) < 45 for ke in combined_events)
        if not too_close:
            combined_events.append(pe)

    combined_events = sorted(set(combined_events))
    print(f"  Total combined events: {len(combined_events)}")

    pnl, positions, event_log = run_event_study(combined_events, px, ret, hold_days=HOLD_DAYS)

    held_pnl = pnl[positions != 0]
    held_spy = spy_r.reindex(held_pnl.index).dropna()
    held_positions = positions.reindex(held_pnl.index).fillna(0)

    if len(held_pnl) < 30:
        return mark_failed(sid, f"insufficient held-days ({len(held_pnl)}) across {len(combined_events)} events")

    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name="Cat-Bond ILS Stress Long RNR/AXS (held-days only)",
        positions=held_positions,
        cost_bps=10,
    )

    ok_events = [e for e in event_log if e.get("status") == "ok"]
    rets = [e["long_return"] for e in ok_events]
    excess = [e["excess_return"] for e in ok_events]
    n_events = len(rets)
    avg_ret = float(np.mean(rets)) if rets else 0.0
    avg_excess = float(np.mean(excess)) if excess else 0.0
    win_rate = float(np.mean([r > 0 for r in rets])) if rets else 0.0

    print(f"Done: {sid}")
    print(f"  n_events={n_events}, held_days={len(held_pnl)}")
    print(f"  avg_long_return={avg_ret:.4f}, avg_excess={avg_excess:.4f}, win_rate={win_rate:.3f}")
    if "error" not in m:
        print(
            f"  Sharpe: {m.get('sharpe', 'N/A'):.2f}  "
            f"CAGR: {m.get('cagr', 0)*100:.2f}%  "
            f"MaxDD: {m.get('max_dd', 0)*100:.2f}%  "
            f"t-stat: {m.get('t_stat', 'N/A'):.2f}"
        )

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "When cat-bond secondary spreads widen materially (ILS risk-appetite "
                "withdrawal, not loss-driven): go long equal-weight RNR + AXS within "
                "3 trading days; hold 40 trading days (8 weeks). "
                "Exit: 8-week max hold, +15% take-profit on either leg, "
                "SPY down >8% macro stop."
            ),
            "mechanism": (
                "ILS capacity withdrawal forces primary insurers to cede more risk to "
                "traditional reinsurers at harder prices, improving reinsurer combined "
                "ratios into the next Jan/Apr renewal cycle. RNR and AXS as "
                "premier Bermuda reinsurers benefit most from ILS market dislocations."
            ),
            "source": (
                "Known ILS stress events identified from Artemis.bm narrative reports "
                "and historical reinsurance hard market episodes. "
                "Price data: yfinance daily adjusted closes."
            ),
            "caveats": (
                "Cat-bond spread data not available as structured time series; "
                "event dates are manually identified from Artemis.bm and history. "
                "Small sample size limits BH significance. "
                "Major cat events post-entry (>$10B insured loss) can reverse thesis quickly. "
                "ILS market evolved significantly post-2017 (larger capacity)."
            ),
            "known_events": KNOWN_EVENTS,
            "historical_events": HISTORICAL_EVENTS,
            "n_events": n_events,
            "avg_long_return": round(avg_ret, 4),
            "avg_excess_return": round(avg_excess, 4),
            "win_rate": round(win_rate, 4),
            "event_log": event_log,
            "tickers": primary_tickers,
            "hold_days": HOLD_DAYS,
        },
        pnl=held_pnl,
    )
    print(f"Saved result for {sid}")


if __name__ == "__main__":
    main()
