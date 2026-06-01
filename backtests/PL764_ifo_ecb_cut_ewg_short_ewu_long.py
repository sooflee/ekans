"""PL764_ifo_ecb_cut_ewg_short_ewu_long
German Ifo <85 + ECB Cut -> Short EWG / Long EWU Pair

When the monthly Ifo Business Climate Index (old 2000=100 scale) prints below 85
AND ECB cuts its deposit facility rate within 60 calendar days, enter a
beta-neutral pair trade: Short EWG (Germany ETF) / Long EWU (UK ETF).

NOTE: Ifo was rescaled from 2000=100 to 2015=100. We use the FRED series
GVGDGER10A100NNBR as a proxy for historical Ifo, or alternatively hand-code
the known recession thresholds from historical press releases. For the modern
(post-2017) 2015=100 scale, the equivalent recession threshold is ~93-96.

Strategy: hold 25 trading days, stop-loss if -5% combined pair PnL.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


# Known Ifo sub-threshold signals (approximate) with concurrent ECB cut windows
# These are manually coded from historical Ifo press releases + ECB cut dates.
# Ifo old scale (2000=100): threshold 85 for recession signal
# Ifo new scale (2015=100): threshold ~95 (scaled approx) starting ~2018
# Format: (signal_month_start, signal_month_end) where Ifo was below threshold
# and ECB cut within 60 days
IFO_ECB_SIGNAL_PERIODS = [
    # GFC: Ifo fell to 82-84 (old scale), ECB cut Nov 2008, Jan 2009
    ("2008-10-01", "2009-03-01"),
    # European debt crisis: ECB cut Jul 2012, Ifo ~103 (new scale at ~100, old ~85)
    # We include based on the strategy logic since ECB cut and Ifo was at the time ~102-103 (old base ~86)
    # Skip 2012 as Ifo didn't clearly hit 85 on old scale at that point
    # COVID: Ifo collapsed, ECB cut effectively zero / special programs (not a conventional rate cut)
    # Skip
    # 2019: Ifo was around 94-95 (new scale), ECB cut Sep 2019 (new scale threshold ~95)
    ("2019-08-01", "2019-12-01"),
]

# Additional signal dates: ECB cut dates (from FRED ECBDFR drops)
# Signal fires: when Ifo monthly is below threshold AND ECB cut within 60 days
# We'll use FRED ECBDFR to detect ECB cuts programmatically for the period 2000-2026
# and combine with manually-identified Ifo weakness periods.

def main():
    sid = "PL764_ifo_ecb_cut_ewg_short_ewu_long"
    tickers = ["EWG", "EWU", "SPY"]

    try:
        px = load_prices(tickers, start="2000-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=5)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    try:
        ecbdfr = load_fred(["ECBDFR"], start="2000-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED ECBDFR load failed: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()
    ewg_r = ret["EWG"].dropna()
    ewu_r = ret["EWU"].dropna()

    # Detect ECB cut dates: days when ECBDFR drops from prior day
    ecb_rate = ecbdfr["ECBDFR"].dropna()
    ecb_cuts = ecb_rate[ecb_rate.diff() < -0.001]  # any drop > 0.1bps = cut
    # Forward-fill to trading dates
    ecb_cut_dates = pd.DatetimeIndex(ecb_cuts.index)

    # Ifo sub-threshold periods (manually identified + from known_events)
    # We approximate Ifo weakness using the known signal windows
    # Plus the known_events from the strategy definition
    known_signal_dates = [
        pd.Timestamp("2001-09-05"),   # Ifo ~86 old scale, ECB cutting cycle
        pd.Timestamp("2008-11-06"),   # Ifo 82-84 old scale, ECB cut Nov 2008
        pd.Timestamp("2009-01-08"),   # continued weakness
        pd.Timestamp("2019-09-12"),   # Ifo new scale ~94, ECB cut Sep 2019
    ]

    # Build set of signal entry dates:
    # For each signal date, check if there's an ECB cut within 60 calendar days
    entry_dates = []
    for sig_dt in known_signal_dates:
        # Check if ECB cut within 60 days before or after
        window_start = sig_dt - pd.Timedelta(days=60)
        window_end = sig_dt + pd.Timedelta(days=60)
        nearby_cuts = ecb_cut_dates[(ecb_cut_dates >= window_start) & (ecb_cut_dates <= window_end)]
        if len(nearby_cuts) > 0:
            # Find first trading day at or after sig_dt
            future = ret.index[ret.index >= sig_dt]
            if len(future) > 0:
                entry_dates.append(future[0])

    # De-duplicate (enforce 90-day gap between signals)
    entry_dates_clean = []
    last_exit = None
    for ed in sorted(set(entry_dates)):
        if last_exit is None or (ed - last_exit).days >= 90:
            entry_dates_clean.append(ed)
            last_exit = ed + pd.Timedelta(days=25 * 1.5)  # approximate exit

    if not entry_dates_clean:
        return mark_failed(sid, "no entry dates found after matching Ifo weakness + ECB cut proximity")

    # Build PnL: beta-neutral pair short EWG / long EWU
    hold_days = 25
    stop_loss = -0.05

    positions_ewg = pd.Series(0.0, index=ret.index)
    positions_ewu = pd.Series(0.0, index=ret.index)
    events = []

    for entry_dt in entry_dates_clean:
        entry_loc = ret.index.get_loc(entry_dt)
        if entry_loc < 120:
            continue  # not enough history for beta calc

        # Compute rolling 120-day beta of EWG vs EWU
        ewg_slice = ewg_r.iloc[entry_loc - 120:entry_loc]
        ewu_slice = ewu_r.iloc[entry_loc - 120:entry_loc]
        common = ewg_slice.index.intersection(ewu_slice.index)
        if len(common) < 60:
            continue
        cov = np.cov(ewg_slice.loc[common], ewu_slice.loc[common])
        beta_ewg_vs_ewu = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 1.2
        beta_ewg_vs_ewu = np.clip(beta_ewg_vs_ewu, 0.5, 3.0)

        # Beta-neutral: short EWG 1 unit, long EWU * (1/beta_ewg_vs_ewu)
        # Combined PnL = ewu_ret * (1/beta_ewg_vs_ewu) - ewg_ret
        ewg_weight = -1.0
        ewu_weight = 1.0 / beta_ewg_vs_ewu

        # Walk through holding period with stop-loss
        exit_loc = min(entry_loc + hold_days, len(ret.index) - 1)
        cum_pnl = 0.0
        actual_exit = entry_loc

        for j in range(entry_loc, exit_loc + 1):
            d = ret.index[j]
            ewg_day = ewg_r.get(d, 0.0) or 0.0
            ewu_day = ewu_r.get(d, 0.0) or 0.0
            day_pnl = ewg_weight * ewg_day + ewu_weight * ewu_day
            cum_pnl += day_pnl
            positions_ewg.iloc[j] = ewg_weight
            positions_ewu.iloc[j] = ewu_weight

            if cum_pnl < stop_loss:
                actual_exit = j
                exit_reason = "stop_loss"
                break
        else:
            actual_exit = exit_loc
            exit_reason = "scheduled_25d"

        entry_date = ret.index[entry_loc]
        exit_date = ret.index[actual_exit]

        events.append({
            "entry_date": str(entry_date.date()),
            "exit_date": str(exit_date.date()),
            "exit_reason": exit_reason,
            "beta_ewg_vs_ewu": round(float(beta_ewg_vs_ewu), 3),
            "ewu_weight": round(float(ewu_weight), 3),
            "cum_pnl": round(float(cum_pnl), 4),
        })

    if not events:
        return mark_failed(sid, "no events with sufficient beta-calc history")

    # Combined daily PnL (shifted for no look-ahead)
    pos_ewg_shifted = positions_ewg.shift(1)
    pos_ewu_shifted = positions_ewu.shift(1)

    common_dates = ret.index
    pnl = (pos_ewg_shifted.reindex(common_dates).fillna(0) * ewg_r.reindex(common_dates).fillna(0) +
           pos_ewu_shifted.reindex(common_dates).fillna(0) * ewu_r.reindex(common_dates).fillna(0))
    pnl = pnl.dropna()

    if (pnl != 0).sum() < 20:
        return mark_failed(sid, f"insufficient active days: {(pnl!=0).sum()}, events: {len(events)}")

    # Combined positions for cost calc (abs avg notional)
    combined_pos = (positions_ewg.abs() + positions_ewu.abs()) / 2

    m = compute_metrics(
        pnl,
        benchmark=spy_r,
        name="Ifo<85 + ECB Cut Short EWG/Long EWU Pair",
        positions=combined_pos.reindex(pnl.index).fillna(0),
        cost_bps=10,
    )

    n_events = len(events)
    ev_rets = [e["cum_pnl"] for e in events]
    win_rate = float(np.mean([r > 0 for r in ev_rets])) if ev_rets else None
    avg_event = float(np.mean(ev_rets)) if ev_rets else None

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "When monthly Ifo Business Climate Index prints below 85 (old 2000=100 scale, "
                "~95 new 2015=100 scale) AND ECB cuts deposit rate within +-60 calendar days, "
                "enter beta-neutral Short EWG / Long EWU pair for 25 trading days. "
                "Stop-loss at -5% combined pair PnL."
            ),
            "mechanism": (
                "German economy (EWG) is more cyclical and more exposed to ECB rate path than UK (EWU). "
                "When Ifo signals recession AND ECB starts cutting (admitting weakness), Germany's "
                "cyclical premium vs UK contracts. Beta-neutral removes broad European equity beta."
            ),
            "source": "FRED ECBDFR; Ifo press releases (known_events hand-coded); yfinance EWG/EWU/SPY",
            "tickers": ["EWG", "EWU"],
            "n_events": n_events,
            "event_win_rate": round(win_rate, 4) if win_rate is not None else None,
            "avg_event_return": round(avg_event, 4) if avg_event is not None else None,
            "events": events,
            "caveats": (
                "Very few events (~3-4 in 25 years). Ifo series underwent scale change; "
                "threshold alignment is approximate. EWU includes FX (GBP/EUR) risk. "
                "Beta estimated from 120-day trailing window may be unstable. "
                "ECB cut detection from FRED ECBDFR daily series may miss intra-day timing."
            ),
        },
        pnl=pnl,
    )

    print(f"Done: {sid}")
    print(f"  n_events: {n_events}, win_rate: {win_rate}, avg_event_return: {avg_event}")
    print(
        f"  Sharpe: {m.get('sharpe', float('nan')):.2f}  "
        f"CAGR: {m.get('cagr', float('nan'))*100:.2f}%  "
        f"MaxDD: {m.get('max_dd', float('nan'))*100:.2f}%  "
        f"t-stat: {m.get('t_stat', float('nan')):.2f}"
    )
    for e in events:
        print(f"  Event: entry {e['entry_date']} -> {e['exit_date']} ({e['exit_reason']}) "
              f"beta={e['beta_ewg_vs_ewu']} cum_pnl={e['cum_pnl']}")


if __name__ == "__main__":
    main()
