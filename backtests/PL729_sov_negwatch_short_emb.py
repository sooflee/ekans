"""PL729_sov_negwatch_short_emb
Sovereign Negative-Watch Cluster -> Short EMB (Counter-Signal)

When 2+ rating agencies (S&P, Moody's, Fitch) place the same EM sovereign on
negative watch within a rolling 30-day window, short EMB for 30 trading days.

This is a counter-signal to long EM credit — coordinated negative watch signals
from multiple agencies precede spread widening and index weight reduction.

Hand-coded sovereign rating action events 2015-2025 from agency press releases.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# -------- Hand-coded EM Sovereign Negative Watch Events 2015-2025 --------
# Source: S&P Global, Moody's, Fitch Ratings press releases
# Each entry = one agency action: negative watch or negative outlook
# placed on an EM sovereign's foreign-currency government bonds.
#
# We look for clusters: 2+ agencies acting on the SAME sovereign within 30 days.
RATING_ACTIONS = [
    # 2015: Brazil downgrade cycle
    {"date": "2015-09-09", "sovereign": "Brazil", "agency": "S&P",    "action": "downgrade_to_junk"},
    {"date": "2015-10-07", "sovereign": "Brazil", "agency": "Fitch",  "action": "negative_watch"},
    {"date": "2016-02-24", "sovereign": "Brazil", "agency": "Moody's","action": "downgrade_to_junk"},
    # 2016: South Africa
    {"date": "2016-12-02", "sovereign": "South Africa", "agency": "S&P",    "action": "negative_watch"},
    {"date": "2016-12-07", "sovereign": "South Africa", "agency": "Fitch",  "action": "negative_watch"},
    # 2018: Turkey crisis
    {"date": "2018-08-17", "sovereign": "Turkey", "agency": "S&P",    "action": "negative_watch"},
    {"date": "2018-08-17", "sovereign": "Turkey", "agency": "Fitch",  "action": "negative_watch"},
    {"date": "2018-08-20", "sovereign": "Turkey", "agency": "Moody's","action": "negative_watch"},
    # 2018: Argentina
    {"date": "2018-05-09", "sovereign": "Argentina", "agency": "S&P",    "action": "negative_watch"},
    {"date": "2018-05-15", "sovereign": "Argentina", "agency": "Fitch",  "action": "negative_watch"},
    # 2020: Ecuador COVID default
    {"date": "2020-03-19", "sovereign": "Ecuador", "agency": "S&P",    "action": "negative_watch"},
    {"date": "2020-03-20", "sovereign": "Ecuador", "agency": "Fitch",  "action": "negative_watch"},
    # 2020: Zambia
    {"date": "2020-03-26", "sovereign": "Zambia", "agency": "Moody's","action": "negative_watch"},
    {"date": "2020-04-01", "sovereign": "Zambia", "agency": "S&P",    "action": "negative_watch"},
    # 2020: Lebanon
    {"date": "2020-02-11", "sovereign": "Lebanon", "agency": "S&P",    "action": "default_rating"},
    {"date": "2020-02-12", "sovereign": "Lebanon", "agency": "Fitch",  "action": "default_rating"},
    # 2022: Sri Lanka
    {"date": "2022-03-19", "sovereign": "Sri Lanka", "agency": "S&P",    "action": "negative_watch"},
    {"date": "2022-03-24", "sovereign": "Sri Lanka", "agency": "Fitch",  "action": "negative_watch"},
    {"date": "2022-04-01", "sovereign": "Sri Lanka", "agency": "Moody's","action": "negative_watch"},
    # 2022: Ukraine (invasion)
    {"date": "2022-02-25", "sovereign": "Ukraine", "agency": "S&P",    "action": "negative_watch"},
    {"date": "2022-02-25", "sovereign": "Ukraine", "agency": "Fitch",  "action": "negative_watch"},
    {"date": "2022-02-28", "sovereign": "Ukraine", "agency": "Moody's","action": "negative_watch"},
    # 2023: Pakistan
    {"date": "2023-01-27", "sovereign": "Pakistan", "agency": "S&P",    "action": "negative_watch"},
    {"date": "2023-02-10", "sovereign": "Pakistan", "agency": "Fitch",  "action": "negative_watch"},
    # 2024: Egypt
    {"date": "2024-02-01", "sovereign": "Egypt", "agency": "Fitch",  "action": "negative_watch"},
    {"date": "2024-02-15", "sovereign": "Egypt", "agency": "Moody's","action": "negative_watch"},
    # 2024: Ethiopia
    {"date": "2024-01-11", "sovereign": "Ethiopia", "agency": "S&P",    "action": "negative_watch"},
    {"date": "2024-01-12", "sovereign": "Ethiopia", "agency": "Fitch",  "action": "negative_watch"},
]


def find_clusters(actions, window_days=30, min_agencies=2):
    """Find dates where min_agencies+ agencies act on the same sovereign within window_days.
    Returns list of cluster trigger dates (the date when the threshold is crossed),
    deduped to avoid overlapping signals within 30 days.
    """
    df = pd.DataFrame(actions)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    clusters = []
    seen_windows = set()  # track (sovereign, trigger_date) to avoid duplicates

    for i, row in df.iterrows():
        sov = row["sovereign"]
        d = row["date"]
        window_start = d - pd.Timedelta(days=window_days)
        # Count unique agencies acting on this sovereign within window
        mask = (df["sovereign"] == sov) & (df["date"] >= window_start) & (df["date"] <= d)
        agencies_in_window = df[mask]["agency"].nunique()
        if agencies_in_window >= min_agencies:
            key = (sov, d.date())
            if key not in seen_windows:
                # Check no cluster within the last 30 days for same sovereign
                recent = any(
                    abs((d - pd.Timestamp(str(prev_date))).days) <= 30
                    for prev_sov, prev_date in seen_windows if prev_sov == sov
                )
                if not recent:
                    clusters.append({
                        "trigger_date": d,
                        "sovereign": sov,
                        "n_agencies": agencies_in_window,
                    })
                    seen_windows.add(key)

    return clusters


def run_event_study(clusters, ret, ticker="EMB", hold_days=30):
    """Simulate SHORT entry on next trading day after trigger, hold hold_days days."""
    idx = ret.index
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)
    asset_ret = ret[ticker].fillna(0)

    event_log = []
    for cluster in clusters:
        trigger = cluster["trigger_date"]
        future = idx[idx > trigger]
        if len(future) == 0:
            event_log.append({**cluster, "trigger_date": str(trigger.date()), "status": "no_data"})
            continue
        entry_dt = future[0]
        entry_pos = idx.get_loc(entry_dt)
        exit_pos = min(entry_pos + hold_days, len(idx))

        # SHORT EMB: negative of EMB return
        slice_r = asset_ret.iloc[entry_pos:exit_pos]
        short_ret = float((1 + (-slice_r)).prod() - 1) if len(slice_r) else None

        ev_rec = {
            "trigger_date": str(trigger.date()),
            "sovereign": cluster["sovereign"],
            "n_agencies": cluster["n_agencies"],
            "entry_date": str(entry_dt.date()),
            "exit_date": str(idx[exit_pos - 1].date()) if exit_pos > entry_pos else None,
            "n_hold_days": int(exit_pos - entry_pos),
            "event_return_short": round(short_ret, 4) if short_ret is not None else None,
            "emb_raw_return": round(float((1 + slice_r).prod() - 1), 4) if len(slice_r) else None,
        }
        event_log.append(ev_rec)

        for j in range(entry_pos, exit_pos):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = -1.0
                pnl.iloc[j] = -asset_ret.iloc[j]  # short

    return pnl, positions, event_log


def main():
    sid = "PL729_sov_negwatch_short_emb"
    ticker = "EMB"
    tickers = [ticker, "SPY"]

    try:
        px = load_prices(tickers, start="2014-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    clusters = find_clusters(RATING_ACTIONS, window_days=30, min_agencies=2)
    print(f"Clusters found: {len(clusters)}")
    for c in clusters:
        print(f"  {c['trigger_date'].date()}: {c['sovereign']} ({c['n_agencies']} agencies)")

    pnl, positions, event_log = run_event_study(clusters, ret, ticker=ticker, hold_days=30)

    n_events = sum(1 for e in event_log if e.get("entry_date"))
    print(f"Events with price data: {n_events}")

    held_pnl = pnl[positions != 0]
    held_spy = spy_r.reindex(held_pnl.index).dropna()

    if len(held_pnl) < 30:
        event_rets = [e["event_return_short"] for e in event_log if e.get("event_return_short") is not None]
        return mark_failed(
            sid,
            f"insufficient held days: {len(held_pnl)} (n_events={n_events})",
            extra={
                "events": event_log,
                "rule": "2+ agencies negative-watch same EM sovereign 30 days -> short EMB 30 days",
            },
        )

    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name="Sovereign Neg-Watch Cluster Short EMB (held-days only)",
        positions=positions.reindex(held_pnl.index).fillna(0),
        cost_bps=10,
    )

    event_rets = [e["event_return_short"] for e in event_log if e.get("event_return_short") is not None]
    event_summary = {
        "n_events": len(event_rets),
        "avg_event_return": round(float(np.mean(event_rets)), 4) if event_rets else None,
        "median_event_return": round(float(np.median(event_rets)), 4) if event_rets else None,
        "win_rate": round(float(np.mean([r > 0 for r in event_rets])), 4) if event_rets else None,
        "best": round(float(np.max(event_rets)), 4) if event_rets else None,
        "worst": round(float(np.min(event_rets)), 4) if event_rets else None,
    }

    def event_vs_spy(log, spy_ret):
        rows = []
        for e in log:
            entry = e.get("entry_date"); exit_d = e.get("exit_date")
            if not entry or not exit_d:
                continue
            entry_dt = pd.Timestamp(entry); exit_dt = pd.Timestamp(exit_d)
            spy_slice = spy_ret.loc[entry_dt:exit_dt]
            if len(spy_slice):
                spy_cum = float((1 + spy_slice).prod() - 1)
                rows.append({
                    "trigger_date": e["trigger_date"],
                    "sovereign": e["sovereign"],
                    "short_emb_return": e.get("event_return_short"),
                    "spy_return": round(spy_cum, 4),
                })
        return rows

    excess = event_vs_spy(event_log, spy_r)

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "When 2+ rating agencies (S&P, Moody's, Fitch) place the same EM "
                "sovereign on negative watch within a rolling 30-day window, short "
                "EMB at the next session open; hold 30 trading days; cover at close."
            ),
            "mechanism": (
                "Coordinated negative-watch actions by 2+ agencies signal imminent "
                "downgrade risk and forced selling by index-constrained funds "
                "(investment grade mandates must exit sub-IG holdings). EMB tracks "
                "the JP Morgan EMBI Global index — a major EM sovereign downgrade "
                "or default increases credit spreads across the index, depressing "
                "EMB price. The 30-day hold captures the immediate spread widening "
                "window before the market fully prices in the action."
            ),
            "source": (
                "S&P Global, Moody's, Fitch Ratings press releases 2015-2025; "
                "EMB/SPY prices via yfinance (auto_adjust=True)."
            ),
            "events": event_log,
            "event_summary": event_summary,
            "event_vs_spy": excess,
            "n_events": n_events,
            "caveats": (
                "Rating action dates are hand-coded approximations — actual timing "
                "requires agency press release timestamps. Some actions (e.g. Ukraine "
                "2022) are predictable from geopolitical events visible before the "
                "formal rating action. Short EMB has financing costs and may have "
                "borrow-cost drag not captured in the backtest. Multiple EM credit "
                "crises cluster in time (e.g. COVID 2020), creating correlated "
                "simultaneous events that amplify apparent signal strength."
            ),
        },
        pnl=held_pnl,
    )

    print(f"Done: {sid}")
    print(f"  n_events={n_events}, event_summary={event_summary}")
    if "error" not in m:
        print(
            f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
            f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
        )
        if "oos_sharpe" in m:
            print(f"  OOS Sharpe: {m['oos_sharpe']:.2f}")


if __name__ == "__main__":
    main()
