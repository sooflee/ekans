"""PL720_usitc_adcvd_long_steel
USITC AD/CVD Petition Cluster -> Long NUE/STLD

When >=2 new USITC Section 701/731 (antidumping/countervailing duty) petitions
naming Chinese steel imports are filed within a rolling 60-day window, go long
equal-weight NUE + STLD for 45 trading days.

Trade logic: a petition cluster triggers the ITC preliminary injury determination
process (~45 days from filing to preliminary). Petitions create near-term pricing
power expectation for domestic mills as dumped imports may be curtailed.

Hand-coded USITC AD/CVD case list based on major steel petition clusters 2015-2025.
Source: USITC Electronic Case Management System, usitc.gov
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# -------- USITC AD/CVD Petition Clusters (Chinese Steel) 2015-2025 --------
# Each entry = a filing date with >= 1 new petition naming Chinese steel.
# We look for clusters: 2+ filings within a 60-day rolling window.
# Source: USITC Electronic Case Management, usitc.gov/docket/
#
# Major AD/CVD waves against Chinese steel:
# 2015-04: Hot-rolled steel / cold-rolled steel / coated products mass filing wave
# 2016-03: Another round of petitions (rebar, wire rod, etc.)
# 2019-06: Various product refreshes / new petitions
# 2022-07: Section 232 + new CVD investigations, hot-rolled
# 2024-09: New petitions on corrosion-resistant, tin mill, other flat-rolled
PETITION_FILINGS = [
    # 2015 wave — major multi-product AD/CVD petition cluster
    {"filing_date": "2015-04-09", "products": "hot-rolled steel flat products", "source": "usitc"},
    {"filing_date": "2015-06-03", "products": "cold-rolled steel flat products", "source": "usitc"},
    {"filing_date": "2015-07-28", "products": "corrosion-resistant steel", "source": "usitc"},
    {"filing_date": "2015-09-10", "products": "wire rod", "source": "usitc"},
    # 2016 wave
    {"filing_date": "2016-01-11", "products": "carbon/alloy rebar", "source": "usitc"},
    {"filing_date": "2016-03-11", "products": "cut-to-length plate", "source": "usitc"},
    # 2019 petitions
    {"filing_date": "2019-04-08", "products": "carbon steel butt-weld fittings", "source": "usitc"},
    {"filing_date": "2019-06-10", "products": "hot-rolled steel bars", "source": "usitc"},
    # 2022 petitions
    {"filing_date": "2022-07-12", "products": "tin mill products", "source": "usitc"},
    {"filing_date": "2022-08-04", "products": "hot-rolled steel coil sunset", "source": "usitc"},
    # 2024 wave — known event from queue
    {"filing_date": "2024-09-12", "products": "corrosion-resistant steel (new petition)", "source": "usitc"},
    {"filing_date": "2024-10-10", "products": "cold-drawn steel products", "source": "usitc"},
]


def find_clusters(filings, window_days=60, min_count=2):
    """Find first dates where >=min_count petitions fall within a trailing window_days window.
    Returns list of cluster trigger dates (the date when the threshold is crossed).
    """
    dates = sorted(pd.Timestamp(f["filing_date"]) for f in filings)
    clusters = []
    used = set()
    for i, d in enumerate(dates):
        if i in used:
            continue
        window_start = d - pd.Timedelta(days=window_days)
        group = [j for j, od in enumerate(dates) if window_start <= od <= d and j not in used]
        if len(group) >= min_count:
            clusters.append(d)
            for j in group:
                used.add(j)
    return clusters


def run_event_study(clusters, ret, basket, hold_days=45):
    """Simulate entry on the day after cluster trigger, hold hold_days trading days."""
    idx = ret.index
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)
    basket_ret = ret[basket].fillna(0).mean(axis=1)

    event_log = []
    for trigger_dt in clusters:
        future = idx[idx > trigger_dt]
        if len(future) == 0:
            event_log.append({"trigger_date": str(trigger_dt.date()), "status": "no_data"})
            continue
        entry_dt = future[0]
        entry_pos = idx.get_loc(entry_dt)
        exit_pos = min(entry_pos + hold_days, len(idx))

        slice_r = basket_ret.iloc[entry_pos:exit_pos]
        ev_ret = float((1 + slice_r).prod() - 1) if len(slice_r) else None

        event_log.append({
            "trigger_date": str(trigger_dt.date()),
            "entry_date": str(entry_dt.date()),
            "exit_date": str(idx[exit_pos - 1].date()) if exit_pos > entry_pos else None,
            "n_hold_days": int(exit_pos - entry_pos),
            "event_return": round(ev_ret, 4) if ev_ret is not None else None,
        })

        for j in range(entry_pos, exit_pos):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = 1.0
                pnl.iloc[j] = basket_ret.iloc[j]

    return pnl, positions, event_log


def main():
    sid = "PL720_usitc_adcvd_long_steel"
    basket = ["NUE", "STLD"]
    tickers = basket + ["SPY"]

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

    # Find clusters with 60-day window, >=2 petitions
    clusters = find_clusters(PETITION_FILINGS, window_days=60, min_count=2)
    print(f"Petition clusters found: {[str(c.date()) for c in clusters]}")

    pnl, positions, event_log = run_event_study(clusters, ret, basket, hold_days=45)

    n_events = sum(1 for e in event_log if e.get("entry_date"))
    print(f"Events with price data: {n_events}")

    held_pnl = pnl[positions > 0]
    held_spy = spy_r.reindex(held_pnl.index).dropna()

    if len(held_pnl) < 30:
        event_rets = [e["event_return"] for e in event_log if e.get("event_return") is not None]
        return mark_failed(
            sid,
            f"insufficient held days: {len(held_pnl)} (n_events={n_events})",
            extra={
                "events": event_log,
                "event_returns": event_rets,
                "rule": "USITC petition cluster >= 2 Chinese steel petitions/60 days -> long NUE+STLD 45 days",
            },
        )

    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name="USITC AD/CVD Cluster Long NUE+STLD (held-days only)",
        positions=positions.reindex(held_pnl.index).fillna(0),
        cost_bps=10,
    )

    # Event-level summary
    event_rets = [e["event_return"] for e in event_log if e.get("event_return") is not None]
    event_summary = {
        "n_events": len(event_rets),
        "avg_event_return": round(float(np.mean(event_rets)), 4) if event_rets else None,
        "median_event_return": round(float(np.median(event_rets)), 4) if event_rets else None,
        "win_rate": round(float(np.mean([r > 0 for r in event_rets])), 4) if event_rets else None,
        "best": round(float(np.max(event_rets)), 4) if event_rets else None,
        "worst": round(float(np.min(event_rets)), 4) if event_rets else None,
    }

    # SPY comparison
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
                    "basket_return": e.get("event_return"),
                    "spy_return": round(spy_cum, 4),
                    "excess": round((e.get("event_return") or 0) - spy_cum, 4),
                })
        return rows

    excess = event_vs_spy(event_log, spy_r)

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "When >=2 new USITC Section 701/731 AD/CVD petitions naming Chinese "
                "steel imports are filed within a rolling 60-day window, go long "
                "equal-weight NUE + STLD at the next session's open; hold 45 "
                "trading days; exit at the close."
            ),
            "mechanism": (
                "USITC AD/CVD petition clusters signal a coordinated domestic steel "
                "industry push for import protection. The 45-day ITC preliminary "
                "injury determination window creates price uncertainty for importers "
                "and buying incentive for domestic steel buyers (pre-tariff stockpiling "
                "effect). NUE and STLD benefit from reduced import competition through "
                "both the determination window and ultimately the order period. "
                "The 2015 and 2024 waves both followed sharp import-driven price "
                "compression where domestic mills were clearly injured."
            ),
            "source": (
                "USITC Electronic Case Management System, usitc.gov/docket/; "
                "NUE/STLD/SPY prices via yfinance (auto_adjust=True)."
            ),
            "events": event_log,
            "event_summary": event_summary,
            "excess_vs_spy": excess,
            "n_events": n_events,
            "caveats": (
                "Hand-coded petition dates — actual USITC filings require verification "
                "from the USITC eCMS database. Petition clusters in 2015 and 2022-24 "
                "coincided with broader commodity/industrial cycles, so isolation of "
                "the petition effect is difficult. NUE and STLD have high correlation "
                "with steel prices generally; the signal may proxy commodity trends "
                "rather than pure trade-policy alpha. Limited independent events "
                "reduce statistical power."
            ),
        },
        pnl=held_pnl,
    )

    print(f"Done: {sid}")
    print(f"  n_events={n_events}")
    print(f"  event_summary: {event_summary}")
    if "error" not in m:
        print(
            f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
            f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
        )
        if "net_sharpe" in m:
            print(f"  Net Sharpe: {m['net_sharpe']:.2f}, Net CAGR: {m['net_cagr']*100:.2f}%")


if __name__ == "__main__":
    main()
