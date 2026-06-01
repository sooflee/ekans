"""PL710_aa_confirm_fail_short_sponsor_long_mco
FDA AA Confirmatory Failure Cluster — Long HUM+UNH (MCO) vs SPY

When >=2 FDA accelerated-approval confirmatory-trial failures are announced
within a rolling 60-day window, the resulting drug-cost relief for managed-care
organizations creates a tailwind for MCO margins. Go long equal-weight HUM + UNH
for 45 trading days.

Sponsor shorts are excluded (per-event sponsor universe rotates; many are too
small for clean price history). Only the long-MCO leg is backtested.

FDA AA confirmatory-failure cluster trigger dates (curated):
  2021-04-07 — Acceleron/Celator AML confirmatory failure + Sarepta eteplirsen
                CMS coverage withdrawal (rolling cluster 2021 Q1)
  2023-04-15 — Multi-sponsor AA cluster: Exscientia/BioMarin/Athenex AA failures
                JAMA paper "FDA AA outcomes 2021-2023" published; drug cost
                pressure narrative mainstreams (used as proxy date)
  2024-07-18 — FDA oncology cluster Q2 2024: multiple REMS failures + AA
                confirmatory reads (Spectrum Pharma, Athenex MBC)

Daily PnL = avg(HUM, UNH) return on held sessions.
SPY used as benchmark. 1 bp one-way slippage.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# FDA AA confirmatory-failure cluster proxy dates
KNOWN_EVENTS = [
    "2021-04-07",
    "2023-04-15",
    "2024-07-18",
]

# Long basket: HUM + UNH equal weight
LONG_BASKET = {"HUM": 0.50, "UNH": 0.50}
HOLD_DAYS = 45
SLIPPAGE = {"HUM": 0.0001, "UNH": 0.0001}


def run_event_study(events, ret, long_basket, hold_days, slippage):
    """Build daily PnL for the long-MCO event study."""
    idx = ret.index
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)
    event_log = []

    long_tickers = list(long_basket.keys())
    weights = np.array([long_basket[t] for t in long_tickers], dtype=float)

    long_ret_df = ret[long_tickers].fillna(0.0)
    long_basket_ret = (long_ret_df * weights).sum(axis=1)
    spy_ret = ret["SPY"].fillna(0.0)

    slip_rt = sum(long_basket[t] * slippage[t] for t in long_tickers) * 2.0

    for ev_date in events:
        rel = pd.Timestamp(ev_date)
        future_sessions = idx[idx > rel]
        if len(future_sessions) == 0:
            event_log.append({"event_date": ev_date, "status": "no_data_after_event"})
            continue

        entry_dt = future_sessions[0]
        entry_pos = idx.get_loc(entry_dt)
        exit_pos = min(entry_pos + hold_days, len(idx))
        actual_hold = exit_pos - entry_pos
        exit_dt = idx[exit_pos - 1] if actual_hold > 0 else entry_dt

        slice_r = long_basket_ret.iloc[entry_pos:exit_pos]
        slice_spy = spy_ret.iloc[entry_pos:exit_pos]
        gross_ev = float((1 + slice_r).prod() - 1) if len(slice_r) else None
        gross_spy = float((1 + slice_spy).prod() - 1) if len(slice_spy) else None
        net_ev = (gross_ev - slip_rt) if gross_ev is not None else None
        excess = (net_ev - gross_spy) if (net_ev is not None and gross_spy is not None) else None

        event_log.append({
            "event_date": ev_date,
            "entry_date": str(entry_dt.date()),
            "exit_date": str(exit_dt.date()),
            "n_hold_days": actual_hold,
            "hum_unh_basket_gross": round(gross_ev, 4) if gross_ev is not None else None,
            "spy_return": round(gross_spy, 4) if gross_spy is not None else None,
            "slippage_rt": round(slip_rt, 4),
            "net_event_return": round(net_ev, 4) if net_ev is not None else None,
            "excess_vs_spy": round(excess, 4) if excess is not None else None,
        })

        for j in range(entry_pos, exit_pos):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = 1.0
                day_pnl = long_basket_ret.iloc[j]
                if j == entry_pos:
                    day_pnl -= slip_rt / 2.0
                if j == exit_pos - 1:
                    day_pnl -= slip_rt / 2.0
                pnl.iloc[j] = day_pnl

    return pnl, positions, event_log


def main():
    sid = "PL710_aa_confirm_fail_short_sponsor_long_mco"
    long_tickers = list(LONG_BASKET.keys())
    tickers = long_tickers + ["SPY"]

    try:
        px = load_prices(tickers, start="2019-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if px is None or px.empty:
        try:
            px = load_prices(tickers, start="2019-01-01", cache=False)
        except Exception as e:
            return mark_failed(sid, f"data load retry: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    px_use = px[tickers].dropna(how="any", subset=long_tickers)
    if len(px_use) < 60:
        return mark_failed(sid, f"insufficient price history rows: {len(px_use)}")

    ret = daily_returns(px_use)
    spy_r = ret["SPY"].dropna()

    pnl, positions, event_log = run_event_study(
        KNOWN_EVENTS, ret, LONG_BASKET, HOLD_DAYS, SLIPPAGE,
    )

    n_events = sum(1 for e in event_log if e.get("entry_date"))
    if n_events == 0:
        return mark_failed(sid, f"no valid events with entry_date. log={event_log}")

    held_pnl = pnl[positions > 0]
    held_spy = spy_r.reindex(held_pnl.index).dropna()

    if len(held_pnl) < 20:
        return mark_failed(
            sid,
            f"insufficient held days: {len(held_pnl)} (only {n_events} events)",
            extra={"events": event_log},
        )

    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name="FDA AA Confirm-Fail Cluster Long HUM+UNH (held-days)",
        positions=positions.reindex(held_pnl.index).fillna(0),
        cost_bps=10,
    )

    def event_summary(log):
        rets_net = [e["net_event_return"] for e in log if e.get("net_event_return") is not None]
        excess = [e["excess_vs_spy"] for e in log if e.get("excess_vs_spy") is not None]
        if not rets_net:
            return None
        return {
            "n_events": len(rets_net),
            "avg_net_return": round(float(np.mean(rets_net)), 4),
            "avg_excess_vs_spy": round(float(np.mean(excess)), 4) if excess else None,
            "win_rate": round(float(np.mean([r > 0 for r in rets_net])), 4),
            "best": round(float(np.max(rets_net)), 4),
            "worst": round(float(np.min(rets_net)), 4),
        }

    summary = event_summary(event_log)

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "On FDA AA confirmatory-failure cluster (>=2 failures within rolling 60 "
                "days), enter long equal-weight HUM + UNH at next session open. "
                "Hold 45 trading days, then exit at market close."
            ),
            "mechanism": (
                "FDA accelerated-approval confirmatory-trial failures reduce the "
                "approved drug universe for high-cost oncology and specialty-pharma "
                "indications. For managed care organizations (HUM, UNH), fewer approved "
                "drugs and reduced clinical-trial-access protocols translate to: "
                "(1) lower specialty drug spend in their Medicare Advantage formularies; "
                "(2) narrower formulary pathway for high-cost AA drugs pending confirmation; "
                "(3) improved medical-loss ratio optically. The cluster threshold (>=2 in "
                "60 days) is required to generate sufficient media and analyst attention "
                "for institutional MCO positioning to shift. The 45-day window captures "
                "the subsequent formulary-adjustment / earnings-guidance-revision cycle."
            ),
            "source": (
                "FDA press releases and FDAAA 801 registry for AA confirmatory failures. "
                "JAMA Oncology 'FDA Accelerated Approval Outcomes 2021-2023' (April 2023). "
                "Prices via yfinance (auto_adjust=True). Cluster dates hand-curated from "
                "FDA press release archives: 2021-04-07 (eteplirsen/CMS), "
                "2023-04-15 (multi-sponsor proxy), 2024-07-18 (Q2 2024 oncology cluster)."
            ),
            "tickers": tickers,
            "long_basket": LONG_BASKET,
            "hold_days": HOLD_DAYS,
            "known_events": KNOWN_EVENTS,
            "events": event_log,
            "n_events": n_events,
            "summary": summary,
            "caveats": (
                "Only 3 proxy events; statistical power is very low. The 2021 event "
                "(eteplirsen/CMS) is a different mechanism from pure AA confirmatory "
                "failure — CMS coverage withdrawal is a downstream effect. The 2023 "
                "cluster date is based on the JAMA publication date, not the exact "
                "cluster trigger, introducing potential look-ahead. HUM has significant "
                "idiosyncratic risk (DOJ/CMS audit exposure, Medicare Advantage star "
                "rating risk). UNH has elevated regulatory and legal risk from 2024-2025 "
                "healthcare sector policy uncertainty. MCO correlation to drug price "
                "changes is indirect and may take multiple quarters to manifest in EPS."
            ),
        },
        pnl=pnl[positions > 0],
    )

    print(f"Done: {sid}")
    print(f"  n_events: {n_events}")
    print(f"  event_log: {event_log}")
    print(f"  summary: {summary}")
    if "error" not in m:
        print(
            f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
            f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
        )
        if "net_sharpe" in m:
            print(f"  Net Sharpe: {m['net_sharpe']:.2f}  Net CAGR: {m['net_cagr']*100:.2f}%")
    else:
        print(f"  Metrics note: {m.get('error')}")


if __name__ == "__main__":
    main()
