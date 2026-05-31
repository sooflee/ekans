"""PL512_lbnl_queue_ipp_short
LBNL 'Queued Up' Annual Release -> Short IPPs vs XLU

On each LBNL Queued Up annual interconnection-queue report release date, enter
at the next session open a short basket of independent power producers
(50% CWEN + 50% ORA) and long equal-notional XLU (sector hedge). Hold 126
trading days (~6 months) then exit at close. Dollar-neutral.

Daily PnL = XLU return MINUS IPP basket return (since we short the IPPs).

Mechanism: LBNL's annual Queued Up report documents the growing backlog of
solar/wind/storage projects in U.S. interconnection queues. The report
historically highlights how long queue times, withdrawal rates, and rising
network-upgrade costs erode IPP project economics — pressuring developer
margins/IRRs at marginal projects. XLU (a utility-heavy ETF dominated by
regulated utilities) is comparatively insulated. The spread isolates this
report-driven repricing of IPP economics.

Event dates curated from LBNL Queued Up annual release announcements:
  2022-04-12, 2023-04-06, 2024-04-10, 2025-04-15
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


KNOWN_EVENTS = [
    "2022-04-12",
    "2023-04-06",
    "2024-04-10",
    "2025-04-15",
]

# Basket weights (SHORT side - independent power producers)
SHORT_BASKET = {"CWEN": 0.50, "ORA": 0.50}
LONG_TICKER = "XLU"  # equal-notional long (weight 1.0)

# Per-leg one-way slippage (decimal). CWEN/ORA are mid-cap utilities/IPPs.
SLIPPAGE = {"CWEN": 0.0005, "ORA": 0.0005, "XLU": 0.0001}

HOLD_DAYS = 126


def run_event_study(events, ret, short_weights, long_ticker, hold_days):
    """Build daily PnL & positions for a long-short event study.

    Long XLU, short IPP basket. Spread return = long XLU return - short basket return.
    When events overlap, positions are NOT stacked (capped at 1 unit of book).
    """
    idx = ret.index
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)  # 1.0 when in a position (book size)
    event_log = []

    short_tickers = list(short_weights.keys())
    weights = np.array([short_weights[t] for t in short_tickers], dtype=float)

    short_ret_df = ret[short_tickers].fillna(0.0)
    short_basket_ret = (short_ret_df * weights).sum(axis=1)
    long_ret = ret[long_ticker].fillna(0.0)

    # Spread = long XLU - short IPP basket (dollar-neutral)
    spread_ret = long_ret - short_basket_ret

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
        exit_pos = min(entry_pos + hold_days, len(idx))
        exit_dt = idx[exit_pos - 1] if exit_pos > entry_pos else entry_dt

        # Slippage: one-way entry + one-way exit on each leg
        slip = (
            sum(short_weights[t] * SLIPPAGE[t] for t in short_tickers)
            + SLIPPAGE[long_ticker]  # long leg weight 1.0
        ) * 2.0  # round trip

        slice_r = spread_ret.iloc[entry_pos:exit_pos]
        long_slice = long_ret.iloc[entry_pos:exit_pos]
        short_slice = short_basket_ret.iloc[entry_pos:exit_pos]
        gross_ev = float((1 + slice_r).prod() - 1) if len(slice_r) else None
        gross_long = float((1 + long_slice).prod() - 1) if len(long_slice) else None
        gross_short = float((1 + short_slice).prod() - 1) if len(short_slice) else None
        net_ev = gross_ev - slip if gross_ev is not None else None

        event_log.append({
            "event_date": ev_date,
            "entry_date": str(entry_dt.date()),
            "exit_date": str(exit_dt.date()),
            "n_hold_days": int(exit_pos - entry_pos),
            "xlu_long_return": round(gross_long, 4) if gross_long is not None else None,
            "ipp_short_basket_return": round(gross_short, 4) if gross_short is not None else None,
            "gross_event_return": round(gross_ev, 4) if gross_ev is not None else None,
            "slippage": round(slip, 4),
            "net_event_return": round(net_ev, 4) if net_ev is not None else None,
        })

        for j in range(entry_pos, exit_pos):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = 1.0
                day_pnl = spread_ret.iloc[j]
                if j == entry_pos:
                    day_pnl -= slip / 2.0  # entry slip
                if j == exit_pos - 1:
                    day_pnl -= slip / 2.0  # exit slip
                pnl.iloc[j] = day_pnl

    return pnl, positions, event_log


def main():
    sid = "PL512_lbnl_queue_ipp_short"
    short_tickers = list(SHORT_BASKET.keys())
    tickers = short_tickers + [LONG_TICKER, "SPY"]

    try:
        # First event 2022-04-12; start 2021 to seed returns and have a buffer.
        px = load_prices(tickers, start="2021-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    px_use = px[tickers].dropna(how="any", subset=short_tickers + [LONG_TICKER])
    if len(px_use) < 60:
        return mark_failed(sid, f"insufficient price history rows: {len(px_use)}")

    ret = daily_returns(px_use)
    spy_r = ret["SPY"].dropna()

    pnl, positions, event_log = run_event_study(
        KNOWN_EVENTS, ret, SHORT_BASKET, LONG_TICKER, HOLD_DAYS,
    )

    n_events = sum(1 for e in event_log if e.get("entry_date"))
    if n_events == 0:
        return mark_failed(sid, "no valid events with entry_date")

    held_pnl = pnl[positions > 0]
    held_spy = spy_r.reindex(held_pnl.index).dropna()

    if len(held_pnl) < 30:
        return mark_failed(
            sid,
            f"insufficient held days: {len(held_pnl)}",
            extra={"events": event_log},
        )

    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name="LBNL Queued Up -> Long XLU / Short IPPs (held-days)",
        positions=positions.reindex(held_pnl.index).fillna(0),
        cost_bps=10,
    )

    def event_summary(log):
        rets_gross = [e["gross_event_return"] for e in log if e.get("gross_event_return") is not None]
        rets_net = [e["net_event_return"] for e in log if e.get("net_event_return") is not None]
        if not rets_gross:
            return None
        return {
            "n_events": len(rets_gross),
            "avg_gross_event_return": round(float(np.mean(rets_gross)), 4),
            "median_gross_event_return": round(float(np.median(rets_gross)), 4),
            "avg_net_event_return": round(float(np.mean(rets_net)), 4) if rets_net else None,
            "win_rate_gross": round(float(np.mean([r > 0 for r in rets_gross])), 4),
            "win_rate_net": round(float(np.mean([r > 0 for r in rets_net])), 4) if rets_net else None,
            "best": round(float(np.max(rets_gross)), 4),
            "worst": round(float(np.min(rets_gross)), 4),
        }

    summary = event_summary(event_log)

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
                    "event_date": e["event_date"],
                    "entry_date": entry,
                    "spread_return": e.get("net_event_return"),
                    "spy_return": round(spy_cum, 4),
                    "excess_vs_spy": round((e.get("net_event_return") or 0) - spy_cum, 4),
                })
        return rows

    excess_vs_spy = event_vs_spy(event_log, spy_r)

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "On each LBNL 'Queued Up' annual interconnection-queue report release "
                "date, enter at next session open a short basket of IPPs "
                "(50% CWEN + 50% ORA) and long equal-notional XLU. Hold 126 trading "
                "days (~6 months); exit at close. Dollar-neutral."
            ),
            "mechanism": (
                "LBNL's annual Queued Up report quantifies the growing U.S. "
                "interconnection-queue backlog: longer wait times, higher network-upgrade "
                "cost allocations, and elevated withdrawal rates. These factors compress "
                "project IRRs and slow deployment for independent power producers / "
                "renewable developers (CWEN, ORA). Regulated utilities in XLU are far "
                "less exposed (allowed-rate-of-return treatment, ratebase cost recovery). "
                "Shorting IPPs while long XLU isolates this queue-friction shock from "
                "broader utility/rate beta. 126 trading days captures the report -> "
                "guidance / regulatory-process propagation through the next earnings cycle."
            ),
            "source": (
                "LBNL Electricity Markets & Policy Group, 'Queued Up' annual report "
                "series (emp.lbl.gov/queues). Prices via yfinance (auto_adjust=True). "
                "CWEN/ORA one-way slip 5 bps; XLU 1 bp."
            ),
            "tickers": short_tickers + [LONG_TICKER],
            "short_basket": SHORT_BASKET,
            "long_ticker": LONG_TICKER,
            "hold_days": HOLD_DAYS,
            "known_events": KNOWN_EVENTS,
            "events": event_log,
            "n_events": n_events,
            "summary": summary,
            "excess_vs_spy": excess_vs_spy,
            "caveats": (
                "Very small event sample (n=4 curated annual releases, 2022-2025). "
                "LBNL Queued Up release dates are hand-curated; alternative curators "
                "might pick adjacent dates (the publication is announced via blog/email "
                "and not always on the report's stamped date). NEP excluded from the "
                "active basket (delisted in 2025), reducing pure-play IPP exposure; "
                "ORA (Ormat) has thermal-renewable / geothermal exposure that is less "
                "queue-sensitive than pure-solar IPPs — this is a conservative basket. "
                "The 126-day horizon is long and overlaps with macro/rate cycles; the "
                "spread is somewhat hedged but utility / IPP both share rate-sensitivity "
                "and the hedge is imperfect. Sharpe / CAGR computed on held days only, "
                "amplifying realized exposure but sensitive to small n. Statistical "
                "significance weak (4 events x 126 days = 504 daily obs, heavily "
                "auto-correlated within events)."
            ),
        },
        pnl=pnl[positions > 0],
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


if __name__ == "__main__":
    main()
