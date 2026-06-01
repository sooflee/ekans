"""PL707_mica_casp_grant_long_coin
MiCA CASP Authorization Grant — Long COIN for 10 Trading Days

When EU competent authority grants a MiCA Title V CASP (Crypto Asset Service
Provider) authorization to a Coinbase-affiliated entity, go long COIN for
10 trading days from the announcement date. SPY used as benchmark.

Known CASP authorization events:
  2024-12-04 — Coinbase Ireland CASP authorization (CBI grants MiCA license)

Note: Very sparse events; this backtest should be viewed as a proof-of-concept
framework. Expanding to other crypto platform CASP grants (Binance.EU, Kraken,
etc.) could widen the sample.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# MiCA CASP authorization dates for Coinbase-affiliated entities
# 2024-12-04: Coinbase Ireland granted CBI/MiCA CASP license
KNOWN_EVENTS = [
    "2024-12-04",
]

HOLD_DAYS = 10
SLIPPAGE_BPS = 5   # 5 bps one-way for COIN (liquid large-cap crypto equity)


def run_event_study(events, ret, hold_days, slippage_bps):
    """Build daily PnL for the COIN long event study."""
    idx = ret.index
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)
    event_log = []

    slip_one_way = slippage_bps / 10000.0
    slip_rt = slip_one_way * 2.0  # round-trip

    coin_ret = ret["COIN"].fillna(0.0)
    spy_ret = ret["SPY"].fillna(0.0)

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

        slice_coin = coin_ret.iloc[entry_pos:exit_pos]
        slice_spy = spy_ret.iloc[entry_pos:exit_pos]
        gross_coin = float((1 + slice_coin).prod() - 1) if len(slice_coin) else None
        gross_spy = float((1 + slice_spy).prod() - 1) if len(slice_spy) else None
        net_coin = (gross_coin - slip_rt) if gross_coin is not None else None
        excess = (net_coin - gross_spy) if (net_coin is not None and gross_spy is not None) else None

        event_log.append({
            "event_date": ev_date,
            "entry_date": str(entry_dt.date()),
            "exit_date": str(exit_dt.date()),
            "n_hold_days": actual_hold,
            "coin_return_gross": round(gross_coin, 4) if gross_coin is not None else None,
            "spy_return": round(gross_spy, 4) if gross_spy is not None else None,
            "slippage_rt": round(slip_rt, 4),
            "coin_return_net": round(net_coin, 4) if net_coin is not None else None,
            "excess_vs_spy": round(excess, 4) if excess is not None else None,
        })

        for j in range(entry_pos, exit_pos):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = 1.0
                day_pnl = coin_ret.iloc[j]
                if j == entry_pos:
                    day_pnl -= slip_one_way
                if j == exit_pos - 1:
                    day_pnl -= slip_one_way
                pnl.iloc[j] = day_pnl

    return pnl, positions, event_log


def main():
    sid = "PL707_mica_casp_grant_long_coin"
    tickers = ["COIN", "SPY"]

    try:
        px = load_prices(tickers, start="2021-04-14")  # COIN IPO date
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if px is None or px.empty:
        try:
            px = load_prices(tickers, start="2021-04-14", cache=False)
        except Exception as e:
            return mark_failed(sid, f"data load retry: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    px_use = px[tickers].dropna(how="any")
    if len(px_use) < 60:
        return mark_failed(sid, f"insufficient price history rows: {len(px_use)}")

    ret = daily_returns(px_use)
    spy_r = ret["SPY"].dropna()

    pnl, positions, event_log = run_event_study(
        KNOWN_EVENTS, ret, HOLD_DAYS, SLIPPAGE_BPS,
    )

    n_events = sum(1 for e in event_log if e.get("entry_date"))
    if n_events == 0:
        return mark_failed(sid, f"no valid events with entry_date. log={event_log}")

    held_pnl = pnl[positions > 0]
    held_spy = spy_r.reindex(held_pnl.index).dropna()

    if len(held_pnl) < 5:
        return mark_failed(
            sid,
            f"insufficient held days: {len(held_pnl)} (only {n_events} events, {HOLD_DAYS}-day hold)",
            extra={"events": event_log},
        )

    # With only 1 event x 10 days, compute_metrics may flag "insufficient data"
    # We still attempt it and capture the result
    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name="MiCA CASP Grant Long COIN (held-days)",
        positions=positions.reindex(held_pnl.index).fillna(0),
        cost_bps=10,
    )

    def event_summary(log):
        rets = [e["coin_return_net"] for e in log if e.get("coin_return_net") is not None]
        excesses = [e["excess_vs_spy"] for e in log if e.get("excess_vs_spy") is not None]
        if not rets:
            return None
        return {
            "n_events": len(rets),
            "avg_net_return": round(float(np.mean(rets)), 4),
            "avg_excess_vs_spy": round(float(np.mean(excesses)), 4) if excesses else None,
            "win_rate": round(float(np.mean([r > 0 for r in rets])), 4),
            "best": round(float(np.max(rets)), 4),
            "worst": round(float(np.min(rets)), 4),
        }

    summary = event_summary(event_log)

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "On EU competent-authority MiCA Title V CASP authorization grant to a "
                "Coinbase-affiliated entity, enter long COIN at next session open. "
                "Hold 10 trading days, then close at market."
            ),
            "mechanism": (
                "MiCA CASP authorization grants formal legal status to crypto exchanges "
                "in the EU, enabling retail-facing services across all 27 member states "
                "under passporting. For Coinbase specifically, an Irish CBI license "
                "represents the primary EU passporting route. Institutional traders "
                "interpret CASP grants as a regulatory de-risking catalyst: forward-looking "
                "EU AUM flows become more bankable, custody mandates from EU institutions "
                "become accessible, and regulatory tail-risk is reduced. The 10-day window "
                "captures the initial institutional repositioning on the catalyst."
            ),
            "source": (
                "EU competent-authority (CBI Ireland) CASP authorization register. "
                "ESMA MiCA supervisory convergence publications. Prices via yfinance "
                "(auto_adjust=True). COIN IPO April 14 2021 (NASDAQ). "
                "Event: 2024-12-04 Coinbase Ireland CASP authorization."
            ),
            "tickers": tickers,
            "hold_days": HOLD_DAYS,
            "slippage_bps": SLIPPAGE_BPS,
            "known_events": KNOWN_EVENTS,
            "events": event_log,
            "n_events": n_events,
            "summary": summary,
            "caveats": (
                "Only 1 qualifying event in the test window — backtest has no statistical "
                "validity whatsoever. COIN is extremely volatile (100%+ annual vol) and "
                "10-day returns are highly noisy. MiCA CASP grants for Coinbase may have "
                "been anticipated and priced in before the formal announcement date. "
                "Expanding the signal to cover all major exchange CASP grants (Binance.EU, "
                "Kraken Europe, etc.) would widen the sample but introduces selection bias "
                "as not all exchanges benefit symmetrically. The strategy should only be "
                "considered with a broader multi-event dataset."
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
    else:
        print(f"  Metrics note: {m.get('error')}")


if __name__ == "__main__":
    main()
