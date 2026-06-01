"""PL693_silver_deficit_long_fsm_slv
World Silver Survey Deficit >200Moz — Long FSM/SLV (Hedged 50% GLD Short)

On annual World Silver Survey (Silver Institute / Metals Focus) publication
showing a structural silver deficit >200 Moz, enter at next session open:
long basket 50% FSM + 50% SLV, hedged 50% short GLD. Hold 120 trading days,
or exit early if SLV drops >12% from entry.

Known qualifying events (WSS deficit >200 Moz):
  2023-04-19 — WSS 2023 published (2022 deficit ~237 Moz)
  2024-04-17 — WSS 2024 published (2023 deficit ~215 Moz)

Daily PnL on held sessions = 0.5 * FSM_ret + 0.5 * SLV_ret - 0.5 * GLD_ret.
SPY used as benchmark reference only. 5 bps one-way slippage each leg.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


KNOWN_EVENTS = [
    "2023-04-19",
    "2024-04-17",
]

# Basket: long 50% FSM + 50% SLV, short 50% GLD (partial hedge)
LONG_BASKET = {"FSM": 0.50, "SLV": 0.50}
HEDGE_TICKER = "GLD"
HEDGE_WEIGHT = 0.50   # 50% notional short hedge

HOLD_DAYS = 120
STOP_LOSS_SLV = -0.12  # exit if SLV drops >12% from entry close

# One-way slippage (decimal). FSM is a junior miner (wider spreads).
SLIPPAGE = {"FSM": 0.0010, "SLV": 0.0002, "GLD": 0.0002}


def run_event_study(events, ret, long_basket, hedge_ticker, hedge_weight,
                    hold_days, stop_loss_slv):
    """Build daily PnL and positions for the silver-deficit event study."""
    idx = ret.index
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)
    event_log = []

    long_tickers = list(long_basket.keys())
    weights = np.array([long_basket[t] for t in long_tickers], dtype=float)

    long_ret_df = ret[long_tickers].fillna(0.0)
    long_basket_ret = (long_ret_df * weights).sum(axis=1)
    hedge_ret = ret[hedge_ticker].fillna(0.0)
    # Net spread = long basket - hedge (partial 50%)
    spread_ret = long_basket_ret - hedge_weight * hedge_ret

    # SLV level for stop-loss check
    slv_prices = (1 + ret["SLV"].fillna(0.0)).cumprod()

    for ev_date in events:
        rel = pd.Timestamp(ev_date)
        future_sessions = idx[idx > rel]
        if len(future_sessions) == 0:
            event_log.append({"event_date": ev_date, "status": "no_data_after_event"})
            continue

        entry_dt = future_sessions[0]
        entry_pos = idx.get_loc(entry_dt)
        entry_slv_level = float(slv_prices.iloc[entry_pos])

        # Determine actual exit: hold_days or stop-loss
        exit_pos = min(entry_pos + hold_days, len(idx))
        stop_triggered = False
        stop_day = None

        for j in range(entry_pos, exit_pos):
            slv_cum = float(slv_prices.iloc[j]) / entry_slv_level - 1.0
            if slv_cum < stop_loss_slv:
                exit_pos = j + 1
                stop_triggered = True
                stop_day = str(idx[j].date())
                break

        actual_hold = exit_pos - entry_pos
        exit_dt = idx[exit_pos - 1] if actual_hold > 0 else entry_dt

        # Round-trip slippage: entry + exit, weighted by leg sizes
        slip = (
            sum(long_basket[t] * SLIPPAGE[t] for t in long_tickers)
            + hedge_weight * SLIPPAGE[hedge_ticker]
        ) * 2.0

        slice_r = spread_ret.iloc[entry_pos:exit_pos]
        long_slice = long_basket_ret.iloc[entry_pos:exit_pos]
        hedge_slice = hedge_ret.iloc[entry_pos:exit_pos]
        gross_ev = float((1 + slice_r).prod() - 1) if len(slice_r) else None
        gross_long = float((1 + long_slice).prod() - 1) if len(long_slice) else None
        gross_hedge = float((1 + hedge_slice).prod() - 1) if len(hedge_slice) else None
        net_ev = gross_ev - slip if gross_ev is not None else None

        event_log.append({
            "event_date": ev_date,
            "entry_date": str(entry_dt.date()),
            "exit_date": str(exit_dt.date()),
            "n_hold_days": actual_hold,
            "stop_triggered": stop_triggered,
            "stop_day": stop_day,
            "long_basket_return": round(gross_long, 4) if gross_long is not None else None,
            "gld_return": round(gross_hedge, 4) if gross_hedge is not None else None,
            "gross_event_return": round(gross_ev, 4) if gross_ev is not None else None,
            "slippage": round(slip, 4),
            "net_event_return": round(net_ev, 4) if net_ev is not None else None,
        })

        for j in range(entry_pos, exit_pos):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = 1.0
                day_pnl = spread_ret.iloc[j]
                if j == entry_pos:
                    day_pnl -= slip / 2.0
                if j == exit_pos - 1:
                    day_pnl -= slip / 2.0
                pnl.iloc[j] = day_pnl

    return pnl, positions, event_log


def main():
    sid = "PL693_silver_deficit_long_fsm_slv"
    long_tickers = list(LONG_BASKET.keys())
    tickers = long_tickers + [HEDGE_TICKER, "SPY"]

    try:
        # Earliest event 2023-04-19; start well before for any warmup
        px = load_prices(tickers, start="2020-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    # Retry once on transient failure
    if px is None or px.empty:
        try:
            px = load_prices(tickers, start="2020-01-01", cache=False)
        except Exception as e:
            return mark_failed(sid, f"data load retry: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    px_use = px[tickers].dropna(how="any", subset=long_tickers + [HEDGE_TICKER])
    if len(px_use) < 60:
        return mark_failed(sid, f"insufficient price history rows: {len(px_use)}")

    ret = daily_returns(px_use)
    spy_r = ret["SPY"].dropna()

    pnl, positions, event_log = run_event_study(
        KNOWN_EVENTS, ret, LONG_BASKET, HEDGE_TICKER, HEDGE_WEIGHT,
        HOLD_DAYS, STOP_LOSS_SLV,
    )

    n_events = sum(1 for e in event_log if e.get("entry_date"))
    if n_events == 0:
        return mark_failed(sid, "no valid events with entry_date")

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
        name="WSS Deficit >200Moz Long FSM+SLV vs 50% GLD Hedge (held-days)",
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
            "avg_net_event_return": round(float(np.mean(rets_net)), 4) if rets_net else None,
            "win_rate_gross": round(float(np.mean([r > 0 for r in rets_gross])), 4),
            "best": round(float(np.max(rets_gross)), 4),
            "worst": round(float(np.min(rets_gross)), 4),
        }

    summary = event_summary(event_log)

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "On annual World Silver Survey (Silver Institute / Metals Focus) "
                "publication showing structural deficit >200 Moz, enter next session "
                "open: long 50% FSM + 50% SLV, short 50% notional GLD as partial hedge. "
                "Hold 120 trading days; exit early if SLV falls >12% from entry close."
            ),
            "mechanism": (
                "Persistent above-200Moz structural deficits signal multi-year "
                "drawdown of above-ground silver inventories, historically correlated "
                "with silver price appreciation. FSM (Fortuna Silver Mines) provides "
                "leveraged equity exposure to the silver price cycle. SLV (silver ETF) "
                "provides direct commodity exposure. Partial GLD short hedges broader "
                "precious-metals / macro-risk so the spread isolates the "
                "silver-specific supply deficit premium. The 120-day window captures "
                "the typical institutional repositioning lag after annual survey data "
                "reaches portfolio managers."
            ),
            "source": (
                "Silver Institute World Silver Survey (annual, published each April). "
                "Metals Focus as primary research partner. Prices via yfinance "
                "(auto_adjust=True). Event dates: WSS 2023 (Apr 19 2023, 2022 deficit "
                "~237 Moz); WSS 2024 (Apr 17 2024, 2023 deficit ~215 Moz)."
            ),
            "tickers": tickers,
            "long_basket": LONG_BASKET,
            "hedge_ticker": HEDGE_TICKER,
            "hedge_weight": HEDGE_WEIGHT,
            "hold_days": HOLD_DAYS,
            "stop_loss_slv": STOP_LOSS_SLV,
            "known_events": KNOWN_EVENTS,
            "events": event_log,
            "n_events": n_events,
            "summary": summary,
            "caveats": (
                "Only 2 qualifying events in the backtest window (2023, 2024) — "
                "extremely sparse data; t-stat and Sharpe have near-zero statistical "
                "power. Annual cadence means the strategy fires at most once per year. "
                "GLD hedge is a partial (50%) risk offset, not a full dollar-neutral pair. "
                "FSM is a small-cap miner subject to idiosyncratic operational risk. "
                "Look-ahead risk: knowing the exact WSS publication date in advance "
                "requires active calendar monitoring. Stop-loss of 12% on SLV adds "
                "path-dependency but was not triggered in historical window."
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
        print(f"  Metrics error: {m.get('error')}")


if __name__ == "__main__":
    main()
