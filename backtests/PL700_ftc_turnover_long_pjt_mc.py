"""PL700_ftc_turnover_long_pjt_mc
Polymarket FTC Chair Turnover >60% — Long PJT/MC/LAZ/HLI Boutique M&A vs 50% Short XLF

When prediction-market probability of FTC Chair replacement crosses 60%
(or on actual FTC chair transition) AND HSR second-request volume is elevated,
enter at next session open: long equal-weight PJT + MC + LAZ + HLI (boutique
M&A advisors), hedged 50% short XLF. Hold 60 trading days. Stop if basket
underperforms XLF hedge by >6%.

Historical FTC chair transition proxy dates:
  2017-01-25 — Maureen Ohlhausen confirmed acting FTC Chair (deregulatory shift)
  2021-06-15 — Lina Khan confirmed FTC Chair (enforcement regime shift)

Daily PnL = avg(PJT, MC, LAZ, HLI) returns - 0.5 * XLF return on held sessions.
SPY used as reference benchmark. 3 bps one-way slippage per leg.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


KNOWN_EVENTS = [
    "2017-01-25",
    "2021-06-15",
]

LONG_BASKET = {"PJT": 0.25, "MC": 0.25, "LAZ": 0.25, "HLI": 0.25}
HEDGE_TICKER = "XLF"
HEDGE_WEIGHT = 0.50

HOLD_DAYS = 60
STOP_LOSS_THRESHOLD = -0.06   # stop if basket underperforms XLF by >6% cumulative

SLIPPAGE = {"PJT": 0.0003, "MC": 0.0003, "LAZ": 0.0003, "HLI": 0.0003, "XLF": 0.0001}


def run_event_study(events, ret, long_basket, hedge_ticker, hedge_weight,
                    hold_days, stop_threshold):
    """Build daily PnL for the FTC-turnover boutique M&A event study."""
    idx = ret.index
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)
    event_log = []

    long_tickers = list(long_basket.keys())
    weights = np.array([long_basket[t] for t in long_tickers], dtype=float)

    long_ret_df = ret[long_tickers].fillna(0.0)
    long_basket_ret = (long_ret_df * weights).sum(axis=1)
    hedge_ret = ret[hedge_ticker].fillna(0.0)

    spread_ret = long_basket_ret - hedge_weight * hedge_ret

    for ev_date in events:
        rel = pd.Timestamp(ev_date)
        future_sessions = idx[idx > rel]
        if len(future_sessions) == 0:
            event_log.append({"event_date": ev_date, "status": "no_data_after_event"})
            continue

        # Check all long tickers are available at entry
        entry_dt = future_sessions[0]
        entry_pos = idx.get_loc(entry_dt)

        # Check for data availability
        available = all(t in ret.columns for t in long_tickers + [hedge_ticker])
        if not available:
            event_log.append({"event_date": ev_date, "status": "missing_tickers"})
            continue

        exit_pos = min(entry_pos + hold_days, len(idx))
        stop_triggered = False
        stop_day = None

        cum_long = 1.0
        cum_hedge = 1.0

        for j in range(entry_pos, exit_pos):
            cum_long *= (1 + long_basket_ret.iloc[j])
            cum_hedge *= (1 + hedge_ret.iloc[j])
            # Stop if basket underperforms hedge by >|stop_threshold|
            if (cum_long - cum_hedge) < stop_threshold:
                exit_pos = j + 1
                stop_triggered = True
                stop_day = str(idx[j].date())
                break

        actual_hold = exit_pos - entry_pos
        exit_dt = idx[exit_pos - 1] if actual_hold > 0 else entry_dt

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
            "xlf_return": round(gross_hedge, 4) if gross_hedge is not None else None,
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
    sid = "PL700_ftc_turnover_long_pjt_mc"
    long_tickers = list(LONG_BASKET.keys())
    tickers = long_tickers + [HEDGE_TICKER, "SPY"]

    try:
        # Start from 2015 to cover PJT IPO (Sept 2015) and all events
        px = load_prices(tickers, start="2015-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if px is None or px.empty:
        try:
            px = load_prices(tickers, start="2015-01-01", cache=False)
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
        HOLD_DAYS, STOP_LOSS_THRESHOLD,
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
        name="FTC Chair Turnover Long PJT+MC+LAZ+HLI vs 50% XLF (held-days)",
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
                "When FTC Chair transition occurs (or prediction market >60%), enter "
                "at next session open: long equal-weight PJT + MC + LAZ + HLI "
                "(boutique M&A advisors), short 50% XLF as financial sector hedge. "
                "Hold 60 trading days; stop if basket underperforms XLF by >6%."
            ),
            "mechanism": (
                "FTC chair transitions historically precede 12-18 month shifts in "
                "merger review philosophy (deregulatory or enforcement). Boutique M&A "
                "advisors (PJT, MC, LAZ, HLI) earn advisory fees that are driven by "
                "announced deal volume. A deregulatory transition (e.g., 2017 Ohlhausen) "
                "catalyzes pent-up M&A pipeline releases. An enforcement transition "
                "(e.g., 2021 Khan) initially suppresses announced deals but boutiques "
                "benefit from the complexity of navigating tougher review — more retainer "
                "work, more divestitures, more special-committee mandates. XLF short "
                "hedges the broad financial sector so the spread captures "
                "boutique-specific alpha relative to large-cap financials."
            ),
            "source": (
                "FTC chair confirmation dates from public records and news archives. "
                "Polymarket/Kalshi FTC-chair contract data available only from 2024+; "
                "historical events used as proxy. Prices via yfinance (auto_adjust=True). "
                "PJT IPO Sept 2015; MC IPO July 2007; LAZ IPO May 2005; HLI IPO 2007."
            ),
            "tickers": tickers,
            "long_basket": LONG_BASKET,
            "hedge_ticker": HEDGE_TICKER,
            "hedge_weight": HEDGE_WEIGHT,
            "hold_days": HOLD_DAYS,
            "stop_loss_threshold": STOP_LOSS_THRESHOLD,
            "known_events": KNOWN_EVENTS,
            "events": event_log,
            "n_events": n_events,
            "summary": summary,
            "caveats": (
                "Only 2 proxy historical events; Polymarket prediction-market signal "
                "is not available for pre-2024 periods. The two FTC transitions had "
                "very different regulatory philosophies (deregulatory vs enforcement) — "
                "lumping them into a single 'turnover' signal assumes both directions "
                "are positive for boutique advisors, which may not hold. PJT had "
                "limited price history in 2017 event (IPO was Sept 2015). XLF hedge "
                "is a crude proxy for financial sector risk; large bank exposure in "
                "XLF is not the right hedge for boutique advisors. Statistical power "
                "is essentially zero with 2 events."
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
