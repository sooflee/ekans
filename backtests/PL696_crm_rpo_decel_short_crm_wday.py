"""PL696_crm_rpo_decel_short_crm_wday
Salesforce RPO Decel + Channel Discount Surge — Short CRM/WDAY vs 50% Long XLK

When CRM quarterly Remaining Performance Obligation (RPO) growth decelerates
>300bps QoQ AND channel-partner surveillance indicates >25% stacked discounts
on Sales/Service Cloud SKUs, enter at next session open: short 50% CRM + 50%
WDAY (equal-weight) hedged 50% long XLK. Hold 40 trading days; stop if
CRM+WDAY outperform XLK by >5%.

Known qualifying events (CRM RPO decel >300bps in earnings release):
  2023-08-30 — CRM Q2 FY2024 earnings (RPO decel after hyper-growth)
  2024-05-29 — CRM Q1 FY2025 earnings (weaker RPO guidance)

Daily PnL = -(0.5 * CRM_ret + 0.5 * WDAY_ret) + 0.5 * XLK_ret on held days.
SPY used as reference benchmark. 2 bps one-way slippage per leg.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


KNOWN_EVENTS = [
    "2023-08-30",
    "2024-05-29",
]

# Short basket: short 50% CRM + 50% WDAY
SHORT_BASKET = {"CRM": 0.50, "WDAY": 0.50}
HEDGE_TICKER = "XLK"
HEDGE_WEIGHT = 0.50    # long 50% XLK as partial hedge

HOLD_DAYS = 40
STOP_OUT_THRESHOLD = 0.05   # stop if short basket outperforms XLK by >5%

SLIPPAGE = {"CRM": 0.0002, "WDAY": 0.0002, "XLK": 0.0001}


def run_event_study(events, ret, short_basket, hedge_ticker, hedge_weight,
                    hold_days, stop_threshold):
    """Build daily PnL for the CRM RPO decel event study.

    PnL = -(short_basket_ret) + hedge_weight * hedge_ret.
    Stop if cumulative (short_basket_cum - hedge_cum) > stop_threshold
    (i.e. shorted names are running against us relative to hedge).
    """
    idx = ret.index
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)
    event_log = []

    short_tickers = list(short_basket.keys())
    weights = np.array([short_basket[t] for t in short_tickers], dtype=float)

    short_ret_df = ret[short_tickers].fillna(0.0)
    short_basket_ret = (short_ret_df * weights).sum(axis=1)
    hedge_ret = ret[hedge_ticker].fillna(0.0)

    # Net daily pnl: we are short basket + long hedge fraction
    spread_ret = -short_basket_ret + hedge_weight * hedge_ret

    for ev_date in events:
        rel = pd.Timestamp(ev_date)
        future_sessions = idx[idx > rel]
        if len(future_sessions) == 0:
            event_log.append({"event_date": ev_date, "status": "no_data_after_event"})
            continue

        entry_dt = future_sessions[0]
        entry_pos = idx.get_loc(entry_dt)
        exit_pos = min(entry_pos + hold_days, len(idx))

        # Stop-loss check: track cumulative (short_basket - hedge) vs stop_threshold
        stop_triggered = False
        stop_day = None
        cum_short = 1.0
        cum_hedge = 1.0

        for j in range(entry_pos, exit_pos):
            cum_short *= (1 + short_basket_ret.iloc[j])
            cum_hedge *= (1 + hedge_ret.iloc[j])
            # Short basket running >5% ahead of hedge = adverse move -> stop
            if (cum_short / cum_hedge - 1.0) > stop_threshold:
                exit_pos = j + 1
                stop_triggered = True
                stop_day = str(idx[j].date())
                break

        actual_hold = exit_pos - entry_pos
        exit_dt = idx[exit_pos - 1] if actual_hold > 0 else entry_dt

        # Round-trip slippage
        slip = (
            sum(short_basket[t] * SLIPPAGE[t] for t in short_tickers)
            + hedge_weight * SLIPPAGE[hedge_ticker]
        ) * 2.0

        slice_r = spread_ret.iloc[entry_pos:exit_pos]
        short_slice = short_basket_ret.iloc[entry_pos:exit_pos]
        hedge_slice = hedge_ret.iloc[entry_pos:exit_pos]
        gross_ev = float((1 + slice_r).prod() - 1) if len(slice_r) else None
        gross_short = float((1 + short_slice).prod() - 1) if len(short_slice) else None
        gross_hedge = float((1 + hedge_slice).prod() - 1) if len(hedge_slice) else None
        net_ev = gross_ev - slip if gross_ev is not None else None

        event_log.append({
            "event_date": ev_date,
            "entry_date": str(entry_dt.date()),
            "exit_date": str(exit_dt.date()),
            "n_hold_days": actual_hold,
            "stop_triggered": stop_triggered,
            "stop_day": stop_day,
            "short_basket_return": round(gross_short, 4) if gross_short is not None else None,
            "xlk_return": round(gross_hedge, 4) if gross_hedge is not None else None,
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
    sid = "PL696_crm_rpo_decel_short_crm_wday"
    short_tickers = list(SHORT_BASKET.keys())
    tickers = short_tickers + [HEDGE_TICKER, "SPY"]

    try:
        px = load_prices(tickers, start="2020-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if px is None or px.empty:
        try:
            px = load_prices(tickers, start="2020-01-01", cache=False)
        except Exception as e:
            return mark_failed(sid, f"data load retry: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    px_use = px[tickers].dropna(how="any", subset=short_tickers + [HEDGE_TICKER])
    if len(px_use) < 60:
        return mark_failed(sid, f"insufficient price history rows: {len(px_use)}")

    ret = daily_returns(px_use)
    spy_r = ret["SPY"].dropna()

    pnl, positions, event_log = run_event_study(
        KNOWN_EVENTS, ret, SHORT_BASKET, HEDGE_TICKER, HEDGE_WEIGHT,
        HOLD_DAYS, STOP_OUT_THRESHOLD,
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
        name="CRM RPO Decel Short CRM+WDAY vs 50% XLK (held-days)",
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
                "When CRM quarterly RPO growth decelerates >300bps QoQ AND channel "
                "surveillance shows >25% stacked discounts on Sales/Service Cloud, "
                "enter at next session open: short 50% CRM + 50% WDAY, long 50% XLK "
                "as partial hedge. Hold 40 trading days; stop if short basket "
                "outperforms XLK hedge by >5%."
            ),
            "mechanism": (
                "Remaining Performance Obligation (RPO) deceleration signals slowing "
                "net new bookings and rising churn risk for SaaS horizontal platforms. "
                "Channel discount escalation above 25% stacked typically precedes "
                "recognition of weaker demand in subsequent quarters. WDAY is included "
                "as it shares similar enterprise software buying cycles and valuation "
                "sensitivity. Shorting XLK (50%) hedges out broad tech sector beta, "
                "leaving the spread to isolate enterprise-software-specific fundamental "
                "deterioration. 40-day window (~2 quarters of price digestion) captures "
                "the typical analyst estimate revision cycle post-earnings."
            ),
            "source": (
                "CRM 10-Q quarterly filings on EDGAR (RPO footnote). Channel discount "
                "intelligence from reseller price lists. Prices via yfinance "
                "(auto_adjust=True). Event dates: CRM Q2 FY2024 earnings 2023-08-30 "
                "(RPO decel observed); CRM Q1 FY2025 earnings 2024-05-29 (guidance cut)."
            ),
            "tickers": tickers,
            "short_basket": SHORT_BASKET,
            "hedge_ticker": HEDGE_TICKER,
            "hedge_weight": HEDGE_WEIGHT,
            "hold_days": HOLD_DAYS,
            "stop_out_threshold": STOP_OUT_THRESHOLD,
            "known_events": KNOWN_EVENTS,
            "events": event_log,
            "n_events": n_events,
            "summary": summary,
            "caveats": (
                "Only 2 qualifying events in the test window; t-stat and Sharpe have "
                "effectively zero statistical power. Channel discount surveillance is "
                "not systematically available in historical data — event selection is "
                "based on post-hoc identification of earnings dates where RPO metrics "
                "deteriorated. WDAY tracks a different fiscal calendar and may not "
                "have been impacted by CRM-specific pricing issues at the same dates. "
                "XLK hedge is partial (50%) and tech-sector risk is not fully neutralized. "
                "High-information-ratio short setups in single-name SaaS can face "
                "violent short squeezes on any positive announcement."
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
