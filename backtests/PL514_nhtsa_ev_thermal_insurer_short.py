"""PL514_nhtsa_ev_thermal_insurer_short
NHTSA EV Battery Thermal Event Surge -> Short PGR/ALL vs Long CPRT

On each curated NHTSA EV-battery-thermal-event acceleration date (NHTSA EWR
quarterly filing release, NHTSA recall campaign opening, or HLDI bulletin
showing >50% YoY rise in EV battery thermal/fire field reports), enter at
the next session open a dollar-neutral pair:
  - Short basket 50% PGR + 50% ALL  (auto insurers absorbing EV body/total-loss claim cost)
  - Long equal-notional CPRT        (salvage auctioneer that benefits from
                                     total-loss volume and EV battery write-offs).
Hold 126 trading days (~2 quarters / 6 months); exit at close.

Daily PnL = CPRT return MINUS (0.5 * PGR return + 0.5 * ALL return) on held days.
Metrics computed on held days only (overlapping windows merged - no stacking).
Per-leg one-way slippage applied: 5 bps on each of CPRT/PGR/ALL (large-cap, liquid).

Event dates curated from NHTSA EWR quarterly publication calendar, NHTSA OVSC
EV battery fire recall investigation openings (TSLA Model S/X 2020; GM Bolt
LG cell fires 2020-2021; Ford Mach-E and F-150 Lightning 2022-2023; Rivian
R1T 2023), and HLDI EV bulletins 2020-2024.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


KNOWN_EVENTS = [
    "2020-10-13",
    "2021-02-08",
    "2021-08-13",
    "2022-02-18",
    "2022-09-23",
    "2023-02-16",
    "2023-04-14",
    "2023-08-25",
    "2023-12-13",
    "2024-04-26",
    "2024-08-13",
    "2024-12-13",
    "2025-04-25",
    "2025-08-15",
]

# Long leg: CPRT (weight 1.0)
LONG_TICKER = "CPRT"
# Short basket weights (sum to 1.0 of short notional)
SHORT_BASKET = {"PGR": 0.5, "ALL": 0.5}

# One-way slippage per leg (decimal)
SLIPPAGE = {"CPRT": 0.0005, "PGR": 0.0005, "ALL": 0.0005}

HOLD_DAYS = 126


def run_event_study(events, ret, long_ticker, short_basket, hold_days):
    """Build daily PnL & a 1/0 position series for a long-short event study.

    Spread daily return = long_ret - sum(weight_i * short_ret_i)
    Overlapping events are not stacked (book size capped at 1).
    Round-trip slippage is amortized half on entry day, half on exit day.
    """
    idx = ret.index
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)
    event_log = []

    short_tickers = list(short_basket.keys())
    short_weights = np.array([short_basket[t] for t in short_tickers], dtype=float)

    long_ret = ret[long_ticker].fillna(0.0)
    short_ret_df = ret[short_tickers].fillna(0.0)
    short_basket_ret = (short_ret_df * short_weights).sum(axis=1)

    # Spread = long CPRT - short (0.5 PGR + 0.5 ALL)
    spread_ret = long_ret - short_basket_ret

    # Round-trip slippage cost as % of book
    # Long leg: 1.0 * slip * 2; Short basket legs: sum(w_i * slip_i) * 2
    slip_round_trip = (
        SLIPPAGE[long_ticker]
        + sum(short_basket[t] * SLIPPAGE[t] for t in short_tickers)
    ) * 2.0

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

        slice_spread = spread_ret.iloc[entry_pos:exit_pos]
        slice_long = long_ret.iloc[entry_pos:exit_pos]
        slice_short = short_basket_ret.iloc[entry_pos:exit_pos]
        gross_ev = float((1 + slice_spread).prod() - 1) if len(slice_spread) else None
        gross_long = float((1 + slice_long).prod() - 1) if len(slice_long) else None
        gross_short = float((1 + slice_short).prod() - 1) if len(slice_short) else None
        net_ev = gross_ev - slip_round_trip if gross_ev is not None else None

        event_log.append({
            "event_date": ev_date,
            "entry_date": str(entry_dt.date()),
            "exit_date": str(exit_dt.date()),
            "n_hold_days": int(exit_pos - entry_pos),
            "long_cprt_return": round(gross_long, 4) if gross_long is not None else None,
            "short_basket_return": round(gross_short, 4) if gross_short is not None else None,
            "gross_event_return": round(gross_ev, 4) if gross_ev is not None else None,
            "slippage": round(slip_round_trip, 4),
            "net_event_return": round(net_ev, 4) if net_ev is not None else None,
        })

        # Fill daily PnL on the held window; cap at 1 book unit
        for j in range(entry_pos, exit_pos):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = 1.0
                day_pnl = spread_ret.iloc[j]
                if j == entry_pos:
                    day_pnl -= slip_round_trip / 2.0
                if j == exit_pos - 1:
                    day_pnl -= slip_round_trip / 2.0
                pnl.iloc[j] = day_pnl

    return pnl, positions, event_log


def main():
    sid = "PL514_nhtsa_ev_thermal_insurer_short"
    # Trading tickers (in basket) + benchmark
    trade_tickers = [LONG_TICKER] + list(SHORT_BASKET.keys())
    tickers = trade_tickers + ["SPY"]

    try:
        # Events start 2020; load from 2019 to give buffer for returns.
        px = load_prices(tickers, start="2019-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    px_use = px[tickers].dropna(how="any", subset=trade_tickers)
    if len(px_use) < 60:
        return mark_failed(sid, f"insufficient price history rows: {len(px_use)}")

    ret = daily_returns(px_use)
    spy_r = ret["SPY"].dropna()

    pnl, positions, event_log = run_event_study(
        KNOWN_EVENTS, ret, LONG_TICKER, SHORT_BASKET, HOLD_DAYS,
    )

    n_events = sum(1 for e in event_log if e.get("entry_date"))
    if n_events == 0:
        return mark_failed(sid, "no valid events with entry_date")

    # Restrict metrics to held days
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
        name="NHTSA EV Thermal Event: Long CPRT / Short PGR+ALL (held-days)",
        positions=positions.reindex(held_pnl.index).fillna(0),
        cost_bps=10,
    )

    # Event-level summary
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
                "On each curated NHTSA EV battery thermal event acceleration date "
                "(NHTSA EWR quarterly filing release, NHTSA OVSC recall campaign on "
                "EV battery fires for TSLA/GM/F/RIVN/LCID, or HLDI bulletin showing "
                ">50% YoY rise in EV battery thermal/fire field reports), enter at "
                "next session open a dollar-neutral pair: short 50% PGR + 50% ALL, "
                "long equal-notional CPRT. Hold 126 trading days (~2 quarters); exit "
                "at close."
            ),
            "mechanism": (
                "EV battery thermal events drive an asymmetric P&L shock for auto "
                "insurers vs salvage auctioneers. Insurers (PGR, ALL) face elevated "
                "total-loss frequency and high battery-pack replacement costs (often "
                "exceeding vehicle ACV), forcing combined-ratio deterioration and "
                "reserve strengthening over 1-2 quarters as claims develop. CPRT, by "
                "contrast, monetizes EV total losses through its salvage auction "
                "marketplace: more thermal-loss vehicles -> more units sold -> higher "
                "fee revenue and ASP. The 126-day window captures the EWR -> recall -> "
                "claims development -> earnings revision propagation."
            ),
            "source": (
                "NHTSA Early Warning Reporting (EWR) quarterly publications, NHTSA "
                "OVSC EV recall investigation openings, HLDI EV thermal/fire "
                "bulletins 2020-2024. Prices via yfinance (auto_adjust=True). "
                "Per-leg one-way slippage: 5 bps on CPRT/PGR/ALL."
            ),
            "tickers": trade_tickers,
            "long_ticker": LONG_TICKER,
            "short_basket": SHORT_BASKET,
            "hold_days": HOLD_DAYS,
            "known_events": KNOWN_EVENTS,
            "events": event_log,
            "n_events": n_events,
            "summary": summary,
            "excess_vs_spy": excess_vs_spy,
            "caveats": (
                "Event dates curated by hand from NHTSA EWR/recall calendar and HLDI "
                "bulletins; different curators could pick different dates, raising "
                "look-ahead risk. The 126-day window is long enough that overlapping "
                "events (several occur within months of each other) merge into a "
                "near-continuously-held book over 2022-2025; the no-stacking cap "
                "limits double-counting but the lack of flat periods reduces the "
                "event-study cleanliness. CPRT, PGR and ALL all have substantial "
                "non-EV-specific drivers (used-car prices, auto insurance pricing "
                "cycle, catastrophe loss season) that contribute most realized "
                "variance. Statistical t-stat caveat: 14 events x 126 days has heavy "
                "auto-correlation within events. SPY benchmark is buy-and-hold over "
                "held days only."
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
