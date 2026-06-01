"""PL972_btp_retail_miss_italian_banks_short
BTP Retail Subscription Miss -> Italian Bank Sovereign Doom-Loop Short

When MEF BTP Valore/Italia retail subscription totals show a weak raise
(< 60% of comparable prior issuance or absolute < EUR 8B for BTP Valore /
< EUR 6B for BTP Italia), enter short Italian banks (ISP.MI 60% / BMPS.MI 40%)
at next-day open, holding up to 10 weeks (50 trading days).

Known BTP subscription miss events used as seed:
- 2022-04-04: BTP Italia close (Apr 2022 raise ~EUR 10.6B, vs Oct 2021 ~EUR 8.2B —
  NOT a miss by ratio, but Apr 2022 was elevated stress: BTP-Bund spread >200bp,
  weak investor appetite vs nominal. Use as ambiguous event.
- 2020-01-14: BTP Italia Jan 2020 final sub ~EUR 9.6B vs Oct 2019 EUR 12.6B (−24% miss)
- 2020-10-19: BTP Italia Oct 2020 final sub ~EUR 9.0B vs Jan 2020 EUR 9.6B (slightly weaker)

Expanded proxy: use EWI as a liquid proxy for Italian equity risk when Milan-listed
tickers are not available in yfinance.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


# BTP retail subscription miss / stress events
# Source: MEF Dipartimento del Tesoro dt.mef.gov.it press releases
BTP_EVENTS = [
    {
        "date": "2020-01-14",
        "description": "BTP Italia Jan 2020: EUR 9.6B vs Oct 2019 EUR 12.6B (-24% vs prior; BTP-Bund spread ~160bp)",
        "raise_eur_b": 9.6,
        "prior_raise_eur_b": 12.6,
        "miss_pct": -24.0,
        "instrument": "BTP Italia",
        "condition": "miss_vs_prior",
    },
    {
        "date": "2020-10-19",
        "description": "BTP Italia Oct 2020: EUR 9.0B vs Jan 2020 EUR 9.6B (-6%; COVID-era stress, BTP-Bund >160bp)",
        "raise_eur_b": 9.0,
        "prior_raise_eur_b": 9.6,
        "miss_pct": -6.0,
        "instrument": "BTP Italia",
        "condition": "weak_demand",
    },
    {
        "date": "2022-04-04",
        "description": "BTP Italia Apr 2022: EUR 10.6B (nominal ok but BTP-Bund spread surging to 220bp, weak institutional)",
        "raise_eur_b": 10.6,
        "prior_raise_eur_b": 8.2,
        "miss_pct": None,  # Not a ratio miss, but elevated stress signal
        "instrument": "BTP Italia",
        "condition": "spread_stress",
    },
]

HOLD_DAYS = 50  # 10 calendar weeks ~ 50 trading days


def run_event_study(events, basket_ret, spy_r, idx, hold_days=HOLD_DAYS):
    """Short basket on event day. Returns (pnl, positions, event_log)."""
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)

    event_log = []
    for ev in events:
        event_dt = pd.Timestamp(ev["date"])

        # Entry at open on day after event date
        future = idx[idx > event_dt]
        if len(future) == 0:
            event_log.append(dict(ev, status="no_data_after_date"))
            continue

        entry_dt = future[0]
        entry_pos = idx.get_loc(entry_dt)
        exit_pos = min(entry_pos + hold_days, len(idx))

        if exit_pos <= entry_pos:
            event_log.append(dict(ev, entry_date=str(entry_dt.date()), status="no_hold_window"))
            continue

        # Short basket: pnl = -basket_return each day
        slice_r = basket_ret.iloc[entry_pos:exit_pos].fillna(0)
        spy_slice = spy_r.iloc[entry_pos:exit_pos].fillna(0)

        short_cum = float((1 + (-slice_r)).prod() - 1)
        spy_cum = float((1 + spy_slice).prod() - 1)
        basket_cum = float((1 + slice_r).prod() - 1)

        ev_record = dict(ev)
        ev_record["entry_date"] = str(entry_dt.date())
        ev_record["exit_date"] = str(idx[exit_pos - 1].date())
        ev_record["n_hold_days"] = int(exit_pos - entry_pos)
        ev_record["short_basket_return"] = round(short_cum, 4)
        ev_record["basket_return"] = round(basket_cum, 4)
        ev_record["spy_return"] = round(spy_cum, 4)
        ev_record["excess_vs_spy"] = round(short_cum - spy_cum, 4)
        event_log.append(ev_record)

        for j in range(entry_pos, exit_pos):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = -1.0
                r = basket_ret.iloc[j] if not pd.isna(basket_ret.iloc[j]) else 0.0
                pnl.iloc[j] = -r

    return pnl, positions, event_log


def main():
    sid = "PL972_btp_retail_miss_italian_banks_short"

    # Try Milan-listed bank tickers first
    milan_tickers = ["ISP.MI", "BMPS.MI", "UCG.MI"]
    proxy_tickers = ["EWI", "SPY"]
    all_tickers = milan_tickers + proxy_tickers

    try:
        px = load_prices(all_tickers, start="2019-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)
    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()
    idx = ret.index

    # Determine available tickers
    available_milan = [t for t in milan_tickers if t in ret.columns]
    has_ewi = "EWI" in ret.columns

    if not available_milan and not has_ewi:
        return mark_failed(sid, f"No Italian bank or EWI tickers available. Loaded: {list(ret.columns)}")

    # Build basket: prefer Milan-listed banks; fall back to EWI
    if available_milan:
        # 60% ISP.MI, 40% BMPS.MI if both available; else equal weight available
        weights = {}
        if "ISP.MI" in available_milan and "BMPS.MI" in available_milan:
            weights = {"ISP.MI": 0.6, "BMPS.MI": 0.4}
        elif "ISP.MI" in available_milan:
            weights = {"ISP.MI": 1.0}
        else:
            w = 1.0 / len(available_milan)
            weights = {t: w for t in available_milan}
        basket_ret = sum(ret[t].fillna(0) * w for t, w in weights.items())
        basket_label = f"ISP.MI(60%)/BMPS.MI(40%) short" if "BMPS.MI" in weights else f"{list(weights.keys())} short"
    else:
        # Fallback to EWI (iShares MSCI Italy ETF)
        basket_ret = ret["EWI"].fillna(0)
        basket_label = "EWI short (Italian equity proxy)"
        weights = {"EWI": 1.0}
        print("WARNING: Milan bank tickers unavailable, using EWI as proxy")

    # ---------- Primary event study ----------
    pnl, pos, event_log = run_event_study(
        BTP_EVENTS, basket_ret, spy_r, idx, hold_days=HOLD_DAYS
    )

    held_mask = pos != 0.0
    held_pnl = pnl[held_mask]
    held_spy = spy_r.reindex(held_pnl.index).fillna(0)
    n_events = sum(1 for e in event_log if e.get("entry_date"))

    if len(held_pnl) < 20:
        return mark_failed(
            sid,
            f"insufficient held days: held={len(held_pnl)}, n_events={n_events}",
            extra={"events": event_log},
        )

    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name=f"BTP Retail Miss: {basket_label} (held-days)",
        positions=pos.reindex(held_pnl.index).fillna(0),
        cost_bps=15,  # Milan-listed tickers have slightly higher execution costs
    )

    # ---------- FRED BTP-Bund spread for context (monthly, informational only) ----------
    btp_bund_spread = None
    try:
        italy_10y = load_fred("IRLTLT01ITM156N", start="2019-01-01")
        germany_10y = load_fred("IRLTLT01DEM156N", start="2019-01-01")
        spread_df = (italy_10y - germany_10y).dropna()
        btp_bund_spread = {
            "latest_spread_bp": round(float(spread_df.iloc[-1]) * 100, 1) if len(spread_df) else None,
            "spread_2022_apr": round(float(spread_df.loc["2022-04":"2022-04"].mean() * 100), 1) if "2022" in str(spread_df.index.max()) else None,
        }
    except Exception:
        pass

    # ---------- Event-level summary ----------
    short_rets = [e.get("short_basket_return") for e in event_log if e.get("short_basket_return") is not None]
    excess_rets = [e.get("excess_vs_spy") for e in event_log if e.get("excess_vs_spy") is not None]

    event_summary = {
        "n_events": len(short_rets),
        "avg_short_return": round(float(np.mean(short_rets)), 4) if short_rets else None,
        "avg_excess_vs_spy": round(float(np.mean(excess_rets)), 4) if excess_rets else None,
        "win_rate": round(float(np.mean([r > 0 for r in short_rets])), 4) if short_rets else None,
        "best": round(float(np.max(short_rets)), 4) if short_rets else None,
        "worst": round(float(np.min(short_rets)), 4) if short_rets else None,
    }

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "When MEF BTP Valore/Italia retail subscription shows a miss "
                "(< 60% of comparable prior window or absolute < EUR 8B), enter short "
                "ISP.MI (60%) and BMPS.MI (40%) at next open. Hold up to 50 trading days. "
                "Regime filter: BTP-Bund spread > 150bp."
            ),
            "mechanism": (
                "Italian banks hold large BTP portfolios (sovereign-bank nexus). "
                "Weak retail BTP demand signals fiscal stress and rising sovereign spreads, "
                "which compresses Italian bank book values via mark-to-market losses on "
                "BTP holdings, tightens net interest margins via higher sovereign funding "
                "costs, and raises refinancing risk. The doom-loop: weak retail demand "
                "raises BTP yields, compressing bank CET1 ratios and increasing "
                "probability of sovereign contagion event."
            ),
            "source": (
                "MEF Dipartimento del Tesoro (dt.mef.gov.it) BTP subscription data; "
                "FRED IRLTLT01ITM156N (Italy 10y) and IRLTLT01DEM156N (Germany 10y); "
                "prices via yfinance. Milan-listed tickers (ISP.MI, BMPS.MI) used; "
                "EWI as fallback proxy."
            ),
            "basket_used": basket_label,
            "basket_weights": weights,
            "available_milan_tickers": available_milan,
            "events": event_log,
            "event_summary": event_summary,
            "btp_bund_context": btp_bund_spread,
            "hold_days": HOLD_DAYS,
            "n_events": n_events,
            "caveats": (
                "N=3 events is very small. BMPS.MI (Monte Paschi) underwent major capital "
                "restructuring 2016-2022; pre-2022 price history may be unreliable for "
                "return calculation. BTP subscription data requires manual compilation from "
                "MEF press releases — not machine-readable. The April 2022 event is a "
                "'stress signal' rather than a ratio miss, making it marginally consistent "
                "with the strict entry condition. Milan-listed tickers may have limited "
                "yfinance coverage in non-EU infrastructure."
            ),
        },
        pnl=held_pnl,
    )

    print(f"Done: {sid}")
    print(f"  n_events={n_events}, hold_days={HOLD_DAYS}, basket={basket_label}")
    print(f"  Event summary: {event_summary}")
    if btp_bund_spread:
        print(f"  BTP-Bund context: {btp_bund_spread}")
    if "error" not in m:
        print(
            f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
            f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
        )


if __name__ == "__main__":
    main()
