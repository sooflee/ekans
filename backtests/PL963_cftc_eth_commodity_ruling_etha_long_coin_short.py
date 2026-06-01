"""PL963_cftc_eth_commodity_ruling_etha_long_coin_short
CFTC ETH Commodity Jurisdiction Assertion: Long ETHA (ETH-USD proxy) / Short COIN Pair

On each CFTC formal jurisdictional assertion over ETH as a commodity:
- Enter long ETH-USD (as ETHA proxy for pre-July-2024 events; ETHA for post)
- Enter short COIN (Coinbase Global) at 1.5x notional (beta-weight to neutralize COIN's ETH exposure)
- Hold 30 trading days, exit at close on day +30

Known events:
- 2023-03-27: CFTC v. Binance order citing ETH as commodity
- 2023-06-14: CFTC Chair Behnam Senate testimony asserting ETH commodity status
- 2024-05-23: CFTC Chair speech on FIT21 jurisdictional split
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# CFTC ETH commodity jurisdiction assertion events
# Source: CFTC press releases, congressional testimony, enforcement orders
CFTC_EVENTS = [
    {
        "date": "2023-03-27",
        "description": "CFTC v. Binance enforcement order citing ETH as commodity under CEA 1a(9)",
        "source": "CFTC press release, CFTC.gov/enforcement",
    },
    {
        "date": "2023-06-14",
        "description": "CFTC Chair Behnam Senate Agriculture Committee testimony asserting exclusive CFTC jurisdiction over spot ETH",
        "source": "US Senate Agriculture Committee hearing, CFTC Chair testimony",
    },
    {
        "date": "2024-05-23",
        "description": "CFTC Chair speech on FIT21 jurisdictional split asserting ETH commodity jurisdiction",
        "source": "CFTC Chair public remarks on FIT21",
    },
]

HOLD_DAYS = 30
COIN_BETA_WEIGHT = 1.5  # short COIN at 1.5x ETHA notional


def run_event_study(events, ret_long, ret_short, idx, hold_days=HOLD_DAYS, short_weight=COIN_BETA_WEIGHT):
    """Event study: long `ret_long`, short `ret_short` at `short_weight` notional.
    Returns (pnl_series, positions_series, event_log).
    """
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)

    event_log = []
    for ev in events:
        event_dt = pd.Timestamp(ev["date"])

        # Find the event day or next trading day
        future = idx[idx >= event_dt]
        if len(future) == 0:
            event_log.append(dict(ev, status="no_data_after_date"))
            continue

        entry_dt = future[0]
        entry_pos = idx.get_loc(entry_dt)
        exit_pos = min(entry_pos + hold_days, len(idx))

        # Compute pair return: long ETH, short COIN at beta_weight
        slice_long = ret_long.iloc[entry_pos:exit_pos].fillna(0)
        slice_short = ret_short.iloc[entry_pos:exit_pos].fillna(0)

        # Daily pair return: long + short_weight * (-short)
        pair_daily = slice_long - short_weight * slice_short

        # Cumulative event return
        event_ret = float((1 + pair_daily).prod() - 1) if len(pair_daily) else None

        ev_record = dict(ev)
        ev_record["entry_date"] = str(entry_dt.date())
        ev_record["exit_date"] = str(idx[exit_pos - 1].date()) if exit_pos > entry_pos else None
        ev_record["n_hold_days"] = int(exit_pos - entry_pos)
        ev_record["event_pair_return"] = round(event_ret, 4) if event_ret is not None else None
        ev_record["eth_return"] = round(float((1 + slice_long).prod() - 1), 4) if len(slice_long) else None
        ev_record["coin_return"] = round(float((1 + slice_short).prod() - 1), 4) if len(slice_short) else None
        event_log.append(ev_record)

        # Fill pnl and positions on hold window
        for j in range(entry_pos, exit_pos):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = 1.0
                pnl.iloc[j] = (ret_long.iloc[j] if not pd.isna(ret_long.iloc[j]) else 0) - \
                               short_weight * (ret_short.iloc[j] if not pd.isna(ret_short.iloc[j]) else 0)

    return pnl, positions, event_log


def main():
    sid = "PL963_cftc_eth_commodity_ruling_etha_long_coin_short"

    # Use ETH-USD as ETHA proxy (ETHA launched July 2024)
    tickers = ["ETH-USD", "COIN", "SPY"]
    try:
        px = load_prices(tickers, start="2021-04-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    # Check which tickers loaded
    missing = [t for t in tickers if t not in px.columns]
    if "ETH-USD" in missing or "COIN" in missing:
        return mark_failed(sid, f"missing critical tickers: {missing}")

    px = px.sort_index().ffill(limit=3)
    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    eth_r = ret["ETH-USD"].fillna(0)
    coin_r = ret["COIN"].fillna(0)
    idx = ret.index

    # ---------- Primary event study: long ETH-USD, short COIN ----------
    pnl, positions, event_log = run_event_study(
        CFTC_EVENTS, eth_r, coin_r, idx, hold_days=HOLD_DAYS, short_weight=COIN_BETA_WEIGHT
    )

    # Extract held-day returns for metrics
    held_mask = positions != 0.0
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
        name="CFTC ETH Commodity Ruling: Long ETH / Short COIN Pair (held-days)",
        positions=positions.reindex(held_pnl.index).fillna(0),
        cost_bps=20,  # higher cost for crypto execution
    )

    # ---------- Event-level summary ----------
    pair_rets = [e["event_pair_return"] for e in event_log if e.get("event_pair_return") is not None]
    event_summary = {
        "n_events": len(pair_rets),
        "avg_pair_return": round(float(np.mean(pair_rets)), 4) if pair_rets else None,
        "win_rate": round(float(np.mean([r > 0 for r in pair_rets])), 4) if pair_rets else None,
        "best": round(float(np.max(pair_rets)), 4) if pair_rets else None,
        "worst": round(float(np.min(pair_rets)), 4) if pair_rets else None,
    }

    # Compute COIN ETH beta over COIN's full history
    # (just for informational reporting)
    coin_data_start = ret["COIN"].first_valid_index()
    if coin_data_start is not None:
        common_idx = eth_r.dropna().index.intersection(coin_r.dropna().index)
        if len(common_idx) > 60:
            eth_aligned = eth_r.loc[common_idx]
            coin_aligned = coin_r.loc[common_idx]
            beta_est = np.cov(coin_aligned, eth_aligned)[0, 1] / np.var(eth_aligned)
        else:
            beta_est = None
    else:
        beta_est = None

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "On each CFTC formal jurisdictional assertion over ETH as a commodity "
                "(enforcement order, Chair Senate testimony, or FIT21-related statement): "
                "enter long ETH-USD (ETHA proxy) and short COIN at 1.5x notional "
                "(beta-weight to neutralize COIN ETH exposure). Hold 30 trading days. "
                "Exit at close on day +30."
            ),
            "mechanism": (
                "CFTC commodity designation is bullish for ETH (legitimizes its use, "
                "removes SEC securities risk) but bearish for Coinbase (whose revenue "
                "depends on maintaining a regulatory moat over crypto exchanges and whose "
                "valuation partly reflects the prospect of SEC oversight reducing "
                "competition). A CFTC victory reduces regulatory uncertainty for ETH "
                "holders but compresses Coinbase's franchise value vs. decentralized "
                "exchange alternatives. The pair trade isolates the differential impact."
            ),
            "source": (
                "CFTC press releases: cftc.gov/PressRoom; "
                "US Senate Agriculture Committee hearings; "
                "CFTC enforcement docket; prices via yfinance (auto_adjust=True). "
                "ETH-USD used as ETHA proxy for pre-July-2024 events."
            ),
            "tickers_long": ["ETH-USD"],
            "tickers_short": ["COIN"],
            "coin_eth_beta_estimate": round(float(beta_est), 4) if beta_est is not None else None,
            "short_weight_applied": COIN_BETA_WEIGHT,
            "events": event_log,
            "event_summary": event_summary,
            "hold_days": HOLD_DAYS,
            "n_events": n_events,
            "caveats": (
                "N=3 events is very small — results are anecdotal, not statistically "
                "significant. CFTC and SEC issued conflicting signals on ETH status 2023-2025; "
                "event contamination from SEC counter-announcements is possible. "
                "ETH-USD used as ETHA proxy for pre-July-2024 events (ETHA launched July 23, 2024). "
                "COIN launched April 2021 — limited overlap with ETH price history. "
                "Crypto liquidity and execution costs are higher than equity assumptions. "
                "The 1.5x short weight is an estimate; realized COIN/ETH beta varies widely."
            ),
        },
        pnl=held_pnl,
    )

    print(f"Done: {sid}")
    print(f"  n_events={n_events}, hold_days={HOLD_DAYS}")
    print(f"  Event summary: {event_summary}")
    print(f"  COIN/ETH beta estimate: {beta_est:.3f}" if beta_est is not None else "  COIN/ETH beta: not computed")
    if "error" not in m:
        print(
            f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
            f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
        )


if __name__ == "__main__":
    main()
