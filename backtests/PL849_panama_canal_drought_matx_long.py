"""PL849_panama_canal_drought_matx_long
Panama Canal Slot Auction Premium Spike: Long MATX (Jones-Act Pacific Monopoly Beneficiary)

When ZIM 20-day return > +15% (proxy for ocean freight rate spike from Panama
congestion) AND MATX 20-day trailing return > 0% AND SPY 20-day return > -5%,
enter LONG MATX for 25 trading days.

Take profit at +20% MATX gain; stop-loss at -10% MATX loss; ZIM freight reversal
exit if ZIM 10-day return drops below -15%.

Primary analog: Aug 2023 - April 2024 Panama Canal drought (Gatun Lake record low,
ACP cut slots from 36 to 22/day, MATX doubled from ~$72 to ~$128).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL849_panama_canal_drought_matx_long"
    tickers = ["MATX", "ZIM", "SPY"]

    try:
        px = load_prices(tickers, start="2021-02-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)

    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()
    idx = ret.index

    # Rolling returns for signal
    matx_20d = px["MATX"].pct_change(20)
    zim_20d = px["ZIM"].pct_change(20)
    spy_20d = px["SPY"].pct_change(20)
    zim_10d = px["ZIM"].pct_change(10)

    # Entry signal conditions (evaluated at close, enter next session)
    cond_matx = matx_20d > 0.00          # MATX not in downtrend
    cond_zim = zim_20d > 0.15            # ZIM >+15% in 20 days (freight spike proxy)
    cond_spy = spy_20d > -0.05           # SPY not in severe drawdown

    signal = (cond_matx & cond_zim & cond_spy).fillna(False)

    matx_ret = ret["MATX"]

    HOLD_DAYS = 25
    TAKE_PROFIT = 0.20
    STOP_LOSS = -0.10
    ZIM_REVERSAL = -0.15  # ZIM 10-day drop threshold for thesis break

    positions = pd.Series(0.0, index=idx)
    pnl = pd.Series(0.0, index=idx)

    in_position = False
    open_until = None
    entry_price = None
    cum_matx = 0.0  # cumulative MATX return from entry

    events = []
    n = len(idx)
    matx_prices = px["MATX"]

    for j in range(1, n):
        dt = idx[j]

        if in_position:
            day_r = matx_ret.iloc[j] if not pd.isna(matx_ret.iloc[j]) else 0.0
            pnl.iloc[j] = day_r
            positions.iloc[j] = 1.0
            cum_matx = (1 + cum_matx) * (1 + day_r) - 1

            scheduled_end = (j + 1 >= open_until)
            take_profit = cum_matx >= TAKE_PROFIT
            stop_loss = cum_matx <= STOP_LOSS
            zim_reversal = float(zim_10d.iloc[j]) < ZIM_REVERSAL if not pd.isna(zim_10d.iloc[j]) else False

            if scheduled_end or take_profit or stop_loss or zim_reversal:
                reason = (
                    "take_profit" if take_profit
                    else "stop_loss" if stop_loss
                    else "zim_reversal" if zim_reversal
                    else "scheduled_25d"
                )
                events[-1]["exit_date"] = str(dt.date())
                events[-1]["exit_reason"] = reason
                events[-1]["matx_return"] = round(cum_matx, 4)
                in_position = False
                open_until = None
                cum_matx = 0.0

        else:
            if signal.iloc[j - 1]:
                in_position = True
                open_until = min(j + HOLD_DAYS, n)
                cum_matx = 0.0
                entry_price = float(matx_prices.iloc[j])
                events.append({
                    "signal_date": str(idx[j - 1].date()),
                    "entry_date": str(dt.date()),
                    "matx_20d": round(float(matx_20d.iloc[j-1]), 4) if not pd.isna(matx_20d.iloc[j-1]) else None,
                    "zim_20d": round(float(zim_20d.iloc[j-1]), 4) if not pd.isna(zim_20d.iloc[j-1]) else None,
                    "spy_20d": round(float(spy_20d.iloc[j-1]), 4) if not pd.isna(spy_20d.iloc[j-1]) else None,
                    "entry_price": round(entry_price, 2),
                })
                day_r = matx_ret.iloc[j] if not pd.isna(matx_ret.iloc[j]) else 0.0
                pnl.iloc[j] = day_r
                positions.iloc[j] = 1.0
                cum_matx = float(day_r)

    # Close any still-open position at end of sample
    if in_position and events and "exit_date" not in events[-1]:
        events[-1]["exit_date"] = str(idx[-1].date())
        events[-1]["exit_reason"] = "end_of_sample"
        events[-1]["matx_return"] = round(cum_matx, 4)

    held_pnl = pnl[positions > 0]
    held_spy = spy_r.reindex(held_pnl.index).dropna()
    n_events = len(events)

    if len(held_pnl) < 30:
        return mark_failed(
            sid,
            f"insufficient held days ({len(held_pnl)}) across {n_events} events",
            extra={"events": events, "signal_count": int(signal.sum())},
        )

    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name="Panama Canal Drought: Long MATX (held-days only)",
        positions=positions.reindex(held_pnl.index).fillna(0),
        cost_bps=10,
    )

    matx_returns = [e.get("matx_return") for e in events if e.get("matx_return") is not None]
    # SPY same-window returns for excess calculation
    spy_excess = []
    for ev in events:
        if not ev.get("entry_date") or not ev.get("exit_date"):
            continue
        entry_dt = pd.Timestamp(ev["entry_date"])
        exit_dt = pd.Timestamp(ev["exit_date"])
        spy_slice = spy_r.loc[entry_dt:exit_dt]
        if len(spy_slice):
            spy_cum = float((1 + spy_slice).prod() - 1)
            ev["spy_return"] = round(spy_cum, 4)
            ev["excess_vs_spy"] = round((ev.get("matx_return") or 0) - spy_cum, 4)
            spy_excess.append(ev["excess_vs_spy"])

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "When ZIM 20-day return > +15% (freight rate spike proxy) AND "
                "MATX 20-day return > 0% AND SPY 20-day return > -5%, enter LONG "
                "MATX at next open; hold 25 trading days; take profit at +20%, "
                "stop-loss at -10%, or exit if ZIM 10-day return drops below -15%."
            ),
            "mechanism": (
                "Panama Canal slot restrictions divert Asia-USEC volume to all-water "
                "routes (via Suez or around Cape Horn), tightening capacity and lifting "
                "spot rates. MATX is the sole Jones-Act expedited Pacific carrier with "
                "Hawaii/Alaska domestic volume (inelastic demand floor) plus China-USEC "
                "express service that benefits directly from Canal diversion. ZIM price "
                "is used as a real-time proxy for SCFI/FBX freight rate spikes."
            ),
            "source": (
                "MATX, ZIM, SPY via yfinance; "
                "known analog: 2023-08 to 2024-04 Panama Canal drought (ACP slot cuts "
                "to 22/day, auction prices ~$2M/slot)."
            ),
            "tickers": ["MATX"],
            "freight_proxy": "ZIM",
            "n_events": n_events,
            "signal_count": int(signal.sum()),
            "events": events[:20],
            "matx_returns": matx_returns,
            "avg_matx_return": round(float(np.mean(matx_returns)), 4) if matx_returns else None,
            "avg_excess_vs_spy": round(float(np.mean(spy_excess)), 4) if spy_excess else None,
            "caveats": (
                "Backtest window limited to 2021-present due to ZIM IPO (Jan 2021). "
                "ZIM is a noisy proxy for SCFI; it reflects global container market "
                "conditions, not just Panama Canal traffic. MATX is a small-cap stock "
                "(~$3B market cap) with limited institutional coverage and wide bid-ask "
                "spreads. The 2023-24 Panama drought is the primary in-sample analog; "
                "the strategy may be overfitted to this single event. Take-profit and "
                "stop-loss triggers reduce the effective hold period. Cost drag applied "
                "at 10bps round-trip."
            ),
        },
        pnl=held_pnl,
    )

    print(f"Done: {sid}")
    print(f"  n_events: {n_events}, signal fires: {int(signal.sum())}, held_days: {len(held_pnl)}")
    print(f"  avg_matx_return: {round(float(np.mean(matx_returns)), 4) if matx_returns else None}")
    print(f"  avg_excess_vs_spy: {round(float(np.mean(spy_excess)), 4) if spy_excess else None}")
    if "error" not in m:
        print(
            f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
            f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
        )
        if "net_sharpe" in m:
            print(f"  Net Sharpe: {m['net_sharpe']:.2f}, Net CAGR: {m['net_cagr']*100:.2f}%")


if __name__ == "__main__":
    main()
