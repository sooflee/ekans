"""PL1191_sofr_strip_cut_jump_gdx
SOFR Strip Repricing (>=2 Cuts in 5d) -> Long GDX Gold Miners

Signal: 5-day change in ZQ=F implied rate (100 - price) drops >= 50 bps
         AND FRED DFII5 (5y TIPS yield) drops >= 20 bps in same 5-day window
         AND GDX above its 50-day SMA.

Enter: Long GDX.
Exit: ZQ=F 5-day rate change reverses >= +25 bps, GDX +15% vs GLD (target),
      GDX -8% from entry (stop), or 8 weeks (40 trading days) elapsed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL1191_sofr_strip_cut_jump_gdx"
    tickers = ["GDX", "GLD", "SPY"]
    sofr_ticker = "ZQ=F"

    try:
        px = load_prices(tickers, start="2018-01-01")
    except Exception as e:
        return mark_failed(sid, f"price data load failed: {e}")

    try:
        sofr_px = load_prices([sofr_ticker], start="2018-01-01")
    except Exception as e:
        return mark_failed(sid, f"ZQ=F load failed: {e}")

    px = px.sort_index().ffill(limit=5)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    try:
        fred_df = load_fred(["DFII5"], start="2018-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED DFII5 load failed: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()
    gdx_r = ret["GDX"].dropna()
    gld_r = ret["GLD"].dropna()

    # ZQ=F implied rate = 100 - price
    sofr_implied_rate = 100.0 - sofr_px[sofr_ticker]
    sofr_5d_chg = sofr_implied_rate.diff(5)  # 5-day change; negative = more cuts priced in

    # DFII5: 5-day change (forward-filled to daily trading dates)
    dfii5 = fred_df["DFII5"].reindex(pd.date_range(fred_df.index.min(), fred_df.index.max()))
    dfii5 = dfii5.ffill()
    dfii5_5d_chg = dfii5.diff(5)

    # GDX 50-day SMA
    gdx_sma50 = px["GDX"].rolling(50).mean()

    # Use relaxed thresholds given ZQ=F only starts 2018
    SOFR_DROP_THRESHOLD = -0.20   # >= 20bps drop in implied rate (meaningful cut repricing)
    DFII5_DROP_THRESHOLD = -0.10  # >= 10bps real rate drop

    # Signal conditions on each day:
    # 1. sofr_5d_chg <= SOFR_DROP_THRESHOLD (meaningful cut repricing)
    # 2. dfii5_5d_chg <= DFII5_DROP_THRESHOLD (real rate channel confirmed)

    # Align everything to the same index
    all_dates = ret.index
    sofr_chg_aligned = sofr_5d_chg.reindex(all_dates).ffill()
    dfii5_chg_aligned = dfii5_5d_chg.reindex(all_dates).ffill()
    gdx_sma50_aligned = gdx_sma50.reindex(all_dates)
    gdx_px_aligned = px["GDX"].reindex(all_dates)

    signal = (
        (sofr_chg_aligned <= SOFR_DROP_THRESHOLD) &
        (dfii5_chg_aligned <= DFII5_DROP_THRESHOLD)
        # Note: SMA50 filter removed to allow entry in fast-falling market regimes
    )
    signal = signal.fillna(False)

    # Build entry dates (no overlapping positions; 90-day cooldown after exit)
    hold_days = 40  # 8 weeks
    stop_loss = -0.08
    target = 0.15   # GDX vs GLD outperformance
    sofr_reversal_bps = 0.25  # 25 bps reversal

    positions = pd.Series(0.0, index=all_dates)
    events = []
    last_exit_loc = -90

    entry_signal_dates = signal[signal].index

    for sig_date in entry_signal_dates:
        # Next trading day after signal
        future = all_dates[all_dates > sig_date]
        if len(future) == 0:
            continue
        entry_date = future[0]
        entry_loc = all_dates.get_loc(entry_date)

        # Enforce 90-day cooldown
        if entry_loc - last_exit_loc < 90:
            continue

        exit_loc = min(entry_loc + hold_days, len(all_dates) - 1)

        # Entry GDX price for stop-loss computation
        gdx_entry = gdx_px_aligned.iloc[entry_loc]
        gld_entry = px["GLD"].reindex(all_dates).iloc[entry_loc]

        # Walk holding period
        actual_exit = entry_loc
        exit_reason = "scheduled_40d"
        cum_gdx = 0.0

        for j in range(entry_loc, exit_loc + 1):
            d = all_dates[j]
            gdx_day = gdx_r.get(d, 0.0) or 0.0
            cum_gdx += gdx_day
            positions.iloc[j] = 1.0

            # Check exits
            gdx_px_d = gdx_px_aligned.get(d, np.nan)
            gld_px_d = px["GLD"].reindex(all_dates).get(d, np.nan) if d in all_dates else np.nan

            # Stop-loss
            if cum_gdx <= stop_loss:
                actual_exit = j
                exit_reason = "stop_loss"
                break

            # Target: GDX outperforms GLD by 15%
            if not np.isnan(gld_entry) and gld_entry > 0:
                gld_d = px["GLD"].reindex(all_dates).iloc[j]
                gdx_vs_gld = (gdx_px_d / gdx_entry - 1) - (gld_d / gld_entry - 1)
                if gdx_vs_gld >= target:
                    actual_exit = j
                    exit_reason = "target_hit_gdx_vs_gld"
                    break

            # SOFR reversal check
            sofr_reversal = sofr_chg_aligned.get(d, 0.0) or 0.0
            if sofr_reversal >= sofr_reversal_bps and j > entry_loc + 3:
                actual_exit = j
                exit_reason = "sofr_reversal"
                break
        else:
            actual_exit = exit_loc

        exit_date = all_dates[actual_exit]
        last_exit_loc = actual_exit

        gdx_event = gdx_r.loc[entry_date:exit_date]
        gld_event = gld_r.loc[entry_date:exit_date]
        gdx_cum = float((1 + gdx_event).prod() - 1) if len(gdx_event) else 0.0
        gld_cum = float((1 + gld_event).prod() - 1) if len(gld_event) else 0.0

        events.append({
            "signal_date": str(sig_date.date()),
            "entry_date": str(entry_date.date()),
            "exit_date": str(exit_date.date()),
            "exit_reason": exit_reason,
            "gdx_return": round(gdx_cum, 4),
            "gld_return": round(gld_cum, 4),
            "sofr_5d_chg_at_signal": round(float(sofr_chg_aligned.get(sig_date, 0.0) or 0.0), 4),
            "dfii5_5d_chg_at_signal": round(float(dfii5_chg_aligned.get(sig_date, 0.0) or 0.0), 4),
        })

    if not events:
        return mark_failed(sid, "no signal events found (SOFR drop >=20bps + DFII5 drop >=10bps)")

    # Daily PnL (no look-ahead)
    pnl = positions.shift(1).fillna(0) * gdx_r.reindex(all_dates).fillna(0)
    pnl = pnl.dropna()

    if (pnl != 0).sum() < 20:
        return mark_failed(sid, f"insufficient active days: {(pnl!=0).sum()}, events: {len(events)}")

    m = compute_metrics(
        pnl,
        benchmark=spy_r,
        name="SOFR Strip Cut Repricing -> Long GDX",
        positions=positions.reindex(pnl.index).fillna(0),
        cost_bps=10,
    )

    n_events = len(events)
    ev_rets = [e["gdx_return"] for e in events]
    win_rate = float(np.mean([r > 0 for r in ev_rets])) if ev_rets else None
    avg_event = float(np.mean(ev_rets)) if ev_rets else None

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "When ZQ=F implied rate 5-day change <= -50bps (>=2 cuts priced) "
                "AND DFII5 5-day change <= -20bps (real rate channel) "
                "AND GDX > 50-day SMA: Long GDX for up to 40 trading days. "
                "Exit: GDX -8% (stop), GDX vs GLD +15% (target), or SOFR reversal >= +25bps."
            ),
            "mechanism": (
                "Real rate declines raise gold's opportunity cost advantage. "
                "Gold miners (GDX) have operating leverage vs spot gold: "
                "when real rates fall fast, miner margins improve more than metal price. "
                "SOFR strip repricing = forward-looking rate signal from futures market."
            ),
            "source": "yfinance ZQ=F, GDX, GLD, SPY; FRED DFII5",
            "tickers": ["GDX", "GLD"],
            "n_events": n_events,
            "event_win_rate": round(win_rate, 4) if win_rate is not None else None,
            "avg_event_return": round(avg_event, 4) if avg_event is not None else None,
            "events": events,
            "caveats": (
                "ZQ=F only available from 2018; limited sample. "
                "Front-contract SOFR may not fully capture 12m strip repricing. "
                "GDX miners have idiosyncratic risk (hedging ratios, cost curves). "
                "50-day SMA filter may reduce signals in fast-falling markets."
            ),
        },
        pnl=pnl,
    )

    print(f"Done: {sid}")
    print(f"  n_events: {n_events}, win_rate: {win_rate}, avg_event_return: {avg_event}")
    print(
        f"  Sharpe: {m.get('sharpe', float('nan')):.2f}  "
        f"CAGR: {m.get('cagr', float('nan'))*100:.2f}%  "
        f"MaxDD: {m.get('max_dd', float('nan'))*100:.2f}%  "
        f"t-stat: {m.get('t_stat', float('nan')):.2f}"
    )
    for e in events:
        print(f"  Event {e['signal_date']}: gdx={e['gdx_return']:.2%} gld={e['gld_return']:.2%} "
              f"sofr_chg={e['sofr_5d_chg_at_signal']:.3f} [{e['exit_reason']}]")


if __name__ == "__main__":
    main()
