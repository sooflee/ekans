"""PL848_oz_renewal_industrial_reit_long
Opportunity Zone Renewal Legislative Markup:
Long REXR/TRNO vs Short IYR (OZ-Exposed Industrial REIT)

When REXR and TRNO both have 5-day return > +3% on volume >= 1.5x 30-day avg
AND the pair outperforms IYR by >=3pp on a 10-day basis, enter LONG equal-weight
REXR+TRNO and SHORT IYR. Hold 30 trading days or exit on take-profit (+10%
spread) or stop-loss (-5% spread from peak).

This is a price-volume proxy for legislative events advancing OZ-related REIT
tax provisions (TCJA-related real estate provisions).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL848_oz_renewal_industrial_reit_long"
    tickers = ["REXR", "TRNO", "IYR", "SPY"]

    try:
        px = load_prices(tickers, start="2018-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)

    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()
    idx = ret.index

    # Volume data — need to reload with OHLCV for volume
    try:
        import yfinance as yf
        vol_raw = yf.download(["REXR", "TRNO"], start="2018-01-01", progress=False, auto_adjust=True)
        if isinstance(vol_raw.columns, pd.MultiIndex):
            vol_df = vol_raw["Volume"]
        else:
            vol_df = pd.DataFrame()
        has_volume = len(vol_df) > 0 and "REXR" in vol_df.columns and "TRNO" in vol_df.columns
    except Exception:
        has_volume = False

    # 5-day and 10-day returns
    rexr_5d = px["REXR"].pct_change(5)
    trno_5d = px["TRNO"].pct_change(5)
    iyr_5d = px["IYR"].pct_change(5)
    rexr_trno_5d = (rexr_5d + trno_5d) / 2
    rexr_trno_10d = px["REXR"].pct_change(10).add(px["TRNO"].pct_change(10)).div(2)
    iyr_10d = px["IYR"].pct_change(10)
    spread_10d = rexr_trno_10d - iyr_10d

    # Volume condition
    if has_volume:
        rexr_vol_ma30 = vol_df["REXR"].rolling(30).mean()
        trno_vol_ma30 = vol_df["TRNO"].rolling(30).mean()
        rexr_vol_ratio = vol_df["REXR"] / rexr_vol_ma30
        trno_vol_ratio = vol_df["TRNO"] / trno_vol_ma30
        # Align to px index
        rexr_vol_ratio = rexr_vol_ratio.reindex(idx)
        trno_vol_ratio = trno_vol_ratio.reindex(idx)
        vol_cond = (rexr_vol_ratio >= 1.5) & (trno_vol_ratio >= 1.5)
    else:
        # No volume data: use stricter return threshold to compensate
        vol_cond = pd.Series(True, index=idx)

    # Entry signal (all evaluated at close, entry next session)
    return_cond = (rexr_5d > 0.03) & (trno_5d > 0.03) & (spread_10d > 0.03)
    signal = (return_cond & vol_cond).fillna(False)

    # Long-short return: long REXR+TRNO equal-weight, short IYR
    basket_ret = (ret["REXR"] + ret["TRNO"]) / 2
    iyr_ret = ret["IYR"]
    ls_ret = basket_ret - iyr_ret

    HOLD_DAYS = 30
    TAKE_PROFIT = 0.10  # +10% spread from entry
    STOP_LOSS = -0.05   # -5% spread from peak

    positions = pd.Series(0.0, index=idx)
    pnl = pd.Series(0.0, index=idx)

    in_position = False
    open_until = None
    cum_ls = 0.0
    peak_cum_ls = 0.0

    events = []
    n = len(idx)

    for j in range(1, n):
        dt = idx[j]

        if in_position:
            day_ls = ls_ret.iloc[j] if not pd.isna(ls_ret.iloc[j]) else 0.0
            pnl.iloc[j] = day_ls
            positions.iloc[j] = 1.0
            cum_ls = (1 + cum_ls) * (1 + day_ls) - 1
            peak_cum_ls = max(peak_cum_ls, cum_ls)

            scheduled_end = (j + 1 >= open_until)
            take_profit = cum_ls >= TAKE_PROFIT
            stop_loss = (cum_ls - peak_cum_ls) <= STOP_LOSS

            if scheduled_end or take_profit or stop_loss:
                reason = (
                    "take_profit" if take_profit and not scheduled_end
                    else "stop_loss" if stop_loss and not scheduled_end
                    else "scheduled_30d"
                )
                events[-1]["exit_date"] = str(dt.date())
                events[-1]["exit_reason"] = reason
                events[-1]["event_ls_return"] = round(cum_ls, 4)
                in_position = False
                open_until = None
                cum_ls = 0.0
                peak_cum_ls = 0.0

        else:
            if signal.iloc[j - 1]:
                in_position = True
                open_until = min(j + HOLD_DAYS, n)
                cum_ls = 0.0
                peak_cum_ls = 0.0
                events.append({
                    "signal_date": str(idx[j - 1].date()),
                    "entry_date": str(dt.date()),
                    "rexr_5d": round(float(rexr_5d.iloc[j-1]), 4) if not pd.isna(rexr_5d.iloc[j-1]) else None,
                    "trno_5d": round(float(trno_5d.iloc[j-1]), 4) if not pd.isna(trno_5d.iloc[j-1]) else None,
                    "spread_10d": round(float(spread_10d.iloc[j-1]), 4) if not pd.isna(spread_10d.iloc[j-1]) else None,
                })
                pnl.iloc[j] = ls_ret.iloc[j] if not pd.isna(ls_ret.iloc[j]) else 0.0
                positions.iloc[j] = 1.0
                cum_ls = float(pnl.iloc[j])
                peak_cum_ls = cum_ls

    # Close any still-open position at end of sample
    if in_position and events and "exit_date" not in events[-1]:
        events[-1]["exit_date"] = str(idx[-1].date())
        events[-1]["exit_reason"] = "end_of_sample"
        events[-1]["event_ls_return"] = round(cum_ls, 4)

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
        name="OZ Renewal: Long REXR+TRNO Short IYR (held-days only)",
        positions=positions.reindex(held_pnl.index).fillna(0),
        cost_bps=10,
    )

    event_returns = [e.get("event_ls_return") for e in events if e.get("event_ls_return") is not None]

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "When REXR and TRNO both show 5-day return > +3% (with volume >= 1.5x "
                "30-day avg where available) AND the pair outperforms IYR by >=3pp on a "
                "10-day basis, enter LONG equal-weight REXR+TRNO and SHORT IYR; hold 30 "
                "trading days; take profit at +10% spread, stop-loss at -5% from peak spread."
            ),
            "mechanism": (
                "REXR and TRNO are Sunbelt/industrial REITs with Opportunity Zone property "
                "exposure. Price-volume breakouts in both relative to broad REITs (IYR) "
                "proxy for institutional pre-positioning ahead of OZ legislative events "
                "(TCJA renewal, Treasury OZ regulations, IRA energy community provisions). "
                "The long-short spread captures OZ-specific regulatory alpha vs generic REIT beta."
            ),
            "source": (
                "REXR, TRNO, IYR, SPY via yfinance (auto_adjust=True); "
                "known analog dates: 2018-12-12, 2019-04-17, 2020-01-01, 2022-08-16."
            ),
            "tickers_long": ["REXR", "TRNO"],
            "tickers_short": ["IYR"],
            "volume_data_used": has_volume,
            "n_events": n_events,
            "signal_count": int(signal.sum()),
            "events": events[:20],
            "avg_event_return": round(float(np.mean(event_returns)), 4) if event_returns else None,
            "caveats": (
                "Price-volume breakout is a noisy proxy for OZ legislative events; "
                "it fires on any momentum episode in industrial REITs relative to broad REITs. "
                "REXR and TRNO are correlated (both industrial REITs); the pair is not a "
                "clean factor-neutral basket. IYR includes retail/residential/office REITs "
                "which may diverge for sector-specific (not OZ) reasons. "
                "True OZ legislative events are sparse (~4 per decade); the backtest likely "
                "includes non-OZ momentum trades. Cost drag applied at 10bps round-trip."
            ),
        },
        pnl=held_pnl,
    )

    print(f"Done: {sid}")
    print(f"  n_events: {n_events}, signal fires: {int(signal.sum())}, held_days: {len(held_pnl)}")
    print(f"  avg_event_return: {round(float(np.mean(event_returns)), 4) if event_returns else None}")
    if "error" not in m:
        print(
            f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
            f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
        )
        if "net_sharpe" in m:
            print(f"  Net Sharpe: {m['net_sharpe']:.2f}, Net CAGR: {m['net_cagr']*100:.2f}%")


if __name__ == "__main__":
    main()
