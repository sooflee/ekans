"""PL778_sofr_overshoot_short_tlt -- SOFR Strip Overshoot -> Short TLT Counter-Signal

Proxy signal: when DFF (overnight rate) - DGS2 (2-year yield) > 100bps,
markets are pricing >4 Fed cuts into the 2-year, often overshooting fundamentals.
Short TLT (long-duration) for up to 8 weeks (40 trading days) on this signal.
Exit: DFF-DGS2 drops back below 50bps (spread normalizes), or 8-week time stop,
or 8% loss on TLT price (stop-loss).

Counter-signal to long-TLT strategies (PL050, PL101, PL126, PL131, PL597).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL778_sofr_overshoot_short_tlt"

    # Load FRED: DFF (daily fed funds effective), DGS2 (2-year Treasury yield)
    try:
        fred = load_fred(["DFF", "DGS2"], start="1995-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED load: {e}")

    if fred is None or fred.empty:
        return mark_failed(sid, "FRED data empty")

    dff = fred["DFF"].dropna()
    dgs2 = fred["DGS2"].dropna()

    # Align on common dates
    combined = pd.DataFrame({"DFF": dff, "DGS2": dgs2}).dropna()
    if len(combined) < 252:
        return mark_failed(sid, "insufficient combined FRED data")

    # Proxy signal: DFF - DGS2 > 100bps (markets pricing >4 cuts of 25bps into 2y)
    combined["spread"] = combined["DFF"] - combined["DGS2"]

    # Find entry dates: first day in each new "overshoot" episode (dedup by 6-month cooldown)
    threshold_enter = 1.0   # 100bps = >4 cuts priced
    threshold_exit = 0.5    # 50bps = overshoot mostly resolved
    trigger_dates = []
    last_trigger = None
    in_overshoot = False

    for i in range(len(combined)):
        d = combined.index[i]
        spread = combined["spread"].iloc[i]
        if spread > threshold_enter and not in_overshoot:
            in_overshoot = True
            if last_trigger is None or (d - last_trigger).days >= 180:
                trigger_dates.append(d)
                last_trigger = d
        elif spread < threshold_exit:
            in_overshoot = False

    if len(trigger_dates) == 0:
        return mark_failed(sid, "no SOFR-strip overshoot events found (DFF-DGS2 > 100bps)")

    print(f"Overshoot episodes: {len(trigger_dates)}")
    for d in trigger_dates:
        print(f"  {d.date()}: DFF={combined.loc[d, 'DFF']:.2f}, DGS2={combined.loc[d, 'DGS2']:.2f}, "
              f"spread={combined.loc[d, 'spread']:.2f}")

    # Load prices: TLT (short = we profit when TLT falls = yields rise)
    try:
        px = load_prices(["TLT", "SPY"], start="2002-01-01")
    except Exception as e:
        try:
            px = load_prices(["TLT", "SPY"], start="2002-01-01")
        except Exception as e2:
            return mark_failed(sid, f"price load: {e2}")

    if px is None or px.empty:
        return mark_failed(sid, "price data empty")

    ret = daily_returns(px)
    if "TLT" not in ret.columns:
        return mark_failed(sid, "TLT not in price data")

    spy_r = ret["SPY"] if "SPY" in ret.columns else None
    tlt_r = ret["TLT"]

    hold_days = 40  # 8 weeks
    stop_loss_pct = 0.08  # 8% loss on short TLT (i.e., TLT rises 8% against us)

    pnl_parts = []
    event_results = []

    for td in trigger_dates:
        # Entry: first trading day on or after trigger date
        entry_mask = tlt_r.index >= td
        if entry_mask.sum() < 10:
            print(f"  Skipping {td.date()}: insufficient TLT data ahead")
            continue

        entry_loc = tlt_r.index.get_loc(tlt_r.index[entry_mask][0])

        # Build DFF-DGS2 spread aligned with TLT trading days
        spread_aligned = combined["spread"].reindex(tlt_r.index, method="ffill")

        # Simulate short TLT position with stop-loss and mean-reversion exit
        short_pnl = []
        actual_days = 0
        stop_triggered = False
        mean_rev_triggered = False
        tlt_cum_return = 0.0  # cumulative TLT return (positive = TLT went up = we lose)

        for i in range(entry_loc, min(entry_loc + hold_days, len(tlt_r) - 1)):
            tlt_day_r = tlt_r.iloc[i]
            if pd.isna(tlt_day_r):
                continue

            short_day_pnl = -tlt_day_r  # short TLT: profit when TLT falls
            tlt_cum_return = (1 + tlt_cum_return) * (1 + tlt_day_r) - 1
            short_pnl.append((tlt_r.index[i], short_day_pnl))
            actual_days += 1

            # Stop-loss: if TLT rises 8% (= we lost 8% on short), exit
            if tlt_cum_return >= stop_loss_pct:
                stop_triggered = True
                break

            # Mean-reversion exit: spread drops below 0.5
            d_spread = spread_aligned.iloc[i]
            if pd.notna(d_spread) and d_spread < threshold_exit:
                mean_rev_triggered = True
                break

        if not short_pnl:
            continue

        pnl_series = pd.Series(
            [p for _, p in short_pnl],
            index=[d for d, _ in short_pnl]
        )
        pnl_parts.append(pnl_series)

        short_cum = float((1 + pnl_series).prod() - 1)
        spy_cum = None
        if spy_r is not None:
            spy_window = spy_r.iloc[entry_loc:entry_loc + actual_days]
            spy_cum = float((1 + spy_window.dropna()).prod() - 1)

        exit_reason = "stop_loss" if stop_triggered else ("mean_reversion" if mean_rev_triggered else "time_stop")

        event_results.append({
            "trigger_date": str(td.date()),
            "entry_date": str(tlt_r.index[entry_loc].date()),
            "hold_days": actual_days,
            "exit_reason": exit_reason,
            "short_tlt_return": round(short_cum, 4),
            "spy_return": round(spy_cum, 4) if spy_cum is not None else None,
        })

    if not event_results:
        return mark_failed(sid, "no valid short-TLT events after price alignment")

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep="first")].dropna()

    if len(all_pnl) < 30:
        return mark_failed(sid, f"insufficient in-position days ({len(all_pnl)})")

    bench = spy_r.reindex(all_pnl.index).dropna() if spy_r is not None else None
    m = compute_metrics(all_pnl, benchmark=bench,
                        name="SOFR Overshoot Counter-Signal -> Short TLT")

    short_rets = [e["short_tlt_return"] for e in event_results]
    win_rate = sum(1 for r in short_rets if r > 0) / len(short_rets)

    save_result(sid, m, extra={
        "rule": "DFF-DGS2 > 100bps (>4 cuts priced into 2-year yield) -> short TLT for up to 40 trading days; exit on spread normalization <50bps or 8% TLT-up stop",
        "mechanism": "When 2-year yields drop >100bps below overnight rate, markets overshoot on Fed cut expectations; mean reversion in rate expectations causes TLT to underperform as cuts get priced out",
        "source": "FRED DFF + DGS2; yfinance TLT/SPY",
        "proxy_note": "SOFR ZQ futures strip not available in FRED; DFF-DGS2 spread used as proxy for implied cut count",
        "n_events": len(event_results),
        "avg_short_return": round(float(np.mean(short_rets)), 4),
        "win_rate": round(win_rate, 3),
        "events": event_results,
    })

    sharpe = m.get("sharpe", 0)
    cagr = m.get("cagr", 0)
    print(f"Done: {len(event_results)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    print(f"  Avg short TLT return: {np.mean(short_rets)*100:.1f}%  Win rate: {win_rate*100:.0f}%")
    for e in event_results:
        flag = "+" if e["short_tlt_return"] > 0 else "-"
        print(f"  {flag} {e['trigger_date']}: short_TLT {e['short_tlt_return']*100:+.1f}% "
              f"({e['hold_days']}d, {e['exit_reason']})")


if __name__ == "__main__":
    main()
