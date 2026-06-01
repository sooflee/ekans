"""PL1022 — Credit Spread Complacency (HY OAS <300bps) -> Short KRE / Long IEF Pair"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL1022_credit_complacency_kre_short_ief_long"

    # Load HY OAS from FRED (BAMLH0A0HYM2 is in percent, 3.00 = 300bps)
    try:
        fred = load_fred("BAMLH0A0HYM2", start="2003-01-01")
        oas = fred.squeeze().dropna()
    except Exception as e:
        return mark_failed(sid, f"FRED load: {e}")

    if oas.empty:
        return mark_failed(sid, "no FRED data")

    # Load prices
    try:
        px = load_prices(["KRE", "IEF", "SPY"], start="2003-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    for t in ["KRE", "IEF", "SPY"]:
        if t not in px.columns or px[t].dropna().empty:
            return mark_failed(sid, f"{t} data missing")

    ret = daily_returns(px)
    kre_r = ret["KRE"]
    ief_r = ret["IEF"]
    spy_r = ret["SPY"]

    # Resample OAS to weekly for trend detection
    oas_weekly = oas.resample("W").last().dropna()

    # Rolling 10-week: count how many weeks OAS declined
    oas_diff = oas_weekly.diff()
    declining_count = oas_diff.rolling(10).apply(lambda x: (x < 0).sum(), raw=True)

    # Signal trigger: OAS < 3.0 (300bps) AND declining >= 8 of last 10 weeks
    signal_mask = (oas_weekly < 3.0) & (declining_count >= 8)

    # Get all signal dates; de-duplicate with minimum 3-week gap
    signal_dates_raw = oas_weekly.index[signal_mask]
    signal_dates = []
    last_entry = None
    for d in signal_dates_raw:
        if last_entry is None or (d - last_entry).days > 21:
            signal_dates.append(d)
            last_entry = d

    print(f"Signal dates found: {len(signal_dates)}")

    pnl_list = []
    events_out = []
    HOLD_DAYS = 42  # calendar days (~6 weeks)

    for sig_date in signal_dates:
        # Find next trading day
        entry_mask = kre_r.index > sig_date
        entry_candidates = kre_r.index[entry_mask]
        if len(entry_candidates) == 0:
            continue
        entry_date = entry_candidates[0]

        exit_cutoff = entry_date + pd.Timedelta(days=HOLD_DAYS)

        # Check if OAS widens above 375bps during the window (exit signal)
        oas_window = oas.loc[(oas.index >= entry_date) & (oas.index <= exit_cutoff)]
        exit_on_widen = oas_window[oas_window >= 3.75]
        if len(exit_on_widen) > 0:
            actual_exit = exit_on_widen.index[0]
            exit_cutoff = min(exit_cutoff, actual_exit)

        # Pair PnL: long IEF, short KRE (dollar neutral)
        window_mask = (kre_r.index >= entry_date) & (kre_r.index <= exit_cutoff)
        kre_window = kre_r.loc[window_mask]
        ief_window = ief_r.reindex(kre_window.index)
        spy_window = spy_r.reindex(kre_window.index)

        if len(kre_window) < 5:
            continue

        # Pair PnL = long IEF + short KRE
        pair_pnl = ief_window.fillna(0) - kre_window.fillna(0)

        # Apply pair stop loss: if KRE outperforms IEF by 10% cumulatively, stop out
        kre_cum = (1 + kre_window).cumprod()
        ief_cum = (1 + ief_window).cumprod()
        kre_vs_ief = kre_cum / ief_cum - 1
        stop_hit = kre_vs_ief[kre_vs_ief >= 0.10]
        stopped_out = len(stop_hit) > 0
        if stopped_out:
            stop_date = stop_hit.index[0]
            pair_pnl = pair_pnl.loc[pair_pnl.index <= stop_date]
            kre_window = kre_window.loc[kre_window.index <= stop_date]

        if len(pair_pnl) < 3:
            continue

        total_pair_return = float((1 + pair_pnl).prod() - 1)
        kre_ret = float((1 + kre_window).prod() - 1)
        spy_cumret = float((1 + spy_window.reindex(pair_pnl.index).fillna(0)).prod() - 1)
        oas_entry = float(oas.asof(entry_date)) if entry_date >= oas.index[0] else None

        pnl_list.append(pair_pnl)
        events_out.append({
            "signal_date": str(sig_date.date()),
            "entry_date": str(entry_date.date()),
            "exit_date": str(pair_pnl.index[-1].date()),
            "n_days": len(pair_pnl),
            "oas_at_entry": round(oas_entry * 100, 1) if oas_entry else None,
            "pair_return": round(total_pair_return, 4),
            "kre_return": round(kre_ret, 4),
            "spy_return": round(spy_cumret, 4),
            "stopped_out": stopped_out,
            "early_exit_spread": len(exit_on_widen) > 0,
        })

    print(f"Valid trade episodes: {len(events_out)}")
    for ev in events_out:
        print(f"  {ev['entry_date']} OAS={ev['oas_at_entry']}bps: pair={ev['pair_return']:.2%}, KRE={ev['kre_return']:.2%}, SPY={ev['spy_return']:.2%} {'STOP' if ev['stopped_out'] else ''}")

    if len(events_out) < 3:
        return mark_failed(sid, f"insufficient trade episodes ({len(events_out)}) — HY OAS rarely sustained below 300bps for 8+ consecutive declining weeks")

    # Build combined PnL
    pnl = pd.concat(pnl_list).sort_index()
    pnl = pnl.groupby(pnl.index).mean()

    ip = pnl[pnl != 0]
    if len(ip) < 30:
        return mark_failed(sid, f"insufficient active days ({len(ip)})")

    spy_aligned = spy_r.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=spy_aligned, name="HY OAS Complacency Short-KRE Long-IEF Pair")

    returns_list = [e["pair_return"] for e in events_out]
    save_result(sid, m, extra={
        "rule": "Short KRE + Long IEF (1:1 dollar neutral) when HY OAS (BAMLH0A0HYM2) < 3.0% AND declining in 8/10 prior weeks (complacency regime); hold 42 calendar days or until OAS widens above 3.75%; stop on 10% adverse pair move",
        "mechanism": "Persistent credit spread compression below 300bps historically precedes spread normalization events; regional banks disproportionately exposed to credit cycle (loan losses, NIM compression) while Treasuries benefit from flight-to-quality on reversion",
        "source": "FRED BAMLH0A0HYM2 (ICE BofA HY OAS); yfinance KRE, IEF, SPY",
        "n_events": len(events_out),
        "avg_event_return": round(float(np.mean(returns_list)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in returns_list])), 4),
        "events": events_out,
        "caveats": "Counter-signal to long-credit consensus. HY OAS rarely sustains persistent decline to <300bps creating small sample. Opposite direction to PL115 which is long KRE on normalization.",
    })
    print("Done.")


if __name__ == "__main__":
    main()
