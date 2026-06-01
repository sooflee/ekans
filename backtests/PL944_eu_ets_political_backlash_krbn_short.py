"""PL944_eu_ets_political_backlash_krbn_short — EU ETS Phase 4 Political Backlash -> Short KRBN"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL944_eu_ets_political_backlash_krbn_short"

    try:
        px = load_prices(["KRBN", "SPY"], start="2020-07-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if "KRBN" not in px.columns or px["KRBN"].dropna().shape[0] < 100:
        return mark_failed(sid, "KRBN data unavailable or insufficient")

    spy_r = daily_returns(px[["SPY"]]).iloc[:, 0].dropna()
    krbn_r = daily_returns(px[["KRBN"]]).iloc[:, 0].dropna()

    # Common index
    common_idx = spy_r.index.intersection(krbn_r.index)
    spy_r = spy_r.reindex(common_idx).fillna(0)
    krbn_r = krbn_r.reindex(common_idx).fillna(0)

    # Since we are SHORTING KRBN, the position return is -krbn_r
    short_krbn_r = -krbn_r

    # Known EU ETS political backlash event dates
    # These are events that signal ETS Phase 4 reform delay or weakening
    known_events = [
        pd.Timestamp("2024-02-01"),   # French/German farmer protest peak: Commission ETS exemption for agriculture announcement
        pd.Timestamp("2024-06-10"),   # EU Parliament election — far-right/Eurosceptic gains; uncertainty on ETS reform
        pd.Timestamp("2025-02-24"),   # German federal election: CDU+AfD outcome signals coalition skeptical of ETS tightening
    ]

    hold_days = 20  # per spec
    pnl = pd.Series(0.0, index=common_idx)
    events = []

    for event_date in known_events:
        # Enter short at T+1 open (next trading day after event)
        future_days = common_idx[common_idx > event_date]
        if len(future_days) < 10:
            print(f"Skipping {event_date}: insufficient future data ({len(future_days)} trading days)")
            continue

        entry_date = future_days[0]
        entry_idx = common_idx.get_loc(entry_date)
        exit_idx = min(entry_idx + hold_days, len(common_idx))

        short_slice = short_krbn_r.iloc[entry_idx:exit_idx]
        krbn_slice = krbn_r.iloc[entry_idx:exit_idx]
        spy_slice = spy_r.iloc[entry_idx:exit_idx]

        # Stop-loss: KRBN moves +10% adverse (i.e., KRBN rises 10% against our short)
        cum_krbn = krbn_slice.cumsum()
        stop_hit = cum_krbn > 0.10  # +10% adverse on KRBN from entry

        if stop_hit.any():
            exit_point = stop_hit.idxmax()
            short_slice = short_krbn_r.reindex(short_slice.index[:short_slice.index.get_loc(exit_point) + 1]).fillna(0)
            spy_slice = spy_r.reindex(short_slice.index).fillna(0)
            krbn_slice = krbn_r.reindex(short_slice.index).fillna(0)

        actual_exit_idx = entry_idx + len(short_slice)
        pnl.iloc[entry_idx:actual_exit_idx] = short_slice.values

        cum_short_total = float((1 + short_slice).prod() - 1)
        cum_krbn_total = float((1 + krbn_slice).prod() - 1)  # underlying (inverted = our PnL)
        cum_spy_total = float((1 + spy_slice).prod() - 1)

        events.append({
            "event_date": str(event_date.date()),
            "entry_date": str(entry_date.date()),
            "hold_days": len(short_slice),
            "short_return": round(cum_short_total, 4),
            "krbn_underlying": round(cum_krbn_total, 4),
            "spy_return": round(cum_spy_total, 4),
            "alpha": round(cum_short_total - cum_spy_total, 4),
            "stop_hit": bool(stop_hit.any()),
        })

    if not events:
        return mark_failed(sid, "no valid signal events found with sufficient data")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active trading days ({len(active_pnl)})")

    print(f"Signal events: {len(events)}, active days: {len(active_pnl)}")
    for ev in events:
        print(f"  {ev['event_date']}: short={ev['short_return']*100:.1f}%, KRBN={ev['krbn_underlying']*100:.1f}%, alpha={ev['alpha']*100:.1f}%, hold={ev['hold_days']}d")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="EU ETS Political Backlash → Short KRBN")
    m["n_events"] = len(events)

    avg_alpha = float(np.mean([e["alpha"] for e in events]))
    win_rate = float(np.mean([1 if e["short_return"] > 0 else 0 for e in events]))

    save_result(sid, m, extra={
        "rule": "Short KRBN at T+1 open following an EU ETS Phase 4 political backlash event (major member-state opposition, EP ENVI vote failure, Council postponement, or election of ETS-skeptic coalition). Hold 20 trading days. Stop-loss: KRBN +10% from entry.",
        "mechanism": "EU ETS price (EUA) is driven by policy credibility. Political events that signal weakening of ETS supply tightening (higher MSR threshold, deferred LRF acceleration, or legislative ETS reform delay) reduce the forward price signal for carbon, causing KRBN (55%+ EUA exposure) to reprice lower. Counter-signal to long-carbon consensus.",
        "source": "yfinance (KRBN, SPY); EU political event dates sourced from news: EC farmer exemption announcement (2024-02-01), EP election results (2024-06-10), German federal election (2025-02-24)",
        "n_events": len(events),
        "avg_event_alpha": round(avg_alpha, 4),
        "event_win_rate": round(win_rate, 4),
        "events": events,
    })
    print(f"Done: Sharpe={m.get('sharpe', 'N/A'):.2f}, CAGR={m.get('cagr', 'N/A')*100:.1f}%, MaxDD={m.get('max_dd', 'N/A')*100:.1f}%")


if __name__ == "__main__":
    main()
