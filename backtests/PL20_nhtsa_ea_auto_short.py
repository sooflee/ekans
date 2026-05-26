"""PL20_nhtsa_ea_auto_short — NHTSA Engineering Analysis → Short Auto OEM
Short the OEM when NHTSA opens an EA on a model with >500K affected units.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL20_nhtsa_ea_auto_short"
    try:
        px = load_prices(["TSLA", "F", "GM", "SPY"], start="2018-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    # NHTSA EA events on models with >500K affected units
    ea_events = [
        ("2019-01-18", "TSLA", "Model S/X suspension EA (PE19-002)"),
        ("2020-10-15", "F", "Bronco Sport/Escape transmission EA"),
        ("2022-06-09", "TSLA", "Autopilot EA (PE22-004, 830K units)"),
        ("2023-03-15", "GM", "Bolt battery recall extension EA"),
    ]

    events = []
    pnl_parts = []
    hold_days = 60

    for date_str, ticker, desc in ea_events:
        if ticker not in ret.columns:
            continue
        entry_date = pd.Timestamp(date_str)
        entry_mask = ret.index >= entry_date
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = ret.index[entry_mask][0]
        entry_loc = ret.index.get_loc(entry_idx)
        exit_loc = min(entry_loc + hold_days, len(ret.index) - 1)

        window = slice(entry_loc, exit_loc)
        # Short the OEM = negative of its return
        short_pnl = -ret[ticker].iloc[window]
        spy_window = spy_r.iloc[window]
        pnl_parts.append(short_pnl)

        short_cum = float((1 + short_pnl).prod() - 1)
        oem_cum = float((1 + ret[ticker].iloc[window]).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)

        events.append({
            "ea_date": date_str,
            "oem": ticker,
            "description": desc,
            "oem_60d_return": round(oem_cum, 4),
            "short_return": round(short_cum, 4),
            "spy_60d_return": round(spy_cum, 4),
            "excess_vs_spy": round(short_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No EA events with price data")

    all_pnl = pd.concat(pnl_parts)
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="NHTSA EA Auto Short")

    avg_short = np.mean([e["short_return"] for e in events])
    win_count = sum(1 for e in events if e["short_return"] > 0)

    save_result(sid, m, extra={
        "rule": "Short OEM on NHTSA EA opening (>500K affected units), hold 60 trading days",
        "mechanism": "EA signals safety-critical defect → recall likely → $500M-$2B warranty cost not yet accrued → EPS compression",
        "source": "NHTSA investigations database; yfinance",
        "n_events": len(events),
        "avg_short_return": round(avg_short, 4),
        "win_rate": f"{win_count}/{len(events)}",
        "events": events,
    })
    sharpe = m.get('sharpe', 0)
    cagr = m.get('cagr', 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    print(f"  Avg short return: {avg_short*100:.1f}%  Win rate: {win_count}/{len(events)}")
    for e in events:
        flag = "+" if e["short_return"] > 0 else "-"
        print(f"  {flag} {e['ea_date']} {e['oem']}: OEM {e['oem_60d_return']*100:+.1f}%, short {e['short_return']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
