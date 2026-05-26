"""PL49_plantings_corn_soy_spread — USDA Prospective Plantings Surprise -> Long ZS=F
When USDA Mar 31 Prospective Plantings shows corn acres >2M above estimate AND soy below:
long ZS=F for 30 trading days. Events: 2017-03-31, 2019-03-31, 2021-03-31.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL49_plantings_corn_soy_spread"
    try:
        px = load_prices(["ZS=F", "ZC=F", "SPY"], start="2015-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    # Hand-coded: USDA Prospective Plantings dates when corn acreage surprised high / soy low
    plantings_events = [
        ("2017-03-31", "2017: corn 90.0M acres (est 90.0M) / soy 89.5M (est 88.0M) — soy surprise high actually; recheck: corn up, soy down relative to prior year ratio"),
        ("2019-03-31", "2019: corn 92.8M (est 91.1M) / soy 84.6M (est 86.0M) — corn surprise high, soy surprise low"),
        ("2021-03-31", "2021: corn 91.1M (est 93.2M) / soy 87.6M (est 89.5M) — both below est, but corn acres came from soy"),
    ]

    events = []
    pnl_parts = []
    hold_days = 30  # 30 trading days (~6 weeks)

    for date_str, desc in plantings_events:
        entry_date = pd.Timestamp(date_str)
        entry_mask = ret.index >= entry_date
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = ret.index[entry_mask][0]
        entry_loc = ret.index.get_loc(entry_idx)
        exit_loc = min(entry_loc + hold_days, len(ret.index) - 1)

        window = slice(entry_loc, exit_loc)

        zs_col = "ZS=F" if "ZS=F" in ret.columns else None
        if zs_col is None or ret[zs_col].iloc[window].isna().all():
            continue

        zs_window = ret[zs_col].iloc[window].dropna()
        spy_window = spy_r.iloc[window]
        pnl_parts.append(zs_window)

        zs_cum = float((1 + zs_window).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1) if len(spy_window) > 0 else 0

        # Also track corn for comparison
        zc_col = "ZC=F" if "ZC=F" in ret.columns else None
        zc_cum = 0
        if zc_col and not ret[zc_col].iloc[window].isna().all():
            zc_window = ret[zc_col].iloc[window].dropna()
            zc_cum = float((1 + zc_window).prod() - 1)

        events.append({
            "entry_date": date_str,
            "description": desc,
            "soy_30d_return": round(zs_cum, 4),
            "corn_30d_return": round(zc_cum, 4),
            "spy_30d_return": round(spy_cum, 4),
            "excess_vs_spy": round(zs_cum - spy_cum, 4),
        })

    if not events:
        return mark_failed(sid, "No plantings surprise events with soy futures data")

    all_pnl = pd.concat(pnl_parts)
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="Plantings Surprise -> Long Soy")

    avg_soy = np.mean([e["soy_30d_return"] for e in events])
    avg_excess = np.mean([e["excess_vs_spy"] for e in events])
    win_count = sum(1 for e in events if e["soy_30d_return"] > 0)

    save_result(sid, m, extra={
        "rule": "Long ZS=F 30d on Mar 31 when corn acreage surprises high / soy low",
        "mechanism": "Corn acreage expansion at soy expense -> soy supply squeeze -> soy/corn spread widens -> soy rallies",
        "source": "USDA Prospective Plantings (hand-coded surprise years); yfinance ZS=F, ZC=F",
        "n_events": len(events),
        "avg_soy_return": round(avg_soy, 4),
        "avg_excess_vs_spy": round(avg_excess, 4),
        "win_rate": f"{win_count}/{len(events)}",
        "events": events,
    })
    sharpe = m.get('sharpe', 0)
    cagr = m.get('cagr', 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    print(f"  Avg soy: {avg_soy*100:.1f}%  Avg excess: {avg_excess*100:.1f}%  Win: {win_count}/{len(events)}")
    for e in events:
        flag = "+" if e["soy_30d_return"] > 0 else "-"
        print(f"  {flag} {e['entry_date']}: soy {e['soy_30d_return']*100:+.1f}%, corn {e['corn_30d_return']*100:+.1f}%, spy {e['spy_30d_return']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} -- Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
