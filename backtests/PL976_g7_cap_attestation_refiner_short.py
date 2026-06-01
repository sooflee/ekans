"""PL976_g7_cap_attestation_refiner_short — G7 Price-Cap Attestation Tightening: Short Indian Refiner Long US Refiner Pair"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL976_g7_cap_attestation_refiner_short"
    # Known G7/EU price-cap attestation tightening events
    # 2022-12-05: G7/EU price cap launch
    # 2024-02-05: EU Attestation 3.0 package
    # 2024-10-11: OFAC tightening + shadow fleet SDN designations
    trigger_dates = pd.to_datetime(["2022-12-05", "2024-02-05", "2024-10-11"])

    try:
        # RELIANCE.NS is the Indian refiner proxy (30-40% Russian crude inputs)
        # VLO is the US refiner long leg
        px = load_prices(["RELIANCE.NS", "VLO", "SPY"], start="2021-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if px.empty or "RELIANCE.NS" not in px.columns or "VLO" not in px.columns:
        return mark_failed(sid, "missing price data for RELIANCE.NS or VLO")

    ret = daily_returns(px)
    reliance_r = ret["RELIANCE.NS"]
    vlo_r = ret["VLO"]
    spy_r = ret["SPY"]

    hold = 40  # 40 trading days (~8 weeks)
    pnl = pd.Series(0.0, index=spy_r.index)
    events = []

    for td in trigger_dates:
        # Find the first trading day on or after the trigger date
        future_mask = spy_r.index >= td
        if future_mask.sum() < hold:
            continue
        entry_idx = spy_r.index[future_mask][0]
        p = spy_r.index.get_loc(entry_idx)
        ep = min(p + hold, len(spy_r))

        # Pair trade: short RELIANCE.NS, long VLO (equal notional)
        # PnL = VLO return - RELIANCE.NS return (long VLO / short RELIANCE)
        if entry_idx not in reliance_r.index or entry_idx not in vlo_r.index:
            continue

        rp = reliance_r.index.get_loc(entry_idx)
        vp = vlo_r.index.get_loc(entry_idx)

        rend = min(rp + hold, len(reliance_r))
        vend = min(vp + hold, len(vlo_r))

        reliance_window = reliance_r.iloc[rp:rend]
        vlo_window = vlo_r.iloc[vp:vend]

        # Align on common dates
        common_dates = reliance_window.index.intersection(vlo_window.index)
        if len(common_dates) < 5:
            continue

        pair_pnl = vlo_window.loc[common_dates] - reliance_window.loc[common_dates]

        rel_ret = float((1 + reliance_window.loc[common_dates]).prod() - 1)
        vlo_ret = float((1 + vlo_window.loc[common_dates]).prod() - 1)
        pair_ret = vlo_ret - rel_ret

        spy_window = spy_r.iloc[p:ep]
        spy_ret = float((1 + spy_window).prod() - 1)

        # Add to pnl series
        for dt, val in pair_pnl.items():
            if dt in pnl.index:
                pnl.loc[dt] += val

        events.append({
            "trigger_date": str(td.date()),
            "entry_date": str(entry_idx.date()),
            "reliance_return": round(rel_ret, 4),
            "vlo_return": round(vlo_ret, 4),
            "pair_return": round(pair_ret, 4),
            "spy_return": round(spy_ret, 4),
            "n_days": len(common_dates),
        })

    print(f"Events: {len(events)}")
    for e in events:
        print(f"  {e['trigger_date']}: pair={e['pair_return']:.2%}, "
              f"RELIANCE={e['reliance_return']:.2%}, VLO={e['vlo_return']:.2%}, "
              f"SPY={e['spy_return']:.2%}")

    if not events:
        return mark_failed(sid, "no valid events found")

    # Use active PnL days only
    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="G7 Cap Attestation: Short RELIANCE.NS / Long VLO")

    pair_returns = [e["pair_return"] for e in events]
    mean_pair = float(np.mean(pair_returns))
    win_rate = float(np.mean([r > 0 for r in pair_returns]))

    save_result(sid, m, extra={
        "rule": "Short RELIANCE.NS / Long VLO 40 trading days when G7/EU price-cap attestation regime tightens with >=2 shadow fleet entity designations",
        "mechanism": "Indian refiners (heavy Russian crude intake) face margin compression as attestation tightens discount access; US refiners benefit from redirected light-sweet barrels",
        "source": "Known events: 2022-12-05 G7 cap launch, 2024-02-05 EU Attestation 3.0, 2024-10-11 OFAC tightening",
        "n_events": len(events),
        "mean_pair_return": round(mean_pair, 4),
        "win_rate": round(win_rate, 4),
        "events": events,
    })
    print(f"Done. Mean pair return: {mean_pair:.2%}, Win rate: {win_rate:.0%}")


if __name__ == "__main__":
    main()
