"""PL943_eu_cbam_scope_expansion_lyb_long — EU CBAM Scope Expansion to Polymers/Chemicals: Long LYB vs Short BASFY"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL943_eu_cbam_scope_expansion_lyb_long"

    try:
        px = load_prices(["LYB", "BASFY", "DOW", "XLB", "SPY"], start="2015-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    for t in ["LYB", "BASFY", "SPY"]:
        if t not in px.columns or px[t].dropna().shape[0] < 100:
            return mark_failed(sid, f"{t} data unavailable or insufficient")

    spy_r = daily_returns(px[["SPY"]]).iloc[:, 0].dropna()
    lyb_r = daily_returns(px[["LYB"]]).iloc[:, 0].dropna()
    basfy_r = daily_returns(px[["BASFY"]]).iloc[:, 0].dropna()

    # DOW IPO'd March 2019 - use as secondary long for post-2019 events
    dow_available = "DOW" in px.columns and px["DOW"].dropna().shape[0] > 100
    if dow_available:
        dow_r = daily_returns(px[["DOW"]]).iloc[:, 0].dropna()
    else:
        dow_r = None

    common_idx = spy_r.index.intersection(lyb_r.index).intersection(basfy_r.index)
    spy_r = spy_r.reindex(common_idx)
    lyb_r = lyb_r.reindex(common_idx)
    basfy_r = basfy_r.reindex(common_idx)

    # Pair: long LYB / short BASFY (dollar-neutral)
    pair_r = lyb_r - basfy_r

    # For 2022 event: add DOW if available
    if dow_available:
        dow_r_common = dow_r.reindex(common_idx).fillna(0)

    # Known CBAM announcement events
    known_events = [
        {"date": pd.Timestamp("2021-07-14"), "name": "Fit-for-55 CBAM proposal", "use_dow": False},
        {"date": pd.Timestamp("2022-06-22"), "name": "EU Council CBAM agreement", "use_dow": True},
    ]

    hold_days = 60  # ~12 weeks

    pnl = pd.Series(0.0, index=common_idx)
    events = []

    for ev_spec in known_events:
        event_date = ev_spec["date"]
        use_dow = ev_spec["use_dow"] and dow_available

        future = common_idx[common_idx >= event_date]
        if len(future) == 0:
            print(f"  No trading days at or after {event_date.date()}")
            continue
        entry_date = future[0]
        entry_idx = common_idx.get_loc(entry_date)
        exit_idx = min(entry_idx + hold_days, len(common_idx))

        if use_dow:
            # For 2022+ event: long 50% LYB + 50% DOW vs short BASFY
            long_r = 0.5 * lyb_r + 0.5 * dow_r_common
        else:
            # For 2021 event: long LYB only vs short BASFY
            long_r = lyb_r

        ev_pair_r = long_r - basfy_r

        pair_slice = ev_pair_r.iloc[entry_idx:exit_idx]
        spy_slice = spy_r.iloc[entry_idx:exit_idx]

        # Stop-loss: long underperforms short by >8%
        cum_pair = pair_slice.cumsum()
        stop_hit = cum_pair < -0.08
        # Take profit: long outperforms short by >15%
        tp_hit = cum_pair > 0.15

        if (stop_hit | tp_hit).any():
            exit_point = (stop_hit | tp_hit).idxmax()
            pair_slice = pair_slice.loc[:exit_point]
            spy_slice = spy_r.reindex(pair_slice.index).fillna(0)

        actual_exit_idx = entry_idx + len(pair_slice)

        # Check for overlaps with prior events
        target_slice = pnl.iloc[entry_idx:actual_exit_idx]
        if (target_slice != 0).sum() > 0:
            print(f"  Skipping {event_date.date()} — overlaps with prior event")
            continue

        pnl.iloc[entry_idx:actual_exit_idx] = pair_slice.values

        cum_pair_total = float((1 + pair_slice).prod() - 1)
        cum_lyb = float((1 + lyb_r.reindex(pair_slice.index).fillna(0)).prod() - 1)
        cum_basfy = float((1 + basfy_r.reindex(pair_slice.index).fillna(0)).prod() - 1)
        cum_spy = float((1 + spy_slice).prod() - 1)

        events.append({
            "event_date": str(event_date.date()),
            "event_name": ev_spec["name"],
            "entry_date": str(entry_date.date()),
            "hold_days": len(pair_slice),
            "pair_return": round(cum_pair_total, 4),
            "lyb_return": round(cum_lyb, 4),
            "basfy_return": round(cum_basfy, 4),
            "spy_return": round(cum_spy, 4),
            "alpha": round(cum_pair_total - cum_spy, 4),
            "stop_hit": bool(stop_hit.any() if len(stop_hit) else False),
            "tp_hit": bool(tp_hit.any() if len(tp_hit) else False),
            "used_dow": use_dow,
        })
        print(f"  {event_date.date()} ({ev_spec['name']}): pair={cum_pair_total*100:.1f}%, LYB={cum_lyb*100:.1f}%, BASFY={cum_basfy*100:.1f}%, SPY={cum_spy*100:.1f}%")

    if not events:
        return mark_failed(sid, "no valid event dates found in data")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active trading days ({len(active_pnl)})")

    print(f"Active PnL days: {len(active_pnl)}")
    m = compute_metrics(active_pnl, benchmark=spy_r, name="EU CBAM Scope → Long LYB / Short BASFY")
    m["n_events"] = len(events)

    avg_alpha = float(np.mean([e["alpha"] for e in events]))
    win_rate = float(np.mean([1 if e["pair_return"] > 0 else 0 for e in events]))

    save_result(sid, m, extra={
        "rule": "At EU CBAM announcement milestones (Fit-for-55 proposal Jul 2021, EU Council agreement Jun 2022), enter dollar-neutral: long LYB (+ DOW for post-2019 events) vs short BASFY. Hold up to 60 trading days. Stop: -8% spread adverse; take-profit: +15% spread.",
        "mechanism": "EU CBAM expansion to chemicals/polymers would disadvantage EU producers (BASF) who face carbon costs on domestic production vs US Gulf petchem producers (LYB, DOW) with lower embedded emissions in feedstocks. Each CBAM announcement milestone signals future cost divergence.",
        "source": "yfinance (LYB, DOW, BASFY, XLB, SPY); EU CBAM milestones from EUR-Lex public docket",
        "tickers_used": ["LYB", "BASFY"] + (["DOW"] if dow_available else []),
        "n_events": len(events),
        "avg_event_alpha": round(avg_alpha, 4),
        "event_win_rate": round(win_rate, 4),
        "events": events,
    })
    print(f"Done: Sharpe={m.get('sharpe', 'N/A'):.2f}, CAGR={m.get('cagr', 'N/A')*100:.1f}%, MaxDD={m.get('max_dd', 'N/A')*100:.1f}%")


if __name__ == "__main__":
    main()
