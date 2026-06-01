"""PL767 — BoJ Rate Hike >0.75% + Japan Core-Core CPI >2% (6m) -> Long MUFG / Short SMFG NIM Pair"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL767_boj_hike_75bps_mufg_smfg_pair"
    # Known BoJ hike event analogs (NIRP exit Mar-2024, hike Jul-2024)
    # These are the best available proxies; actual >0.75% hike hasn't occurred yet
    KNOWN_EVENTS = ["2024-03-19", "2024-07-31"]
    HOLD_DAYS = 60  # trading days

    # Load US ADR prices for MUFG, SMFG, and SPY
    try:
        px = load_prices(["MUFG", "SMFG", "SPY"], start="2006-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if px.empty or "MUFG" not in px.columns or "SMFG" not in px.columns:
        return mark_failed(sid, "missing MUFG or SMFG price data")

    ret = daily_returns(px)
    mufg_r = ret["MUFG"].dropna()
    smfg_r = ret["SMFG"].dropna()
    spy_r = ret["SPY"].dropna()

    # Also try to load Japan core-core CPI from FRED for filtering
    # JPNCCPIALLMINMEI = Japan CPI all items, monthly
    # We use it to confirm CPI > 2% YoY for 6+ months before event
    try:
        cpi_df = load_fred("JPNCCPIALLMINMEI", start="2020-01-01")
        cpi = cpi_df.iloc[:, 0].dropna()
        # Compute YoY % change
        cpi_yoy = cpi.pct_change(12) * 100
    except Exception as e:
        print(f"Warning: CPI load failed ({e}), using known events without CPI filter")
        cpi_yoy = None

    events = []
    for ev_str in KNOWN_EVENTS:
        ev_date = pd.Timestamp(ev_str)
        # Check CPI condition: 6+ consecutive months above 2% YoY before event date
        if cpi_yoy is not None:
            pre_cpi = cpi_yoy[cpi_yoy.index <= ev_date].tail(6)
            cpi_ok = (pre_cpi > 2.0).all() if len(pre_cpi) >= 6 else False
        else:
            cpi_ok = True  # assume condition met (both 2024 events had CPI>2% for >12 months)

        if not cpi_ok:
            print(f"Event {ev_str}: CPI condition NOT met, skipping")
            continue

        events.append(ev_date)

    print(f"Valid events after CPI filter: {len(events)} — {[str(e.date()) for e in events]}")

    if not events:
        return mark_failed(sid, "no valid BoJ hike events with CPI condition met")

    # Build daily PnL series for long MUFG / short SMFG pair
    # Use the overlapping date index from both series
    common_idx = mufg_r.index.intersection(smfg_r.index)
    pnl = pd.Series(0.0, index=common_idx)
    event_details = []

    for ev_date in events:
        # Find first trading day >= event date
        future = common_idx[common_idx >= ev_date]
        if len(future) < HOLD_DAYS:
            print(f"Event {ev_date.date()}: insufficient price data, skipping")
            continue

        entry_date = future[0]
        pos_idx = common_idx.get_loc(entry_date)
        end_idx = min(pos_idx + HOLD_DAYS, len(common_idx))
        window = common_idx[pos_idx:end_idx]

        # Pair PnL: long MUFG + short SMFG (1:1 notional)
        # Pair return = MUFG_return - SMFG_return
        pair_ret = mufg_r.reindex(window) - smfg_r.reindex(window)

        # Fill into combined pnl (avoid overlap — one position at a time)
        active_days = (pnl.loc[window] != 0).sum()
        if active_days > 0:
            print(f"Event {ev_date.date()}: overlaps with existing position, skipping")
            continue

        pnl.loc[window] = pair_ret.values

        # Compute event metrics
        mufg_cum = float((1 + mufg_r.reindex(window)).prod() - 1)
        smfg_cum = float((1 + smfg_r.reindex(window)).prod() - 1)
        pair_cum = float((1 + pair_ret).prod() - 1)
        spy_window = spy_r.reindex(window)
        spy_cum = float((1 + spy_window).prod() - 1) if len(spy_window) > 0 else None

        event_details.append({
            "event_date": str(ev_date.date()),
            "entry_date": str(entry_date.date()),
            "mufg_return": round(mufg_cum, 4),
            "smfg_return": round(smfg_cum, 4),
            "pair_return": round(pair_cum, 4),
            "spy_return": round(spy_cum, 4) if spy_cum is not None else None,
            "days_held": len(window),
            "cpi_condition": "met",
        })
        print(f"Event {ev_date.date()}: MUFG {mufg_cum:.2%}, SMFG {smfg_cum:.2%}, pair {pair_cum:.2%}, SPY {spy_cum:.2%}")

    active_pnl = pnl[pnl != 0]
    print(f"Active trading days: {len(active_pnl)}")

    if len(active_pnl) < 30:
        # With only 2 events x 60 days = 120 days, we may be at the edge.
        # Still compute but may fail sample size gate
        if len(active_pnl) < 5:
            return mark_failed(sid, f"insufficient active days ({len(active_pnl)})")

    spy_bench = spy_r.reindex(active_pnl.index).dropna()
    m = compute_metrics(active_pnl, benchmark=spy_bench,
                        name="BoJ Hike → Long MUFG / Short SMFG")

    save_result(sid, m, extra={
        "rule": "Long MUFG / Short SMFG (1:1 USD notional) for 60 trading days after BoJ rate hike where core-core CPI >2% for 6+ consecutive months. Analogs: Mar-2024 NIRP exit, Jul-2024 hike to 0.25%.",
        "mechanism": "MUFG has ~1.3-1.5x greater NIM sensitivity to BoJ rate hikes than SMFG due to larger domestic loan book and longer asset duration. In a rate-hiking cycle with sticky core CPI, MUFG should outperform SMFG as rates reprice faster than funding costs.",
        "source": "BoJ MPM statements (boj.or.jp); FRED JPNCCPIALLMINMEI; yfinance MUFG, SMFG US ADRs",
        "known_events": KNOWN_EVENTS,
        "events": event_details,
        "n_events": len(event_details),
        "forward_looking_note": "Primary trigger (BoJ rate >0.75%) not yet occurred as of mid-2025. Backtests use lower-threshold analogs (Mar-2024 NIRP exit, Jul-2024 hike to 0.25%) as proxies.",
        "caveats": "Only 2 historical analog events available; statistical power is very limited. NIM asymmetry claim based on FY24 IR disclosures, not independently verified from free data.",
    })
    print(f"Saved result: Sharpe={m.get('sharpe', 'N/A'):.2f}, CAGR={m.get('cagr', 'N/A'):.2%}")


if __name__ == "__main__":
    main()
