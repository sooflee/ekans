"""PL890 — India DGFT Sugar Export Quota Release OGL >1 MMT -> Short CANE / Long Indian Sugar Mills Pair"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL890_india_dgft_sugar_quota_sb_short"

    # Load price data
    # CANE = Teucrium Sugar ETF (USD), Indian sugar mills in INR
    # USDINR=X for FX conversion
    try:
        px_usd = load_prices(["CANE", "SPY", "USDINR=X"], start="2011-01-01")
        if "CANE" not in px_usd.columns or px_usd["CANE"].dropna().empty:
            return mark_failed(sid, "CANE price data unavailable")
    except Exception as e:
        return mark_failed(sid, f"USD data load: {e}")

    # Try to load Indian sugar mill stocks
    try:
        px_inr = load_prices(["BALRAMCHIN.NS", "EIDPARRY.NS"], start="2011-01-01")
    except Exception as e:
        px_inr = pd.DataFrame()  # Will handle below

    # CANE returns (USD)
    cane_r = daily_returns(px_usd[["CANE"]]).iloc[:, 0].dropna()
    spy_r = daily_returns(px_usd[["SPY"]]).iloc[:, 0].dropna()

    # USDINR for FX conversion
    usdinr = px_usd.get("USDINR=X", pd.Series(dtype=float)).dropna()

    # Indian mill returns (INR-denominated)
    have_mills = False
    mill_r = None
    if not px_inr.empty:
        balram_col = "BALRAMCHIN.NS" if "BALRAMCHIN.NS" in px_inr.columns else None
        eidparry_col = "EIDPARRY.NS" if "EIDPARRY.NS" in px_inr.columns else None
        if balram_col and eidparry_col:
            balram_r = daily_returns(px_inr[[balram_col]]).iloc[:, 0].dropna()
            eidparry_r = daily_returns(px_inr[[eidparry_col]]).iloc[:, 0].dropna()
            # Equal-weight mills basket
            common_idx = balram_r.index.intersection(eidparry_r.index)
            if len(common_idx) > 100:
                mill_r = 0.5 * balram_r.reindex(common_idx).fillna(0) + 0.5 * eidparry_r.reindex(common_idx).fillna(0)
                have_mills = True
        elif balram_col:
            mill_r = daily_returns(px_inr[[balram_col]]).iloc[:, 0].dropna()
            have_mills = True
        elif eidparry_col:
            mill_r = daily_returns(px_inr[[eidparry_col]]).iloc[:, 0].dropna()
            have_mills = True

    # Known DGFT OGL release events (sugar export quota release = short CANE / long mills)
    # and ban events (suspend exports = long CANE / short mills)
    # Strategy: OGL-release events = short CANE, long mills
    release_events = [
        pd.Timestamp("2021-11-01"),  # DGFT allowed 6 MMT sugar export for MY2021-22
        pd.Timestamp("2022-05-24"),  # DGFT 10 MMT quota for FY2022-23
        pd.Timestamp("2024-09-15"),  # DGFT MY2024-25 policy announcement
    ]

    # Ban events (inverse signal: long CANE, short mills) - mark as negative alpha opportunity
    # Not included in primary signal (direction is reversed, but note in events)
    ban_events = [
        pd.Timestamp("2023-10-15"),  # DGFT suspended ALL exports for MY2023-24
    ]

    # Since Indian mill data may be sparse, also add a proxy approach:
    # Use CANE rolling z-score vs sugar season fundamentals (Oct-Feb peak demand)
    # Seasonal signal: India harvest/export announcement windows overlap with
    # CANE being elevated before export release brings supply
    try:
        # Monthly proxy: CANE price 4-week z-score during Oct-Jan (peak harvest window)
        cane_px = px_usd["CANE"].dropna()
        cane_monthly = cane_px.resample("W").last().dropna()
        cane_roll = cane_monthly.rolling(8)  # 8-week z-score
        cane_zscore = (cane_monthly - cane_roll.mean()) / cane_roll.std()

        # Proxy signal: CANE elevated (z > 1.0) during Oct-Jan (harvest/export window)
        # This captures times when India export quota release would have price impact
        seasonal_mask = cane_zscore.index.month.isin([10, 11, 12, 1, 2])
        proxy_signal_weeks = cane_zscore[(cane_zscore > 1.0) & seasonal_mask].index

        # Convert to daily
        proxy_entries_raw = []
        last_entry = None
        for wk in proxy_signal_weeks:
            future = cane_r.index[cane_r.index >= wk]
            if len(future) > 0:
                d = future[0]
                if last_entry is None or (d - last_entry).days >= 56:  # min 8-week gap
                    proxy_entries_raw.append(d)
                    last_entry = d
    except Exception as e:
        proxy_entries_raw = []

    # Combine known events + proxy entries
    all_entries = []
    for ev in release_events:
        future = cane_r.index[cane_r.index >= ev]
        if len(future) > 0:
            all_entries.append((future[0], "release", True))

    for ev in proxy_entries_raw:
        # Check not too close to a known event
        near_known = any(abs((ev - ke).days) < 28 for ke in release_events + ban_events)
        if not near_known:
            all_entries.append((ev, "proxy", False))

    # Sort and deduplicate
    all_entries = sorted(all_entries, key=lambda x: x[0])
    deduped = []
    last_date = None
    for d, etype, is_known in all_entries:
        if last_date is None or (d - last_date).days >= 56:
            deduped.append((d, etype, is_known))
            last_date = d

    if not deduped:
        return mark_failed(sid, "no signal events found")

    print(f"Signal events: {len(deduped)} ({sum(1 for x in deduped if x[2])} known, {sum(1 for x in deduped if not x[2])} proxy)")

    # Build daily PnL: hold 15 trading days
    # Primary: short CANE only (if no mill data)
    # Full pair: short CANE + long mills (if mill data available)
    hold = 15
    pnl = pd.Series(0.0, index=cane_r.index)
    events = []

    for entry_date, etype, is_known in deduped:
        future_days = cane_r.index[cane_r.index >= entry_date]
        if len(future_days) < 5:
            continue

        entry_idx = cane_r.index.get_loc(future_days[0])
        exit_idx = min(entry_idx + hold, len(cane_r))

        cane_slice = cane_r.iloc[entry_idx:exit_idx]
        spy_slice = spy_r.reindex(cane_slice.index).fillna(0)

        if len(cane_slice) < 5:
            continue

        # Short CANE
        short_cane = -cane_slice

        # Stop-loss: if CANE rises >5% cumulative, stop out
        cum_cane = cane_slice.cumsum()
        stop_hit = cum_cane > 0.05
        take_profit_hit = (-cum_cane) > 0.08  # CANE down >8%

        stop_or_tp = stop_hit | take_profit_hit
        if stop_or_tp.any():
            exit_point = stop_or_tp.idxmax()
            cane_slice = cane_slice.loc[:exit_point]
            short_cane = -cane_slice
            spy_slice = spy_r.reindex(cane_slice.index).fillna(0)

        actual_end_idx = cane_r.index.get_loc(cane_slice.index[-1]) + 1

        if have_mills and mill_r is not None:
            mill_slice = mill_r.reindex(cane_slice.index).fillna(0)
            # Pair: short CANE + long mills (USD-normalized, equal notional)
            # Indian returns in INR; approximate USD conversion via USDINR daily changes
            usdinr_r_slice = usdinr.pct_change().reindex(cane_slice.index).fillna(0)
            mill_usd = mill_slice - usdinr_r_slice  # approximate USD return of INR asset
            pair_pnl = short_cane + mill_usd
        else:
            # CANE-only short
            pair_pnl = short_cane

        pnl.iloc[entry_idx:actual_end_idx] = pair_pnl.values

        cum_pair = float((1 + pair_pnl).prod() - 1)
        cum_spy = float((1 + spy_slice).prod() - 1)
        events.append({
            "entry_date": str(entry_date.date()),
            "event_type": etype,
            "hold_days": len(pair_pnl),
            "pair_return": round(cum_pair, 4),
            "spy_return": round(cum_spy, 4),
            "alpha": round(cum_pair - cum_spy, 4),
            "stop_hit": bool(stop_hit.any() if len(stop_hit) else False),
        })

    if not events:
        return mark_failed(sid, "no valid events with sufficient data")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 30:
        return mark_failed(sid, f"insufficient active days ({len(active_pnl)})")

    print(f"Active days: {len(active_pnl)}, events: {len(events)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="India DGFT Sugar Quota → Short CANE + Long Mills")
    m["n_events"] = len(events)

    avg_alpha = float(np.mean([e["alpha"] for e in events]))
    win_rate = float(np.mean([1 if e["pair_return"] > 0 else 0 for e in events]))

    save_result(sid, m, extra={
        "rule": "Short CANE ETF (+ long BALRAMCHIN.NS/EIDPARRY.NS mills pair when available) for 15 trading days following India DGFT OGL sugar export quota release >= 1 MMT; stop-loss if CANE +5%, take-profit if CANE -8%",
        "mechanism": "India OGL sugar export quota release floods global sugar supply (India is #1-2 world producer); ICE NY11 / CANE depresses on additional supply, while Indian mill stocks benefit from export revenue; the pair trade captures both legs",
        "source": "yfinance (CANE, SPY, BALRAMCHIN.NS, EIDPARRY.NS, USDINR=X); DGFT OGL notification dates from USDA FAS GAIN India sugar reports",
        "n_events": len(events),
        "avg_event_alpha": round(avg_alpha, 4),
        "event_win_rate": round(win_rate, 4),
        "have_mills_data": have_mills,
        "events": events,
    })
    print(f"Done: {len(events)} events, Sharpe={m.get('sharpe', 'N/A'):.2f}, CAGR={m.get('cagr', 'N/A')*100:.1f}%")


if __name__ == "__main__":
    main()
