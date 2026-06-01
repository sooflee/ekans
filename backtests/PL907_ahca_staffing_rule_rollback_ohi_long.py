"""PL907 — CMS SNF Staffing Rule Rollback -> OHI Tenant Coverage Relief Event Long"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL907_ahca_staffing_rule_rollback_ohi_long"

    # Load price data
    try:
        px = load_prices(["OHI", "SBRA", "XLRE", "SPY"], start="2015-01-01")
        if "OHI" not in px.columns or px["OHI"].dropna().empty:
            return mark_failed(sid, "OHI price data unavailable")
        if "SBRA" not in px.columns or px["SBRA"].dropna().empty:
            return mark_failed(sid, "SBRA price data unavailable")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    spy_r = daily_returns(px[["SPY"]]).iloc[:, 0].dropna()
    ohi_r = daily_returns(px[["OHI"]]).iloc[:, 0].dropna()
    sbra_r = daily_returns(px[["SBRA"]]).iloc[:, 0].dropna()
    xlre_r = daily_returns(px[["XLRE"]]).iloc[:, 0].dropna() if "XLRE" in px.columns else None

    # Known CMS staffing rule events
    # Positive catalyst: rollback/vacatur events -> long OHI, short SBRA
    # Negative catalyst: rule finalization -> short OHI, long SBRA
    rollback_events = [
        pd.Timestamp("2024-06-28"),   # N.D. Tex. vacates HPRD components
        # Forward: 5th Circuit confirmation if/when it occurs - use proxy below
    ]

    negative_events = [
        pd.Timestamp("2024-04-22"),   # CMS Final Rule 89 FR 40876 finalized - negative for OHI
    ]

    # Proxy signal: regulatory relief for SNF operators is visible in OHI vs SBRA spread
    # Use OHI/SBRA relative strength: when OHI has been underperforming SBRA materially
    # (due to staffing cost fears), a recovery trade can be constructed
    try:
        ohi_px = px["OHI"].dropna()
        sbra_px = px["SBRA"].dropna()

        # Align
        common = ohi_px.index.intersection(sbra_px.index)
        ohi_aligned = ohi_px.loc[common]
        sbra_aligned = sbra_px.loc[common]

        # Rolling 60-day relative performance z-score (OHI vs SBRA)
        rel_ret = ohi_r.reindex(common) - sbra_r.reindex(common)
        cum_rel = rel_ret.cumsum()
        roll_60 = cum_rel.rolling(60)
        rel_zscore = (cum_rel - roll_60.mean()) / roll_60.std()

        # Signal: OHI significantly underperforming SBRA (z < -1.5) AND
        # OHI beginning to recover (rel_ret > 0 for last 3 days) -> mean reversion long OHI
        # This proxies for regulatory relief events where OHI was oversold due to staffing fears
        recovery_signal = (rel_zscore < -1.0) & (rel_ret.rolling(3).mean() > 0)

        # Entry dates (min 6-week gap)
        proxy_entries = []
        last_entry = None
        in_signal = False
        prev_signal = False
        for d in recovery_signal.index:
            curr = recovery_signal.get(d, False)
            if curr and not prev_signal:  # signal turns on
                if last_entry is None or (d - last_entry).days >= 42:
                    proxy_entries.append(d)
                    last_entry = d
            prev_signal = curr
    except Exception as e:
        proxy_entries = []

    # Build all entry events
    all_entries = []
    for ev in rollback_events:
        future = ohi_r.index[ohi_r.index >= ev]
        if len(future) > 0:
            all_entries.append((future[0], "rollback", 1))  # +1 = long OHI

    for ev in negative_events:
        future = ohi_r.index[ohi_r.index >= ev]
        if len(future) > 0:
            all_entries.append((future[0], "rule_final", -1))  # -1 = short OHI

    for ev in proxy_entries:
        near_known = any(abs((ev - ke).days) < 28 for ke in rollback_events + negative_events)
        if not near_known:
            all_entries.append((ev, "proxy", 1))  # long OHI mean reversion

    # Sort and deduplicate
    all_entries = sorted(all_entries, key=lambda x: x[0])
    deduped = []
    last_date = None
    for d, etype, direction in all_entries:
        if last_date is None or (d - last_date).days >= 28:
            deduped.append((d, etype, direction))
            last_date = d

    if not deduped:
        return mark_failed(sid, "no signal events found")

    print(f"Signal events: {len(deduped)}")

    # Build daily PnL: hold 10 trading days
    hold = 10
    pnl = pd.Series(0.0, index=ohi_r.index)
    events = []

    for entry_date, etype, direction in deduped:
        future_days = ohi_r.index[ohi_r.index >= entry_date]
        if len(future_days) < 5:
            continue

        entry_idx = ohi_r.index.get_loc(future_days[0])
        exit_idx = min(entry_idx + hold, len(ohi_r))

        ohi_slice = ohi_r.iloc[entry_idx:exit_idx]
        sbra_slice = sbra_r.reindex(ohi_slice.index).fillna(0)
        spy_slice = spy_r.reindex(ohi_slice.index).fillna(0)

        if len(ohi_slice) < 3:
            continue

        # Long OHI short SBRA pair (direction 1 = long OHI; -1 = short OHI)
        # Include XLRE hedge (50% notional short of OHI)
        pair_r = direction * (ohi_slice - sbra_slice)
        if xlre_r is not None:
            xlre_slice = xlre_r.reindex(ohi_slice.index).fillna(0)
            # Net: OHI long + SBRA short - 0.5 XLRE (to dampen broad REIT beta)
            pair_r = direction * ohi_slice - direction * sbra_slice * 0.5 - 0.5 * xlre_slice

        # Stop-loss at -8%, take-profit at +15% (for long OHI)
        cum_spread = (direction * (ohi_slice - sbra_slice)).cumsum()
        stop_hit = cum_spread < -0.08
        tp_hit = cum_spread > 0.15
        stop_or_tp = stop_hit | tp_hit
        if stop_or_tp.any():
            exit_point = stop_or_tp.idxmax()
            pair_r = pair_r.loc[:exit_point]
            ohi_slice = ohi_slice.loc[:exit_point]
            spy_slice = spy_r.reindex(pair_r.index).fillna(0)

        actual_end_idx = ohi_r.index.get_loc(pair_r.index[-1]) + 1
        pnl.iloc[entry_idx:actual_end_idx] = pair_r.values

        cum_pair = float((1 + pair_r).prod() - 1)
        cum_spy = float((1 + spy_slice).prod() - 1)
        events.append({
            "entry_date": str(entry_date.date()),
            "event_type": etype,
            "direction": direction,
            "hold_days": len(pair_r),
            "pair_return": round(cum_pair, 4),
            "spy_return": round(cum_spy, 4),
            "alpha": round(cum_pair - cum_spy, 4),
        })

    if not events:
        return mark_failed(sid, "no valid events with sufficient data")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 30:
        return mark_failed(sid, f"insufficient active days ({len(active_pnl)})")

    print(f"Active days: {len(active_pnl)}, events: {len(events)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="CMS SNF Staffing Rollback → Long OHI vs SBRA Pair")
    m["n_events"] = len(events)

    avg_alpha = float(np.mean([e["alpha"] for e in events]))
    win_rate = float(np.mean([1 if e["pair_return"] > 0 else 0 for e in events]))

    save_result(sid, m, extra={
        "rule": "Long OHI / short SBRA (equal notional) + short XLRE 50% for 10 trading days on CMS SNF staffing rule rollback/vacatur events (court orders, CMS withdrawal) OR OHI vs SBRA mean reversion signal (z-score < -1.0 then recovering); stop-loss -8%, take-profit +15%",
        "mechanism": "CMS SNF minimum staffing mandates (3.48 HPRD) impose significant cost burdens on OHI tenants; rollback events reduce the cost burden, improving tenant EBITDARM coverage ratios and OHI dividend safety; the OHI/SBRA spread reverts as OHI re-rates to full NAV",
        "source": "yfinance (OHI, SBRA, XLRE, SPY); PACER docket N.D. Tex. 2:24-cv-00114; CMS regulations.gov CMS-3442-F; AHCA press releases",
        "n_events": len(events),
        "avg_event_alpha": round(avg_alpha, 4),
        "event_win_rate": round(win_rate, 4),
        "events": events,
        "caveat": "Only 2 confirmed regulatory events (2024); proxy signal extends sample but with lower specificity",
    })
    print(f"Done: {len(events)} events, Sharpe={m.get('sharpe', 'N/A'):.2f}, CAGR={m.get('cagr', 'N/A')*100:.1f}%")


if __name__ == "__main__":
    main()
