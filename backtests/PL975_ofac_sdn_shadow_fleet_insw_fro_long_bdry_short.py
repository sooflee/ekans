"""PL975 — OFAC Shadow-Fleet SDN Tranche (>=8 Crude Tankers) -> Long INSW + FRO / Short BDRY"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL975_ofac_sdn_shadow_fleet_insw_fro_long_bdry_short"

    # Load price data
    try:
        px = load_prices(["INSW", "FRO", "BDRY", "STNG", "SPY"], start="2016-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    # Check required tickers
    for ticker in ["INSW", "FRO", "SPY"]:
        if ticker not in px.columns or px[ticker].dropna().empty:
            return mark_failed(sid, f"{ticker} price data unavailable")

    spy_r = daily_returns(px[["SPY"]]).iloc[:, 0].dropna()
    insw_r = daily_returns(px[["INSW"]]).iloc[:, 0].dropna()
    fro_r = daily_returns(px[["FRO"]]).iloc[:, 0].dropna()

    bdry_r = None
    if "BDRY" in px.columns and not px["BDRY"].dropna().empty:
        bdry_r = daily_returns(px[["BDRY"]]).iloc[:, 0].dropna()

    # OFAC Russia shadow-fleet tranche designation events
    # Each event: OFAC designated >=8 crude tankers in a single Russia-evasion tranche
    known_events = [
        pd.Timestamp("2023-10-12"),  # OFAC Sun Ship Mgmt + ~18 vessels (t=0)
        pd.Timestamp("2023-12-20"),  # OFAC Sovcomflot-linked vessels (t=0)
        pd.Timestamp("2024-02-23"),  # OFAC Sovcomflot subsidiaries ~10 tankers (t=0)
        pd.Timestamp("2025-01-10"),  # OFAC largest tranche ~180 vessels (t=0)
    ]

    # Build all entries (t+1 from event date)
    all_entries = []
    for ev in known_events:
        # Find next trading day after event
        future = insw_r.index[insw_r.index > ev]
        if len(future) > 0:
            all_entries.append((future[0], "known"))

    if not all_entries:
        return mark_failed(sid, "no signal events found in price data range")

    # Deduplicate (min 30 days apart)
    all_entries = sorted(all_entries, key=lambda x: x[0])
    deduped = []
    last_date = None
    for d, etype in all_entries:
        if last_date is None or (d - last_date).days >= 30:
            deduped.append((d, etype))
            last_date = d

    print(f"Signal events: {len(deduped)}")

    # Build daily PnL:
    # Long INSW (40%) + Long FRO (40%) + Short BDRY (20%)
    # If BDRY not available, just Long INSW 50% + Long FRO 50%
    # Hold 40 trading days with individual stop-losses
    hold = 40
    pnl = pd.Series(0.0, index=insw_r.index)
    events = []

    for entry_date, etype in deduped:
        future_days = insw_r.index[insw_r.index >= entry_date]
        if len(future_days) < 5:
            continue

        entry_idx = insw_r.index.get_loc(future_days[0])
        exit_idx = min(entry_idx + hold, len(insw_r))

        insw_slice = insw_r.iloc[entry_idx:exit_idx]
        fro_slice = fro_r.reindex(insw_slice.index).fillna(0)
        spy_slice = spy_r.reindex(insw_slice.index).fillna(0)

        if len(insw_slice) < 5:
            continue

        # Stop-loss: exit if INSW or FRO individually falls >12% from entry
        cum_insw = insw_slice.cumsum()
        cum_fro = fro_slice.cumsum()
        stop_insw = cum_insw < -0.12
        stop_fro = cum_fro < -0.12
        stop_hit = stop_insw | stop_fro

        if stop_hit.any():
            exit_point = stop_hit.idxmax()
            insw_slice = insw_slice.loc[:exit_point]
            fro_slice = fro_slice.reindex(insw_slice.index).fillna(0)
            spy_slice = spy_r.reindex(insw_slice.index).fillna(0)

        actual_end_idx = insw_r.index.get_loc(insw_slice.index[-1]) + 1

        # Build portfolio return
        if bdry_r is not None:
            bdry_slice = bdry_r.reindex(insw_slice.index).fillna(0)
            port_r = 0.4 * insw_slice + 0.4 * fro_slice - 0.2 * bdry_slice
        else:
            port_r = 0.5 * insw_slice + 0.5 * fro_slice

        pnl.iloc[entry_idx:actual_end_idx] = port_r.values

        cum_port = float((1 + port_r).prod() - 1)
        cum_spy = float((1 + spy_slice).prod() - 1)
        cum_insw_tot = float((1 + insw_slice).prod() - 1)
        cum_fro_tot = float((1 + fro_slice).prod() - 1)
        events.append({
            "entry_date": str(entry_date.date()),
            "event_type": etype,
            "hold_days": len(insw_slice),
            "port_return": round(cum_port, 4),
            "insw_return": round(cum_insw_tot, 4),
            "fro_return": round(cum_fro_tot, 4),
            "spy_return": round(cum_spy, 4),
            "alpha": round(cum_port - cum_spy, 4),
        })

    if not events:
        return mark_failed(sid, "no valid events with sufficient data")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active days ({len(active_pnl)})")

    print(f"Active days: {len(active_pnl)}, events: {len(events)}")
    for e in events:
        print(f"  {e['entry_date']} ({e['event_type']}): port={e['port_return']*100:.1f}% INSW={e['insw_return']*100:.1f}% FRO={e['fro_return']*100:.1f}% SPY={e['spy_return']*100:.1f}% alpha={e['alpha']*100:.1f}%")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="OFAC Shadow-Fleet SDN -> Long INSW+FRO")
    m["n_events"] = len(events)

    avg_alpha = float(np.mean([e["alpha"] for e in events]))
    win_rate = float(np.mean([1 if e["port_return"] > 0 else 0 for e in events]))

    save_result(sid, m, extra={
        "rule": "Long INSW (40%) + FRO (40%), short BDRY (20%) for 40 trading days when OFAC designates >=8 crude tankers in a single Russia shadow-fleet SDN tranche; stop-loss INSW or FRO down >12%",
        "mechanism": "Large OFAC SDN tranches suddenly remove shadow-fleet tankers from global supply, tightening effective tanker availability for compliant operators like INSW and FRO; simultaneously signals heightened scrutiny that suppresses shadow-fleet capacity, benefiting Western tanker rates",
        "source": "yfinance (INSW, FRO, BDRY, SPY); OFAC SDN delta CSV (treasury.gov); US Treasury Russia shipping press releases",
        "n_events": len(events),
        "avg_event_alpha": round(avg_alpha, 4),
        "event_win_rate": round(win_rate, 4),
        "events": events,
    })
    print(f"Done: {len(events)} events, Sharpe={m.get('sharpe', 'N/A'):.2f}, CAGR={m.get('cagr', 'N/A')*100:.1f}%")


if __name__ == "__main__":
    main()
