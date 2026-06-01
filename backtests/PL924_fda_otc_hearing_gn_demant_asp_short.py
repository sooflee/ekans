"""PL924_fda_otc_hearing_gn_demant_asp_short — FDA OTC Hearing Aid Disruption: Short GN/Demant ASP Compression"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL924_fda_otc_hearing_gn_demant_asp_short"

    # GN.CO=GN Store Nord (Copenhagen), DEMANT.CO=Demant (Copenhagen), SOON.SW=Sonova (SIX), SPY=benchmark
    try:
        px = load_prices(["GN.CO", "DEMANT.CO", "SOON.SW", "SPY"], start="2017-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    # Check availability of key tickers
    available = {}
    for ticker in ["GN.CO", "DEMANT.CO", "SOON.SW"]:
        if ticker in px.columns and px[ticker].dropna().shape[0] >= 50:
            available[ticker] = True
            print(f"  {ticker}: {px[ticker].dropna().shape[0]} trading days of data")
        else:
            available[ticker] = False
            print(f"  {ticker}: UNAVAILABLE or insufficient data")

    # Need at least the two primary short legs
    if not (available.get("GN.CO") or available.get("DEMANT.CO")):
        return mark_failed(sid, "Neither GN.CO nor DEMANT.CO data available via yfinance")

    spy_r = daily_returns(px[["SPY"]]).iloc[:, 0].dropna()

    # Build return series for available tickers
    gn_r = daily_returns(px[["GN.CO"]]).iloc[:, 0].dropna() if available["GN.CO"] else None
    demant_r = daily_returns(px[["DEMANT.CO"]]).iloc[:, 0].dropna() if available["DEMANT.CO"] else None
    soon_r = daily_returns(px[["SOON.SW"]]).iloc[:, 0].dropna() if available["SOON.SW"] else None

    # Build common index from available series
    common_idx = spy_r.index
    if gn_r is not None:
        common_idx = common_idx.intersection(gn_r.index)
    if demant_r is not None:
        common_idx = common_idx.intersection(demant_r.index)
    if soon_r is not None:
        common_idx = common_idx.intersection(soon_r.index)

    if len(common_idx) < 100:
        # Try fallback: use US-listed proxies for hearing aid companies
        # WS Audiology (not listed), Sonova ADR might work; try alternative approach
        # If common index is too small, mark failed
        return mark_failed(sid, f"Insufficient overlapping trading data ({len(common_idx)} days) for GN.CO/DEMANT.CO/SOON.SW")

    spy_r = spy_r.reindex(common_idx).fillna(0)
    gn_r = gn_r.reindex(common_idx).fillna(0) if gn_r is not None else pd.Series(0.0, index=common_idx)
    demant_r = demant_r.reindex(common_idx).fillna(0) if demant_r is not None else pd.Series(0.0, index=common_idx)
    soon_r = soon_r.reindex(common_idx).fillna(0) if soon_r is not None else pd.Series(0.0, index=common_idx)

    # Strategy: short 40% GN.CO + 40% DEMANT.CO, long 20% SOON.SW
    # Net position: -(0.4*GN + 0.4*DEMANT) + 0.2*SOON
    # Since we're shorting: pair_r = -(0.4*gn_r + 0.4*demant_r) + 0.2*soon_r
    # But note if a ticker is unavailable, adjust weights
    if available["GN.CO"] and available["DEMANT.CO"]:
        short_wt_gn = 0.4
        short_wt_demant = 0.4
    elif available["GN.CO"]:
        short_wt_gn = 0.8
        short_wt_demant = 0.0
    else:  # only DEMANT
        short_wt_gn = 0.0
        short_wt_demant = 0.8

    long_wt_soon = 0.2 if available["SOON.SW"] else 0.0

    # Pair return = (long SOON) - (short GN + short DEMANT)
    pair_r = (long_wt_soon * soon_r) - (short_wt_gn * gn_r + short_wt_demant * demant_r)

    # Known OTC hearing aid disruption event dates
    known_events = [
        pd.Timestamp("2018-10-01"),   # Costco KS9 launch
        pd.Timestamp("2024-09-13"),   # Apple AirPods Pro OTC hearing aid FDA clearance
    ]

    hold_days = 120  # per spec
    entry_lag = 5    # T+5 per spec (let initial knee-jerk settle)
    pnl = pd.Series(0.0, index=common_idx)
    events = []

    for event_date in known_events:
        # Find T+5 entry date
        future_days = common_idx[common_idx > event_date]
        if len(future_days) < entry_lag + 10:
            print(f"Skipping {event_date}: insufficient future data (only {len(future_days)} days)")
            continue

        entry_date = future_days[entry_lag - 1]  # 0-indexed: future_days[4] = T+5
        entry_idx = common_idx.get_loc(entry_date)
        exit_idx = min(entry_idx + hold_days, len(common_idx))

        pair_slice = pair_r.iloc[entry_idx:exit_idx]
        spy_slice = spy_r.iloc[entry_idx:exit_idx]

        if len(pair_slice) < 20:
            print(f"Skipping {event_date}: too few trading days in hold window ({len(pair_slice)})")
            continue

        # Stop-loss: exit if either short leg rises >20% from entry
        # Proxy: monitor cumulative pair loss
        cum_pair = pair_slice.cumsum()
        stop_hit = cum_pair < -0.20   # pair down 20% from entry

        if stop_hit.any():
            exit_point = stop_hit.idxmax()
            pair_slice = pair_slice.loc[:exit_point]
            spy_slice = spy_r.reindex(pair_slice.index).fillna(0)

        actual_exit_idx = entry_idx + len(pair_slice)
        pnl.iloc[entry_idx:actual_exit_idx] = pair_slice.values

        cum_pair_total = float((1 + pair_slice).prod() - 1)
        cum_gn_total = float((1 + gn_r.reindex(pair_slice.index).fillna(0)).prod() - 1)
        cum_demant_total = float((1 + demant_r.reindex(pair_slice.index).fillna(0)).prod() - 1)
        cum_soon_total = float((1 + soon_r.reindex(pair_slice.index).fillna(0)).prod() - 1)
        cum_spy_total = float((1 + spy_slice).prod() - 1)

        events.append({
            "event_date": str(event_date.date()),
            "entry_date": str(entry_date.date()),
            "hold_days": len(pair_slice),
            "pair_return": round(cum_pair_total, 4),
            "gn_return": round(cum_gn_total, 4),
            "demant_return": round(cum_demant_total, 4),
            "soon_return": round(cum_soon_total, 4),
            "spy_return": round(cum_spy_total, 4),
            "alpha": round(cum_pair_total - cum_spy_total, 4),
            "stop_hit": bool(stop_hit.any()),
        })

    if not events:
        return mark_failed(sid, "no valid signal events found with sufficient data")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active trading days ({len(active_pnl)})")

    print(f"Signal events: {len(events)}, active days: {len(active_pnl)}")
    for ev in events:
        print(f"  {ev['event_date']} (entry {ev['entry_date']}): pair={ev['pair_return']*100:.1f}%, GN={ev['gn_return']*100:.1f}%, DEMANT={ev['demant_return']*100:.1f}%, SOON={ev['soon_return']*100:.1f}%, alpha={ev['alpha']*100:.1f}%, hold={ev['hold_days']}d")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="FDA OTC Hearing Aid Disruption → Short GN/DEMANT + Long SOON")
    m["n_events"] = len(events)

    avg_alpha = float(np.mean([e["alpha"] for e in events]))
    win_rate = float(np.mean([1 if e["pair_return"] > 0 else 0 for e in events]))

    save_result(sid, m, extra={
        "rule": "Short GN.CO 40% + DEMANT.CO 40% / Long SOON.SW 20% pair trade. Enter T+5 after OTC hearing aid disruption event (FDA 510(k) QDB clearance for major consumer brand OR Costco Kirkland Signature series refresh). Hold 120 trading days. Stop-loss: pair down >20%.",
        "mechanism": "OTC hearing aids certified by FDA for self-fit reduce the licensed audiologist distribution moat, compressing ASP and volumes for premium hearing aid manufacturers (GN, Demant). Sonova (SOON) has diversified exposure (cochlear implants, B2B channels) and may benefit from scale advantages in OTC-compatible products. Counter-signal to long-aging-demographics consensus.",
        "source": "yfinance (GN.CO, DEMANT.CO, SOON.SW, SPY); FDA 510(k) product code QDB database; Costco hearing aid product page",
        "n_events": len(events),
        "avg_event_alpha": round(avg_alpha, 4),
        "event_win_rate": round(win_rate, 4),
        "events": events,
    })
    print(f"Done: Sharpe={m.get('sharpe', 'N/A'):.2f}, CAGR={m.get('cagr', 'N/A')*100:.1f}%, MaxDD={m.get('max_dd', 'N/A')*100:.1f}%")


if __name__ == "__main__":
    main()
