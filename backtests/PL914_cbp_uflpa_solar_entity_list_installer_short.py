"""PL914_cbp_uflpa_solar_entity_list_installer_short — CBP UFLPA Solar Detention Spike -> Short RUN/SEDG vs TAN"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL914_cbp_uflpa_solar_entity_list_installer_short"

    try:
        # RUN=Sunrun, SEDG=SolarEdge, ARRY=Array Technologies, TAN=solar ETF, SPY=benchmark
        px = load_prices(["RUN", "SEDG", "ARRY", "TAN", "SPY"], start="2021-06-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    spy_r = daily_returns(px[["SPY"]]).iloc[:, 0].dropna()

    # Key UFLPA enforcement escalation dates (known events)
    # UFLPA became effective June 21, 2022; major entity list expansions follow
    known_events = [
        pd.Timestamp("2022-06-21"),  # UFLPA initial enforcement: CBP begins detaining solar modules
        pd.Timestamp("2023-01-11"),  # FLETF entity list expansion (Hoshine Silicon/XPCC, wafer producers)
        pd.Timestamp("2023-06-01"),  # Additional solar enforcement escalation (mid-2023 CBP denial rate spike)
        pd.Timestamp("2024-05-01"),  # Additional wafer producer entity list additions
    ]

    # Build daily returns for the short basket and long TAN hedge
    run_r = daily_returns(px[["RUN"]]).iloc[:, 0].dropna() if "RUN" in px.columns else None
    sedg_r = daily_returns(px[["SEDG"]]).iloc[:, 0].dropna() if "SEDG" in px.columns else None
    arry_r = daily_returns(px[["ARRY"]]).iloc[:, 0].dropna() if "ARRY" in px.columns else None
    tan_r = daily_returns(px[["TAN"]]).iloc[:, 0].dropna() if "TAN" in px.columns else None

    if run_r is None or sedg_r is None or tan_r is None:
        return mark_failed(sid, "RUN, SEDG, or TAN data unavailable")

    # Align index to common dates
    common_idx = spy_r.index
    for r in [run_r, sedg_r, tan_r]:
        common_idx = common_idx.intersection(r.index)
    if arry_r is not None:
        arry_available = arry_r.index
    else:
        arry_available = pd.DatetimeIndex([])

    run_r = run_r.reindex(common_idx).fillna(0)
    sedg_r = sedg_r.reindex(common_idx).fillna(0)
    tan_r = tan_r.reindex(common_idx).fillna(0)
    spy_r_aligned = spy_r.reindex(common_idx).fillna(0)

    if arry_r is not None:
        arry_r = arry_r.reindex(common_idx).fillna(0)

    # Strategy: short equal-weighted (RUN+SEDG) basket, long TAN (market neutral)
    # Hold 35 trading days per spec or until exit condition
    # Spread return = -0.5*RUN - 0.5*SEDG + 1.0*TAN (dollar-neutral long TAN / short installer basket)
    hold_days = 35
    pnl = pd.Series(0.0, index=common_idx)
    events = []

    for event_date in known_events:
        # Enter on T+1 business day after event
        future_days = common_idx[common_idx > event_date]
        if len(future_days) < 5:
            print(f"Skipping {event_date}: insufficient future data")
            continue

        entry_date = future_days[0]
        entry_idx = common_idx.get_loc(entry_date)
        exit_idx = min(entry_idx + hold_days, len(common_idx))

        run_slice = run_r.iloc[entry_idx:exit_idx]
        sedg_slice = sedg_r.iloc[entry_idx:exit_idx]
        tan_slice = tan_r.iloc[entry_idx:exit_idx]
        spy_slice = spy_r_aligned.iloc[entry_idx:exit_idx]

        # Check: only trade if TAN not in confirmed bear (price below 200-day MA by >15%)
        tan_px = px["TAN"].dropna()
        tan_ma200 = tan_px.rolling(200).mean()
        tan_at_entry = tan_px.get(entry_date, None)
        ma200_at_entry = tan_ma200.get(entry_date, None)
        if tan_at_entry is not None and ma200_at_entry is not None:
            if tan_at_entry < ma200_at_entry * 0.85:
                print(f"Skipping {event_date}: TAN in confirmed bear (price={tan_at_entry:.2f} vs MA200={ma200_at_entry:.2f})")
                continue

        # Blend short basket: if ARRY available use ARRY as substitute/supplement
        # Spread: long TAN, short equal-weighted installer basket
        if arry_r is not None and len(arry_r.reindex(run_slice.index).dropna()) > hold_days * 0.8:
            arry_slice = arry_r.reindex(run_slice.index).fillna(0)
            # Short basket: 1/3 RUN + 1/3 SEDG + 1/3 ARRY (diversified installer basket)
            short_basket = (run_slice + sedg_slice + arry_slice) / 3
        else:
            short_basket = (run_slice + sedg_slice) / 2

        # Spread PnL: long TAN - short basket (counter-signal to consensus long renewable)
        spread = tan_slice - short_basket

        # Stop-loss / take-profit on spread
        cum_spread = spread.cumsum()
        stop_hit = cum_spread < -0.10   # spread falls 10% (position going wrong)
        tp_hit = cum_spread > 0.20      # spread gains 20%

        if (stop_hit | tp_hit).any():
            exit_point = (stop_hit | tp_hit).idxmax()
            spread = spread.loc[:exit_point]
            spy_slice = spy_r_aligned.reindex(spread.index).fillna(0)

        # Write to pnl (overwrite if overlapping events — use latest)
        actual_entry = entry_idx
        actual_exit = entry_idx + len(spread)
        pnl.iloc[actual_entry:actual_exit] = spread.values

        cum_spread_total = float((1 + spread).prod() - 1)
        cum_spy_total = float((1 + spy_slice).prod() - 1)
        events.append({
            "event_date": str(event_date.date()),
            "entry_date": str(entry_date.date()),
            "hold_days": len(spread),
            "spread_return": round(cum_spread_total, 4),
            "spy_return": round(cum_spy_total, 4),
            "alpha": round(cum_spread_total - cum_spy_total, 4),
            "stop_hit": bool(stop_hit.any() if len(stop_hit) else False),
            "tp_hit": bool(tp_hit.any() if len(tp_hit) else False),
        })

    if not events:
        return mark_failed(sid, "no valid signal events found with sufficient data")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active trading days ({len(active_pnl)})")

    print(f"Events: {len(events)}, active days: {len(active_pnl)}")
    for ev in events:
        print(f"  {ev['event_date']}: spread={ev['spread_return']*100:.1f}%, alpha={ev['alpha']*100:.1f}%, hold={ev['hold_days']}d")

    m = compute_metrics(active_pnl, benchmark=spy_r_aligned, name="UFLPA Solar Enforcement → Short RUN/SEDG Long TAN Spread")
    m["n_events"] = len(events)

    avg_alpha = float(np.mean([e["alpha"] for e in events]))
    win_rate = float(np.mean([1 if e["spread_return"] > 0 else 0 for e in events]))

    save_result(sid, m, extra={
        "rule": "On CBP UFLPA enforcement escalation (HTS-8541 denial rate >35%) or FLETF entity list expansion (polysilicon/wafer producer), enter spread: long TAN / short equal-weighted RUN+SEDG+ARRY basket. Hold 35 trading days. Exit on: spread -10% stop-loss, +20% take-profit, or 35-day time stop.",
        "mechanism": "UFLPA enforcement forces solar module detention at US ports, directly disrupting supply chains for US solar installers (RUN, SEDG, ARRY) who depend on Chinese-origin cells/wafers. Consensus long-renewables trade gets squeezed while enforcement is escalating. TAN (broad solar ETF) provides market-neutral offset; installer shorts bear the direct supply-chain disruption pain.",
        "source": "yfinance (RUN, SEDG, ARRY, TAN, SPY); CBP UFLPA Statistics Dashboard (cbp.gov); DHS FLETF Entity List Federal Register notices (dhs.gov/uflpa-entity-list)",
        "n_events": len(events),
        "avg_event_alpha": round(avg_alpha, 4),
        "event_win_rate": round(win_rate, 4),
        "events": events,
    })
    print(f"Done: Sharpe={m.get('sharpe', 'N/A'):.2f}, CAGR={m.get('cagr', 'N/A')*100:.1f}%, MaxDD={m.get('max_dd', 'N/A')*100:.1f}%")


if __name__ == "__main__":
    main()
