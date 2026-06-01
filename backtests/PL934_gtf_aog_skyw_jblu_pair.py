"""PL934_gtf_aog_skyw_jblu_pair — P&W GTF AOG Disclosure -> Long SKYW / Short JBLU Pair"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL934_gtf_aog_skyw_jblu_pair"

    try:
        # SKYW=SkyWest (regional CPA, no GTF exposure), JBLU=JetBlue (high GTF A320neo exposure)
        # ALK=Alaska Airlines (alternative short leg), RTX=Raytheon (GTF OEM), SPY=benchmark
        px = load_prices(["SKYW", "JBLU", "ALK", "RTX", "SPY"], start="2022-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    for ticker in ["SKYW", "JBLU"]:
        if ticker not in px.columns or px[ticker].dropna().shape[0] < 100:
            return mark_failed(sid, f"{ticker} data unavailable or insufficient")

    spy_r = daily_returns(px[["SPY"]]).iloc[:, 0].dropna()
    skyw_r = daily_returns(px[["SKYW"]]).iloc[:, 0].dropna()
    jblu_r = daily_returns(px[["JBLU"]]).iloc[:, 0].dropna()

    # Common index
    common_idx = spy_r.index.intersection(skyw_r.index).intersection(jblu_r.index)
    spy_r = spy_r.reindex(common_idx).fillna(0)
    skyw_r = skyw_r.reindex(common_idx).fillna(0)
    jblu_r = jblu_r.reindex(common_idx).fillna(0)

    # Dollar-neutral pair: long SKYW / short JBLU
    pair_r = skyw_r - jblu_r

    # RTX GTF AOG disclosure event dates (from SEC EDGAR 8-Ks, earnings releases)
    known_events = [
        pd.Timestamp("2023-09-11"),   # RTX initial $3B provision, GTF AOG disclosure
        pd.Timestamp("2023-10-26"),   # Q3 2023 earnings — quantified AOG fleet count
        pd.Timestamp("2024-02-01"),   # Q4 2023 earnings — ongoing AOG; JBLU Spirit-block anomaly
        pd.Timestamp("2024-04-23"),   # Q1 2024 earnings — AOG count update
        pd.Timestamp("2024-07-23"),   # Q2 2024 earnings — 600+ AOG disclosed
        pd.Timestamp("2024-10-23"),   # Q3 2024 earnings — further AOG detail
    ]

    hold_days = 20  # per spec: 20 trading days fixed hold
    pnl = pd.Series(0.0, index=common_idx)
    events = []

    for event_date in known_events:
        # Enter at T+1 (next trading day after event disclosure)
        future_days = common_idx[common_idx > event_date]
        if len(future_days) < hold_days:
            print(f"Skipping {event_date}: insufficient future data ({len(future_days)} days)")
            continue

        entry_date = future_days[0]
        entry_idx = common_idx.get_loc(entry_date)
        exit_idx = min(entry_idx + hold_days, len(common_idx))

        pair_slice = pair_r.iloc[entry_idx:exit_idx]
        skyw_slice = skyw_r.iloc[entry_idx:exit_idx]
        jblu_slice = jblu_r.iloc[entry_idx:exit_idx]
        spy_slice = spy_r.iloc[entry_idx:exit_idx]

        # Stop-loss: pair down >20% from entry
        cum_pair = pair_slice.cumsum()
        stop_hit = cum_pair < -0.20

        if stop_hit.any():
            exit_point = stop_hit.idxmax()
            pair_slice = pair_slice.loc[:exit_point]
            skyw_slice = skyw_r.reindex(pair_slice.index).fillna(0)
            jblu_slice = jblu_r.reindex(pair_slice.index).fillna(0)
            spy_slice = spy_r.reindex(pair_slice.index).fillna(0)

        actual_exit_idx = entry_idx + len(pair_slice)
        pnl.iloc[entry_idx:actual_exit_idx] = pair_slice.values

        cum_pair_total = float((1 + pair_slice).prod() - 1)
        cum_skyw_total = float((1 + skyw_slice).prod() - 1)
        cum_jblu_total = float((1 + jblu_slice).prod() - 1)
        cum_spy_total = float((1 + spy_slice).prod() - 1)
        is_anomalous = (event_date == pd.Timestamp("2024-02-01"))  # Spirit-block overlap

        events.append({
            "event_date": str(event_date.date()),
            "entry_date": str(entry_date.date()),
            "hold_days": len(pair_slice),
            "pair_return": round(cum_pair_total, 4),
            "skyw_return": round(cum_skyw_total, 4),
            "jblu_return": round(cum_jblu_total, 4),
            "spy_return": round(cum_spy_total, 4),
            "alpha": round(cum_pair_total - cum_spy_total, 4),
            "stop_hit": bool(stop_hit.any()),
            "anomalous_flag": is_anomalous,
        })

    if not events:
        return mark_failed(sid, "no valid signal events found with sufficient data")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active trading days ({len(active_pnl)})")

    print(f"Signal events: {len(events)}, active days: {len(active_pnl)}")
    for ev in events:
        flag = " [ANOMALOUS-Spirit]" if ev.get("anomalous_flag") else ""
        print(f"  {ev['event_date']}: pair={ev['pair_return']*100:.1f}%, SKYW={ev['skyw_return']*100:.1f}%, JBLU={ev['jblu_return']*100:.1f}%, alpha={ev['alpha']*100:.1f}%, hold={ev['hold_days']}d{flag}")

    # Also compute without the anomalous Feb 2024 event
    non_anomalous = [e for e in events if not e.get("anomalous_flag")]
    if non_anomalous:
        avg_pair_excl = float(np.mean([e["pair_return"] for e in non_anomalous]))
        win_rate_excl = float(np.mean([1 if e["pair_return"] > 0 else 0 for e in non_anomalous]))
        print(f"  Excl. Spirit anomaly ({len(non_anomalous)} events): avg pair={avg_pair_excl*100:.1f}%, win rate={win_rate_excl*100:.0f}%")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="GTF AOG Disclosure → Long SKYW / Short JBLU")
    m["n_events"] = len(events)

    avg_alpha = float(np.mean([e["alpha"] for e in events]))
    win_rate = float(np.mean([1 if e["pair_return"] > 0 else 0 for e in events]))

    save_result(sid, m, extra={
        "rule": "Long SKYW / Short JBLU (dollar-neutral 1:1 pair) at T+1 market open following RTX GTF/PW1100G grounded-aircraft-count disclosure (8-K, earnings). Hold 20 trading days. Stop-loss: pair down >20%.",
        "mechanism": "RTX GTF inspections force A320neo-family aircraft groundings, disproportionately hitting JBLU (highest US A320neo fleet exposure, ~45%) while capacity reduction on mainline routes flows incremental passengers to SkyWest CPA regional partners. SKYW's fixed-fee capacity purchase agreements provide revenue insulation from fare pressure while benefiting from demand spillover. Counter-signal to buy-the-dip on airline disruption.",
        "source": "yfinance (SKYW, JBLU, ALK, RTX, SPY); RTX SEC EDGAR 8-K filings and earnings releases (GTF AOG count disclosures, 2023-09-11 through 2024-10-23)",
        "n_events": len(events),
        "avg_event_alpha": round(avg_alpha, 4),
        "event_win_rate": round(win_rate, 4),
        "events": events,
    })
    print(f"Done: Sharpe={m.get('sharpe', 'N/A'):.2f}, CAGR={m.get('cagr', 'N/A')*100:.1f}%, MaxDD={m.get('max_dd', 'N/A')*100:.1f}%")


if __name__ == "__main__":
    main()
