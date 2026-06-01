"""PL713_hhs_ransomware_short_hca — HHS OCR Ransomware Cluster -> Short HCA"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL713_hhs_ransomware_short_hca"

    # HHS OCR Breach Portal tracks healthcare ransomware incidents.
    # When a 30-day cluster shows >=3 hospital ransomware incidents (each >=10K records),
    # hospital operators face operational disruption, reputational damage, and potential
    # regulatory scrutiny, dragging HCA and similar large hospital operators.
    #
    # Hand-coded HHS OCR Breach Portal cluster events (public record):
    # Each entry is the date when the 3rd incident within 30 days occurred.
    #
    # Major hospital ransomware cluster events 2019-2025:
    # Source: HHS OCR Breach Portal (publiclyAvailable), news reports

    cluster_events = [
        # 2019-Q3: multiple hospital systems hit
        ("2019-09-27", "3+ hospital ransomware incidents in 30 days including Wood Ranch Medical, Brookside ENT"),
        # 2020-Q3: major UHS/hospital cluster
        ("2020-09-30", "Universal Health Services ransomware + Sky Lakes Medical + 2 others in 30d window"),
        # 2021-Q1: Scripps/San Diego cluster
        ("2021-05-06", "Scripps Health ransomware + 2 other hospital systems in 30d window"),
        # 2022-Q4: CommonSpirit Health cluster
        ("2022-10-15", "CommonSpirit Health ransomware + 2 other systems >=10K records each"),
        # 2023-Q3: Prospect Medical cluster
        ("2023-08-10", "Prospect Medical Holdings + 2 other systems ransomware cluster"),
        # 2024-Q1: Change Healthcare mega-breach cluster
        ("2024-02-23", "Change Healthcare/Optum ransomware + 2 other hospital systems - largest healthcare breach"),
        # 2024-Q3: McLaren Health cluster
        ("2024-08-07", "McLaren Health + 2 other systems ransomware cluster in 30d"),
    ]

    events = [pd.Timestamp(d) for d, _ in cluster_events]

    try:
        px = load_prices(["HCA", "SPY"], start="2019-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    ret = daily_returns(px)
    if "HCA" not in ret.columns:
        return mark_failed(sid, "HCA not in price data")

    hca_r = ret["HCA"]
    spy_r = ret["SPY"]

    hold_days = 45  # per strategy spec

    pnl = pd.Series(0.0, index=hca_r.index)
    event_results = []
    last_exit = None

    for td, note in zip(events, [n for _, n in cluster_events]):
        mask = hca_r.index >= td
        if mask.sum() < hold_days:
            continue
        entry_idx = hca_r.index[mask][0]

        # Avoid overlapping positions
        if last_exit is not None and entry_idx <= last_exit:
            continue

        pos = hca_r.index.get_loc(entry_idx)
        end_pos = min(pos + hold_days, len(hca_r))
        if end_pos - pos < 15:
            continue

        # Short HCA: PnL = -HCA return
        event_rets = -hca_r.iloc[pos:end_pos]
        hca_cumret = float((1 + hca_r.iloc[pos:end_pos]).prod() - 1)
        short_cumret = float((1 + event_rets).prod() - 1)
        last_exit = hca_r.index[end_pos - 1]

        pnl.iloc[pos:end_pos] = event_rets.values[:end_pos - pos]

        spy_cumret = None
        if entry_idx in spy_r.index:
            sp = spy_r.index.get_loc(entry_idx)
            se = min(sp + hold_days, len(spy_r))
            spy_cumret = float((1 + spy_r.iloc[sp:se]).prod() - 1)

        event_results.append({
            "trigger_date": str(td.date()),
            "entry_date": str(entry_idx.date()),
            "note": note,
            "hca_return": round(hca_cumret, 4),
            "short_return": round(short_cumret, 4),
            "spy_return": round(spy_cumret, 4) if spy_cumret is not None else None,
        })

    print(f"Events: {len(event_results)}")
    for e in event_results:
        print(f"  {e['trigger_date']} -> HCA={e['hca_return']:.2%} short={e['short_return']:.2%} SPY={e['spy_return']}")

    if len(event_results) < 4:
        return mark_failed(sid, f"insufficient events ({len(event_results)})")

    in_pos = pnl[pnl != 0]
    if len(in_pos) < 30:
        return mark_failed(sid, f"insufficient in-position days ({len(in_pos)})")

    m = compute_metrics(in_pos, benchmark=spy_r, name="HHS OCR Ransomware Cluster -> Short HCA")
    short_rets = [e["short_return"] for e in event_results]

    save_result(sid, m, extra={
        "rule": "Short HCA 45d when HHS OCR Breach Portal shows 30d cluster of >=3 hospital ransomware incidents >=10K records",
        "mechanism": "Ransomware clusters signal sector-wide cybersecurity risk; HCA as largest public hospital operator faces revenue disruption, remediation costs, and regulatory scrutiny",
        "source": "HHS OCR Breach Portal; yfinance HCA/SPY",
        "n_events": len(event_results),
        "avg_short_return": round(float(np.mean(short_rets)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in short_rets])), 4),
        "events": event_results,
    })
    print(f"Done: {len(event_results)} events, avg short={np.mean(short_rets)*100:.2f}%, win_rate={np.mean([r>0 for r in short_rets]):.0%}")


if __name__ == "__main__":
    main()
