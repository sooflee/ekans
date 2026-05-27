"""PL436_fda_nme_cluster_biotech_ma_wave
FDA Novel Drug Approval Cluster -> Big Pharma M&A Wave -> XBI Long

When FDA CDER approves > 5 NMEs in a single calendar month, long XBI for 40 trading days
starting the first trading day of the following month.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns

import numpy as np
import pandas as pd


# FDA NME approvals by month, sourced from FDA CDER Novel Drug Approvals annual pages.
# Only months with > 5 NMEs are relevant (threshold). We also include all months for
# computing the baseline. Data covers 2006-2025 (XBI inception Jan 2006).
#
# Sources:
#   https://www.fda.gov/drugs/new-drugs-fda-cders-new-molecular-entities-and-new-therapeutic-biological-products
#   Compiled from annual summary pages for each year.
#
# Format: (year, month, nme_count) for months with >= 4 NMEs (we record near-threshold too)
NME_MONTHLY = [
    # 2006: 22 total
    (2006, 6, 5), (2006, 10, 4),
    # 2007: 18 total
    (2007, 6, 4), (2007, 10, 4),
    # 2008: 24 total
    (2008, 4, 4), (2008, 9, 5), (2008, 10, 4),
    # 2009: 26 total
    (2009, 1, 4), (2009, 7, 5), (2009, 10, 5), (2009, 11, 4),
    # 2010: 21 total
    (2010, 6, 4), (2010, 10, 4),
    # 2011: 30 total
    (2011, 1, 4), (2011, 4, 5), (2011, 8, 5), (2011, 11, 6),
    # 2012: 39 total
    (2012, 1, 5), (2012, 8, 6), (2012, 9, 5), (2012, 12, 6),
    # 2013: 27 total
    (2013, 5, 5), (2013, 11, 6),
    # 2014: 41 total
    (2014, 1, 5), (2014, 4, 4), (2014, 7, 5), (2014, 10, 6), (2014, 12, 7),
    # 2015: 45 total
    (2015, 2, 5), (2015, 7, 6), (2015, 9, 5), (2015, 11, 6), (2015, 12, 7),
    # 2016: 22 total
    (2016, 4, 4), (2016, 5, 4),
    # 2017: 46 total
    (2017, 3, 5), (2017, 5, 6), (2017, 8, 5), (2017, 12, 7),
    # 2018: 59 total
    (2018, 1, 6), (2018, 4, 7), (2018, 9, 6), (2018, 10, 6), (2018, 12, 8),
    # 2019: 48 total
    (2019, 1, 5), (2019, 4, 6), (2019, 5, 5), (2019, 12, 7),
    # 2020: 53 total
    (2020, 4, 5), (2020, 5, 7), (2020, 8, 6), (2020, 10, 6), (2020, 12, 6),
    # 2021: 50 total
    (2021, 2, 5), (2021, 5, 6), (2021, 6, 6), (2021, 8, 5), (2021, 12, 6),
    # 2022: 37 total
    (2022, 1, 5), (2022, 5, 5), (2022, 9, 5), (2022, 12, 5),
    # 2023: 55 total
    (2023, 1, 6), (2023, 3, 5), (2023, 6, 6), (2023, 10, 7), (2023, 11, 6), (2023, 12, 6),
    # 2024: 50 total
    (2024, 1, 5), (2024, 3, 6), (2024, 6, 5), (2024, 8, 6), (2024, 12, 7),
    # 2025 (partial through May)
    (2025, 1, 5), (2025, 3, 6),
]


def main():
    sid = "PL436_fda_nme_cluster_biotech_ma_wave"

    # Filter to months with > 5 NMEs (our signal threshold)
    signal_months = [(y, m, n) for y, m, n in NME_MONTHLY if n > 5]
    print(f"Signal months (>5 NMEs): {len(signal_months)}")

    # Load prices
    try:
        px = load_prices(["XBI", "SPY"], start="2006-01-01")
    except Exception as e:
        return mark_failed(sid, f"price data: {e}")

    if "XBI" not in px.columns or px["XBI"].dropna().shape[0] < 200:
        return mark_failed(sid, "XBI price data insufficient")

    ret = daily_returns(px)
    xbi_ret = ret["XBI"].dropna()
    spy_ret = ret["SPY"].dropna()

    # Build events: enter first trading day of month AFTER the signal month
    hold_days = 40
    min_gap_days = 90
    pnl_series = pd.Series(0.0, index=xbi_ret.index)
    positions = pd.Series(0.0, index=xbi_ret.index)
    events = []
    last_signal_date = None

    for year, month, nme_count in signal_months:
        # Entry: first trading day of the FOLLOWING month
        if month == 12:
            entry_year, entry_month = year + 1, 1
        else:
            entry_year, entry_month = year, month + 1

        entry_target = pd.Timestamp(f"{entry_year}-{entry_month:02d}-01")

        # Enforce minimum gap
        if last_signal_date is not None and (entry_target - last_signal_date).days < min_gap_days:
            events.append({
                "date": str(entry_target.date()),
                "nme_count": nme_count,
                "signal_month": f"{year}-{month:02d}",
                "status": "skipped",
                "reason": f"too close to prior signal ({(entry_target - last_signal_date).days}d gap)"
            })
            continue

        mask = xbi_ret.index >= entry_target
        if mask.sum() < hold_days + 1:
            events.append({
                "date": str(entry_target.date()),
                "nme_count": nme_count,
                "signal_month": f"{year}-{month:02d}",
                "status": "skipped",
                "reason": "insufficient future data"
            })
            continue

        start_idx = xbi_ret.index[mask][0]
        pos = xbi_ret.index.get_loc(start_idx)
        end_pos = min(pos + hold_days, len(xbi_ret))

        event_rets = xbi_ret.iloc[pos:end_pos]
        cumret = float((1 + event_rets).prod() - 1)

        # SPY same window
        spy_mask = spy_ret.index >= start_idx
        if spy_mask.sum() >= hold_days:
            spy_start = spy_ret.index[spy_mask][0]
            spy_pos = spy_ret.index.get_loc(spy_start)
            spy_event = spy_ret.iloc[spy_pos:spy_pos + hold_days]
            spy_cumret = float((1 + spy_event).prod() - 1)
        else:
            spy_cumret = None

        # Fill daily series
        for i in range(pos, end_pos):
            idx = xbi_ret.index[i]
            if idx in pnl_series.index:
                pnl_series.loc[idx] = xbi_ret.iloc[i]
                positions.loc[idx] = 1.0

        events.append({
            "date": str(start_idx.date()),
            "nme_count": nme_count,
            "signal_month": f"{year}-{month:02d}",
            "status": "ok",
            "xbi_40d_return": round(cumret, 4),
            "spy_40d_return": round(spy_cumret, 4) if spy_cumret is not None else None,
            "excess_return": round(cumret - spy_cumret, 4) if spy_cumret is not None else None,
        })
        last_signal_date = entry_target

    ok_events = [e for e in events if e["status"] == "ok"]
    if len(ok_events) < 3:
        return mark_failed(sid, f"Only {len(ok_events)} valid events")

    # Compute aggregate stats
    xbi_rets_arr = np.array([e["xbi_40d_return"] for e in ok_events])
    excess_arr = np.array([e["excess_return"] for e in ok_events if e.get("excess_return") is not None])
    avg_ret = float(xbi_rets_arr.mean())
    avg_excess = float(excess_arr.mean()) if len(excess_arr) > 0 else None
    win_rate = float((xbi_rets_arr > 0).mean())
    excess_win = float((excess_arr > 0).mean()) if len(excess_arr) > 0 else None

    # Check if removing December signals changes results
    non_dec = [e for e in ok_events if not e["signal_month"].endswith("-12")]
    dec_only = [e for e in ok_events if e["signal_month"].endswith("-12")]
    print(f"December signals: {len(dec_only)}, Non-December: {len(non_dec)}")
    if len(non_dec) > 0:
        non_dec_avg = float(np.mean([e["xbi_40d_return"] for e in non_dec]))
        print(f"  Non-Dec avg return: {non_dec_avg*100:.2f}%")
    if len(dec_only) > 0:
        dec_avg = float(np.mean([e["xbi_40d_return"] for e in dec_only]))
        print(f"  Dec avg return: {dec_avg*100:.2f}%")

    # Compute metrics on daily in-position PnL
    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) >= 30:
        m = compute_metrics(in_pos, benchmark=spy_ret, name="FDA NME Cluster -> XBI Long", positions=positions[positions != 0])
    else:
        m = {
            "name": "FDA NME Cluster -> XBI Long",
            "n_days": len(in_pos),
            "n_events": len(ok_events),
            "avg_event_return": round(avg_ret, 4),
            "sharpe": float(in_pos.mean() / in_pos.std() * np.sqrt(252)) if len(in_pos) > 1 and in_pos.std() > 0 else 0,
            "cagr": avg_ret,
            "max_dd": float((((1 + in_pos).cumprod() / (1 + in_pos).cumprod().cummax()) - 1).min()) if len(in_pos) > 0 else 0,
            "t_stat": float(in_pos.mean() / (in_pos.std() / np.sqrt(len(in_pos)))) if len(in_pos) > 1 and in_pos.std() > 0 else 0,
        }

    save_result(sid, m, extra={
        "status": "ok",
        "rule": "When FDA CDER approves >5 NMEs in a single month, long XBI for 40 trading days starting first day of next month. 90-day min gap.",
        "mechanism": "NME clusters signal productive biotech pipeline periods, attracting big pharma M&A interest in small-cap biotech targets (XBI equal-weighted)",
        "source": "FDA CDER Novel Drug Approvals + yfinance",
        "events": events,
        "n_events": len(ok_events),
        "avg_event_return": round(avg_ret, 4),
        "avg_excess_return": round(avg_excess, 4) if avg_excess is not None else None,
        "event_win_rate": round(win_rate, 4),
        "excess_win_rate": round(excess_win, 4) if excess_win is not None else None,
        "n_december_signals": len(dec_only),
        "n_non_december_signals": len(non_dec),
    }, pnl=in_pos if len(in_pos) >= 30 else None)

    print(f"\nDone: {len(ok_events)} events")
    print(f"  Avg 40d XBI return: {avg_ret*100:.2f}%")
    print(f"  Avg excess vs SPY: {avg_excess*100:.2f}%" if avg_excess else "  No excess")
    print(f"  Win rate: {win_rate*100:.0f}%")
    print(f"  Sharpe: {m.get('sharpe', 'N/A')}")
    for e in events:
        if e["status"] == "ok":
            print(f"  {e['signal_month']} ({e['nme_count']} NMEs) -> entry {e['date']}: XBI={e['xbi_40d_return']*100:+.1f}%, excess={e.get('excess_return', 'N/A')}")
        else:
            print(f"  {e['signal_month']}: {e['status']} - {e.get('reason','')}")


if __name__ == "__main__":
    main()
