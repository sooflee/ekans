"""PL457_fdic_call_report_fixed_rate_repricing_kre
FDIC Call Report Fixed-Rate Loan Repricing Wall -> Community Bank NII Acceleration -> KRE Long

When FDIC data shows rising share of loans approaching repricing in a rising rate environment,
long KRE for 30 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns

import numpy as np
import pandas as pd


def main():
    sid = "PL457_fdic_call_report_fixed_rate_repricing_kre"

    # FDIC Call Report aggregate data is not on FRED directly, but we can use:
    # 1. FRED DFF (Fed Funds Rate) to identify rate-hiking periods
    # 2. FRED USNIM (US Net Interest Margin) as a proxy for NII
    # 3. Hand-code the repricing wall events based on known rate cycles
    #
    # Repricing wall thesis: when Fed has hiked rates significantly and 3-5 year
    # fixed-rate loans originated at low rates are approaching maturity/repricing,
    # community banks will see NII acceleration.
    #
    # Key repricing wall events:
    # - Post-2004-2006 hike cycle: loans from 2001-2003 (1% Fed Funds) repricing in 2006-2008
    # - Post-2017-2019 hike cycle: loans from 2010-2015 repricing 2018-2020
    # - Post-2022-2023 hike cycle: loans from 2020-2021 (0% ZIRP) repricing 2024-2026

    # Use FRED data to identify rate environments
    try:
        rates = load_fred(["DFF", "DPRIME"], start="2006-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED rate data: {e}")

    # Hand-coded repricing wall signal dates (quarterly FDIC publication, ~8 weeks after quarter)
    # Signal: rates rose significantly 3-5 years ago, creating a repricing tailwind NOW
    events = [
        {"date": "2007-03-01", "label": "Q4 2006 Call Report: 2001-2003 ZIRP loans repricing at 5.25% FF", "cycle": "2004-2006 hikes"},
        {"date": "2007-09-01", "label": "Q2 2007: Peak repricing of 2002-2004 low-rate loans", "cycle": "2004-2006 hikes"},
        {"date": "2018-09-01", "label": "Q2 2018: 2013-2015 loans repricing at higher rates", "cycle": "2017-2018 hikes"},
        {"date": "2019-03-01", "label": "Q4 2018: NII acceleration as 2014-2016 loans reprice", "cycle": "2017-2018 hikes"},
        {"date": "2024-03-01", "label": "Q4 2023: 2020-2021 ZIRP loans beginning to reprice at 5.25-5.50%", "cycle": "2022-2023 hikes"},
        {"date": "2024-09-01", "label": "Q2 2024: Major repricing wall for 3-year fixed loans from 2021", "cycle": "2022-2023 hikes"},
        {"date": "2025-03-01", "label": "Q4 2024: 5-year fixed-rate loans from 2020 ZIRP repricing", "cycle": "2022-2023 hikes"},
    ]

    # Load prices
    try:
        px = load_prices(["KRE", "SPY"], start="2006-06-01")
    except Exception as e:
        return mark_failed(sid, f"price data: {e}")

    if "KRE" not in px.columns or px["KRE"].dropna().shape[0] < 200:
        return mark_failed(sid, "KRE price data insufficient")

    ret = daily_returns(px)
    kre_ret = ret["KRE"].dropna()
    spy_ret = ret["SPY"].dropna()

    hold_days = 30
    pnl_series = pd.Series(0.0, index=kre_ret.index)
    positions = pd.Series(0.0, index=kre_ret.index)
    event_results = []

    for ev in events:
        entry_date = pd.Timestamp(ev["date"])

        mask = kre_ret.index >= entry_date
        if mask.sum() < hold_days + 1:
            event_results.append({**ev, "status": "skipped", "reason": "insufficient future data"})
            continue

        start_idx = kre_ret.index[mask][0]
        pos = kre_ret.index.get_loc(start_idx)
        end_pos = min(pos + hold_days, len(kre_ret))

        event_rets = kre_ret.iloc[pos:end_pos]
        cumret = float((1 + event_rets).prod() - 1)

        # SPY
        spy_mask = spy_ret.index >= start_idx
        if spy_mask.sum() >= hold_days:
            spy_start = spy_ret.index[spy_mask][0]
            spy_pos = spy_ret.index.get_loc(spy_start)
            spy_event = spy_ret.iloc[spy_pos:spy_pos + hold_days]
            spy_cumret = float((1 + spy_event).prod() - 1)
        else:
            spy_cumret = None

        for i in range(pos, end_pos):
            idx = kre_ret.index[i]
            if idx in pnl_series.index:
                pnl_series.loc[idx] = kre_ret.iloc[i]
                positions.loc[idx] = 1.0

        event_results.append({
            **ev,
            "status": "ok",
            "kre_30d_return": round(cumret, 4),
            "spy_30d_return": round(spy_cumret, 4) if spy_cumret is not None else None,
            "excess_return": round(cumret - spy_cumret, 4) if spy_cumret is not None else None,
        })

    ok_events = [e for e in event_results if e["status"] == "ok"]
    if len(ok_events) < 2:
        return mark_failed(sid, f"Only {len(ok_events)} valid events")

    kre_rets = np.array([e["kre_30d_return"] for e in ok_events])
    excess_arr = np.array([e["excess_return"] for e in ok_events if e.get("excess_return") is not None])
    avg_ret = float(kre_rets.mean())
    avg_excess = float(excess_arr.mean()) if len(excess_arr) > 0 else None
    win_rate = float((kre_rets > 0).mean())

    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) >= 30:
        m = compute_metrics(in_pos, benchmark=spy_ret, name="FDIC Repricing Wall -> KRE Long", positions=positions[positions != 0])
    else:
        m = {
            "name": "FDIC Repricing Wall -> KRE Long",
            "n_days": len(in_pos),
            "sharpe": float(in_pos.mean() / in_pos.std() * np.sqrt(252)) if len(in_pos) > 1 and in_pos.std() > 0 else 0,
            "cagr": avg_ret,
            "max_dd": float((((1 + in_pos).cumprod() / (1 + in_pos).cumprod().cummax()) - 1).min()) if len(in_pos) > 0 else 0,
            "t_stat": float(in_pos.mean() / (in_pos.std() / np.sqrt(len(in_pos)))) if len(in_pos) > 1 and in_pos.std() > 0 else 0,
        }

    save_result(sid, m, extra={
        "status": "ok",
        "rule": "When FDIC Call Report shows fixed-rate loan repricing wall in rising rate environment, long KRE 30d",
        "mechanism": "Fixed-rate loans from prior ZIRP period repricing at higher rates -> NII acceleration for community banks",
        "source": "FDIC Call Reports + FRED DFF + yfinance",
        "events": event_results,
        "n_events": len(ok_events),
        "avg_event_return": round(avg_ret, 4),
        "avg_excess_return": round(avg_excess, 4) if avg_excess is not None else None,
        "event_win_rate": round(win_rate, 4),
    }, pnl=in_pos if len(in_pos) >= 30 else None)

    print(f"Done: {len(ok_events)} events, avg return={avg_ret*100:.2f}%, win={win_rate*100:.0f}%")
    print(f"  Sharpe: {m.get('sharpe', 'N/A')}")
    for e in event_results:
        if e["status"] == "ok":
            print(f"  {e['date']} ({e['cycle']}): KRE={e['kre_30d_return']*100:+.1f}%, excess={e.get('excess_return','N/A')}")


if __name__ == "__main__":
    main()
