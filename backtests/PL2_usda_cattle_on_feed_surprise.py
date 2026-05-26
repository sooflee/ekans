"""PL2_usda_cattle_on_feed_surprise — USDA Cattle on Feed Report Week vs Non-Report Week
Compare feeder cattle proxy returns during report week (3rd Friday of month) to non-report weeks.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns

import numpy as np
import pandas as pd


def third_friday(year, month):
    """Return the 3rd Friday of a given month/year."""
    import calendar
    c = calendar.Calendar(firstweekday=calendar.MONDAY)
    fridays = [d for d in c.itermonthdays2(year, month) if d[0] != 0 and d[1] == 4]
    return pd.Timestamp(year=year, month=month, day=fridays[2][0])


def main():
    sid = "PL2_usda_cattle_on_feed_surprise"

    # Try GF=F first (feeder cattle futures)
    ticker = None
    px = None
    for t in ["GF=F", "COW"]:
        try:
            test = load_prices([t], start="2015-01-01")
            if test.dropna().shape[0] > 100:
                ticker = t
                px = test
                break
        except Exception:
            continue

    if ticker is None or px is None:
        return mark_failed(sid, "Neither GF=F nor COW available on yfinance with sufficient data")

    ret = daily_returns(px)
    if isinstance(ret, pd.DataFrame):
        ret = ret.iloc[:, 0]
    ret = ret.dropna()

    # Generate 3rd Friday dates for each month in range
    report_dates = []
    for yr in range(2015, 2027):
        for mo in range(1, 13):
            try:
                rd = third_friday(yr, mo)
                if rd <= ret.index[-1]:
                    report_dates.append(rd)
            except Exception:
                continue

    # For each report date, compute 5-day forward return
    report_week_rets = []
    report_events = []
    for rd in report_dates:
        # Find the next trading day on or after rd
        mask = ret.index >= rd
        if mask.sum() < 6:
            continue
        start_idx = ret.index[mask][0]
        pos = ret.index.get_loc(start_idx)
        if pos + 5 > len(ret):
            continue
        five_day = ret.iloc[pos:pos+5]
        cumret = float((1 + five_day).prod() - 1)
        report_week_rets.append(cumret)
        report_events.append({"date": str(rd.date()), "5d_return": round(cumret, 4)})

    # Compute non-report-week 5-day returns (rolling weekly, skip report weeks)
    report_set = set(pd.Timestamp(r) for r in report_dates)
    non_report_rets = []
    for i in range(0, len(ret) - 5, 5):
        week_start = ret.index[i]
        # Check if this week overlaps with a report week
        is_report = any(abs((week_start - rd).days) < 5 for rd in report_dates if abs((week_start - rd).days) < 10)
        if is_report:
            continue
        five_day = ret.iloc[i:i+5]
        cumret = float((1 + five_day).prod() - 1)
        non_report_rets.append(cumret)

    if len(report_week_rets) < 10 or len(non_report_rets) < 10:
        return mark_failed(sid, f"insufficient events: {len(report_week_rets)} report weeks, {len(non_report_rets)} non-report weeks")

    report_arr = np.array(report_week_rets)
    non_report_arr = np.array(non_report_rets)

    # Build a daily PnL series: hold the cattle proxy only during report weeks
    pnl_series = pd.Series(0.0, index=ret.index)
    for rd in report_dates:
        mask = ret.index >= rd
        if mask.sum() < 6:
            continue
        start_idx = ret.index[mask][0]
        pos = ret.index.get_loc(start_idx)
        if pos + 5 > len(ret):
            continue
        pnl_series.iloc[pos:pos+5] = ret.iloc[pos:pos+5].values

    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) < 30:
        # Build metrics from event returns instead
        pass

    from scipy import stats
    t_test = stats.ttest_ind(report_arr, non_report_arr)

    m = compute_metrics(pnl_series, name=f"Cattle on Feed Report Week ({ticker})")

    save_result(sid, m, extra={
        "rule": f"Long {ticker} during USDA Cattle on Feed report week (3rd Friday of month, 5 trading days) vs flat otherwise",
        "mechanism": "USDA Cattle on Feed report creates short-term price discovery in feeder cattle",
        "source": "USDA NASS; yfinance",
        "ticker_used": ticker,
        "n_report_weeks": len(report_week_rets),
        "n_non_report_weeks": len(non_report_rets),
        "avg_report_week_return": round(float(report_arr.mean()), 5),
        "avg_non_report_week_return": round(float(non_report_arr.mean()), 5),
        "report_vs_non_report_t_stat": round(float(t_test.statistic), 3),
        "report_vs_non_report_p_value": round(float(t_test.pvalue), 4),
        "events": report_events[:24],
    })
    print(f"Done: Sharpe={m.get('sharpe',0):.2f}, CAGR={m.get('cagr',0)*100:.1f}%, "
          f"Report-week avg={report_arr.mean()*100:.2f}% vs Non-report avg={non_report_arr.mean()*100:.2f}%, "
          f"t={t_test.statistic:.2f}, p={t_test.pvalue:.3f}")


if __name__ == "__main__":
    main()
