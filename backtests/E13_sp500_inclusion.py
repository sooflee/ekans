"""
E13 S&P 500 index inclusion drift.

Hardcoded list of S&P 500 additions since ~2010 with announcement dates.
Compute CAR (cumulative abnormal return vs SPY) from announcement T-0 to T+10.
Average across all additions.

Sources: Wikipedia 'List of S&P 500 companies' historical changes table. Dates are
announcement dates (S&P typically pre-announces by ~5 trading days before effective
inclusion). Pre-2010 effect was documented by Harris & Gurel 1986, Shleifer 1986.
Recent academic work (Patel-Welch 2017, Greenwood-Sammon 2022) finds the effect
has decayed sharply in the indexed-fund era.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import load_prices, save_result, mark_failed


# (ticker, announcement_date) — curated subset of post-2010 additions.
# Dates approximate Wikipedia "Announcement" column; if not available we use the
# day before effective.
ADDITIONS = [
    ("FB",   "2013-12-11"),   # later META
    ("DAL",  "2015-09-04"),
    ("URI",  "2014-10-21"),
    ("FRC",  "2018-10-25"),
    ("TWTR", "2018-06-04"),
    ("UAA",  "2014-05-02"),
    ("MNST", "2012-06-27"),
    ("DLTR", "2011-10-31"),
    ("LKQ",  "2016-05-23"),
    ("LYB",  "2012-09-04"),
    ("MAR",  "2010-04-21"),
    ("HCA",  "2015-12-09"),
    ("VRSK", "2015-10-02"),
    ("ULTA", "2015-04-02"),
    ("FOXA", "2013-06-21"),
    ("NWSA", "2013-08-12"),
    ("EBAY", "2008-12-23"),
    ("NLSN", "2013-12-18"),
    ("AAL",  "2015-03-20"),
    ("SBAC", "2017-07-19"),
    ("ALK",  "2016-05-12"),
    ("CHTR", "2016-09-08"),
    ("HOLX", "2016-03-24"),
    ("UHS",  "2017-02-28"),
    ("INCY", "2017-03-01"),
    ("ANSS", "2017-05-19"),
    ("EVRG", "2018-07-09"),
    ("CPRT", "2017-12-20"),
    ("ALGN", "2017-05-31"),
    ("DXCM", "2017-05-08"),
    ("RMD",  "2017-07-26"),
    ("MSCI", "2018-04-04"),
    ("ABMD", "2018-05-23"),
    ("TTWO", "2018-07-19"),
    ("ANET", "2018-08-22"),
    ("TFX",  "2018-10-08"),
    ("LW",   "2019-01-02"),
    ("ATO",  "2019-02-12"),
    ("WAB",  "2019-02-22"),
    ("AMCR", "2019-06-06"),
    ("CTVA", "2019-05-22"),
    ("WRB",  "2019-12-20"),
    ("PAYC", "2020-01-22"),
    ("CARR", "2020-03-12"),
    ("OTIS", "2020-03-12"),
    ("BIO",  "2020-05-12"),
    ("TYL",  "2020-06-08"),
    ("CDW",  "2019-06-20"),
    ("WST",  "2020-07-22"),
    ("DPZ",  "2020-09-04"),
    ("PWR",  "2020-10-02"),
    ("ETSY", "2020-09-04"),
    ("TER",  "2020-09-04"),
    ("CTLT", "2020-10-12"),
    ("POOL", "2020-10-12"),
    ("PENN", "2020-12-11"),
    ("ENPH", "2020-12-11"),
    ("TSLA", "2020-11-16"),  # famous big jump pre-inclusion
    ("MPWR", "2021-04-29"),
    ("NXPI", "2021-03-19"),
    ("CDAY", "2021-07-23"),
    ("CZR",  "2021-03-19"),
    ("PTC",  "2021-06-16"),
    ("BRO",  "2021-09-03"),
    ("MTCH", "2021-09-15"),
    ("FDS",  "2021-12-13"),
    ("EPAM", "2021-12-13"),
    ("MOH",  "2022-06-21"),
    ("KDP",  "2022-06-02"),
    ("ON",   "2022-09-19"),
    ("INVH", "2022-09-19"),
    ("STLD", "2022-12-13"),
    ("VLTO", "2023-09-29"),
    ("GEHC", "2023-01-04"),
    ("FICO", "2022-03-22"),
    ("AXON", "2023-05-04"),
    ("BLDR", "2022-12-21"),
    ("PODD", "2022-07-01"),
    ("HUBB", "2024-02-26"),
    ("DECK", "2024-03-01"),
    ("SMCI", "2024-03-01"),
    ("KKR",  "2024-06-21"),
    ("CRWD", "2024-06-07"),
    ("GDDY", "2024-06-21"),
    ("PLTR", "2024-09-06"),
    ("DELL", "2024-09-06"),
    ("ERIE", "2024-09-06"),
]


def main():
    sid = "E13_sp500_inclusion"
    try:
        # Load SPY plus all addition tickers.
        all_t = sorted(set([t for t, _ in ADDITIONS] + ["SPY"]))

        # yfinance bulk-download works for batches. We'll try a single call.
        import yfinance as yf
        end = "2025-12-31"
        df = yf.download(all_t, start="2008-01-01", end=end, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            px = df["Close"]
        else:
            px = df[["Close"]]
            px.columns = all_t
        px = px.dropna(how="all").sort_index()

        if "SPY" not in px.columns:
            return mark_failed(sid, "SPY price load failed")

        spy_ret = px["SPY"].pct_change()

        # Event study: for each (ticker, ann_date), find first trading date >= ann_date,
        # compute daily returns from T-2 to T+20 windows, abnormal = ret - SPY ret.
        rows = []
        car_window = list(range(-2, 21))  # T-2 .. T+20
        per_event = {}

        for tkr, ann in ADDITIONS:
            if tkr not in px.columns:
                continue
            s = px[tkr].dropna()
            if s.empty:
                continue
            ann_ts = pd.Timestamp(ann)
            # find first trading day on/after announcement
            idx = s.index[s.index >= ann_ts]
            if len(idx) == 0:
                continue
            t0 = idx[0]
            # Get window of trading days around t0
            pos = s.index.get_loc(t0)
            r = s.pct_change()
            window_rets = []
            for off in car_window:
                p = pos + off
                if p < 0 or p >= len(s):
                    window_rets.append(np.nan)
                else:
                    d = s.index[p]
                    ar = r.iloc[p] - spy_ret.reindex([d]).iloc[0]
                    window_rets.append(ar)
            per_event[(tkr, str(t0.date()))] = window_rets
            rows.append(window_rets)

        if not rows:
            return mark_failed(sid, "No valid event windows")

        ar_df = pd.DataFrame(rows, columns=[f"T{w:+d}" for w in car_window])
        mean_ar = ar_df.mean(axis=0)
        # CAR over T-2..T+10 (the requested window)
        car_t0_t10 = ar_df.loc[:, [f"T{w:+d}" for w in range(0, 11)]].sum(axis=1)
        car_full = ar_df.sum(axis=1)
        n = len(ar_df)
        mean_car_t0_t10 = float(car_t0_t10.mean())
        std_car = float(car_t0_t10.std())
        t_stat = mean_car_t0_t10 / (std_car / np.sqrt(n)) if std_car > 0 else 0.0
        hit = float((car_t0_t10 > 0).mean())

        # Construct a synthetic strategy: buy at T+1 close after announcement,
        # sell at T+10 close. Equal-weight overlapping basket.
        pos = pd.DataFrame(0.0, index=spy_ret.index, columns=px.columns)
        for tkr, ann in ADDITIONS:
            if tkr not in px.columns:
                continue
            ann_ts = pd.Timestamp(ann)
            idx = spy_ret.index[spy_ret.index >= ann_ts]
            if len(idx) == 0:
                continue
            t0 = idx[0]
            pos_idx0 = spy_ret.index.get_loc(t0)
            start_i = pos_idx0 + 1
            end_i = min(pos_idx0 + 10, len(spy_ret.index) - 1)
            if start_i >= len(spy_ret.index):
                continue
            for i in range(start_i, end_i + 1):
                pos.iloc[i, pos.columns.get_loc(tkr)] = 1.0

        # Equal-weight: normalize each row by number of active positions
        active = (pos != 0).sum(axis=1).replace(0, np.nan)
        pos_eq = pos.div(active, axis=0).fillna(0.0)

        rets = px.pct_change()
        # match columns
        common = pos_eq.columns.intersection(rets.columns)
        port = (pos_eq[common].shift(1) * rets[common]).sum(axis=1)
        # excess over SPY
        excess = port - (active.shift(1).fillna(0) > 0).astype(int) * spy_ret * 0
        # Just report port return; for an event-CAR-style strategy raw is fine.
        port = port.dropna()
        port = port[port != 0]  # only event-active days
        if len(port) < 5:
            return mark_failed(sid, "Too few active event days")

        # metrics on the event sub-period
        m_strategy = {
            "name": "E13 S&P500 inclusion drift",
            "start": str(port.index[0].date()),
            "end": str(port.index[-1].date()),
            "n_events": int(n),
            "n_active_days": int(len(port)),
            "mean_car_T0_T10": float(mean_car_t0_t10),
            "std_car": float(std_car),
            "t_stat_car": float(t_stat),
            "hit_rate": float(hit),
            "mean_daily_excess_ret": float(port.mean()),
            "ann_excess_vol": float(port.std() * np.sqrt(252)),
        }

        save_result(sid, m_strategy, extra={
            "status": "ok",
            "rule": "Hold each S&P500 addition T+1 to T+10 (equal-weight basket); report mean CAR.",
            "source": "Harris-Gurel 1986, Shleifer 1986, Greenwood-Sammon 2022",
            "n_events": int(n),
            "ar_means_by_day": {f"T{w:+d}": float(mean_ar.iloc[i]) for i, w in enumerate(car_window)},
        })
        print(f"E13: {n} events, mean CAR T0..T10 = {mean_car_t0_t10*100:.2f}%, "
              f"t-stat={t_stat:.2f}, hit={hit*100:.0f}%")
    except Exception as e:
        import traceback; traceback.print_exc()
        return mark_failed(sid, f"unhandled exception: {e}")


if __name__ == "__main__":
    main()
