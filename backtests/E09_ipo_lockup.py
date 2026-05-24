"""
E09 IPO lockup expiry short.

Hardcoded list of well-known IPOs since 2015. Lockup expiry estimated as IPO date
+ 180 days (standard 180-day lockup). Short at lockup_date - 5 trading days, cover
at lockup_date + 5 trading days. Equal-weight basket.

Source: Field-Hanka 2001, Brav-Gompers 2003 documented price pressure & negative
abnormal returns around lockup expiry.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import save_result, mark_failed


# (ticker, ipo_date YYYY-MM-DD)
IPOS = [
    ("FIT",   "2015-06-18"),
    ("ETSY",  "2015-04-16"),
    ("SHOP",  "2015-05-21"),
    ("WIX",   "2013-11-06"),
    ("LC",    "2014-12-11"),
    ("GPRO",  "2014-06-26"),
    ("TWLO",  "2016-06-23"),
    ("NTNX",  "2016-09-30"),
    ("SNAP",  "2017-03-02"),
    ("ROKU",  "2017-09-28"),
    ("APRN",  "2017-06-29"),
    ("MULE",  "2017-03-17"),
    ("YEXT",  "2017-04-13"),
    ("SE",    "2017-10-20"),
    ("STNE",  "2018-10-25"),
    ("DBX",   "2018-03-23"),
    ("SPOT",  "2018-04-03"),
    ("DOCU",  "2018-04-27"),
    ("ZS",    "2018-03-16"),
    ("PINS",  "2019-04-18"),
    ("ZM",    "2019-04-18"),
    ("UBER",  "2019-05-10"),
    ("LYFT",  "2019-03-29"),
    ("WORK",  "2019-06-20"),  # direct listing, technically no lockup, skip later
    ("PD",    "2019-04-11"),
    ("CRWD",  "2019-06-12"),
    ("FVRR",  "2019-06-13"),
    ("CHWY",  "2019-06-14"),
    ("BYND",  "2019-05-02"),
    ("DDOG",  "2019-09-19"),
    ("PTON",  "2019-09-26"),
    ("CLDR",  "2017-04-28"),
    ("PLTR",  "2020-09-30"),
    ("ABNB",  "2020-12-10"),
    ("DASH",  "2020-12-09"),
    ("U",     "2020-09-18"),
    ("SNOW",  "2020-09-16"),
    ("ASAN",  "2020-09-30"),
    ("AI",    "2020-12-09"),
    ("BMBL",  "2021-02-11"),
    ("RBLX",  "2021-03-10"),
    ("COIN",  "2021-04-14"),
    ("AFRM",  "2021-01-13"),
    ("GTLB",  "2021-10-14"),
    ("RIVN",  "2021-11-10"),
    ("HOOD",  "2021-07-29"),
    ("DOCN",  "2021-03-24"),
    ("PATH",  "2021-04-21"),
    ("S",     "2021-06-30"),
    ("EXFY",  "2021-05-26"),
    ("DUOL",  "2021-07-28"),
    ("OLO",   "2021-03-17"),
    ("RSKD",  "2021-07-15"),
    ("FRSH",  "2021-09-22"),
    ("SAMG",  "2021-08-04"),
    ("LCID",  "2021-07-26"),  # via SPAC
    ("OSCR",  "2021-03-03"),
    ("CART",  "2023-09-19"),
    ("KVUE",  "2023-05-04"),
    ("ARM",   "2023-09-14"),
    ("KLAR",  "2024-09-23"),  # placeholder, may not have data
    ("RDDT",  "2024-03-21"),
    ("ASTR",  "2021-06-30"),
    ("BIRK",  "2023-10-11"),
    ("MBLY",  "2022-10-26"),
    ("CAVA",  "2023-06-15"),
    ("CART_ALT", "2023-09-19"),
]


def main():
    sid = "E09_ipo_lockup"
    events = [(t, d) for t, d in IPOS if "_ALT" not in t]
    # dedupe
    seen = set(); clean = []
    for t, d in events:
        if t in seen: continue
        seen.add(t); clean.append((t, d))
    events = clean

    try:
        import yfinance as yf
        tickers = sorted(set([t for t, _ in events] + ["SPY"]))
        df = yf.download(tickers, start="2013-01-01", end="2025-12-31",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            px = df["Close"]
        else:
            px = df.copy()
        px = px.dropna(how="all").sort_index()
        spy = px["SPY"] if "SPY" in px.columns else None

        results = []
        excess_rets = []
        for tkr, d in events:
            if tkr not in px.columns:
                continue
            s = px[tkr].dropna()
            if len(s) < 200:
                continue
            ipo_ts = pd.Timestamp(d)
            lockup_ts = ipo_ts + pd.Timedelta(days=180)
            # Find trading day on/after lockup_ts
            idx = s.index[s.index >= lockup_ts]
            if len(idx) == 0:
                continue
            lockup_td = idx[0]
            lockup_i = s.index.get_loc(lockup_td)
            entry_i = max(lockup_i - 5, 0)
            exit_i = min(lockup_i + 5, len(s) - 1)
            if exit_i - entry_i < 3:
                continue
            entry_d = s.index[entry_i]
            exit_d = s.index[exit_i]
            # Short return = entry/exit - 1 (we short at entry, cover at exit)
            stock_ret = s.iloc[exit_i] / s.iloc[entry_i] - 1
            short_ret = -stock_ret
            spy_ret = spy.loc[entry_d:exit_d]
            if len(spy_ret) >= 2:
                spy_change = spy_ret.iloc[-1] / spy_ret.iloc[0] - 1
            else:
                spy_change = 0
            excess_short = short_ret - (-spy_change)  # short stock excess over short SPY
            results.append({
                "ticker": tkr,
                "ipo": d,
                "lockup_event": str(lockup_td.date()),
                "stock_ret_window": float(stock_ret),
                "short_ret": float(short_ret),
                "excess_vs_short_spy": float(excess_short),
            })
            excess_rets.append(excess_short)

        if not excess_rets:
            return mark_failed(sid, "no valid IPO lockup events")
        arr = np.array(excess_rets)
        mean_e = float(arr.mean())
        std_e = float(arr.std(ddof=1))
        n = len(arr)
        t_stat = mean_e / (std_e / np.sqrt(n)) if std_e > 0 else 0.0
        hit = float((arr > 0).mean())

        m = {
            "name": "E09 IPO lockup short (T-5 to T+5)",
            "n_events": n,
            "mean_excess_short_ret": mean_e,
            "median_excess": float(np.median(arr)),
            "std_excess": std_e,
            "t_stat": float(t_stat),
            "hit_rate": hit,
        }
        save_result(sid, m, extra={
            "status": "ok",
            "rule": "Short ticker basket at lockup-5 trading days, cover at lockup+5; excess vs short-SPY.",
            "source": "Field-Hanka 2001, Brav-Gompers 2003; lockup = IPO+180d (standard).",
            "events": results[:30],  # cap
        })
        print(f"E09: n={n}, mean 10-day short excess={mean_e*100:.2f}%, hit={hit*100:.0f}%, t={t_stat:.2f}")
    except Exception as e:
        import traceback; traceback.print_exc()
        return mark_failed(sid, f"unhandled exception: {e}")


if __name__ == "__main__":
    main()
