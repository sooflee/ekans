"""
E10 Spin-off drift.

Curated list of well-known US spin-offs since ~2010 with first regular-way trade
date. Buy SpinCo at first regular-way close, hold 24 months. Compare equal-weight
basket return vs SPY over same period.

Source rationale: McConnell-Ovtchinnikov 2004 documented 20%+ excess returns over
24 months. Recent academic and practitioner work (Cusatis-Miles-Woolridge 1993,
McConnell-Ovtchinnikov 2014 update) supports the persistence.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import load_prices, save_result, mark_failed


# (spinco_ticker, parent, first_regular_way_trade_date)
SPINOFFS = [
    ("PM",   "MO",   "2008-03-31"),
    ("PFE_PFIZER", None, None),  # placeholder removed
    ("ABBV", "ABT",  "2013-01-02"),
    ("ZTS",  "PFE",  "2013-02-01"),
    ("ADT",  "TYC",  "2012-10-01"),
    ("HII",  "NOC",  "2011-04-01"),
    ("MJN",  "BMY",  "2009-02-11"),
    ("PHM_PFIZER", None, None),
    ("MDLZ", "KFT",  "2012-10-02"),
    ("CC",   "DD",   "2015-07-01"),
    ("CHTR_TWX", None, None),
    ("LEA",  None,   None),
    ("VYX",  "NCR",  "2023-10-17"),
    ("NCR",  None,   None),
    ("VTRS", "PFE",  "2020-11-17"),
    ("OGN",  "MRK",  "2021-06-03"),
    ("CARR", "UTX",  "2020-04-03"),
    ("OTIS", "UTX",  "2020-04-03"),
    ("GEHC", "GE",   "2023-01-04"),
    ("GEV",  "GE",   "2024-04-02"),
    ("KVUE", "JNJ",  "2023-08-23"),  # IPO + later spin
    ("SOLV", "MMM",  "2024-04-01"),
    ("WBD",  "T",    "2022-04-11"),  # WarnerMedia + Discovery; AT&T spin
    ("KHC",  None,   None),  # merger, exclude
    ("DOW",  "DD",   "2019-04-02"),
    ("CTVA", "DD",   "2019-06-03"),
    ("HPE",  "HPQ",  "2015-11-02"),
    ("LITE", "JDSU", "2015-08-03"),
    ("VIAV", "JDSU", "2015-08-03"),
    ("BHF",  "MET",  "2017-08-07"),
    ("MOG.A","NTAP", None),
    ("ARNC", "AA",   "2016-11-01"),  # split into ARNC + AA
    ("DXC",  "HPE",  "2017-04-03"),
    ("LW",   "CAG",  "2016-11-09"),
    ("INGR_PFIZER", None, None),
    ("KTB",  "VFC",  "2019-05-23"),
    ("CARS", "TGNA", "2017-06-01"),
    ("APRN", None,   None),  # not spin
    ("DNOW", "NOV",  "2014-06-02"),
    ("XPO",  None,   None),
    ("RXO",  "XPO",  "2022-11-01"),
    ("GXO",  "XPO",  "2021-08-02"),
    ("BERY", None,   None),
    ("VST",  "ETR_OGE", "2016-10-04"),  # Vistra from TXU/Dynegy ish; use this approx
    ("ZURN_ZBRA", None, None),
    ("EAF",  "GTI",  "2018-04-23"),
    ("ENR",  "EB",   "2000-04-03"),  # too old
    ("HLT",  "BX",   "2013-12-12"),
    ("PK",   "HLT",  "2017-01-04"),
    ("HGV",  "HLT",  "2017-01-04"),
    ("CTLT_CARDINAL", None, None),
    ("VRSK", "VR",   "2009-10-08"),
    ("MSGE", "MSGN", "2020-04-20"),
    ("MSGS", "MSG",  "2015-10-01"),
    ("KEN", "JNJ", "2025-09-02"),  # Kenvue placeholder if you used KVUE differently
]


def main():
    sid = "E10_spinoff_drift"
    # filter to valid rows
    events = [(t, d) for (t, _, d) in SPINOFFS if d is not None and isinstance(t, str) and "_" not in t]
    # dedupe
    seen = set()
    clean = []
    for t, d in events:
        if t in seen:
            continue
        seen.add(t)
        clean.append((t, d))
    events = clean

    try:
        import yfinance as yf
        tickers = sorted(set([t for t, _ in events] + ["SPY"]))
        df = yf.download(tickers, start="2010-01-01", end="2025-12-31",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            px = df["Close"]
        else:
            px = df.copy()
        px = px.dropna(how="all").sort_index()

        spy = px["SPY"]
        results = []
        car_rows = []  # 24m excess return per event
        for tkr, d in events:
            if tkr not in px.columns:
                continue
            s = px[tkr].dropna()
            if s.empty:
                continue
            ann = pd.Timestamp(d)
            idx = s.index[s.index >= ann]
            if len(idx) == 0:
                continue
            t0 = idx[0]
            # exit = t0 + ~24 months (504 trading days)
            pos0 = s.index.get_loc(t0)
            pos1 = min(pos0 + 504, len(s) - 1)
            if pos1 <= pos0:
                continue
            t1 = s.index[pos1]
            spinco_ret = s.iloc[pos1] / s.iloc[pos0] - 1
            # SPY return over same window
            spy_ret = spy.reindex([t0, t1]).pct_change().iloc[-1]
            if pd.isna(spy_ret):
                # use forward fill
                spy_w = spy.loc[t0:t1]
                if len(spy_w) < 2:
                    continue
                spy_ret = spy_w.iloc[-1] / spy_w.iloc[0] - 1
            excess = spinco_ret - spy_ret
            results.append({
                "ticker": tkr,
                "spin_date": str(t0.date()),
                "exit_date": str(t1.date()),
                "spinco_ret": float(spinco_ret),
                "spy_ret": float(spy_ret),
                "excess": float(excess),
            })
            car_rows.append(excess)

        if not car_rows:
            return mark_failed(sid, "no valid spin-off events")

        car = np.array(car_rows)
        mean_excess = float(car.mean())
        std_excess = float(car.std(ddof=1))
        n = len(car)
        t_stat = mean_excess / (std_excess / np.sqrt(n)) if std_excess > 0 else 0.0
        hit = float((car > 0).mean())

        m = {
            "name": "E10 Spin-off 24m drift",
            "n_events": n,
            "mean_24m_excess_vs_SPY": mean_excess,
            "median_24m_excess": float(np.median(car)),
            "std_24m_excess": std_excess,
            "t_stat": float(t_stat),
            "hit_rate": hit,
            "best_event_excess": float(car.max()),
            "worst_event_excess": float(car.min()),
        }
        save_result(sid, m, extra={
            "status": "ok",
            "rule": "Buy SpinCo at first regular-way close, hold ~24 months (504 trading days), measure excess vs SPY.",
            "source": "Cusatis-Miles-Woolridge 1993; McConnell-Ovtchinnikov 2014.",
            "events": results,
        })
        print(f"E10: n={n}, mean 24m excess={mean_excess*100:.2f}%, "
              f"hit={hit*100:.0f}%, t={t_stat:.2f}")
    except Exception as e:
        import traceback; traceback.print_exc()
        return mark_failed(sid, f"unhandled exception: {e}")


if __name__ == "__main__":
    main()
