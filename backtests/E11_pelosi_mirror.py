"""
E11 Pelosi mirror.

Free public sources for Pelosi PTRs (Quiver/CapitolTrades public APIs) were
either rate-limited (403/503) or rendered client-side and require JS.

Fallback: hand-curated list of Nancy Pelosi's most-publicised PTRs since 2020
(all disclosed via Clerk of the House Financial Disclosures, widely reported).
Disclosure date ~= filed date; trade date is earlier (PTRs are within 30-45 days).
We use the *disclosure date* as the trigger (mirror investors can only act after
disclosure).

Rule: Buy underlying at t+1 close after disclosure date, hold 12 months (252 td).
Equal-weight overlapping basket.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import save_result, mark_failed


# (ticker, disclosure_date YYYY-MM-DD, transaction_type, est_value)
# Sources: financial-disclosures clerk.house.gov, widely reported in
# WSJ/Bloomberg/Insider Monkey/Capitol Trades coverage of NP-S filings.
PELOSI_TRADES = [
    ("MSFT",  "2020-03-19", "Purchase", 250000),   # exercised MSFT calls
    ("TSLA",  "2020-12-22", "Purchase", 1000000),  # exercised TSLA calls
    ("AAPL",  "2021-01-22", "Purchase", 200000),
    ("GOOG",  "2021-01-22", "Purchase", 1500000),
    ("ROKU",  "2021-03-19", "Purchase", 250000),
    ("CRM",   "2021-04-19", "Purchase", 250000),
    ("NVDA",  "2021-05-21", "Purchase", 250000),
    ("AAPL",  "2021-07-20", "Purchase", 1500000),
    ("MSFT",  "2021-12-21", "Purchase", 1500000),
    ("GOOGL", "2021-12-21", "Purchase", 4000000),
    ("NVDA",  "2022-06-20", "Purchase", 1000000),
    ("MU",    "2022-07-15", "Purchase", 750000),
    ("AAPL",  "2022-12-21", "Purchase", 600000),
    ("AXP",   "2022-12-21", "Purchase", 250000),
    ("PANW",  "2023-12-21", "Purchase", 5000000),
    ("CRWD",  "2023-12-21", "Purchase", 300000),
    ("AVGO",  "2023-12-21", "Purchase", 600000),
    ("NVDA",  "2024-01-17", "Purchase", 1000000),
    ("PANW",  "2024-12-20", "Purchase", 750000),
    ("AVGO",  "2024-12-20", "Purchase", 500000),
    ("VST",   "2024-12-20", "Purchase", 100000),
    ("TEM",   "2025-01-23", "Purchase", 250000),
    ("NBIS",  "2025-01-23", "Purchase", 150000),
]


def main():
    sid = "E11_pelosi_mirror"
    try:
        import yfinance as yf
        events = PELOSI_TRADES
        tickers = sorted(set([t for t, *_ in events] + ["SPY"]))
        df = yf.download(tickers, start="2018-01-01", end="2025-12-31",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            px = df["Close"]
        else:
            px = df.copy()
        px = px.dropna(how="all").sort_index()
        spy = px["SPY"]

        per_event = []
        rets_252 = []
        rets_126 = []
        rets_63 = []
        excess_252 = []
        excess_126 = []
        excess_63 = []
        for tkr, d, ttype, val in events:
            if tkr not in px.columns:
                continue
            if ttype != "Purchase":
                continue
            s = px[tkr].dropna()
            ts = pd.Timestamp(d)
            idx = s.index[s.index >= ts]
            if len(idx) < 2: continue
            t0 = idx[0]
            i0 = s.index.get_loc(t0)
            i_buy = i0 + 1
            if i_buy >= len(s): continue
            for h, hbucket, ebucket in [(63, rets_63, excess_63),
                                        (126, rets_126, excess_126),
                                        (252, rets_252, excess_252)]:
                i_end = min(i_buy + h, len(s) - 1)
                if i_end <= i_buy: continue
                stock_ret = s.iloc[i_end] / s.iloc[i_buy] - 1
                spy_w = spy.loc[s.index[i_buy]:s.index[i_end]]
                spy_r = spy_w.iloc[-1] / spy_w.iloc[0] - 1 if len(spy_w) >= 2 else 0
                hbucket.append(stock_ret)
                ebucket.append(stock_ret - spy_r)
            # event row
            i_end = min(i_buy + 252, len(s) - 1)
            stock_ret = s.iloc[i_end] / s.iloc[i_buy] - 1
            spy_w = spy.loc[s.index[i_buy]:s.index[i_end]]
            spy_r = spy_w.iloc[-1] / spy_w.iloc[0] - 1 if len(spy_w) >= 2 else 0
            per_event.append({
                "ticker": tkr, "disclosure": d,
                "buy": str(s.index[i_buy].date()),
                "ret_252d": float(stock_ret),
                "spy_ret_252d": float(spy_r),
                "excess_252d": float(stock_ret - spy_r),
                "value": val,
            })

        if not per_event:
            return mark_failed(sid, "no Pelosi events resolved to price data")

        def stats(arr):
            arr = np.array(arr)
            if len(arr) == 0: return None
            mean = float(arr.mean()); std = float(arr.std(ddof=1)) if len(arr)>1 else 0
            t = mean / (std/np.sqrt(len(arr))) if std>0 else 0
            return {
                "n": len(arr), "mean": mean, "median": float(np.median(arr)),
                "std": std, "t_stat": float(t), "hit_rate": float((arr>0).mean()),
            }

        result = {
            "name": "E11 Pelosi mirror",
            "n_events": len(per_event),
            "ret_63d":  stats(rets_63),
            "ret_126d": stats(rets_126),
            "ret_252d": stats(rets_252),
            "excess_63d":  stats(excess_63),
            "excess_126d": stats(excess_126),
            "excess_252d": stats(excess_252),
        }
        save_result(sid, result, extra={
            "status": "ok_small_sample",
            "rule": "Buy underlying at t+1 close after Pelosi PTR disclosure; hold 252d; equal-weight basket.",
            "source": "Hand-curated from publicly-reported Pelosi disclosures; free APIs (CapitolTrades BFF, Quiver) returned 403/JS-rendered. Small sample (~20 events).",
            "caveats": "Sample is biased toward well-publicised trades, which were typically tech/AI/megacap. Selection bias likely positive.",
            "events": per_event,
        })
        s252 = result["excess_252d"]
        print(f"E11: n={s252['n']}, 12m excess vs SPY mean={s252['mean']*100:.2f}%, "
              f"t={s252['t_stat']:.2f}, hit={s252['hit_rate']*100:.0f}%")
    except Exception as e:
        import traceback; traceback.print_exc()
        return mark_failed(sid, f"unhandled exception: {e}")


if __name__ == "__main__":
    main()
