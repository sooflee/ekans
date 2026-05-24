"""
Y9 Post-unlock mean-reversion long.

Idea: After a cliff unlock has hit and the price has drawn down into the
event (see Y7), the overshoot reverses as forced sellers exhaust and
the supply shock is digested. Buy at T+10, sell at T+60.

Rule:
  - Use the same curated cliff-unlock event set as Y7.
  - Filter to events where the T-7 → T return was below -5% (i.e. a
    real pre-event draw-down occurred); otherwise skip.
  - Long the token at close of T+10, sell at close of T+60.
  - Excess vs long-BTC hedge.

Source:
  - Allen, Chen, Lo, Tao (2024 WP) document reversion 2-8 weeks
    post-unlock for major-cap tokens.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import save_result, mark_failed


# Same event set as Y7
EVENTS = [
    ("ARB-USD",   "2024-03-16", 0.766, "ARB 1y team+investor cliff"),
    ("OP-USD",    "2023-05-31", 0.12,  "OP team/contrib cliff"),
    ("OP-USD",    "2024-05-31", 0.10,  "OP year-2 batch"),
    ("APT-USD",   "2023-10-12", 0.20,  "APT 1y cliff"),
    ("APT-USD",   "2024-10-12", 0.15,  "APT 2y cliff"),
    ("SUI-USD",   "2024-05-03", 0.075, "SUI 1y series A cliff"),
    ("SUI-USD",   "2024-08-01", 0.06,  "SUI community reserve unlock"),
    ("AVAX-USD",  "2022-07-15", 0.095, "AVAX team+foundation tranche"),
    ("AVAX-USD",  "2024-02-09", 0.075, "AVAX strategic partners"),
    ("SOL-USD",   "2023-03-01", 0.05,  "SOL early-stage tranche"),
    ("SOL-USD",   "2024-03-01", 0.054, "SOL 2024 vesting tranche"),
    ("ADA-USD",   "2022-09-22", 0.04,  "ADA reserve emission cliff"),
    ("MATIC-USD", "2022-04-26", 0.067, "MATIC foundation cliff"),
    ("MATIC-USD", "2023-04-26", 0.052, "MATIC year-2 batch"),
]

PRE_DRAWDOWN_THRESHOLD = -0.05  # require T-7→T return below -5%


def main():
    sid = "Y9_post_unlock_meanrev"
    try:
        import yfinance as yf
        tickers = sorted(set([t for t, *_ in EVENTS]))
        tickers.append("BTC-USD")
        df = yf.download(
            tickers, start="2021-01-01", end="2025-06-01",
            progress=False, auto_adjust=True, threads=True,
        )
        if isinstance(df.columns, pd.MultiIndex):
            px = df["Close"]
        else:
            px = df.copy()
        px = px.dropna(how="all").sort_index()
        btc = px["BTC-USD"].dropna() if "BTC-USD" in px.columns else None

        results = []
        excess_long = []
        skipped_no_drawdown = 0
        for tkr, d, pct, label in EVENTS:
            if tkr not in px.columns:
                continue
            s = px[tkr].dropna()
            if len(s) < 80:
                continue
            event_ts = pd.Timestamp(d)
            on_or_before = s.index[s.index <= event_ts]
            if len(on_or_before) == 0:
                continue
            tunlock = on_or_before[-1]
            i_unlock = s.index.get_loc(tunlock)
            i_pre = i_unlock - 7
            if i_pre < 0:
                continue
            pre_ret = float(s.iloc[i_unlock] / s.iloc[i_pre] - 1.0)
            if pre_ret > PRE_DRAWDOWN_THRESHOLD:
                skipped_no_drawdown += 1
                continue
            i_entry = i_unlock + 10
            i_exit = i_unlock + 60
            if i_exit >= len(s):
                continue
            entry = float(s.iloc[i_entry])
            exit_ = float(s.iloc[i_exit])
            if entry <= 0:
                continue
            tok_ret = exit_ / entry - 1.0
            d_entry, d_exit = s.index[i_entry], s.index[i_exit]
            btc_change = 0.0
            if btc is not None:
                bw = btc.loc[d_entry:d_exit]
                if len(bw) >= 2:
                    btc_change = float(bw.iloc[-1] / bw.iloc[0] - 1.0)
            excess = tok_ret - btc_change
            results.append({
                "token": tkr, "unlock_date": d, "pct_circ": pct,
                "pre_event_ret": pre_ret,
                "entry_T+10": entry, "exit_T+60": exit_,
                "tok_ret": float(tok_ret),
                "excess_vs_btc": float(excess),
                "label": label,
            })
            excess_long.append(excess)

        if not excess_long:
            return mark_failed(
                sid,
                f"no events met pre-drawdown filter (< {PRE_DRAWDOWN_THRESHOLD}); "
                f"skipped {skipped_no_drawdown} events with no real drawdown",
            )
        arr = np.array(excess_long)
        n = len(arr)
        mean_e = float(arr.mean())
        std_e = float(arr.std(ddof=1)) if n > 1 else float("nan")
        t_stat = (mean_e / (std_e / np.sqrt(n))) if std_e and std_e > 0 else 0.0
        hit = float((arr > 0).mean())
        m = {
            "name": "Y9 post-unlock mean-rev long (T+10 -> T+60, vs BTC)",
            "n_events": n,
            "n_skipped_no_drawdown": skipped_no_drawdown,
            "mean_excess_long_ret": mean_e,
            "median_excess": float(np.median(arr)),
            "std_excess": std_e,
            "hit_rate": hit,
            "t_stat": float(t_stat),
        }
        save_result(sid, m, extra={
            "status": "ok",
            "rule": ("Same curated cliff-unlock events as Y7. Only kept "
                     "events where T-7→T drawdown < -5%. Long token at "
                     "T+10, exit T+60; excess vs long-BTC hedge."),
            "mechanism": ("Forced selling exhausts post-unlock; new "
                          "circulating float is digested by the marginal "
                          "buyer once the supply shock is priced in."),
            "source": ("Allen-Chen-Lo-Tao (2024 WP) 'Token Unlocks and "
                       "Returns' documents 2-8 week reversion in major "
                       "tokens."),
            "events": results[:30],
        })
        print(f"Y9: n={n}, skipped={skipped_no_drawdown}, mean long excess={mean_e*100:.2f}%, hit={hit*100:.0f}%, t={t_stat:.2f}")
    except Exception as e:
        import traceback; traceback.print_exc()
        return mark_failed(sid, f"unhandled exception: {e}")


if __name__ == "__main__":
    main()
