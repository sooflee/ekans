"""
X2 Royalty premium compression — long FNV/WPM/RGLD when streamers cheapen vs miner.

Original rule: When (FNV/WPM/RGLD) P/NAV premium vs NEM compresses to 2y low decile,
long royalty basket. Requires NAV estimates from quarterly filings — heavy.

Substitution: yfinance .info P/B is only a current snapshot, no history. We use
the FNV/NEM price-ratio z-score as a proxy for relative valuation. The streamers
are structurally lower-cost (no AISC, leveraged to gold price), so when the
price ratio drops to the 2y bottom decile, that's the streaming-vs-mining
spread compressing — analogous to P/NAV premium compression.

Rule: Compute log(FNV/NEM) z-score over 504 trading days (2y). When z < 10th pctile
of the rolling 2y window, go long equal-weight FNV+WPM+RGLD for 90 trading days
(non-overlapping; reset position on new entry).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, daily_returns, compute_metrics, print_metrics, save_result, mark_failed


def main():
    sid = "X2_royalty_premium"
    try:
        px = load_prices(["FNV", "WPM", "RGLD", "NEM", "GLD", "SPY"], start="2008-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load failed: {e}")

    px = px.dropna(how="all").sort_index()
    needed = ["FNV", "WPM", "RGLD", "NEM"]
    if not all(c in px.columns for c in needed):
        return mark_failed(sid, f"missing tickers; have {list(px.columns)}")

    # Use log ratio (mean-reverting under cointegration assumption)
    ratio = np.log(px["FNV"] / px["NEM"]).dropna()
    if len(ratio) < 600:
        return mark_failed(sid, f"insufficient ratio history ({len(ratio)} days)")

    win = 504  # ~2 years
    rmean = ratio.rolling(win).mean()
    rstd = ratio.rolling(win).std()
    z = (ratio - rmean) / rstd

    # Decile threshold via rolling: z < -1.28 corresponds to ~10th percentile under normal
    # (also: use rolling 10th percentile for robustness)
    rpct = ratio.rolling(win).quantile(0.10)
    trigger = (ratio <= rpct) & (z < -1.28)
    # First day of each trigger spell (avoid daily re-triggering)
    first = trigger & ~trigger.shift(1, fill_value=False)

    entries = ratio.index[first]
    # filter so entries don't overlap within hold window
    hold = 90
    rets = daily_returns(px[["FNV", "WPM", "RGLD", "NEM", "GLD", "SPY"]]).reindex(ratio.index)
    basket_ret = rets[["FNV", "WPM", "RGLD"]].mean(axis=1)  # equal-weight long
    bench = rets["SPY"]

    pos = pd.Series(0.0, index=basket_ret.index)
    last_exit_idx = -1
    n_events = 0
    event_returns = []
    for t in entries:
        if t not in pos.index:
            continue
        i = pos.index.get_loc(t)
        if i <= last_exit_idx:
            continue
        i_entry = i + 1
        i_exit = min(i_entry + hold, len(pos) - 1)
        if i_exit <= i_entry + 5:
            continue
        pos.iloc[i_entry:i_exit] = 1.0
        last_exit_idx = i_exit
        n_events += 1
        # event return
        sub = basket_ret.iloc[i_entry:i_exit].fillna(0)
        event_returns.append(float((1 + sub).prod() - 1))

    if n_events < 3:
        return mark_failed(sid, f"only {n_events} trigger events in 2y-decile spec",
                           extra={"n_events": n_events})

    pnl = (pos * basket_ret).dropna()
    pnl = pnl.loc[pnl.ne(0).cumsum() > 0]  # start from first trade

    if len(pnl) < 60:
        return mark_failed(sid, f"too few active trade days ({len(pnl)})")

    m = compute_metrics(pnl, benchmark=bench.reindex(pnl.index), name="X2 royalty premium proxy long-90d")
    m["n_events"] = n_events
    m["entries"] = [str(t.date()) for t in entries[:20]]
    m["event_returns"] = event_returns
    m["event_mean_ret"] = float(np.mean(event_returns))
    m["event_hit_rate"] = float(np.mean([r > 0 for r in event_returns]))
    print(f"X2: n_events={n_events}, event-mean={m['event_mean_ret']*100:+.2f}%, hit={m['event_hit_rate']*100:.0f}%")
    print_metrics(m)
    save_result(sid, m, extra={
        "status": "ok",
        "rule": "When log(FNV/NEM) hits 10th percentile of trailing 504-day distribution AND z < -1.28, long equal-weight FNV+WPM+RGLD for 90 trading days.",
        "mechanism": "Streamer/miner price ratio compression is a price-only proxy for P/NAV premium compression in royalty stocks (original NAV-based signal not feasible without 10-Q parsing).",
        "source": "yfinance daily closes 2008-2025; original premium-vs-NAV signal substituted with relative-price z-score.",
        "substitution_note": "Original spec required P/NAV from quarterly filings (heavy). Used log(FNV/NEM) z-score over 504 days as a public-data proxy.",
        "universe": ["FNV", "WPM", "RGLD", "NEM"],
        "hold_days": 90,
    })


if __name__ == "__main__":
    main()
