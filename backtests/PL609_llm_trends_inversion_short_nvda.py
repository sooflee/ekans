"""
PL609 — Google Trends LLM share inversion (ChatGPT/Gemini/Claude) -> Short NVDA

When the trailing-26w-leader among ChatGPT / Gemini / Claude loses #1 rank by
4w MA Trends share for 2 consecutive weeks, short NVDA at next-day close.
Hold 30 trading days; exit early on NVDA earnings proxy or +15% (stop).

Counter-signal against long_NVDA / long_semis defaults.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics, save_result,
                     mark_failed, daily_returns, ROOT)


SIGNAL_ID = "PL609_llm_trends_inversion_short_nvda"
NAME = "LLM Trends Leader Inversion -> Short NVDA (30d)"

TERMS = ["chatgpt", "gemini", "claude"]

CACHE = ROOT / "data" / "pytrends_llm_share.parquet"

# NVDA quarterly earnings proxy anchors (Feb 20, May 22, Aug 27, Nov 20)
NVDA_EARNINGS_ANCHORS = [(2, 20), (5, 22), (8, 27), (11, 20)]


def fetch_trends():
    if CACHE.exists():
        try:
            df = pd.read_parquet(CACHE)
            return df
        except Exception:
            pass
    try:
        from pytrends.request import TrendReq
    except ImportError:
        return None
    try:
        pt = TrendReq(hl="en-US", tz=0, timeout=(10, 25))
        # Use 'today 5-y' for weekly resolution covering 2021-present
        pt.build_payload(TERMS, cat=0, timeframe="today 5-y", geo="US", gprop="")
        df = pt.interest_over_time()
        if df.empty:
            return None
        df = df[TERMS].astype(float)
        df.index = pd.to_datetime(df.index)
        df.to_parquet(CACHE)
        return df
    except Exception as e:
        print(f"pytrends fetch failed: {e}")
        return None


def main():
    sid = SIGNAL_ID
    try:
        px = load_prices(["NVDA", "SPY"], start="2022-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if "NVDA" not in px.columns or "SPY" not in px.columns:
        return mark_failed(sid, f"missing tickers; got {list(px.columns)}")

    px = px.sort_index().ffill(limit=3)
    ret = daily_returns(px)
    nvda_r = ret["NVDA"]
    spy_r = ret["SPY"]

    trends = fetch_trends()
    if trends is None or trends.empty:
        return mark_failed(
            sid,
            "pytrends rate-limited or returned empty; LLM share series not available",
            extra={
                "rule": "Short NVDA 30d on LLM Trends leader inversion",
                "mechanism": "Loss of leadership in LLM consumer share reduces incremental GPU demand expectation",
                "source": "PL609 catalog",
            },
        )

    # Share-of-three
    total = trends[TERMS].sum(axis=1)
    share = trends[TERMS].div(total, axis=0)
    share = share.dropna(how="any")
    if len(share) < 40:
        return mark_failed(sid, f"insufficient weekly Trends share data: {len(share)}")

    # 4-week MA
    ma4 = share.rolling(4, min_periods=2).mean()

    # Identify trailing-26-week leader and check inversion
    events = []  # list of pd.Timestamp (weekly index of inversion confirmation)
    inv_flag = pd.Series(False, index=ma4.index)

    for i in range(26, len(ma4)):
        trailing = ma4.iloc[i-26:i]
        leader_means = trailing.mean()
        L = leader_means.idxmax()
        # Need L to have been #1 (rank 1) for every one of the prior 26 weeks
        ranks = trailing.rank(axis=1, ascending=False)
        if (ranks[L] != 1).any():
            continue
        # Current week and prior week ranks
        cur_ranks = ma4.iloc[i].rank(ascending=False)
        prev_ranks = ma4.iloc[i-1].rank(ascending=False)
        if cur_ranks[L] > 1 and prev_ranks[L] > 1:
            inv_flag.iloc[i] = True

    fire_dates = ma4.index[inv_flag]

    if len(fire_dates) == 0:
        # Not a backtest "failure" per se — record as ok with zero events,
        # but mark_failed is more honest because no PnL can be measured.
        return mark_failed(
            sid,
            "no LLM-leader inversions detected in usable Trends history",
            extra={
                "rule": "Short NVDA 30d when LLM share leader inverts for 2 consec weeks",
                "mechanism": "Loss of incumbent LLM share signals slowing AI GPU demand expectation",
                "source": "PL609 catalog + pytrends",
                "n_trends_weeks": len(ma4),
            },
        )

    hold_days = 30
    stop_pct = 0.15  # NVDA closes >+15% above entry close -> stop for short
    trading_idx = ret.index
    positions = pd.Series(0.0, index=trading_idx)

    def next_nvda_earnings_date(after):
        # find next anchor date strictly after `after`
        candidates = []
        for y in [after.year, after.year + 1]:
            for (m, d) in NVDA_EARNINGS_ANCHORS:
                cand = pd.Timestamp(y, m, d)
                if cand > after:
                    candidates.append(cand)
        return min(candidates) if candidates else after + pd.Timedelta(days=60)

    in_pos_until_loc = -1
    out_events = []

    for fd in fire_dates:
        loc = trading_idx.searchsorted(fd)
        if loc >= len(trading_idx):
            continue
        # entry next trading day (close-to-close)
        entry_loc = loc + 1
        if entry_loc >= len(trading_idx) - 1:
            continue
        if entry_loc <= in_pos_until_loc:
            continue  # overlap
        if pd.isna(px["NVDA"].iloc[entry_loc]):
            continue
        end_loc = min(entry_loc + hold_days, len(trading_idx) - 1)
        # earnings cutoff
        earn = next_nvda_earnings_date(trading_idx[entry_loc])
        earn_loc = trading_idx.searchsorted(earn)
        if earn_loc < end_loc:
            end_loc = max(entry_loc + 1, earn_loc)

        entry_price = px["NVDA"].iloc[entry_loc]
        nvda_win = px["NVDA"].iloc[entry_loc:end_loc + 1]
        cumret = nvda_win / entry_price - 1.0
        exit_loc = end_loc
        exit_reason = "max_hold_or_earnings"
        for k, val in enumerate(cumret.values):
            if pd.notna(val) and val >= stop_pct:
                exit_loc = entry_loc + k
                exit_reason = "stop_loss"
                break

        positions.iloc[entry_loc:exit_loc + 1] = -1.0  # short
        in_pos_until_loc = exit_loc
        out_events.append({
            "trends_date": str(fd.date()),
            "entry_date": str(trading_idx[entry_loc].date()),
            "exit_date": str(trading_idx[exit_loc].date()),
            "exit_reason": exit_reason,
            "nvda_return": float(cumret.iloc[min(exit_loc - entry_loc, len(cumret) - 1)]),
        })

    if not out_events:
        return mark_failed(sid, "fire dates outside NVDA price coverage")

    pnl = positions.shift(1).fillna(0.0) * nvda_r.reindex(positions.index).fillna(0.0)
    pnl = pnl.dropna()
    first_entry = pd.Timestamp(out_events[0]["entry_date"])
    pnl = pnl.loc[pnl.index >= first_entry]
    active_pos = positions.loc[positions.index >= first_entry]

    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient PnL: {len(pnl)} days")

    m = compute_metrics(
        pnl,
        benchmark=spy_r.reindex(pnl.index),
        name=NAME,
        positions=active_pos.abs(),
        cost_bps=10,
    )
    m["n_events"] = len(out_events)
    m["pct_in_market"] = float(active_pos.abs().mean())
    m["events"] = out_events
    m["status"] = "ok"

    save_result(
        sid,
        m,
        extra={
            "rule": "Compute weekly Google Trends US 4-week MA share of {chatgpt, gemini, claude}. Identify trailing-26w leader L. When L loses #1 rank for 2 consecutive weeks (and was #1 every week of trailing 26w), short NVDA at next-day close for up to 30 trading days. Exit early on NVDA next earnings (anchor: Feb 20 / May 22 / Aug 27 / Nov 20) or +15% NVDA close (stop).",
            "mechanism": "Loss of consumer LLM leadership for the incumbent signals slowing user-side AI compute demand growth; market re-rates NVDA's forward GPU revenue trajectory. Counter-signal to long-NVDA / long-semis defaults that bake in continuing market-share dominance.",
            "source": "PL609 idea catalog; pytrends Google Trends US",
            "caveats": "Small N (Trends share usable only post Dec 2023 when all 3 terms have meaningful volume). Counter-signal against widely-held NVDA long thesis. Earnings dates are anchor approximations.",
        },
        pnl=pnl,
    )
    print(f"Saved {sid}: n_events={len(out_events)}  Sharpe={m.get('sharpe',0):.2f}  CAGR={m.get('cagr',0)*100:.2f}%")


if __name__ == "__main__":
    main()
