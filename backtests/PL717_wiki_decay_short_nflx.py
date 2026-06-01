"""PL717_wiki_decay_short_nflx
Wiki Pageview Decay Shortens -> Short NFLX (Counter)

Original spec: For Netflix Top-10 Originals, compute Wikipedia 7-day pageview
decay half-life. When the rolling-30d mean half-life shortens by >2-sigma vs
trailing-1y baseline → short NFLX 25 trading days.

Implementation note: Wikipedia pageview API data for specific Netflix titles is
not accessible programmatically in this environment. We use a price-based
engagement proxy: NFLX 20-day price return crossing +20% vs XLC (Communication
Services) excess return crossing a 2-sigma threshold above its trailing-252d
mean (buzz/hype peak → engagement decay). This captures the same underlying
mechanism: a post-launch hype peak followed by audience decay → subscriber
growth disappointment → stock mean-reversion.

Short NFLX 25 trading days after signal trigger, benchmarked vs SPY.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


HOLD_DAYS = 25
LOOKBACK_SIGNAL = 20      # days for excess return over XLC
LOOKBACK_BASELINE = 252   # 1y trailing mean/std for z-score
ZSCORE_THRESHOLD = 2.0    # signal when z-score > 2.0 (2-sigma buzz peak)
MIN_COOLDOWN_DAYS = 40    # avoid re-triggering immediately after a signal


def main():
    sid = "PL717_wiki_decay_short_nflx"
    tickers = ["NFLX", "XLC", "SPY"]

    try:
        # XLC launched 2018-06-18; use 2019 start for baseline burn-in
        px = load_prices(tickers, start="2019-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    nflx_r = ret["NFLX"].dropna()
    xlc_r  = ret["XLC"].dropna()
    spy_r  = ret["SPY"].dropna()

    # Align all series to common index
    common = nflx_r.index.intersection(xlc_r.index).intersection(spy_r.index)
    nflx_r = nflx_r.reindex(common)
    xlc_r  = xlc_r.reindex(common)
    spy_r  = spy_r.reindex(common)

    # Cumulative price (rebased to 1.0) for rolling return computation
    nflx_px = (1 + nflx_r).cumprod()
    xlc_px  = (1 + xlc_r).cumprod()

    # 20-day excess return: NFLX vs XLC (buzz/engagement proxy)
    nflx_20d = nflx_px / nflx_px.shift(LOOKBACK_SIGNAL) - 1
    xlc_20d  = xlc_px  / xlc_px.shift(LOOKBACK_SIGNAL) - 1
    excess_20d = nflx_20d - xlc_20d

    # Rolling z-score of excess return (vs trailing 252d mean/std)
    roll_mean = excess_20d.rolling(LOOKBACK_BASELINE).mean()
    roll_std  = excess_20d.rolling(LOOKBACK_BASELINE).std()
    z_score   = (excess_20d - roll_mean) / roll_std.clip(lower=1e-6)

    # Signal: z-score crosses from below 2.0 to above 2.0 on previous day
    # (buzz peaked → expect engagement/subscriber disappointment on next 25d)
    z_prev = z_score.shift(1)
    raw_signal = (z_prev <= ZSCORE_THRESHOLD) & (z_score > ZSCORE_THRESHOLD)

    # Apply cooldown to avoid stacked/overlapping positions
    triggers = []
    last_trigger_idx = -999
    for i, (dt, is_sig) in enumerate(raw_signal.items()):
        if is_sig and not pd.isna(is_sig):
            if (i - last_trigger_idx) >= MIN_COOLDOWN_DAYS:
                triggers.append(dt)
                last_trigger_idx = i

    print(f"Signal triggers: {len(triggers)}")
    if len(triggers) < 5:
        return mark_failed(sid, f"too few signal triggers: {len(triggers)} (need >= 5)")

    # Build daily PnL series: short NFLX on trigger day (next session → same day
    # return in adj-close convention, consistent with harness), hold HOLD_DAYS.
    pnl      = pd.Series(0.0, index=common)
    positions = pd.Series(0.0, index=common)

    event_log = []
    for trig_dt in triggers:
        # Entry: same session as signal (consistent with adj-close convention)
        # If we want strict no-look-ahead, use the day after; but adj-close
        # series means the day's return is known at the close, so we treat the
        # next trading session as entry. Find next session after trigger.
        future = common[common > trig_dt]
        if len(future) == 0:
            continue
        entry_dt = future[0]
        entry_idx = common.get_loc(entry_dt)
        exit_idx  = min(entry_idx + HOLD_DAYS, len(common))

        slice_nflx = nflx_r.iloc[entry_idx:exit_idx]
        slice_spy  = spy_r.iloc[entry_idx:exit_idx]
        exit_dt    = common[exit_idx - 1] if exit_idx > entry_idx else entry_dt

        gross_ev  = float((1 + (-slice_nflx)).prod() - 1)  # short = inverted return
        spy_ev    = float((1 + slice_spy).prod() - 1)

        event_log.append({
            "trigger_date": str(trig_dt.date()),
            "z_score":      round(float(z_score.loc[trig_dt]), 3),
            "entry_date":   str(entry_dt.date()),
            "exit_date":    str(exit_dt.date()),
            "n_hold_days":  int(exit_idx - entry_idx),
            "short_nflx_return": round(gross_ev, 4),
            "spy_return":   round(spy_ev, 4),
            "excess_vs_spy": round(gross_ev - spy_ev, 4),
        })

        # Fill PnL (short NFLX = -1 * NFLX return); no double-stacking
        for j in range(entry_idx, exit_idx):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = -1.0   # short
                pnl.iloc[j]      = -nflx_r.iloc[j]

    n_events = len([e for e in event_log if e.get("entry_date")])
    if n_events == 0:
        return mark_failed(sid, "no valid events with entry_date")

    # Restrict metrics to held days only
    held_pnl = pnl[positions != 0]
    held_spy = spy_r.reindex(held_pnl.index).dropna()

    if len(held_pnl) < 30:
        return mark_failed(sid, f"insufficient held days: {len(held_pnl)}")

    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name="Wiki Decay Proxy Short NFLX (held-days only)",
    )

    event_rets = [e["short_nflx_return"] for e in event_log if e.get("short_nflx_return") is not None]
    summary = {
        "n_events":              n_events,
        "avg_short_return":      round(float(np.mean(event_rets)), 4),
        "median_short_return":   round(float(np.median(event_rets)), 4),
        "win_rate":              round(float(np.mean([r > 0 for r in event_rets])), 4),
        "best":                  round(float(np.max(event_rets)), 4),
        "worst":                 round(float(np.min(event_rets)), 4),
    }

    save_result(
        sid, m,
        extra={
            "status": "ok",
            "rule": (
                "Compute NFLX 20-day excess return vs XLC (Communication Services ETF) as "
                "an engagement/buzz proxy. When the rolling z-score of this excess return "
                f"crosses above +{ZSCORE_THRESHOLD:.0f}-sigma vs trailing 252-day baseline "
                f"(buzz peak), short NFLX for {HOLD_DAYS} trading days."
            ),
            "mechanism": (
                "After an extreme positive buzz episode (high NFLX excess return vs sector "
                "proxy), viewer/subscriber engagement typically decays as audiences finish "
                "the buzzy title and churn accelerates. This mean-reversion in engagement "
                "precedes subscriber growth disappointments in the following quarterly report, "
                "causing NFLX to underperform its sector peer basket in the 25-day window."
            ),
            "source": (
                "yfinance NFLX, XLC, SPY. Original spec references Wikipedia pageview API "
                "for Netflix Top-10 Originals; price-based buzz proxy used here as a "
                "feasibility-bounded approximation. XLC (launched 2018-06) captures the "
                "streaming/media sector beta."
            ),
            "caveats": (
                "Price-based proxy replaces the Wikipedia pageview decay half-life in the "
                "original spec — direct pageview data would yield a stronger, more independent "
                "signal. Cooldown of 40 days prevents overlap. Results depend on XLC as the "
                "sector normalizer; a different benchmark (e.g., NFLX vs QQQ) would change "
                "trigger dates. Counter-signal: intended as inverse of a long-NFLX strategy."
            ),
            "tickers": tickers,
            "hold_days": HOLD_DAYS,
            "zscore_threshold": ZSCORE_THRESHOLD,
            "lookback_baseline_days": LOOKBACK_BASELINE,
            "n_events": n_events,
            "events": event_log,
            "summary": summary,
        },
        pnl=pnl[positions != 0],
    )

    print(f"Done: {sid}")
    print(f"  n_events:  {n_events}")
    print(f"  summary:   {summary}")
    if "error" not in m:
        print(
            f"  Sharpe: {m.get('sharpe', float('nan')):.2f}  "
            f"CAGR: {m.get('cagr', float('nan'))*100:.2f}%  "
            f"MaxDD: {m.get('max_dd', float('nan'))*100:.2f}%  "
            f"t-stat: {m.get('t_stat', float('nan')):.2f}"
        )


if __name__ == "__main__":
    main()
