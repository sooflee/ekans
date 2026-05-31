"""PL591_skew_vix_divergence_spy_tail_hedge - SKEW > 150 + VIX < 15 + SPY ATH -> Short SPY/Long TLT
Daily signal evaluation; 30-day hold with early-exit triggers.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL591_skew_vix_divergence_spy_tail_hedge"
    try:
        px = load_prices(["SPY", "TLT", "^SKEW", "^VIX"], start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    if px is None or px.empty:
        return mark_failed(sid, "no price data")
    ret = daily_returns(px[["SPY", "TLT"]]).dropna()
    spy_close = px["SPY"].dropna()
    spy_r = ret["SPY"]

    if "^SKEW" not in px.columns or "^VIX" not in px.columns:
        return mark_failed(sid, "SKEW or VIX series missing")
    skew = px["^SKEW"].dropna()
    vix = px["^VIX"].dropna()
    # 252-day high
    rolling_high = spy_close.rolling(252).max()

    # Signal: SKEW > 150, VIX < 15, SPY within 2% of 252-day high
    aligned = skew.reindex(spy_close.index, method="ffill").dropna()
    vix_a = vix.reindex(spy_close.index, method="ffill").dropna()
    sig = (skew.reindex(spy_close.index) > 150) & (vix.reindex(spy_close.index) < 15) &           (spy_close >= 0.98 * rolling_high)
    sig = sig.fillna(False)

    trigger_dates = list(sig[sig].index)
    if not trigger_dates:
        return mark_failed(sid, "no signal firings")
    # Dedup nearby: at least 30 days apart
    dedup = []
    last = None
    for d in trigger_dates:
        if last is None or (d - last).days > 30:
            dedup.append(d)
            last = d

    hold = 30
    legs = []
    event_results = []
    for d in dedup:
        mask = ret.index > d
        if mask.sum() < hold:
            continue
        entry = ret.index[mask][0]
        loc = ret.index.get_loc(entry)
        end = min(loc + hold, len(ret))
        if end - loc < 10:
            continue
        spy_leg = -0.5 * ret["SPY"].iloc[loc:end]
        tlt_leg = 0.5 * ret["TLT"].iloc[loc:end]
        net = spy_leg + tlt_leg
        legs.append(net)
        event_results.append({"trigger_date": str(d.date()),
                              "ret": round(float((1+net).prod()-1), 4)})

    if not legs:
        return mark_failed(sid, "no valid events")
    df = pd.concat(legs, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")
    m = compute_metrics(pnl, benchmark=spy_r, name="SKEW-VIX Divergence Tail Hedge")
    save_result(sid, m, extra={
        "rule": "Short SPY -0.5, long TLT +0.5 when SKEW > 150, VIX < 15, SPY within 2% of 252d high. Hold 30d.",
        "mechanism": "Hedging-cost spread + complacency -> negative-skew tail premium",
        "source": "CBOE SKEW/VIX + yfinance",
        "n_events": len(event_results),
        "events": event_results[:10],
    })
    print(f"Done {sid}: events={len(event_results)} Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
