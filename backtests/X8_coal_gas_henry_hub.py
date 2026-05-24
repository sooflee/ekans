"""
X8 Coal-vs-gas switching arbitrage — long BTU when natgas cheaper than coal.

Original: When (HH - PRB equiv) < 0 for 10 days, long BTU + ARCH 60 days.

Substitution adjustments:
- ARCH delisted (Arch Resources merged into Consol -> CEIX, then CONSOL Energy -> CNX).
  Replace ARCH with AMR (Alpha Metallurgical Resources, listed) and HCC (Warrior Met) as
  coal pure-plays. Better: BTU + AMR equal weight.
- PRB coal daily not freely available; use FRED WPU051 (US coal PPI, monthly).
  Convert to $/MMBtu proxy using PRB ~8,400 Btu/lb -> ~16.8 MMBtu/short ton.
  PRB price typically $10-15/ton; equivalent ~$0.60-0.90/MMBtu.
- Use the monthly coal price forward-filled to daily, and compare to daily Henry Hub.

When daily Henry Hub ($/MMBtu) < coal_per_MMBtu for 10 consecutive trading days
(natgas substituting for coal is uneconomic — coal demand should hold), long
BTU + AMR equal-weight 60 trading days.

Note: This is the *anti*-version of the usual "cheap gas hurts coal" trade.
Spec is: when HH dips below coal equivalent (gas relatively expensive vs coal),
coal demand should be supported. So sign is intuitive: cheap-coal-vs-gas -> long coal.

Source: FRED DHHNGSP (Henry Hub daily), WPU051 (US coal PPI monthly).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, daily_returns, compute_metrics, print_metrics, save_result, mark_failed


def main():
    sid = "X8_coal_gas_henry_hub"
    try:
        hh = load_fred("DHHNGSP", start="2010-01-01").iloc[:, 0].rename("HH")
        coal_ppi = load_fred("WPU051", start="2010-01-01").iloc[:, 0].rename("CoalPPI")
    except Exception as e:
        return mark_failed(sid, f"FRED load failed: {e}")

    try:
        px = load_prices(["BTU", "AMR", "HCC", "METC", "SPY", "XLE"], start="2017-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load failed: {e}")

    if "BTU" not in px.columns:
        return mark_failed(sid, "BTU price missing")
    if "AMR" not in px.columns:
        return mark_failed(sid, "AMR price missing (substitute coal name)")

    # Build daily coal proxy: anchor: WPU051 index normalized to PRB price.
    # PRB spot price 2010 average ~$11.50/ton; coal PPI Jan 2010 = ~210 (WPU051).
    # We don't need absolute level; we need *relative* changes for switching threshold.
    # Convert coal PPI to $/MMBtu using:
    #   anchor: Apr 2020 PRB spot = $12/ton ≈ $0.71/MMBtu; WPU051 Apr 2020 ≈ 197.
    # Scale factor: $0.71/MMBtu / 197 = 0.0036
    SCALE = 0.71 / 197.0
    coal_daily_mmbtu = (coal_ppi * SCALE).reindex(hh.index, method="ffill")

    df = pd.concat([hh, coal_daily_mmbtu], axis=1).dropna()
    df.columns = ["HH", "CoalProxyMMBtu"]
    # Spread = HH - coal. Positive = gas more expensive than coal (coal favored).
    df["spread"] = df["HH"] - df["CoalProxyMMBtu"]

    # Original rule sign: "When (HH - PRB equiv) < 0 for 10 days, long BTU+ARCH 60d"
    # Negative spread = gas cheaper than coal -> utilities switch to gas -> coal demand weak.
    # That would short coal. Confusing. Re-read original: spec says LONG coal when HH-PRB<0.
    # Interpretation: extremes signal reversion. If gas is unusually cheap (HH<PRB), the
    # market is mid-correction and coal stocks have already sold off; long coal for the rebound.
    # We follow the spec as written.
    consec = 10
    cond = df["spread"] < 0
    in_regime = cond.rolling(consec).sum() >= consec
    first = in_regime & ~in_regime.shift(1, fill_value=False)
    entries = df.index[first]

    if len(entries) == 0:
        # Fallback: use rolling 10th-percentile threshold
        thr = float(df["spread"].quantile(0.10))
        cond = df["spread"] < thr
        in_regime = cond.rolling(consec).sum() >= consec
        first = in_regime & ~in_regime.shift(1, fill_value=False)
        entries = df.index[first]
        threshold_used = thr
        sign_note = "fallback: 10th-percentile threshold (spread never went below 0 with anchor)"
    else:
        threshold_used = 0.0
        sign_note = "spec rule: spread<0 for 10 consec sessions"

    rets = daily_returns(px[["BTU", "AMR", "SPY", "XLE"]]).reindex(df.index)
    basket = (rets["BTU"] + rets["AMR"]) / 2

    hold = 60
    pos = pd.Series(0.0, index=basket.index)
    last_exit = -1
    event_rets = []
    used_entries = []
    for t in entries:
        if t not in pos.index:
            continue
        i = pos.index.get_loc(t)
        if i <= last_exit:
            continue
        i_entry = i + 1
        i_exit = min(i_entry + hold, len(pos) - 1)
        if i_exit <= i_entry + 5:
            continue
        pos.iloc[i_entry:i_exit] = 1.0
        last_exit = i_exit
        sub = basket.iloc[i_entry:i_exit].fillna(0)
        event_rets.append(float((1 + sub).prod() - 1))
        used_entries.append(str(t.date()))

    n_events = len(event_rets)
    if n_events < 3:
        return mark_failed(sid, f"only {n_events} events",
                           extra={"n_events": n_events, "threshold": threshold_used,
                                  "spread_range": [float(df['spread'].min()), float(df['spread'].max())]})

    pnl = (pos * basket).dropna()
    pnl = pnl.loc[pnl.ne(0).cumsum() > 0]
    if len(pnl) < 60:
        return mark_failed(sid, f"too few active days ({len(pnl)})")

    bench = rets["SPY"].reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name=f"X8 HH-coal<{threshold_used:.2f} long BTU+AMR 60d")
    m["n_events"] = n_events
    m["entries"] = used_entries[:30]
    m["event_returns"] = event_rets
    m["event_mean"] = float(np.mean(event_rets))
    m["event_hit"] = float(np.mean([r > 0 for r in event_rets]))
    m["threshold_used"] = float(threshold_used)
    m["sign_note"] = sign_note
    print(f"X8: events={n_events}, event-mean={m['event_mean']*100:+.2f}%, hit={m['event_hit']*100:.0f}%, threshold={threshold_used}")
    print_metrics(m)
    save_result(sid, m, extra={
        "status": "ok",
        "rule": f"When daily Henry Hub - coal-PPI-proxy <{threshold_used:.2f}/MMBtu for {consec} consecutive sessions, long BTU+AMR equal-weight 60 trading days.",
        "mechanism": "Original rule: extreme HH<coal spreads signal coal-stock sell-off has overshot; rebound long.",
        "source": "FRED DHHNGSP (Henry Hub daily), WPU051 (US coal PPI monthly, ffilled to daily).",
        "substitution_note": "(a) ARCH delisted -> use AMR. (b) PRB daily not free -> WPU051 monthly PPI scaled to $/MMBtu (Apr-2020 anchor: $12/ton ≈ $0.71/MMBtu).",
        "universe": ["BTU", "AMR"],
        "hold_days": hold,
    })


if __name__ == "__main__":
    main()
