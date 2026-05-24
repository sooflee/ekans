"""
P12 Steno Global Liquidity flip -> SPY+BTC.

Aggregate Fed + ECB + BOJ (FRED-available CB total assets) in USD.
Compute 4-week change. When it flips negative -> positive, long SPY + BTC
for 8 weeks (equal-weight pair).

Substitutions: PBOC and SNB total-asset series are not available via FRED.
We use a Fed+ECB+BOJ proxy and document. Fallback to Fed-only if BOJ/ECB
unavailable.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, load_fred, daily_returns,
    compute_metrics, print_metrics, save_result,
)


def main():
    fed = load_fred(["WALCL"], start="2010-01-01")["WALCL"] / 1e3   # B$
    try:
        ecb = load_fred(["ECBASSETSW"], start="2010-01-01")["ECBASSETSW"]
        eur_usd = load_fred(["DEXUSEU"], start="2010-01-01")["DEXUSEU"]
    except Exception:
        ecb = None
    try:
        boj = load_fred(["JPNASSETS"], start="2010-01-01")["JPNASSETS"]
        jpy_per_usd = load_fred(["DEXJPUS"], start="2010-01-01")["DEXJPUS"]
    except Exception:
        boj = None

    weekly_idx = pd.date_range(start="2010-02-01", end=pd.Timestamp.today(), freq="W-FRI")
    components = {"fed": fed.reindex(weekly_idx, method="ffill")}
    if ecb is not None:
        eur_usd_w = eur_usd.reindex(weekly_idx, method="ffill")
        components["ecb"] = ecb.reindex(weekly_idx, method="ffill") * eur_usd_w / 1e3
    if boj is not None:
        jpy_w = jpy_per_usd.reindex(weekly_idx, method="ffill")
        components["boj"] = boj.reindex(weekly_idx, method="ffill") / jpy_w  # B$ ~ (B JPY / JPY-per-$)

    liq = pd.concat(components, axis=1).dropna()
    total = liq.sum(axis=1)
    # 4-week change (level diff, in B$)
    delta4 = total.diff(4)

    # Trigger: prior delta4 < 0 and current delta4 > 0 (flip neg -> pos)
    flip = (delta4 > 0) & (delta4.shift(1) < 0)

    # Build signal series: when flip True at week W, set on for 8 weeks (W .. W+7).
    state = pd.Series(False, index=total.index)
    on_until = None
    for d in total.index:
        if bool(flip.loc[d]):
            ix = total.index.searchsorted(d)
            target_ix = min(ix + 8, len(total.index) - 1)
            on_until = total.index[target_ix]
        if on_until is not None and d <= on_until:
            state.loc[d] = True
        else:
            state.loc[d] = False

    # Apply to SPY+BTC pair (equal-weight, half each)
    spy = load_prices(["SPY"], start="2014-01-01")["SPY"]
    btc = load_prices(["BTC-USD"], start="2014-01-01")["BTC-USD"]
    spy_rets = spy.pct_change()
    btc_rets = btc.pct_change()
    common_idx = spy_rets.index.intersection(btc_rets.index)
    pair_rets = 0.5 * spy_rets.reindex(common_idx) + 0.5 * btc_rets.reindex(common_idx)

    # Daily pos with weekly state ffilled and a 1-week pub lag.
    weekly_pos = state.astype(float).shift(1).fillna(0.0)
    pos_daily = weekly_pos.reindex(common_idx, method="ffill").fillna(0.0)
    pnl = (pos_daily.shift(1).fillna(0) * pair_rets).dropna()

    bench = spy_rets.dropna()
    m = compute_metrics(pnl, benchmark=bench, name="P12 Steno Liquidity Flip -> SPY+BTC")
    print_metrics(m)
    n_flips = int(flip.sum())
    save_result("P12_steno_global_liquidity", m, extra={
        "status": "ok",
        "rule": "4-week change of Fed+ECB+BOJ total assets (in USD). When flips negative -> positive, long SPY+BTC (50/50) for 8 weeks.",
        "mechanism": "Steno Research: global liquidity inflection signals risk-on regime; SPY+BTC most beta-sensitive.",
        "universe": "SPY + BTC-USD pair (50/50); signal: FRED WALCL + ECBASSETSW + JPNASSETS in USD.",
        "components_used": list(components.keys()),
        "n_flips": n_flips,
        "substitution_notes": "PBOC and SNB total-asset series not on FRED; omitted. Fed+ECB+BOJ proxy used. SPY-only benchmark.",
        "source": "Steno Research (YouTube/podcasts, Phase 1P)",
    })


if __name__ == "__main__":
    main()
