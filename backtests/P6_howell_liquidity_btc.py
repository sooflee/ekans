"""
P6 Howell Global Liquidity -> BTC.

Construct simplified global liquidity proxy:
- Fed total assets (FRED WALCL, weekly USD).
- ECB total assets (FRED ECBASSETSW, weekly EUR; convert to USD via DEXUSEU).
- BOJ total assets (FRED JPNASSETS, monthly JPY; convert via DEXJPUS).
- PBOC / SNB: not available cleanly in FRED, omit and document.

Aggregate to weekly USD; compute 13-week % change ("liquidity impulse").
Signal: long BTC-USD when impulse > 0 AND 2nd derivative > 0 (accelerating);
exit when both turn negative.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, load_fred, daily_returns,
    compute_metrics, print_metrics, save_result, mark_failed,
)


def main():
    # Fetch series. WALCL: weekly USD millions. ECBASSETSW: weekly EUR millions.
    # JPNASSETS: monthly JPY (in units of 100M JPY per FRED docs - actually monthly billions JPY).
    fed = load_fred(["WALCL"], start="2005-01-01")["WALCL"]                 # USD millions
    try:
        ecb_eur = load_fred(["ECBASSETSW"], start="2005-01-01")["ECBASSETSW"]   # EUR millions
        eur_usd = load_fred(["DEXUSEU"], start="2005-01-01")["DEXUSEU"]        # USD per EUR
    except Exception:
        ecb_eur = None
        eur_usd = None
    try:
        boj_jpy = load_fred(["JPNASSETS"], start="2005-01-01")["JPNASSETS"]    # JPY (units)
        jpy_per_usd = load_fred(["DEXJPUS"], start="2005-01-01")["DEXJPUS"]   # JPY per USD
    except Exception:
        boj_jpy = None
        jpy_per_usd = None

    # Build weekly USD-denominated balances. Align everything to a weekly (W-FRI) grid.
    w_idx_start = max(fed.index.min(),
                      (ecb_eur.index.min() if ecb_eur is not None else fed.index.min()))
    weekly_idx = pd.date_range(start=w_idx_start, end=pd.Timestamp.today(), freq="W-FRI")

    # Fed: weekly USD millions
    fed_w = fed.reindex(weekly_idx, method="ffill") / 1e3   # billions USD

    components = {"fed": fed_w}
    if ecb_eur is not None and eur_usd is not None:
        eur_usd_w = eur_usd.reindex(weekly_idx, method="ffill")
        ecb_usd = ecb_eur.reindex(weekly_idx, method="ffill") * eur_usd_w / 1e3  # billions USD
        components["ecb"] = ecb_usd
    if boj_jpy is not None and jpy_per_usd is not None:
        # FRED JPNASSETS unit: "Billions of Yen" per docs; we'll assume billions JPY.
        jpy_w = jpy_per_usd.reindex(weekly_idx, method="ffill")
        boj_usd = boj_jpy.reindex(weekly_idx, method="ffill") / jpy_w   # billions JPY -> billion USD
        components["boj"] = boj_usd

    liq = pd.concat(components, axis=1).dropna()
    # Sum across central banks (in billions USD)
    total = liq.sum(axis=1)

    # 13-week % change (liquidity impulse) and acceleration (2nd diff of impulse)
    impulse = total.pct_change(13) * 100
    accel = impulse.diff(1)

    # Signal: long when impulse > 0 AND accel > 0; exit when both < 0.
    # Persistent regime: enter on positive&accel, exit only on both negative.
    raw_long = (impulse > 0) & (accel > 0)
    raw_exit = (impulse < 0) & (accel < 0)
    state = pd.Series(False, index=impulse.index)
    on = False
    for d in impulse.index:
        if on and bool(raw_exit.loc[d]):
            on = False
        elif (not on) and bool(raw_long.loc[d]):
            on = True
        state.loc[d] = on

    # Apply to BTC daily returns
    btc = load_prices(["BTC-USD"], start="2014-01-01")["BTC-USD"]
    rets = btc.pct_change()
    idx = btc.index

    # Build daily position from weekly signal, with 1-week publication lag.
    weekly_pos = state.astype(float).shift(1).fillna(0.0)  # 1-week lag
    pos_daily = weekly_pos.reindex(idx, method="ffill").fillna(0.0)
    pnl = (pos_daily.shift(1).fillna(0) * rets).dropna()

    spy = load_prices(["SPY"], start="2014-01-01")["SPY"].pct_change()
    m = compute_metrics(pnl, benchmark=spy.dropna(), name="P6 Howell Liquidity BTC")
    print_metrics(m)
    save_result("P6_howell_liquidity_btc", m, extra={
        "status": "ok",
        "rule": "Long BTC-USD when 13-week change of global-CB liquidity proxy > 0 AND accelerating; exit when both negative.",
        "mechanism": "Michael Howell: global liquidity drives risk-asset cycles, BTC most sensitive.",
        "universe": "BTC-USD; signal: FRED WALCL + ECBASSETSW (USD) + JPNASSETS (USD).",
        "components_used": list(components.keys()),
        "substitution_notes": "PBOC and SNB total-asset series not available in FRED; omitted. Liquidity proxy is Fed+ECB+BOJ only.",
        "source": "Michael Howell, CrossBorder Capital (YouTube/podcasts, Phase 1P)",
    })


if __name__ == "__main__":
    main()
