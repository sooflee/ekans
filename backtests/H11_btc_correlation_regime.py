"""
H11 BTC correlation-regime sizing.

Compute 60d rolling correlations of BTC daily returns vs ^NDX, GLD, DXY.

Regime-conditional sizing of long BTC:
  - corr(BTC, NDX) > 0.4  AND not in 'digital gold' regime -> half-weight (0.5)
  - corr(BTC, GLD) > 0.3  AND corr(BTC, DXY) < -0.3        -> full-weight (1.0)
  - default                                                -> 0.75
Apply position to next-day BTC return. Compare to BTC buy-and-hold.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from harness import (
    load_prices, compute_metrics, print_metrics, save_result,
)


def main():
    px = load_prices(["BTC-USD", "^NDX", "GLD", "DX-Y.NYB"], start="2014-01-01")
    # Some tickers may not exist; rename
    cols = px.columns.tolist()
    # DX-Y.NYB might be the spot DXY index ticker on yfinance
    if "DX-Y.NYB" not in cols:
        # fallback to UUP ETF
        px2 = load_prices(["UUP"], start="2014-01-01")
        px = px.join(px2.rename(columns={"UUP": "DX-Y.NYB"}), how="outer")
    rets = px.pct_change()

    btc = rets["BTC-USD"]
    ndx = rets.get("^NDX")
    gld = rets.get("GLD")
    dxy = rets.get("DX-Y.NYB")

    # daily index from BTC (24/7); join with equities (gap-fill with 0 = no return on closed days)
    common_idx = btc.dropna().index
    def align(s):
        return s.reindex(common_idx).fillna(0.0)
    ndx_a, gld_a, dxy_a = align(ndx), align(gld), align(dxy)

    win = 60
    cor_ndx = btc.rolling(win).corr(ndx_a)
    cor_gld = btc.rolling(win).corr(gld_a)
    cor_dxy = btc.rolling(win).corr(dxy_a)

    pos = pd.Series(0.75, index=common_idx)
    digital_gold = (cor_gld > 0.3) & (cor_dxy < -0.3)
    ndx_high = (cor_ndx > 0.4) & ~digital_gold
    pos = pos.where(~digital_gold, 1.0)
    pos = pos.where(~ndx_high, 0.5)

    pnl = (pos.shift(1) * btc).dropna()
    pnl = pnl.loc[cor_ndx.dropna().index[0]:]

    m = compute_metrics(pnl, benchmark=btc.loc[pnl.index],
                        name="H11 BTC corr-regime sized")
    print_metrics(m)

    # also report fraction of time in each regime
    fraction = {
        "digital_gold": float(digital_gold.mean()),
        "ndx_high_only": float(ndx_high.mean()),
        "default_0p75": float(((~digital_gold) & (~ndx_high)).mean()),
    }

    save_result("H11_btc_correlation_regime", m, extra={
        "status": "ok",
        "rule": "Size BTC long: 0.5x when corr(NDX)>0.4 (high beta); 1.0x when corr(GLD)>0.3 AND corr(DXY)<-0.3 (digital-gold); 0.75x otherwise.",
        "data_source": "BTC-USD, ^NDX, GLD, DX-Y.NYB (or UUP fallback) from yfinance.",
        "regime_time_fraction": fraction,
    })


if __name__ == "__main__":
    main()
