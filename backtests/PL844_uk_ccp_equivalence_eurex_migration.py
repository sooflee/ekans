"""PL844_uk_ccp_equivalence_eurex_migration — UK CCP Equivalence Expiry / Eurex Migration: Long DB1.DE / Short LSEG.L"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL844_uk_ccp_equivalence_eurex_migration"

    # Rolling signal: long DB1.DE / short LSEG.L when:
    # 1. DB1.DE 30-day trailing return - LSEG.L 30-day trailing return >= 5pp
    # 2. EWU 5-day return < -1.0%
    # Note: DB1.DE trades in EUR, LSEG.L in GBP - yfinance provides prices in local currency
    # We use returns directly (currency-adjusted implicitly via return series)

    HOLD_DAYS = 30
    MOMENTUM_THRESHOLD = 0.05  # 5pp spread in 30-day returns
    EWU_THRESHOLD = -0.01      # -1% EWU 5-day return

    try:
        px = load_prices(["DB1.DE", "LSEG.L", "EWU", "SPY"], start="2016-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    # Drop columns where we have no data
    available = [c for c in ["DB1.DE", "LSEG.L", "EWU", "SPY"] if c in px.columns]
    if "DB1.DE" not in available or "LSEG.L" not in available:
        return mark_failed(sid, "DB1.DE or LSEG.L not available in yfinance")

    ret = daily_returns(px)
    db1_r = ret["DB1.DE"]
    lseg_r = ret["LSEG.L"]
    ewu_r = ret["EWU"]
    spy_r = ret["SPY"]

    # Align all series
    common_idx = db1_r.index.intersection(lseg_r.index).intersection(ewu_r.index)
    db1_r = db1_r.reindex(common_idx)
    lseg_r = lseg_r.reindex(common_idx)
    ewu_r = ewu_r.reindex(common_idx)

    # 30-day rolling returns
    db1_roll30 = (1 + db1_r).rolling(30).apply(lambda x: x.prod() - 1, raw=True)
    lseg_roll30 = (1 + lseg_r).rolling(30).apply(lambda x: x.prod() - 1, raw=True)
    mom_spread = db1_roll30 - lseg_roll30

    # 5-day EWU returns
    ewu_roll5 = (1 + ewu_r).rolling(5).apply(lambda x: x.prod() - 1, raw=True)

    # Signal: momentum spread >= 5pp AND EWU 5d < -1%
    signal = (mom_spread >= MOMENTUM_THRESHOLD) & (ewu_roll5 < EWU_THRESHOLD)

    pnl = pd.Series(0.0, index=common_idx)
    in_trade = False
    trade_start = None
    trades = []

    for i, date in enumerate(common_idx):
        if not in_trade:
            if signal.iloc[i]:
                in_trade = True
                trade_start = i
        else:
            days_in = i - trade_start
            if days_in >= HOLD_DAYS:
                in_trade = False
                trades.append((trade_start, i))

    # Build PnL: long DB1.DE / short LSEG.L
    for ts, te in trades:
        seg_db1 = db1_r.iloc[ts:te]
        seg_lseg = lseg_r.iloc[ts:te]
        seg_pnl = 0.5 * seg_db1 - 0.5 * seg_lseg  # equal notional
        pnl.iloc[ts:te] += seg_pnl.values

    if pnl.abs().sum() == 0 or len(trades) == 0:
        return mark_failed(sid, f"no qualifying trades found (signal fired: {signal.sum()} times)")

    spy_r_aligned = spy_r.reindex(common_idx)
    m = compute_metrics(pnl, benchmark=spy_r_aligned, name="UK CCP Equivalence Eurex Migration Long DB1 Short LSEG")
    save_result(sid, m, extra={
        "rule": (
            "When DB1.DE 30-day return minus LSEG.L 30-day return >= 5pp AND EWU 5-day return < -1.0%, "
            "enter LONG DB1.DE / SHORT LSEG.L equal notional. Hold 30 trading days."
        ),
        "mechanism": (
            "UK CCP equivalence expiry forces EU-domiciled clearing to shift from LCH.Clearnet (LSEG) "
            "to Eurex Clearing (DB1). Volume migration boosts DB1 clearing revenue while reducing LSEG "
            "clearing throughput. UK political risk (EWU proxy) amplifies the migration pressure."
        ),
        "source": (
            "DB1.DE, LSEG.L, EWU, SPY via yfinance. Rolling momentum proxy. "
            "Key dates: Brexit vote 2016-06-24, EU equivalence extensions 2022-12-30, 2025-01-01."
        ),
        "n_trades": len(trades),
        "momentum_threshold": MOMENTUM_THRESHOLD,
        "ewu_threshold": EWU_THRESHOLD,
        "note": "DB1.DE (EUR) vs LSEG.L (GBP) - returns used directly, FX risk embedded in return series",
    })


if __name__ == "__main__":
    main()
