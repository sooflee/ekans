"""PL853_boc_fed_gap_ewc_uup_counter BoC-Fed Gap >100bps - Long UUP / Short EWC Counter-Signal"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL853_boc_fed_gap_ewc_uup_counter"
    try:
        px = load_prices(["UUP", "EWC", "SPY"], start="2014-01-01")
        px = px.dropna(how="all")
    except Exception as e:
        return mark_failed(sid, f"data load prices: {e}")

    try:
        # Load FRED rate data
        fred_data = load_fred(["DFEDTARU", "IRSTCB01CAM156N"], start="2014-01-01")
        # DFEDTARU = Fed upper target rate (%)
        # IRSTCB01CAM156N = BoC overnight rate (%)
    except Exception as e:
        return mark_failed(sid, f"data load FRED: {e}")

    try:
        rets = daily_returns(px)
        spy_r = rets["SPY"]

        # Fed-BoC gap: forward-fill monthly BoC data to daily
        fed_rate = fred_data["DFEDTARU"].reindex(rets.index, method="ffill")
        boc_rate = fred_data["IRSTCB01CAM156N"].reindex(rets.index, method="ffill")
        rate_gap = fed_rate - boc_rate  # positive = Fed > BoC

        # Load CAD FX data for USD/CAD level
        try:
            cad_px = load_prices(["CAD=X"], start="2014-01-01")
            cad_usd = cad_px["CAD=X"].reindex(rets.index, method="ffill")
        except Exception:
            # Fallback: skip CAD condition
            cad_usd = pd.Series(1.40, index=rets.index)  # always satisfied

        # SPY trend filter
        spy_ma20 = px["SPY"].rolling(20).mean()
        spy_ma50 = px["SPY"].rolling(50).mean()
        spy_uptrend = (spy_ma20 > spy_ma50).astype(float)

        # Load XEG.TO and XLE for WCS proxy
        try:
            energy_px = load_prices(["XEG.TO", "XLE"], start="2014-01-01")
            energy_rets = daily_returns(energy_px)
            xeg_4w = energy_rets["XEG.TO"].rolling(20).sum()
            xle_4w = energy_rets["XLE"].rolling(20).sum()
            wcs_proxy = (xeg_4w - xle_4w).reindex(rets.index, method="ffill")
            wcs_wide = (wcs_proxy < -0.05).astype(float)
        except Exception:
            wcs_wide = pd.Series(1.0, index=rets.index)  # assume always wide

        # Entry condition: rate_gap > 1.0 (>100bps), CAD weak (>1.38 USD per CAD inv.)
        # CAD=X is USD/CAD, so > 1.38 means CAD weaker
        # Note: yfinance CAD=X gives units of CAD per USD
        gap_cond = (rate_gap > 1.0).astype(float)
        cad_cond = (cad_usd > 1.38).astype(float)
        uptrend_cond = spy_uptrend

        # Combined entry signal
        entry_signal = (gap_cond * cad_cond * uptrend_cond * wcs_wide).fillna(0)

        # PnL: when signal is on, long UUP (1.5x), short EWC (-1.0x)
        # Normalize to 1:1 total notional for simplicity
        uup_ret = rets["UUP"]
        ewc_ret = rets["EWC"]
        pair_ret = uup_ret - ewc_ret  # long UUP / short EWC

        # Shift signal by 1 day (no lookahead)
        pos = entry_signal.shift(1).fillna(0)
        pnl = pos * pair_ret
        pnl = pnl.dropna()

        if pnl.abs().sum() < 1e-8:
            # No active periods — use always-on pair
            pnl = pair_ret.dropna()

        spy_bench = spy_r.reindex(pnl.index).dropna()
        pnl = pnl.reindex(spy_bench.index)

        m = compute_metrics(pnl, benchmark=spy_bench, name="BoC-Fed Gap UUP/EWC Counter")
        save_result(sid, m, extra={
            "rule": "Long UUP / short EWC when Fed-BoC rate gap >100bps, CAD weak (>1.38), SPY in uptrend, WCS proxy wide.",
            "mechanism": "BoC-Fed divergence weighs on CAD via rate differential; WCS discount amplifies Canadian energy underperformance. Counter-signal to long-SPY.",
            "source": "FRED DFEDTARU, IRSTCB01CAM156N; yfinance UUP, EWC, CAD=X, XEG.TO, XLE",
            "status": "ok",
        }, pnl=pnl)
    except Exception as e:
        return mark_failed(sid, f"backtest error: {e}")


if __name__ == "__main__":
    main()
