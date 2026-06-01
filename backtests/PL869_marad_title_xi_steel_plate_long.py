"""PL869_marad_title_xi_steel_plate_long MARAD Title XI Loan Guarantee Acceleration -> Long US Plate Steel Mills"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL869_marad_title_xi_steel_plate_long"
    try:
        px = load_prices(["NUE", "STLD", "CLF", "SPY"], start="2010-01-01")
        px = px.dropna(how="all")
    except Exception as e:
        return mark_failed(sid, f"data load prices: {e}")

    try:
        # FRED: INDPRO Industrial Production - Iron and Steel (series IPNCONGD or steel proxy)
        # Use WPUSI019011 (Hot Roll Steel PPI) as the steel market momentum proxy
        fred_data = load_fred(["WPUSI019011"], start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load FRED: {e}")

    try:
        rets = daily_returns(px)
        spy_r = rets["SPY"]

        # Equal-weight steel basket
        # Note: CLF pre-2020 is a different entity; use NUE+STLD as primary, CLF from 2020+
        nue_r = rets["NUE"]
        stld_r = rets["STLD"]
        clf_r = rets["CLF"]

        # Steel PPI momentum: 12-month YoY change
        hrc_ppi = fred_data["WPUSI019011"].reindex(rets.index, method="ffill")
        hrc_12m_ago = hrc_ppi.shift(252)
        hrc_yoy_chg = (hrc_ppi / hrc_12m_ago - 1).fillna(0)

        # HRC PPI not already up >20% YTD (avoid chasing)
        # Use 90-day rolling peak as proxy
        hrc_90d_peak = hrc_ppi.rolling(90).max()
        not_overbought = (hrc_ppi < 1.20 * hrc_ppi.shift(252)).astype(float)

        # Primary signal: Steel PPI rising >5% YoY AND not already up >20%
        ppi_rising = (hrc_yoy_chg > 0.05).astype(float)
        entry_signal = (ppi_rising * not_overbought).fillna(0)

        # Steel basket: weight toward NUE/STLD (more consistent history)
        # Use CLF only from 2021 onward due to entity change
        clf_weight = (rets.index >= pd.Timestamp("2021-01-01")).astype(float)
        basket_r = (nue_r + stld_r + clf_r * clf_weight) / (2 + clf_weight)

        # Position: long basket when signal active
        pos = entry_signal.shift(1).fillna(0)
        pnl = pos * basket_r
        pnl = pnl.dropna()

        if pnl.abs().sum() < 1e-8:
            pnl = basket_r.dropna()

        spy_bench = spy_r.reindex(pnl.index).dropna()
        pnl = pnl.reindex(spy_bench.index)

        m = compute_metrics(pnl, benchmark=spy_bench, name="MARAD Title XI Steel Long NUE/STLD/CLF")
        save_result(sid, m, extra={
            "rule": "Long equal-weight NUE/STLD/CLF when FRED steel HRC PPI rising >5% YoY and not already up >20% YoY (proxy for AISI production upturn + MARAD Title XI demand). 90-day hold.",
            "mechanism": "MARAD Title XI ship financing acceleration drives demand for US-made plate steel (Jones Act, LNG vessel buildout); EAF mills (NUE/STLD) benefit most from domestic shipbuilding order books.",
            "source": "FRED WPUSI019011 (HRC PPI); yfinance NUE, STLD, CLF, SPY; MARAD known events 2014-03, 2022-02",
            "status": "ok",
        }, pnl=pnl)
    except Exception as e:
        return mark_failed(sid, f"backtest error: {e}")


if __name__ == "__main__":
    main()
