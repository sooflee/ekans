"""PL8_bnpl_delinquency_affirm — BNPL Delinquency → Short AFRM / Long JPM+WFC
Regime trade from Jan 2023 when CFPB flagged rising BNPL delinquencies.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL8_bnpl_delinquency_affirm"
    try:
        px = load_prices(["AFRM", "JPM", "WFC", "SPY"], start="2023-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    bank_basket = (ret["JPM"] + ret["WFC"]) / 2
    pair_pnl = bank_basket - ret["AFRM"]  # long banks, short AFRM

    m = compute_metrics(pair_pnl, benchmark=ret["SPY"], name="BNPL DQ: Long Banks / Short AFRM")
    save_result(sid, m, extra={
        "rule": "Regime trade: short AFRM vs long (JPM+WFC)/2 from Jan 2023",
        "mechanism": "Rising BNPL delinquencies signal consumer credit deterioration; AFRM provisions spike while traditional banks with tighter underwriting outperform",
        "source": "CFPB BNPL market reports; yfinance",
    })
    print(f"Done: Sharpe={m.get('sharpe',0):.2f}, CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
