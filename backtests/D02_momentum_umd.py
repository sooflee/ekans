"""
D02 Cross-sectional momentum (UMD/Mom) — Ken French daily Mom factor.

Universe-shortcut: Same as D01 — we use Ken French's published daily Momentum
factor rather than building from CRSP returns.

Window: 1990-01-01 -> present.
Comparison: same metrics on MKT-RF over the same window.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import pandas_datareader.data as web

from harness import compute_metrics, print_metrics, save_result, mark_failed


def main():
    try:
        mom = web.DataReader("F-F_Momentum_Factor_daily",
                             "famafrench", start="1990-01-01")[0] / 100.0
        ff3 = web.DataReader("F-F_Research_Data_Factors_daily",
                             "famafrench", start="1990-01-01")[0] / 100.0
    except Exception as e:
        mark_failed("D02_momentum_umd", f"FF data fetch failed: {e}")
        return

    umd = mom.iloc[:, 0].dropna()       # column 'Mom'
    mkt = ff3["Mkt-RF"].dropna()
    umd.index = pd.to_datetime(umd.index)
    mkt.index = pd.to_datetime(mkt.index)

    m_umd = compute_metrics(umd, benchmark=mkt, name="D02 UMD (Momentum)")
    m_mkt = compute_metrics(mkt, name="MKT-RF (reference)")
    print_metrics(m_umd)
    print_metrics(m_mkt)

    save_result("D02_momentum_umd", m_umd, extra={
        "status": "ok",
        "rule": "Long UMD (winners minus losers) per Ken French daily factor.",
        "universe": "Ken French U.S. equity universe.",
        "source": "Carhart 1997 / pandas_datareader.famafrench 'F-F_Momentum_Factor_daily'.",
        "reference_mkt_rf": {
            "cagr": m_mkt["cagr"],
            "sharpe": m_mkt["sharpe"],
            "max_dd": m_mkt["max_dd"],
        },
        "shortcut_note": "Used Ken French published UMD rather than replicating from CRSP."
    })


if __name__ == "__main__":
    main()
