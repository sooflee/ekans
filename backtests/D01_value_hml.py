"""
D01 Value (HML) — Ken French daily HML factor.

Universe-shortcut: We do NOT replicate HML from CRSP/Compustat. Instead we use
Ken French's published daily HML factor series via pandas_datareader.famafrench.
This is a long-short return series already (zero-cost portfolio in pct/day).

Window: 1990-01-01 -> present.
Comparison: same metrics on the MKT-RF factor over the same window.
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
        ff = web.DataReader("F-F_Research_Data_Factors_daily",
                            "famafrench", start="1990-01-01")[0] / 100.0
    except Exception as e:
        mark_failed("D01_value_hml", f"FF data fetch failed: {e}")
        return

    hml = ff["HML"].dropna()
    mkt = ff["Mkt-RF"].dropna()
    # align HML to same index as MKT
    hml.index = pd.to_datetime(hml.index)
    mkt.index = pd.to_datetime(mkt.index)

    m_hml = compute_metrics(hml, benchmark=mkt, name="D01 HML (Value)")
    m_mkt = compute_metrics(mkt, name="MKT-RF (reference)")
    print_metrics(m_hml)
    print_metrics(m_mkt)

    save_result("D01_value_hml", m_hml, extra={
        "status": "ok",
        "rule": "Long HML (high B/M minus low B/M) per Ken French daily factor.",
        "universe": "Ken French U.S. equity universe (CRSP+Compustat).",
        "source": "Fama-French; pandas_datareader.famafrench 'F-F_Research_Data_Factors_daily'.",
        "reference_mkt_rf": {
            "cagr": m_mkt["cagr"],
            "sharpe": m_mkt["sharpe"],
            "max_dd": m_mkt["max_dd"],
        },
        "shortcut_note": "Used Ken French published HML rather than replicating from CRSP/Compustat."
    })


if __name__ == "__main__":
    main()
