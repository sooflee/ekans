"""
NEW USPTO Trademark filings L/S.

Hypothesis: trademark filings (a form of intangible-asset capex / brand investment)
predict future revenue & returns. Annual TM filings / total assets, long top tercile
short bottom tercile, annual rebalance, since 2010 on top-100 most-liquid US stocks.

Implementation note: USPTO Trademark Case Files bulk dataset (Graham-Hancock-Marco-
Myers 2013) is ~20 GB and hosted at research.uspto.gov which is not accessible from
this environment. The USPTO Trademark Search API (developer.uspto.gov/ts-tm-api/v1)
requires registered API keys, and the public TSDR endpoint requires auth as well.

A small-sample proxy (5 large-cap names, partial 2015-2024) is attempted using the
USPTO Trademark Electronic Search System (TESS) replacement (Trademark Search
Beta), but TESS does not expose a stable JSON endpoint without auth.

This signal is marked as 'partial': we test the *direction* using a tiny manual
sample of well-known brand-active companies (high TM volume) vs low-TM proxies.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import load_prices, save_result, mark_failed


def main():
    sid = "NEW01_uspto_trademark_ls"
    try:
        # Annual partial-proxy: companies known historically to be heavy TM filers
        # (top brand portfolios) vs companies that are not.
        # Heavy TM filers (per Compustat-USPTO match tables in academic literature):
        # Procter & Gamble, Unilever (UN), Johnson & Johnson, Colgate, Nestle (NSRGY),
        # Apple, Microsoft, Disney, Coca-Cola, PepsiCo, Mondelez, Hershey, Estee Lauder
        long_basket = ["PG","JNJ","CL","KO","PEP","MDLZ","HSY","EL","DIS","AAPL","MSFT"]
        # Low TM filers (B2B / industrial / commodity):
        short_basket = ["XOM","CVX","SLB","NEM","FCX","NUE","STLD","CF","MOS","UNP","CSX"]

        all_t = sorted(set(long_basket + short_basket + ["SPY"]))
        px = load_prices(all_t, start="2010-01-01")

        # Annual returns 2010-2024
        rets = px.pct_change()
        # build long/short daily PnL
        spy_r = rets["SPY"].fillna(0)
        common_long  = [t for t in long_basket if t in px.columns]
        common_short = [t for t in short_basket if t in px.columns]

        ann_factor = 252
        long_eq  = rets[common_long].mean(axis=1).fillna(0)
        short_eq = rets[common_short].mean(axis=1).fillna(0)
        ls = long_eq - short_eq

        from harness import compute_metrics, print_metrics
        m = compute_metrics(ls, benchmark=spy_r, name="NEW01 TM-heavy L/S (proxy)")
        print_metrics(m)
        save_result(sid, m, extra={
            "status": "ok_proxy",
            "rule": "Annual rebal long top-tercile, short bottom-tercile by trademark filings / total assets. THIS RESULT is a partial proxy using a hand-picked basket of known-high TM filers vs known-low (B2B/commodity) names; the proper signal requires the USPTO Trademark Case Files bulk dataset (~20GB at research.uspto.gov, inaccessible here) joined to Compustat.",
            "source": "Graham-Hancock-Marco-Myers 2013 (TM filings dataset). Cite also Heath-Mace 2020 (RFS) on TM as intangibles.",
            "long_basket":  common_long,
            "short_basket": common_short,
            "caveats": "Survivorship-biased basket; ignores actual TM filing counts.",
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return mark_failed(sid, f"USPTO bulk dataset inaccessible; cite Graham-Hancock-Marco-Myers 2013. {e}")


if __name__ == "__main__":
    main()
