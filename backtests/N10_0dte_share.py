"""
N10 0DTE option-share daily metric.
CBOE publishes daily SPX volume but the by-expiry breakdown that yields 0DTE share
is only available via CBOE LiveVol or DataShop (paid). Public daily aggregates do
not separate 0DTE expiries historically.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "N10_0dte_share",
        "0DTE share of SPX option volume not available as free historical time series; CBOE DataShop/LiveVol paywall.",
        extra={
            "rule_intended": "Mean-revert SPY when 0DTE share spikes > 2 sigma above 60d mean.",
            "source": "CBOE DataShop / LiveVol",
            "universe": "SPY / SPX options",
        },
    )
    print("N10 0DTE share: marked failed (data paywalled).")


if __name__ == "__main__":
    main()
