"""
R-M1 ETH staking APR (retry).

Original spec: long ETH-USD when 30d MA staking APR drops below 2.8%.
Attempted sources:
1) Lido https://stake.lido.fi/api/lido-stats -- 404.
2) eth-api-v2.lido.fi / api.lido.fi -- DNS / connection blocked.
3) Beaconchain https://beaconcha.in/api/v1/* -- 401 (API key required).
4) rated.network / dappnode endpoints -- 404 / unreachable.

A historical APR time series spanning multiple regimes (>=12 months) cannot
be assembled from any free open endpoint reachable from this sandbox.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import mark_failed


SIGNAL_ID = "R-M1_eth_staking_apr"


def main():
    mark_failed(
        SIGNAL_ID,
        ("No free ETH staking APR time series accessible from sandbox: Lido APIs 404 or "
         "blocked, Beaconchain requires API key, alternative aggregators 404 / DNS-blocked."),
        extra={
            "rule": "Long ETH-USD when 30d MA staking APR drops below 2.8%.",
            "mechanism": "Low APR coincides with high participation rates / network maturity; valuation reflects security premium.",
            "source_attempted": "stake.lido.fi/api, beaconcha.in/api, api.rated.network, dappnode staking API.",
            "data_substitution": "None successful.",
        },
    )


if __name__ == "__main__":
    main()
