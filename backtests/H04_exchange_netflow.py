"""
H04 Exchange netflow.

Status: FAILED — paywalled.
Tried free endpoints:
  - CryptoQuant API (api.cryptoquant.com): 401 Unauthorized, requires API key.
  - cryptoquant.com web charts: Cloudflare bot challenge (403).
  - Glassnode "free" metrics: requires login token even for limited daily series.
  - Coinglass open-api: 30001 "API key missing".
Other free sources (mempool.space, blockchain.info) do not publish per-exchange
on-chain netflow as a labelled time series — they would require us to maintain
an exchange-address tag set and reprocess on-chain data.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "H04_exchange_netflow",
        "Exchange-netflow time series is paywalled across all free providers "
        "(CryptoQuant API: 401; cryptoquant.com web: Cloudflare 403; Glassnode "
        "free tier requires login; Coinglass open-api requires key).",
        extra={
            "sources_tried": [
                "api.cryptoquant.com/v1/btc/exchange-flows/all-exchanges/netflow",
                "cryptoquant.com/asset/btc/chart/exchange-flows",
                "open-api.coinglass.com/public/v2/funding",
            ],
            "rule": "Exchange netflow > +X BTC -> sell pressure (short); netflow < -X -> demand (long). NOT IMPLEMENTED.",
        },
    )


if __name__ == "__main__":
    main()
