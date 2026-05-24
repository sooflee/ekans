"""
H10 Coin Days Destroyed.

Status: FAILED — no free, unrestricted CDD time series found.
Tried:
  - blockchain.info charts API: no coin-days-destroyed series (404).
  - Blockchair: data exists but free tier is heavily rate-limited (430 from
    this IP); aggregate-transactions endpoint also rate-limited.
  - Glassnode free metrics: CDD is paywalled.
  - CryptoQuant: paywalled.

Computing CDD from raw on-chain UTXO data is feasible but would require:
  (a) Pulling ~10+ years of block-level UTXO sets (terabytes), or
  (b) An indexed UTXO db (which is what paid providers sell).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "H10_coin_days_destroyed",
        "No free unrestricted CDD time series. blockchain.info does not expose "
        "the series; Blockchair free tier is rate-limited (HTTP 430); Glassnode "
        "/ CryptoQuant CDD is paywalled. Building CDD from raw chain data "
        "requires an indexed UTXO DB, which is the product paid providers sell.",
        extra={
            "sources_tried": [
                "api.blockchain.info/charts/coin-days-destroyed (404)",
                "api.blockchair.com/bitcoin/* (HTTP 430)",
                "glassnode/cryptoquant (paywalled)",
            ],
            "rule": "Long BTC when 7d CDD < 30d MA - 1.5*sigma (HODLer accumulation); flat when CDD spikes > +2 sigma (distribution). NOT IMPLEMENTED.",
        },
    )


if __name__ == "__main__":
    main()
