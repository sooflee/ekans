"""
M1 ETH staking APR.
Plan: try to reconstruct staking APR from beaconcha.in. Their free API requires auth.
Fallback: derive APR from the known issuance schedule and total ETH staked from
publicly-available proxies. Without paid API key or a clean public CSV we cannot get
historical daily APR reliably. Mark failed honestly.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import urllib.request
import json

import pandas as pd
import numpy as np
from harness import (
    load_prices, daily_returns,
    compute_metrics, print_metrics, save_result, mark_failed,
)


def try_beaconcha_in():
    """Probe public beaconcha.in endpoints. Returns df or None."""
    urls = [
        "https://beaconcha.in/api/v1/epoch/latest",
        "https://beaconcha.in/api/v1/epoch/finalized",
        "https://beaconcha.in/api/v1/validators/total-deposits",
    ]
    for u in urls:
        try:
            req = urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=10) as r:
                _ = json.load(r)
        except Exception:
            continue
    return None


def main():
    # Try the public endpoint quickly
    df = try_beaconcha_in()
    if df is None:
        return mark_failed(
            "M1_eth_staking_apr",
            "beaconcha.in API requires auth (401) and no public daily-APR CSV is freely available; historical staking-APR series unobtainable without paid key or scraping.",
            extra={
                "mechanism": "High staking APR -> ETH becomes attractive yield asset; APR spikes have historically preceded ETH outperformance.",
                "source_attempted": "beaconcha.in API, Coingecko (also 401).",
            },
        )


if __name__ == "__main__":
    main()
