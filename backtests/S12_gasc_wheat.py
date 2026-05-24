"""
S12 GASC wheat tender results - historical archive limited.

Egypt's GASC (General Authority for Supply Commodities) is the world's
largest single wheat importer. Tender results (volume, origin, price) are
published as Reuters / Agricensus news flashes. Aggregated historical
panel data sits behind Reuters / S&P-Platts paywalls.

In 2024 Egypt restructured GASC tendering into the new Mostakbal Misr
agency with even less reporting transparency.

No public free historical tender-by-tender database was found after
attempts on Reuters, Agricensus, World-Grain.com, and FAO GIEWS.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "S12_gasc_wheat",
        "GASC tender-result historical panel is paywalled (Reuters/Agricensus). "
        "From late 2024 the new Mostakbal Misr agency has reduced reporting frequency further.",
        extra={
            "rule": "When GASC tender clears > $20/ton above prior tender same origin, long ZW=F 30 days.",
            "mechanism": "GASC clearing higher = global wheat tightness leakage into the world's largest single-buyer benchmark.",
            "source_attempted": "Reuters GASC desk (paywall); Agricensus (paywall); World Grain (editorial); FAO GIEWS (monthly only).",
        },
    )


if __name__ == "__main__":
    main()
