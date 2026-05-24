"""
S8 SHFE rebar inventory (Mysteel weekly) - data paywalled.

Same constraint as S7: Mysteel's weekly rebar inventory series at major
Chinese steel trader warehouses is subscription-only. SHFE itself does
not publish a free aggregated inventory CSV (only daily exchange-warehouse
stocks for the SHFE-deliverable contract, which is a small fraction of
the physical market).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "S8_shfe_rebar",
        "Mysteel weekly rebar warehouse inventory series is paid subscription only; "
        "no free historical archive identified.",
        extra={
            "rule": "Short steel-mill equities (e.g. STLD, NUE) when Mysteel weekly rebar inventory rises > 30% YoY for 4 weeks.",
            "mechanism": "Steel inventory build flags falling property / construction demand and a forward price drag on steel margins.",
            "source_attempted": "mysteel.net (paywall); SHFE official (exchange-warehouse only, not representative); SMM (paywall).",
        },
    )


if __name__ == "__main__":
    main()
