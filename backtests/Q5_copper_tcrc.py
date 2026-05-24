"""
Q5 Copper TC/RC (treatment/refining charges).

TC/RC is a smelter spot benchmark series published by Fastmarkets, Argus, SMM (Shanghai
Metals Market) and CRU. All four are subscription / paywalled. There is no free
historical CSV; only sporadic news mentions.

Marking failed.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    return mark_failed(
        "Q5_copper_tcrc",
        "Copper smelter TC/RC data is published only by paywalled providers (Fastmarkets, "
        "Argus, SMM/metal.com, CRU). No free historical timeseries exists.",
        extra={
            "source_attempted": [
                "https://www.metal.com (paywalled)",
                "https://www.fastmarkets.com (paywalled)",
            ],
            "rule_intended": "Long FCX or HG=F when TC/RC plunges (signal of concentrate tightness).",
            "mechanism": "Concentrate scarcity hammers smelter margins; tight upstream supply lifts refined copper.",
        },
    )


if __name__ == "__main__":
    main()
