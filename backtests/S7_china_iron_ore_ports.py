"""
S7 China 45-port iron ore inventory (Mysteel) - data paywalled.

Mysteel publishes a weekly survey of iron ore stocks held at China's
45 major ports. Historical archives are behind a paid Mysteel subscription
(>$10k/yr). sxcoal.com mirrors current snapshots but no historical
download. Steel Home, mysteel.net, and SMM all restrict the time series
to subscribers.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "S7_china_iron_ore_ports",
        "Mysteel 45-port iron ore weekly inventory series is paid subscription only; "
        "no public free historical archive identified after attempts on mysteel.net and sxcoal.com.",
        extra={
            "rule": "Short iron ore (BHP / RIO equity proxy) when 45-port stock exceeds 150 Mt threshold for 4 consecutive weeks.",
            "mechanism": "Glut at Chinese ports flags weak steel-mill demand and a forward iron-ore price drag.",
            "source_attempted": "mysteel.net (paywall); sxcoal.com (current only); Steel Home (paywall).",
        },
    )


if __name__ == "__main__":
    main()
