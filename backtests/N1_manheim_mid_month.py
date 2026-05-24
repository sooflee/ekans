"""
N1 Manheim Used Vehicle Value Index - mid-month estimate.
Manheim mid-month and monthly UVI prints are gated behind Cox Automotive's site
(login walls + JS-rendered pages). The historical mid-month series isn't exposed
as a free machine-readable feed. Marking failed; would need Cox/Manheim data
license or aggressive headless scrape with brittle DOM dependencies.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "N1_manheim_mid_month",
        "Manheim mid-month UVI not available as a free historical time series (Cox Automotive paywall / JS-rendered).",
        extra={
            "rule_intended": "Long AN/KMX/CVNA short basket when mid-month UVI YoY accelerates / decelerates beyond threshold.",
            "source": "manheim.com/services/consulting/used-vehicle-value-index",
            "universe": "Auto retail (AN, KMX, CVNA)",
        },
    )
    print("N1 Manheim mid-month: marked failed (data paywalled).")


if __name__ == "__main__":
    main()
