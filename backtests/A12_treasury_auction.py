"""
A12 Treasury auction cycle.
Per spec: skip — auction calendar parsing is heavier than a quick test allows.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "A12_treasury_auction",
        "auction calendar parsing too heavy for this pass; cite Lou-Yan-Zhang 2013",
        extra={
            "source": "Lou, Yan, Zhang (RFS 2013)",
            "universe": "TLT / Treasury",
        },
    )
    print("A12 treasury auction: marked failed per spec.")


if __name__ == "__main__":
    main()
