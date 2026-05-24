"""
V6 Bob Elliott CTA replication
Original rule:
  Fetch HFRX Macro/CTA Index monthly NAVs (HFR site).
  Regress 24m rolling against Fung-Hsieh PTFS (Primitive Trend-Following
  Strategy) factors — straddle-replicated bond, FX, commodity, stock, and
  short-rate trends.
  Build an ETF basket of UUP/GLD/DBC/TLT/SHY rebalanced monthly with the
  regressed loadings.
Status: marked failed.
Reasons:
  1. HFRX index NAVs are not freely downloadable; HFR's free site shows only
     a public scrolling chart, and historical CSV requires a paid login.
  2. The Fung-Hsieh PTFS factors require option-straddle simulation across
     five asset classes (or the David Hsieh research files, which are
     credentialled).
  3. Without both inputs, the rolling regression produces meaningless
     loadings; we won't ship a regression on noise.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "V6_bob_elliott_replication",
        "HFRX Macro/CTA NAVs require paid HFR access; Fung-Hsieh PTFS factors require credentialled Hsieh data files. Replication infeasible without both.",
        extra={
            "source": "Bob Elliott (Unlimited Funds), YouTube interview round 2 (Phase 1V).",
            "mechanism": "Replicate CTA index by regressing its returns onto trend-following factors and trading a transparent ETF basket.",
            "next_step": "Subscribe to HFR Database + David Hsieh PTFS factors, or build PTFS lookbacks from CRSP futures data.",
        },
    )


if __name__ == "__main__":
    main()
