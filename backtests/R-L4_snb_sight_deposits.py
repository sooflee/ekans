"""
R-L4 SNB sight deposits (retry).

Original spec: SNB weekly sight deposits change -> short EUR/CHF when 2sigma
above 52w mean. Attempted substitutions:
1) data.snb.ch CSV API -- unreachable from this env (HTTPSConnectionPool/DNS
   blocked).
2) FRED's Swiss M1 (MANMM101CHM189S) -- series discontinued at 2018-12;
   only ~10 years available, p95 of z-score = 0.84 so the 2-sigma rule
   never triggers (n_events=0). Below 1.5sigma also gave 0.
3) Other Swiss reserve series (SWITERESM, TRESEGCHM052N, BIS series) are
   either unavailable on FRED or non-existent.

Marking failed -- no clean SNB sight-deposit series accessible.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import mark_failed


SIGNAL_ID = "R-L4_snb_sight_deposits"


def main():
    mark_failed(
        SIGNAL_ID,
        ("SNB direct API blocked from sandbox; FRED Swiss M1 (MANMM101CHM189S) "
         "ends 2018-12 and yields 0 trigger events at 2 sigma (or 1.5 sigma); "
         "no alternative Swiss sight-deposit series on FRED."),
        extra={
            "rule": "Long FXF / short FXE 4 weeks when 2sigma above 52w mean change in SNB sight deposits.",
            "mechanism": "SNB intervention-driven CHF weakness mean-reverts after sight-deposit growth normalizes.",
            "source_attempted": "data.snb.ch (network-gated); FRED MANMM101CHM189S (discontinued 2018-12).",
            "data_substitution": "None successful.",
        },
    )


if __name__ == "__main__":
    main()
