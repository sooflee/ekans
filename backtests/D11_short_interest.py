"""
D11 Short interest — not implemented as a quant backtest in this batch.

FINRA publishes bi-monthly short-interest data (Reg SHO) but the bulk download
is non-trivial:
  - per-symbol, per-settlement-date flat files at
    https://cdn.finra.org/equity/regsho/monthly/ (consolidated short interest)
  - schema changed over time; matching to a CRSP-grade universe requires
    a security-master join.

For a tractable backtest one would replicate Boehmer/Jones/Zhang 2008
('Which Shorts Are Informed?') by:
  - obtain semimonthly NYSE/Nasdaq short interest as % of float
  - rank cross-sectionally; long bottom quintile, short top quintile
  - hold 1 month; rebalance on each settlement date

We mark this signal as failed with an explicit citation rather than ship
a half-built ETL that won't produce trustworthy numbers.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "D11_short_interest",
        ("FINRA short-interest bi-monthly bulk download is non-trivial: per-symbol "
         "settlement-date files at cdn.finra.org/equity/regsho/monthly/ require a "
         "schema-versioned ETL and security-master join. Cite Boehmer, Jones & Zhang 2008 "
         "'Which Shorts Are Informed?' for the published estimate."),
        extra={
            "rule": "INTENDED: rank stocks by short-interest-as-pct-float; long bottom quintile, short top, monthly.",
            "universe": "Intended: NYSE+Nasdaq listed common stock.",
            "source": "FINRA Reg SHO short interest; Boehmer, Jones & Zhang 2008.",
            "shortcut_note": "Not run. Reference: Boehmer/Jones/Zhang 2008 finds ~16% annualized for the top-vs-bottom decile portfolio (value-weight, 1988-2005).",
        },
    )


if __name__ == "__main__":
    main()
