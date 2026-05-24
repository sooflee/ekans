"""
W11 Option IV-slope time-series long-minus-short (TLM).

Original signal (Cao-Han-Wang, JFE 2023): a cross-sectional long-short on the
time-series of each stock's implied-volatility slope (ATM-OTM put IV gradient)
generates significant alpha. Requires daily option chains across the entire
cross-section of stocks for many years.

We mark this failed because:
  - No free daily option-chain source covers a useful universe historically.
    Yahoo's per-ticker option endpoint returns only the current chain, not
    a historical surface.
  - OptionMetrics / Cboe DataShop are paywalled.
  - A SPY-only IV-slope proxy would not capture the cross-sectional alpha
    the paper documents.

Source: Cao, Han, Wang (JFE 2023) 'The Cross-Section of Option-Implied Slopes'
        and follow-ups.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "W11_option_iv_slope_tlm",
        ("Requires daily IV-slope (ATM-OTM put gradient) across the cross-"
         "section of optionable stocks. No free historical option-chain panel "
         "exists; OptionMetrics and Cboe DataShop are paywalled. A SPY-only "
         "proxy collapses to a time-series test and is not the published "
         "cross-sectional result."),
        extra={
            "rule": ("INTENDED: estimate each optionable stock's IV-slope "
                      "(ATM minus 25-delta OTM put), monthly rank, long top "
                      "decile / short bottom decile."),
            "universe": "Intended: Russell 3000 optionable common stock.",
            "source": ("Cao, Han, Wang (JFE 2023) 'The Cross-Section of "
                        "Option-Implied Slopes'."),
            "infra_required": ("Historical option chain panel (OptionMetrics / "
                                "Cboe DataShop) or equivalent."),
        },
    )
    print("W11 IV-slope TLM: marked failed (no free option panel).")


if __name__ == "__main__":
    main()
