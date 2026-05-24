"""
Z3 SEC whistleblower bounty / award-order reverse engineering.

Why we mark this failed:
- The SEC's whistleblower program publishes Final Orders only after
  the underlying enforcement action has closed. The orders are
  redacted (employer name, dates, $ amount of fraud) -- by statute
  the SEC must protect the whistleblower's identity, which means
  the issuer involved is intentionally obscured. Reverse-engineering
  an issuer from a redacted order is speculative at best; in many
  cases multiple plausible defendant companies match the award text.
- Even with a complete mapping, the *trade* window is ambiguous --
  the precipitating event is the tip itself, the public enforcement
  action, or the bounty announcement, each of which has a different
  signature.

Intended replication (if mapping were tractable):
  1. For each Whistleblower Final Order 2012-2024, identify the
     underlying enforcement action by cross-referencing redacted
     award amount / time period against the SEC litigation release
     index.
  2. From the enforcement action map to a public issuer (CIK lookup).
  3. Short the issuer 20 trading days starting on the date of the
     PUBLIC bounty announcement.

Source / paper:
  Call, A. et al. "Whistleblowers and Outcomes of Financial
  Misrepresentation Enforcement Actions" JAR 2018.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "Z3_whistleblower_bounty",
        ("SEC whistleblower Final Orders are statutorily redacted (issuer "
         "name, dates, $ size of underlying fraud). Mapping an award back "
         "to a public issuer is speculative -- typically multiple defendant "
         "candidates fit, and the relevant trade window is ambiguous "
         "(tip date vs enforcement action date vs bounty announcement)."),
        extra={
            "rule": ("INTENDED: for each SEC Whistleblower Final Order 2012+, "
                     "identify the underlying enforcement action, then map to "
                     "a public-issuer CIK, then short the issuer 20 trading "
                     "days from the bounty announcement."),
            "mechanism": ("Bounty announcement is a clean public-information "
                          "shock that the whistleblower tip was the "
                          "precipitating mechanism for an enforcement, often "
                          "with follow-on litigation."),
            "source": ("SEC OWB Final Orders page; SEC litigation releases "
                       "index; EDGAR for issuer mapping."),
            "data_obstacle": ("Final Orders are redacted by statute under "
                              "Section 21F(h)(2) of the Exchange Act. "
                              "Reverse-engineering issuer identity is "
                              "non-falsifiable for this backtest."),
        },
    )
    print("Z3 whistleblower bounty: marked failed (redacted by statute).")


if __name__ == "__main__":
    main()
