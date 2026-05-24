"""
Y1 Estimize beat-street signal.

Idea: Estimize crowd-sourced earnings estimates often lead the Wall Street
consensus. When the Estimize consensus is materially above I/B/E/S consensus
("beat-street") ahead of an earnings print, the stock has a positive drift
into and over the announcement.

Why we mark this failed:
  - Estimize's public API (api.estimize.com) was restricted to analyst-tier
    paying subscribers (~mid-2020 onwards), and the public dataset on
    Kaggle stops in 2019.
  - Even the analyst-tier API now requires an enterprise license through
    Nasdaq Data Link (Estimize was acquired by Quandl/Nasdaq).
  - Without a continuous Estimize-vs-IBES delta panel we cannot construct
    the signal in this batch.

Intended replication:
  1. For each earnings event, compute Δ = (Estimize EPS consensus −
     IBES EPS consensus) / |IBES EPS consensus|.
  2. Long top quintile Δ, short bottom quintile, position window
     [t−5, t+1] around announcement.
  3. Equal-weight, daily rebalance, market-neutral.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "Y1_estimize_beat_street",
        ("Estimize beat-street signal requires (a) Estimize crowd-EPS "
         "consensus and (b) IBES Wall-Street consensus, both per earnings "
         "event. Estimize was acquired by Nasdaq (Quandl) and now sits "
         "behind an enterprise paywall; the legacy public API has been "
         "shut down. No free path to a full beat-street panel."),
        extra={
            "rule": ("INTENDED: long top-quintile Estimize-minus-IBES EPS "
                     "delta, short bottom-quintile, hold from t-5 to t+1 "
                     "around earnings announcement, equal-weight."),
            "mechanism": ("Crowd consensus reflects retail / buy-side "
                          "private estimates that anticipate sell-side "
                          "revisions; positive drift into earnings."),
            "source": ("Jame, Johnston, Markov, Wolfe (2016) 'The Value of "
                       "Crowdsourced Earnings Forecasts' Journal of "
                       "Accounting Research 54(4); Da-Huang (RFS 2019)."),
            "data_required": ("Estimize crowd-EPS consensus per event "
                              "(Nasdaq Data Link enterprise); IBES Wall-St "
                              "consensus (Refinitiv); ~3000 stocks × 4 "
                              "quarters × ~10 years = 120k event panel."),
            "api_paywall": "Estimize → Nasdaq Data Link enterprise tier; IBES → Refinitiv.",
        },
    )
    print("Y1 Estimize beat-street: marked failed (paywalled API).")


if __name__ == "__main__":
    main()
