"""
W12 ML which-anomalies-survive.

Original line of research: Chen-Zimmermann (RAPS 2022) 'Open Source Asset
Pricing', Jensen-Kelly-Pedersen (JF 2023) 'Is There a Replication Crisis in
Finance', and Bryzgalova-Pelger-Zhu (2023) 'Forest Through the Trees'.
They run an ML pipeline (random-forest / shrinkage / FDR) on hundreds of
published anomaly signals to identify which ones survive multiple-testing
corrections out of sample.

A real replication requires:
  - The full Open Asset Pricing csv dump (Chen & Zimmermann) — 200+ signals
    at monthly frequency, ~2.5GB.
  - Re-implementation of each signal on CRSP-grade data for live extension.
  - A multi-fold ML pipeline (lasso / gradient boosting) with FDR control.

End-to-end this is a few hours of compute and many hundreds of lines of code.
We mark this failed with the citation and skip.

Source: Chen-Zimmermann (RAPS 2022); Jensen-Kelly-Pedersen (JF 2023);
        Bryzgalova-Pelger-Zhu (JFE 2023).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "W12_ml_survivor_anomalies",
        ("Replicating the 'which anomalies survive ML / FDR' pipeline requires "
         "downloading the full Chen-Zimmermann Open Asset Pricing CSV dump "
         "(~200 signals at monthly frequency, ~2.5 GB) and running a multi-"
         "fold cross-validated ML pipeline with multiple-testing correction. "
         "This is hours of compute and out of scope for this batch — marked "
         "failed with the cite."),
        extra={
            "rule": ("INTENDED: run lasso / gradient-boosting on the cross-"
                      "section of ~200 published anomalies; apply FDR control "
                      "(Benjamini-Hochberg). Long top-quintile of surviving "
                      "anomaly composite, short bottom."),
            "universe": "Intended: CRSP US common stock 1980-2024.",
            "source": ("Chen & Zimmermann (RAPS 2022) 'Open Source Asset Pricing'; "
                        "Jensen, Kelly, Pedersen (JF 2023) 'Is There a Replication "
                        "Crisis in Finance?'; Bryzgalova, Pelger, Zhu (2023) "
                        "'Forest Through the Trees'."),
            "infra_required": ("Open Asset Pricing csv dump; ML pipeline "
                                "(lasso / xgboost / FDR); 8-16 GB RAM."),
        },
    )
    print("W12 ML survivor anomalies: marked failed (compute cost).")


if __name__ == "__main__":
    main()
