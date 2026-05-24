"""
W1 FinGPT MD&A sentiment.

Paper: Wang, Zhang, Liu (2023-2024) 'FinGPT: Open-Source Financial LLM' and
the follow-up 'Instruct-FinGPT' (arXiv:2306.06031, 2308.08555) show that an
LLM fine-tuned on financial text produces tradable cross-sectional sentiment
scores from 10-K Management Discussion & Analysis (MD&A) sections.

Why we mark this failed:
  - Requires running a multi-billion-parameter LLM (FinGPT / Llama2-7B base)
    on the GPU.
  - Requires a corpus of thousands of 10-K MD&A sections downloaded from
    EDGAR and parsed (Item 7).
  - Requires a CRSP-grade universe + alignment of filing dates → trading dates.
  - End-to-end ETL is several GB of disk + many hours of GPU inference, well
    beyond the scope of a quick public-data backtest.

A clean replication would need:
  1. EDGAR full-text search to enumerate 10-K filings 2010-2025 for the
     Russell 3000.
  2. Extract Item 7 MD&A from each filing (parsing variants since 2002).
  3. Run FinGPT-Forecaster / Instruct-FinGPT on each MD&A to get a
     sentiment / forecast token.
  4. Cross-sectional rank → top/bottom decile long-short, 6-month hold.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "W1_fingpt_mda_sentiment",
        ("Running FinGPT on 10-K MD&A text requires a GPU, the FinGPT/Llama2-7B "
         "checkpoint from HuggingFace, and parsing thousands of EDGAR Item 7 "
         "filings. Infrastructure cost is well outside a quick backtest pass."),
        extra={
            "rule": ("INTENDED: Apply FinGPT-Forecaster to each Russell-3000 "
                     "10-K MD&A. Cross-sectional decile portfolios, 6-month "
                     "rebalance, long top decile / short bottom decile."),
            "universe": "Intended: Russell 3000 common stock, 2010-2025.",
            "source": ("Wang, Zhang, Liu (2023) 'FinGPT: Open-Source Financial "
                       "LLM' arXiv:2306.06031; Yang et al. (2023) "
                       "'Instruct-FinGPT' arXiv:2308.08555."),
            "infra_required": ("GPU (>=24GB), HuggingFace FinGPT checkpoint, "
                                "EDGAR 10-K corpus + Item 7 extractor, ~thousands "
                                "of filings to process."),
        },
    )
    print("W1 FinGPT MD&A sentiment: marked failed (infra cost).")


if __name__ == "__main__":
    main()
