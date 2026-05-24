"""
NEW Director Network Interlock.

Hypothesis: when a director at company A buys (Form 4) shares of company B where
another B-director is on A's board, long the buy. Hold 6 months. (JBF 2020
director-network paper).

Implementation requires parsing SEC DEF 14A filings since 2015 to build the
director→company bipartite graph (~5000 DEF 14A's/year × parsing tables of
director names). This is heavy NLP work — DEF 14A's are unstructured PDFs/HTML
with non-standard director-bio sections. Tractable open datasets that pre-build
this graph (BoardEx, ISS, MSCI) are paywalled.

Per the user's instructions, marking as infeasible-in-this-pass.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    sid = "NEW02_director_interlock"
    return mark_failed(
        sid,
        "DEF14A bulk parsing infeasible in this pass; cite Larcker-So-Wang 2013 "
        "(JBF 2020 director-network paper). BoardEx/ISS director-graph datasets "
        "required for clean implementation. Form 4 buy-by-director-of-A-on-B-board "
        "events are <50/yr at S&P 1500 scale even with the right data.",
        extra={
            "name": "NEW02 Director network interlock",
            "rule": "Director at A buys B Form 4, where B-director sits on A-board → long buy, hold 6m.",
            "source": "Larcker-So-Wang 2013, Hwang-Kim 2009 (director networks); JBF 2020 paper cited.",
            "blocker": "DEF14A parsing for ~75k filings since 2015 + name-resolution for ~50k directors.",
        }
    )


if __name__ == "__main__":
    main()
