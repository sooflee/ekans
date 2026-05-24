"""
J03 Super Bowl Indicator (null test).

Rule:
- Hardcode Super Bowl winners since 1995, mapping each to NFC vs AFC (and the historical legacy
  conference for pre-merger teams; we mark Ravens/Steelers as AFC, etc).
- For each year y: SB played early-Feb. If winner is NFC -> long SPY from Feb 1 (y) to Dec 31 (y).
  If AFC -> stay in cash.
- Daily returns: pos*ret series.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from harness import load_prices, compute_metrics, print_metrics, save_result

# Super Bowl winners and their conference (NFC/AFC), 1995-2025.
# The Super Bowl played in February of year Y is "for the prior NFL season".
# Convention: indexed by calendar year of game.
SB_WINNERS = {
    1995: ("San Francisco 49ers", "NFC"),
    1996: ("Dallas Cowboys", "NFC"),
    1997: ("Green Bay Packers", "NFC"),
    1998: ("Denver Broncos", "AFC"),
    1999: ("Denver Broncos", "AFC"),
    2000: ("St. Louis Rams", "NFC"),
    2001: ("Baltimore Ravens", "AFC"),
    2002: ("New England Patriots", "AFC"),
    2003: ("Tampa Bay Buccaneers", "NFC"),
    2004: ("New England Patriots", "AFC"),
    2005: ("New England Patriots", "AFC"),
    2006: ("Pittsburgh Steelers", "AFC"),
    2007: ("Indianapolis Colts", "AFC"),
    2008: ("New York Giants", "NFC"),
    2009: ("Pittsburgh Steelers", "AFC"),
    2010: ("New Orleans Saints", "NFC"),
    2011: ("Green Bay Packers", "NFC"),
    2012: ("New York Giants", "NFC"),
    2013: ("Baltimore Ravens", "AFC"),
    2014: ("Seattle Seahawks", "NFC"),
    2015: ("New England Patriots", "AFC"),
    2016: ("Denver Broncos", "AFC"),
    2017: ("New England Patriots", "AFC"),
    2018: ("Philadelphia Eagles", "NFC"),
    2019: ("New England Patriots", "AFC"),
    2020: ("Kansas City Chiefs", "AFC"),
    2021: ("Tampa Bay Buccaneers", "NFC"),
    2022: ("Los Angeles Rams", "NFC"),
    2023: ("Kansas City Chiefs", "AFC"),
    2024: ("Kansas City Chiefs", "AFC"),
    2025: ("Philadelphia Eagles", "NFC"),
}


def main():
    px = load_prices(["SPY"], start="1995-01-01")["SPY"]
    rets = px.pct_change()
    idx = rets.index

    pos = pd.Series(0.0, index=idx)
    for d in idx:
        y = d.year
        if y in SB_WINNERS:
            _, conf = SB_WINNERS[y]
            # Long from Feb 1 to Dec 31 if NFC
            if conf == "NFC" and d.month >= 2:
                pos.loc[d] = 1.0
    pnl = (pos.shift(1).fillna(0) * rets).dropna()
    m = compute_metrics(pnl, benchmark=rets.dropna(), name="J03 Super Bowl Indicator")
    print_metrics(m)
    n_nfc = sum(1 for v in SB_WINNERS.values() if v[1] == "NFC")
    n_afc = sum(1 for v in SB_WINNERS.values() if v[1] == "AFC")
    save_result("J03_super_bowl", m, extra={
        "status": "ok",
        "rule": "NFC wins SB -> long SPY Feb 1 to Dec 31. AFC -> cash.",
        "universe": "SPY daily",
        "n_nfc_years": n_nfc,
        "n_afc_years": n_afc,
        "warning": "Known-debunked null indicator; provided for documentation only.",
        "source": "Folk indicator (Krzysztof Borowski et al.)",
    })


if __name__ == "__main__":
    main()
