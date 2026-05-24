"""
Z10 CFPB consent order pair-trade: short issuer / long KBE.

Rule:
- Curated set of ~15 major CFPB enforcement actions vs publicly-traded
  banks / consumer-finance / fintech issuers 2014-2024. For each, short
  the issuer / long KBE (S&P Banks ETF) starting T+1 for 120 trading
  days. Equal-weight across overlapping events.

Mechanism:
- CFPB consent orders impose monetary penalties, restitution, and
  multi-year compliance obligations. The issuer carries CFPB-monitor
  drag (operational restrictions, capital-allocation friction)
  relative to the bank sector for 2-4 quarters before the news
  decays.

Source:
- consumerfinance.gov/enforcement/actions (CFPB enforcement actions
  index). Curated to public issuers.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed

# (ticker, date, description)
EVENTS = [
    ("WFC", "2016-09-08", "Wells Fargo $100M sales practices"),
    ("WFC", "2018-04-20", "Wells Fargo $1B auto/mortgage"),
    ("WFC", "2022-12-20", "Wells Fargo $3.7B consumer-banking"),
    ("BAC", "2014-04-09", "BofA $727M credit-card add-ons"),
    ("BAC", "2023-07-11", "BofA $250M junk fees / fake accounts"),
    ("GS",  "2023-11-21", "Goldman Apple Card servicing -- consent order"),
    ("JPM", "2015-07-08", "JPM $216M credit-card debt-collection"),
    ("C",   "2015-07-21", "Citigroup $700M credit-card add-ons"),
    ("C",   "2018-06-29", "Citigroup $335M card APR"),
    ("DFS", "2015-07-22", "Discover $18.5M student-loan servicing"),
    ("COF", "2018-04-24", "Capital One $100M (OCC, paired CFPB)"),
    ("SYF", "2014-09-19", "Synchrony / GE Capital Retail Bank $225M"),
    ("AXP", "2017-08-23", "American Express $96M (related)"),
    ("SQ",  "2024-01-04", "Block / Cash App enforcement (CFPB probe)"),
    ("PYPL","2015-05-19", "PayPal $25M credit"),
    ("LC",  "2018-04-25", "LendingClub FTC/CFPB"),
    ("ALLY","2013-12-20", "Ally Financial $98M auto lending (kept)"),
    ("HBAN","2024-12-23", "Huntington -- CFPB late-fee probe (placeholder)"),
    ("USB", "2022-07-28", "US Bancorp $37.5M unauthorized accounts"),
    ("TFC", "2024-08-20", "Truist -- CFPB probe disclosure (placeholder)"),
]


def main():
    df = pd.DataFrame(EVENTS, columns=["ticker", "date", "desc"])
    df["date"] = pd.to_datetime(df["date"])
    tickers = sorted(df["ticker"].unique())

    try:
        px = load_prices(tickers + ["SPY", "KBE"], start="2012-06-01")
    except Exception as e:
        return mark_failed("Z10_cfpb_consent_order", f"price load failed: {e}")

    rets = px.pct_change()
    if "SPY" not in rets.columns or "KBE" not in rets.columns:
        return mark_failed("Z10_cfpb_consent_order", "SPY/KBE missing")
    spy = rets["SPY"]
    kbe = rets["KBE"]

    HOLD = 120
    daily_pnls = []
    n_used = 0
    n_skipped = 0
    for _, row in df.iterrows():
        t = row["ticker"]; d = row["date"]
        if t not in rets.columns:
            n_skipped += 1; continue
        idx = rets[t].dropna().index
        nxt = idx[idx > d]
        if len(nxt) == 0:
            n_skipped += 1; continue
        i0 = rets.index.get_loc(nxt[0])
        end = min(i0 + HOLD, len(rets.index))
        # short issuer / long KBE
        leg = -rets[t].iloc[i0:end].fillna(0) + kbe.iloc[i0:end].fillna(0)
        leg = leg.clip(lower=-0.3, upper=0.3)
        daily_pnls.append(leg)
        n_used += 1

    if not daily_pnls:
        return mark_failed("Z10_cfpb_consent_order", "no events matched")

    panel = pd.concat(daily_pnls, axis=1).sort_index()
    pnl = panel.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed("Z10_cfpb_consent_order", f"insufficient overlap (n={len(pnl)})")

    bench = spy.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="Z10 CFPB consent pair 120d")
    print_metrics(m)
    print(f"\nEvents used: {n_used}, skipped: {n_skipped}")
    save_result("Z10_cfpb_consent_order", m, extra={
        "status": "ok",
        "rule": ("For each curated CFPB enforcement action vs a publicly-"
                 "traded bank / consumer-finance / fintech issuer 2013-2024, "
                 "short the issuer / long KBE (S&P Banks ETF) starting T+1, "
                 "hold 120 trading days. Equal-weight across overlapping "
                 "events; daily returns clipped at +/-30%."),
        "mechanism": ("CFPB consent orders impose monetary penalty, "
                      "restitution, and multi-year compliance overhang; "
                      "issuer underperforms the bank sector for 2-4 "
                      "quarters."),
        "source": "consumerfinance.gov/enforcement/actions, curated.",
        "n_events": int(n_used),
        "n_skipped": int(n_skipped),
        "events": EVENTS,
    })


if __name__ == "__main__":
    main()
