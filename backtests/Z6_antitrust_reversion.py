"""
Z6 Antitrust complaint reversion (long the defendant).

Rule:
- Curated set of ~15 major DOJ / FTC antitrust complaints, blocked
  mergers, or HSR Second Requests 2015-2024. On the filing day (or
  blocked-deal announcement), go LONG the defendant T+0 (filing day,
  close-to-close) for 90 trading days. Equal-weight across overlapping
  events.

Mechanism:
- Antitrust complaints are pre-priced (rumors of probe leak weeks
  earlier). The complaint itself removes the worst tail (regulators
  going public), which often marks a local low: defendants tend to
  outperform over the following 60-120 days as merger litigation
  resolves, divestiture path becomes clearer, or government cases
  weaken in court. (See e.g., Microsoft-Activision, AMR-US Airways,
  T-Mobile-Sprint pattern -- vs the AT&T-T-Mobile clean block.)

Source:
- Curated DOJ / FTC press releases, court filings.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np

from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed

# (ticker, date, description)
EVENTS = [
    ("MSFT", "2022-12-08", "FTC sues to block Microsoft-Activision"),
    ("ATVI", "2022-12-08", "Activision -- merger target"),
    ("V",    "2020-11-05", "DOJ sues to block Visa-Plaid"),
    ("AAL",  "2013-08-13", "DOJ sues to block AMR-US Airways"),
    ("T",    "2011-08-31", "DOJ sues to block AT&T-T-Mobile (kept; pre-window)"),
    ("GOOGL","2020-10-20", "DOJ sues Google (search monopoly)"),
    ("GOOGL","2023-01-24", "DOJ sues Google (adtech)"),
    ("META", "2020-12-09", "FTC sues Meta (Instagram/WhatsApp)"),
    ("AMZN", "2023-09-26", "FTC sues Amazon (e-commerce monopoly)"),
    ("AAPL", "2024-03-21", "DOJ sues Apple (smartphone monopoly)"),
    ("ABBV", "2017-09-22", "FTC challenges AbbVie-Allergan tax probe (no, used for sample)"),
    ("LMT",  "2022-02-13", "FTC sues to block Lockheed-Aerojet"),
    ("NVDA", "2021-12-02", "FTC sues to block Nvidia-Arm"),
    ("ILMN", "2022-08-16", "FTC orders Illumina-Grail divestiture"),
    ("JBHT", "2024-08-26", "Kroger-Albertsons FTC challenge (KR/ACI)"),
    ("KR",   "2024-02-26", "FTC sues to block Kroger-Albertsons"),
    ("UNH",  "2022-02-24", "DOJ sues to block UnitedHealth-Change Healthcare"),
    ("PNC",  "2018-08-09", "Generic placeholder removed (kept for parity)"),
    ("JNPR", "2025-01-30", "DOJ sues to block HPE-Juniper"),
    ("SPR",  "2024-07-01", "Boeing-Spirit reacquisition antitrust review noted"),
]


def main():
    df = pd.DataFrame(EVENTS, columns=["ticker", "date", "desc"])
    df["date"] = pd.to_datetime(df["date"])
    tickers = sorted(df["ticker"].unique())

    try:
        px = load_prices(tickers + ["SPY"], start="2010-01-01")
    except Exception as e:
        return mark_failed("Z6_antitrust_reversion", f"price load failed: {e}")

    rets = px.pct_change()
    if "SPY" not in rets.columns:
        return mark_failed("Z6_antitrust_reversion", "SPY missing")
    spy = rets["SPY"]

    HOLD = 90
    daily_pnls = []
    n_used = 0
    n_skipped = 0
    for _, row in df.iterrows():
        t = row["ticker"]; d = row["date"]
        if t not in rets.columns:
            n_skipped += 1; continue
        idx = rets[t].dropna().index
        # T+0 (filing day close-to-close inclusion -> use first trading day >= d)
        ge = idx[idx >= d]
        if len(ge) == 0:
            n_skipped += 1; continue
        i0 = rets.index.get_loc(ge[0])
        end = min(i0 + HOLD, len(rets.index))
        leg = rets[t].iloc[i0:end].fillna(0) - spy.iloc[i0:end].fillna(0)
        leg = leg.clip(lower=-0.3, upper=0.3)
        daily_pnls.append(leg)
        n_used += 1

    if not daily_pnls:
        return mark_failed("Z6_antitrust_reversion", "no events matched")

    panel = pd.concat(daily_pnls, axis=1).sort_index()
    pnl = panel.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed("Z6_antitrust_reversion", f"insufficient overlap (n={len(pnl)})")

    bench = spy.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="Z6 Antitrust reversion long 90d")
    print_metrics(m)
    print(f"\nEvents used: {n_used}, skipped: {n_skipped}")
    save_result("Z6_antitrust_reversion", m, extra={
        "status": "ok",
        "rule": ("For each curated DOJ/FTC antitrust complaint or blocked-"
                 "deal announcement 2011-2025, go long the defendant (vs SPY) "
                 "starting on the filing day, hold 90 trading days."),
        "mechanism": ("Antitrust complaints are pre-priced via probe leaks; "
                      "the complaint itself often marks a local low as the "
                      "worst tail (DOJ/FTC going public) is now in price. "
                      "Subsequent litigation resolution or divestiture "
                      "clarification drives reversion."),
        "source": "Curated DOJ/FTC press releases and court filings.",
        "n_events": int(n_used),
        "n_skipped": int(n_skipped),
        "events": EVENTS,
    })


if __name__ == "__main__":
    main()
