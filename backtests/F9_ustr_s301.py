"""
F9 USTR Section 301 comment window closure.

When the USTR publishes a Section 301 action and the public comment window
closes (typically 30-60 days after Federal Register publication), a final
decision usually follows 30+ days later. Long an equal-weight basket of
discretionary consumer / import-heavy retailers for 5 days after comment close.

Curated comment-close dates from Federal Register USTR notices (2018-2025).
Assumption: final action is >30 days away (true for all listed events).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed

# Curated USTR Section 301 comment-window close dates.
COMMENT_CLOSE = [
    "2018-05-22",   # List 1 hearings/comments closed
    "2018-07-23",   # List 2 comment close
    "2018-09-06",   # List 3 comment close
    "2019-06-17",   # List 4A/4B hearings
    "2019-07-31",   # List 4A comment close
    "2020-01-06",   # Phase 1 implementation comments
    "2022-07-05",   # Statutory 4-year review comment period
    "2024-07-14",   # Final 2024 tariff increases comment close
    "2024-12-22",   # Connected-vehicle / shipbuilding 301
    "2025-01-17",   # Semiconductor 301 comment close (proposed)
]


def main():
    sid = "F9_ustr_s301"
    try:
        tickers = ["AAPL", "NKE", "BBY", "WSM"]
        px = load_prices(tickers + ["SPY"], start="2017-06-01")
        rets = px.pct_change()
        idx = rets.index

        pos = pd.DataFrame(0.0, index=idx, columns=tickers)
        used = []
        for d in COMMENT_CLOSE:
            Dts = pd.Timestamp(d)
            loc = idx.searchsorted(Dts, side="right")
            if loc >= len(idx):
                continue
            start = loc
            end = min(loc + 5, len(idx) - 1)
            for t in tickers:
                pos.iloc[start:end + 1, pos.columns.get_loc(t)] = 1.0 / len(tickers)
            used.append(d)

        port = (pos.shift(1) * rets[tickers]).sum(axis=1)
        active = (pos.abs().sum(axis=1) > 0)
        port_active = port[active.shift(1).fillna(False)].dropna()
        if len(port_active) < 5:
            return mark_failed(sid, "Too few active days")

        m = compute_metrics(port_active, benchmark=rets["SPY"].reindex(port_active.index),
                            name="F9 USTR Section 301 comment-close basket")
        print_metrics(m)
        save_result(sid, m, extra={
            "status": "ok",
            "rule": "5-day long equal-weight AAPL+NKE+BBY+WSM after each USTR Section 301 "
                    "comment-window close (assumes final action >30d away).",
            "mechanism": "Removal of near-term tariff escalation tail relieves discretionary "
                         "import retailers and consumer-tech multiples.",
            "universe": "AAPL, NKE, BBY, WSM",
            "n_events": len(used),
            "events": used,
            "source": "Federal Register USTR Section 301 notices (curated)",
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return mark_failed(sid, f"unhandled exception: {e}")


if __name__ == "__main__":
    main()
