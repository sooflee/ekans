"""
F1 IRA Sec 48E/45Y solar guidance event drift.

For each IRS / Treasury guidance release related to the Inflation Reduction Act
(Sec 48E investment tax credit, Sec 45Y production tax credit, domestic content
adders, prevailing wage rules, transferability), long an equal-weight basket of
TAN (solar ETF), RUN (Sunrun), and FSLR (First Solar) from T-5 to T+3.

Curated dates from Federal Register IRS notices and Treasury press releases.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed

# Curated guidance release dates (publication / press-release dates).
# Source: Federal Register, Treasury press releases, IRS notices.
EVENTS = [
    "2022-11-29",  # IRS Notice 2022-61 prevailing wage / apprenticeship (init guidance)
    "2023-02-13",  # Notice 2023-17 low-income community bonus 48(e)
    "2023-05-12",  # Notice 2023-38 domestic content rules (48/45/45Y/48E)
    "2023-06-14",  # Treasury proposed regs on transferability/direct pay 6417/6418
    "2023-08-29",  # Proposed prevailing wage & apprenticeship NPRM
    "2024-04-25",  # Final regs on transferability of clean energy credits (TD 9993)
    "2024-05-16",  # Notice 2024-41 expanded domestic content safe harbor
    "2024-12-04",  # Final regs Sec 48 investment tax credit (TD 10015)
    "2025-01-15",  # Final regs Sec 45Y/48E technology-neutral credits
]


def main():
    sid = "F1_ira_solar_guidance"
    try:
        tickers = ["TAN", "RUN", "FSLR"]
        px = load_prices(tickers + ["SPY"], start="2021-06-01")
        rets = px.pct_change()

        # For each event, mark T-5 to T+3 trading days inclusive across the basket.
        idx = rets.index
        pos = pd.DataFrame(0.0, index=idx, columns=tickers)
        used = []
        for d in EVENTS:
            D = pd.Timestamp(d)
            loc = idx.searchsorted(D, side="left")
            if loc >= len(idx):
                continue
            start = max(loc - 5, 0)
            end = min(loc + 3, len(idx) - 1)
            for t in tickers:
                pos.iloc[start:end + 1, pos.columns.get_loc(t)] = 1.0 / len(tickers)
            used.append(d)

        # Apply positions (shift 1 for no look-ahead).
        port = (pos.shift(1) * rets[tickers]).sum(axis=1)
        # Restrict to active-event days for proper event metrics.
        active = (pos.sum(axis=1) > 0)
        port_active = port[active.shift(1).fillna(False)].dropna()

        spy_r = rets["SPY"]
        m = compute_metrics(port_active, benchmark=spy_r.reindex(port_active.index),
                            name="F1 IRA solar guidance basket")
        print_metrics(m)
        save_result(sid, m, extra={
            "status": "ok",
            "rule": "Long equal-weight TAN+RUN+FSLR from T-5 to T+3 around each IRA solar "
                    "guidance release.",
            "mechanism": "Policy clarity reduces project finance discount rate for solar developers; "
                         "ITC/PTC transferability liquidity premium.",
            "universe": "TAN, RUN, FSLR",
            "n_events": len(used),
            "events": used,
            "source": "Federal Register IRS notices, Treasury press releases (curated)",
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return mark_failed(sid, f"unhandled exception: {e}")


if __name__ == "__main__":
    main()
