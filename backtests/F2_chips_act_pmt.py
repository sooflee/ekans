"""
F2 CHIPS Act Preliminary Memorandum of Terms (PMT) announcements.

Commerce Department / NIST CHIPS Program Office announces PMT awards 2024-2026.
For each, long the named recipient ticker and short SOXX for 2 trading days
(announcement day + next day open / 2-day hold from next session).

Curated from NIST CHIPS Act announcements / Commerce press releases.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed

# (announcement_date, ticker) — recipient must be publicly traded.
# Source: Commerce Department / NIST CHIPS Program Office press releases.
EVENTS = [
    ("2024-02-19", "GFS"),   # GlobalFoundries (first PMT)
    ("2024-03-20", "INTC"),  # Intel ($8.5B)
    ("2024-04-08", "TSM"),   # TSMC Arizona ($6.6B)
    ("2024-04-15", "MU"),    # Micron ($6.1B)
    ("2024-04-25", "MCHP"),  # Microchip ($162M)
    ("2024-08-16", "TXN"),   # Texas Instruments ($1.6B)
    ("2024-09-04", "POLAR"), # Polar Semi — placeholder; not public
    ("2024-10-15", "WOLF"),  # Wolfspeed ($750M)
    ("2024-11-26", "INTC"),  # Intel final award $7.86B
    ("2024-12-09", "MU"),    # Micron final award
    ("2024-12-20", "TSM"),   # TSMC final award $6.6B
    ("2025-01-10", "GFS"),   # GlobalFoundries final
    ("2025-01-14", "AMKR"),  # Amkor Technology ($407M)
]


def main():
    sid = "F2_chips_act_pmt"
    try:
        # Drop non-public placeholders.
        events = [(d, t) for d, t in EVENTS if t not in ("POLAR",)]
        unique_tk = sorted({t for _, t in events})
        all_t = unique_tk + ["SOXX", "SPY"]
        px = load_prices(all_t, start="2023-06-01")
        rets = px.pct_change()
        idx = rets.index

        # For each event: from T+1 open to T+2 close (i.e., 2 trading days starting
        # the session after the announcement). Long recipient (+1) and short SOXX (-1).
        pos_recip = pd.Series(0.0, index=idx)  # net (combined) return holder
        recip_ret = pd.Series(0.0, index=idx)
        soxx_ret = pd.Series(0.0, index=idx)
        used = []

        for d, tk in events:
            if tk not in rets.columns:
                continue
            D = pd.Timestamp(d)
            loc = idx.searchsorted(D, side="right")
            if loc >= len(idx) - 2:
                continue
            # 2 trading day hold starting at loc (T+1 session).
            r_t = rets[tk].iloc[loc:loc + 2].values
            r_s = rets["SOXX"].iloc[loc:loc + 2].values
            for i, day_loc in enumerate(range(loc, loc + 2)):
                if day_loc < len(idx):
                    recip_ret.iloc[day_loc] += r_t[i]
                    soxx_ret.iloc[day_loc] += r_s[i]
            used.append((d, tk))

        # Combine: long recipient - short SOXX (since potentially overlapping events).
        port = (recip_ret - soxx_ret)
        # Keep only event-active days.
        active = (port != 0)
        port_active = port[active]
        if len(port_active) < 5:
            return mark_failed(sid, "Too few active event days")

        spy_r = rets["SPY"]
        # Use full period (zero on non-event days) for sharpe; also event-only stats.
        m = compute_metrics(port, benchmark=spy_r.reindex(port.index),
                            name="F2 CHIPS PMT recipient vs SOXX")
        m["n_active_days"] = int(len(port_active))
        m["mean_active_ret"] = float(port_active.mean())
        m["std_active_ret"] = float(port_active.std())
        m["hit_rate_active"] = float((port_active > 0).mean())
        m["t_stat_active"] = (
            float(port_active.mean() / (port_active.std() / np.sqrt(len(port_active))))
            if port_active.std() > 0 else 0.0
        )
        print_metrics(m)
        save_result(sid, m, extra={
            "status": "ok",
            "rule": "On each CHIPS Act PMT announcement, long named recipient and short SOXX "
                    "for 2 trading days starting the session after.",
            "mechanism": "Award reduces equity dilution risk and de-risks fab capex; ETF leg "
                         "isolates idiosyncratic award news.",
            "universe": "INTC, MU, TSM, GFS, MCHP, TXN, WOLF, AMKR vs SOXX",
            "n_events": len(used),
            "events": [list(e) for e in used],
            "source": "Commerce Dept / NIST CHIPS Program Office press releases (curated)",
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return mark_failed(sid, f"unhandled exception: {e}")


if __name__ == "__main__":
    main()
