"""
G-19 China NPC two-sessions post-meeting drift.

NPC typically convenes March 5 and closes ~10 days later. Hardcode start/end dates 2010-2025.
Strategy: short FXI 5 trading days before NPC start; cover at NPC close. Then flip long FXI for
10 trading days post-close.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from harness import load_prices, compute_metrics, print_metrics, save_result

# NPC plenary session start/close dates 2010-2025. Sources: NPC press releases / Wikipedia.
NPC = [
    ("2010-03-05", "2010-03-14"),
    ("2011-03-05", "2011-03-14"),
    ("2012-03-05", "2012-03-14"),
    ("2013-03-05", "2013-03-17"),
    ("2014-03-05", "2014-03-13"),
    ("2015-03-05", "2015-03-15"),
    ("2016-03-05", "2016-03-16"),
    ("2017-03-05", "2017-03-15"),
    ("2018-03-05", "2018-03-20"),
    ("2019-03-05", "2019-03-15"),
    ("2020-05-22", "2020-05-28"),  # postponed due to COVID
    ("2021-03-05", "2021-03-11"),
    ("2022-03-05", "2022-03-11"),
    ("2023-03-05", "2023-03-13"),
    ("2024-03-05", "2024-03-11"),
    ("2025-03-05", "2025-03-11"),
]


def main():
    px = load_prices(["FXI"], start="2009-06-01")["FXI"]
    rets = px.pct_change()
    idx = rets.index

    pos = pd.Series(0.0, index=idx)
    used = []
    for s, e in NPC:
        S = pd.Timestamp(s); E = pd.Timestamp(e)
        i_s = idx.searchsorted(S)
        i_e = idx.searchsorted(E)
        if i_s >= len(idx) or i_e >= len(idx):
            continue
        S_loc = idx.get_loc(idx[i_s]) if idx[i_s] == S else i_s
        E_loc = idx.get_loc(idx[i_e]) if idx[i_e] == E else i_e
        # Short 5 days before S to E close
        sh_start = max(0, S_loc - 5)
        sh_end = E_loc
        pos.iloc[sh_start:sh_end + 1] = -1.0
        # Long for 10 trading days after E
        lo_start = E_loc + 1
        lo_end = min(E_loc + 10, len(idx) - 1)
        if lo_start <= lo_end:
            pos.iloc[lo_start:lo_end + 1] = 1.0
        used.append((s, e))

    pnl = (pos.shift(1).fillna(0) * rets).dropna()
    spy = load_prices(["SPY"], start="2009-06-01")["SPY"].pct_change()
    m = compute_metrics(pnl, benchmark=spy, name="G19 NPC two-sessions")
    print_metrics(m)
    save_result("G19_npc_two_sessions", m, extra={
        "status": "ok",
        "rule": "Short FXI 5 trading days before NPC start through close; long FXI 10 trading days "
                "after NPC close.",
        "universe": "FXI",
        "n_events": len(NPC),
        "events": used,
        "pct_days_short_long_net": float(pos.mean()),
        "source": "NPC plenary calendar (Wikipedia / Xinhua)",
    })


if __name__ == "__main__":
    main()
