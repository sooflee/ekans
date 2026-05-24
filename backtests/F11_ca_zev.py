"""
F11 California ACC II ZEV compliance threshold start dates.

CARB's Advanced Clean Cars II requires increasing ZEV % share starting MY2026.
Compliance thresholds reset Jan 1 of each model year:
  MY2026 = 35%, MY2027 = 43%, MY2028 = 51% (etc.).

Strategy: long TSLA+RIVN / short F+GM equal-weight, 10 trading days before
Jan 1 of each compliance year.

Only 1-3 realized events; tiny N — interpret with caution.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed

# Compliance threshold start dates (Jan 1 of model-year start, MY begins prior
# Oct typically but rule cites Jan 1).
EVENTS = [
    ("2026-01-01", 0.35),
    ("2027-01-01", 0.43),
    ("2028-01-01", 0.51),
]


def main():
    sid = "F11_ca_zev"
    try:
        longs = ["TSLA", "RIVN"]
        shorts = ["F", "GM"]
        all_t = longs + shorts + ["SPY"]
        px = load_prices(all_t, start="2024-01-01")
        rets = px.pct_change()
        idx = rets.index

        pos = pd.DataFrame(0.0, index=idx, columns=longs + shorts)
        used = []
        for d, threshold in EVENTS:
            Dts = pd.Timestamp(d)
            loc = idx.searchsorted(Dts, side="left")
            if loc >= len(idx):
                # event in future of available data — skip
                continue
            start = max(loc - 10, 0)
            end = loc
            for t in longs:
                pos.iloc[start:end, pos.columns.get_loc(t)] = 1.0 / len(longs)
            for t in shorts:
                pos.iloc[start:end, pos.columns.get_loc(t)] = -1.0 / len(shorts)
            used.append({"date": d, "threshold": threshold})

        port = (pos.shift(1) * rets[longs + shorts]).sum(axis=1)
        active = (pos.abs().sum(axis=1) > 0)
        port_active = port[active.shift(1).fillna(False)].dropna()
        if len(port_active) < 3:
            return mark_failed(sid, f"Too few active days (got {len(port_active)}); "
                                    f"only {len(used)} events realized in data window")

        # Use compute_metrics on full period to get sharpe; manual on active.
        m_full = compute_metrics(port.fillna(0).dropna(),
                                 benchmark=rets["SPY"].reindex(port.index),
                                 name="F11 CA ACC II ZEV threshold pre-Jan1")
        if "error" in m_full:
            # Tiny N — compute manual metrics.
            import numpy as _np
            mean_r = float(port_active.mean())
            std_r = float(port_active.std())
            cum = float((1 + port_active).prod() - 1)
            m = {
                "name": "F11 CA ACC II ZEV threshold pre-Jan1",
                "n_active_days": int(len(port_active)),
                "mean_daily_ret": mean_r,
                "std_daily_ret": std_r,
                "cum_return": cum,
                "t_stat": (mean_r / (std_r / _np.sqrt(len(port_active)))) if std_r > 0 else 0.0,
                "hit_rate": float((port_active > 0).mean()),
                "start": str(port_active.index[0].date()),
                "end": str(port_active.index[-1].date()),
            }
        else:
            m = m_full
            m["n_active_days"] = int(len(port_active))
            m["cum_return_active"] = float((1 + port_active).prod() - 1)
        print(m)
        save_result(sid, m, extra={
            "status": "ok_tiny_n",
            "rule": "10 trading days before Jan 1 of each ACC II compliance year, "
                    "long TSLA+RIVN / short F+GM equal-weight.",
            "mechanism": "Compliance fleet mix shock favors pure-play BEV vs legacy OEMs; "
                         "ZEV credit market value asymmetry.",
            "universe": "TSLA, RIVN, F, GM",
            "n_events": len(used),
            "events": used,
            "data_caveat": "Tiny N (1-3 events realized in data window); interpret cautiously.",
            "source": "CARB ACC II Final Regulation Order (2022)",
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return mark_failed(sid, f"unhandled exception: {e}")


if __name__ == "__main__":
    main()
