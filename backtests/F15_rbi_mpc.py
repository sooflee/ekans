"""
F15 RBI Monetary Policy Committee calendar.

The RBI MPC holds 6 scheduled meetings per year. Long INDA 3 trading days
before each MPC decision IF the most recent Indian CPI YoY is within ±50bp
of the 4% target.

We use FRED INDCPIALLMINMEI (India CPI all items index) to compute YoY %.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import load_prices, load_fred, compute_metrics, print_metrics, save_result, mark_failed

# Curated RBI MPC scheduled meeting dates (post-MPC-formation; MPC began 2016).
# Sources: RBI press releases, MPC resolution archives.
MEETINGS = [
    # 2018
    "2018-02-07", "2018-04-05", "2018-06-06", "2018-08-01", "2018-10-05", "2018-12-05",
    # 2019
    "2019-02-07", "2019-04-04", "2019-06-06", "2019-08-07", "2019-10-04", "2019-12-05",
    # 2020
    "2020-02-06", "2020-03-27", "2020-05-22", "2020-08-06", "2020-10-09", "2020-12-04",
    # 2021
    "2021-02-05", "2021-04-07", "2021-06-04", "2021-08-06", "2021-10-08", "2021-12-08",
    # 2022
    "2022-02-10", "2022-04-08", "2022-05-04", "2022-06-08", "2022-08-05",
    "2022-09-30", "2022-12-07",
    # 2023
    "2023-02-08", "2023-04-06", "2023-06-08", "2023-08-10", "2023-10-06", "2023-12-08",
    # 2024
    "2024-02-08", "2024-04-05", "2024-06-07", "2024-08-08", "2024-10-09", "2024-12-06",
    # 2025
    "2025-02-07", "2025-04-09", "2025-06-06", "2025-08-06", "2025-10-01", "2025-12-05",
]


def main():
    sid = "F15_rbi_mpc"
    try:
        px = load_prices(["INDA", "SPY"], start="2017-06-01")
        rets = px.pct_change()
        idx = rets.index

        # India CPI from FRED. INDCPIALLMINMEI is index-level (monthly).
        try:
            cpi = load_fred(["INDCPIALLMINMEI"], start="2010-01-01")
        except Exception as e:
            return mark_failed(sid, f"FRED India CPI series unavailable: {e}")
        cpi = cpi.dropna()
        cpi_col = cpi.columns[0]
        cpi_yoy = cpi[cpi_col].pct_change(12) * 100  # % YoY
        cpi_yoy = cpi_yoy.dropna()

        pos = pd.Series(0.0, index=idx)
        used = []
        for d in MEETINGS:
            Dts = pd.Timestamp(d)
            loc = idx.searchsorted(Dts, side="left")
            if loc < 3 or loc >= len(idx):
                continue
            # Last available CPI YoY observation prior to meeting.
            cidx = cpi_yoy.index[cpi_yoy.index <= Dts]
            if len(cidx) == 0:
                continue
            yoy = float(cpi_yoy.loc[cidx[-1]])
            if abs(yoy - 4.0) > 0.5:
                continue
            start = max(loc - 3, 0)
            end = loc
            pos.iloc[start:end] = 1.0
            used.append({"meeting": d, "cpi_yoy": yoy})

        port = pos.shift(1) * rets["INDA"]
        active = (pos != 0)
        port_active = port[active.shift(1).fillna(False)].dropna()
        if len(port_active) < 5:
            return mark_failed(sid, f"Too few active days ({len(port_active)}); "
                                    f"only {len(used)} meetings met CPI band")

        m_full = compute_metrics(port.fillna(0).dropna(),
                                 benchmark=rets["SPY"].reindex(port.index),
                                 name="F15 RBI MPC + CPI-in-band -> INDA")
        if "error" in m_full:
            import numpy as _np
            mean_r = float(port_active.mean()); std_r = float(port_active.std())
            m = {
                "name": "F15 RBI MPC + CPI-in-band -> INDA",
                "n_active_days": int(len(port_active)),
                "mean_daily_ret": mean_r, "std_daily_ret": std_r,
                "cum_return": float((1 + port_active).prod() - 1),
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
            "status": "ok",
            "rule": "3 trading days before each RBI MPC meeting, long INDA IF prior CPI YoY "
                    "within ±50bp of 4% target.",
            "mechanism": "When inflation is on-target, MPC has policy space and meetings reduce "
                         "tail risk; equity rallies into well-anchored decisions.",
            "universe": "INDA",
            "n_meetings": len(MEETINGS),
            "n_events_traded": len(used),
            "cpi_series": "INDCPIALLMINMEI (FRED)",
            "events": used,
            "source": "RBI MPC resolution archives + FRED INDCPIALLMINMEI",
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return mark_failed(sid, f"unhandled exception: {e}")


if __name__ == "__main__":
    main()
