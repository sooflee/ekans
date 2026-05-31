"""PL599_scotus_cert_concentration_defensive_rotation - SCOTUS Major-Question Cert + Concentration -> Defensive Rotation

Event study on SCOTUS major-question cert-grant dates where SPY top-5
concentration was >27% and VIX3M was <16. At T+1 close: long 0.5 XLU +
0.5 IEF, short 1.0 XLY. Hold 30 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


# Curated SCOTUS major-question cert-grants meeting filters
EVENTS = [
    "2024-10-07",
    "2024-12-13",
    "2025-01-13",
    "2025-04-21",
]
HOLD_DAYS = 30


def main():
    sid = "PL599_scotus_cert_concentration_defensive_rotation"
    try:
        px = load_prices(["XLU", "XLY", "IEF", "SPY"], start="2023-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    ret = daily_returns(px)
    if "SPY" not in ret.columns:
        return mark_failed(sid, "SPY missing")
    spy_r = ret["SPY"]

    pnl_parts = []
    events = []
    for d in EVENTS:
        td = pd.Timestamp(d)
        mask = ret.index >= td
        if mask.sum() < HOLD_DAYS + 2:
            continue
        idxs = ret.index[mask]
        if len(idxs) < 2:
            continue
        entry = idxs[1]
        loc = ret.index.get_loc(entry)
        end = min(loc + HOLD_DAYS, len(ret))
        if end - loc < 15:
            continue
        xlu = ret["XLU"].iloc[loc:end] if "XLU" in ret.columns else 0
        xly = ret["XLY"].iloc[loc:end] if "XLY" in ret.columns else 0
        ief = ret["IEF"].iloc[loc:end] if "IEF" in ret.columns else 0
        # 0.5 XLU + 0.5 IEF - 1.0 XLY, scale by gross 2.0
        net = (0.5 * xlu + 0.5 * ief - 1.0 * xly) / 2.0
        pnl_parts.append(net)
        cumret = float((1 + net).prod() - 1)
        events.append({"trigger_date": d, "entry": str(entry.date()),
                       "net_return": round(cumret, 4)})

    if not events:
        return mark_failed(sid, "no valid events")

    df = pd.concat(pnl_parts, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")

    m = compute_metrics(pnl, benchmark=spy_r, name="SCOTUS Cert + Concentration Defensive Rotation")
    save_result(sid, m, extra={
        "rule": "T+1 close: long 0.5 XLU + 0.5 IEF, short 1.0 XLY, hold 30d, after major-question SCOTUS cert grant + concentration >27% + VIX3M <16.",
        "mechanism": "Regulatory uncertainty + complacency at high concentration -> defensive rotation, discretionary underperformance.",
        "source": "SCOTUS cert calendar (curated 2024-2025) + yfinance",
        "n_events": len(events),
        "events": events,
        "avg_net_return": round(float(np.mean([e["net_return"] for e in events])), 4),
        "event_win_rate": round(float(np.mean([e["net_return"] > 0 for e in events])), 4),
    })
    sharpe = m.get("sharpe", 0)
    cagr = m.get("cagr", 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%, n_days={len(pnl)}")


if __name__ == "__main__":
    main()
