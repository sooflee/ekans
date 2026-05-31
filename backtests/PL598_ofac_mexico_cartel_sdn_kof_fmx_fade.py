"""PL598_ofac_mexico_cartel_sdn_kof_fmx_fade - OFAC Mexican-Cartel SDN Cluster -> Short KOF+FMX+CX / Long XLP

Event study on OFAC Mexican-cartel SDN cluster dates (>=4 designations in
a weekly window). At next open: short equal-weight KOF/FMX/CX, long XLP
(dollar-neutral). Hold 15 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


EVENTS = [
    "2022-03-08", "2022-09-29", "2022-12-15", "2023-04-14",
    "2023-10-26", "2024-02-13", "2024-05-21", "2024-09-12",
    "2025-01-23",
]
HOLD_DAYS = 15


def main():
    sid = "PL598_ofac_mexico_cartel_sdn_kof_fmx_fade"
    try:
        px = load_prices(["KOF", "FMX", "CX", "XLP", "SPY"], start="2020-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    ret = daily_returns(px)
    if "SPY" not in ret.columns:
        return mark_failed(sid, "SPY missing")
    spy_r = ret["SPY"]

    shorts = [t for t in ["KOF", "FMX", "CX"] if t in ret.columns]
    if len(shorts) < 2 or "XLP" not in ret.columns:
        return mark_failed(sid, "missing required tickers")

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
        if end - loc < 8:
            continue
        s_basket = -1.0 * ret[shorts].mean(axis=1).iloc[loc:end]
        l_xlp = ret["XLP"].iloc[loc:end]
        # Dollar-neutral pair: average of short basket and long XLP
        net = (s_basket + l_xlp) / 2.0
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

    m = compute_metrics(pnl, benchmark=spy_r, name="OFAC Mexico Cartel SDN Pair Fade")
    save_result(sid, m, extra={
        "rule": "Short KOF/FMX/CX + long XLP at T+1 after >=4 Mexico-jurisdiction SDN designations clustered in a single week. Hold 15d, dollar-neutral.",
        "mechanism": "Sanctions cluster increases Mexico-jurisdiction risk premium; consumer staples (XLP) outperform as defensive offset.",
        "source": "OFAC SDN list (curated Mexico-cartel cluster dates 2022-2025) + yfinance",
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
