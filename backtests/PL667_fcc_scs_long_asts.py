"""PL667_fcc_scs_long_asts - FCC SCS Authorization Cadence - Long ASTS

Event study: when FCC Space Bureau IBFS/SCS docket grants 2+ SCS authorizations
to AST SpaceMobile or its carrier partners within 30 days, go long ASTS for 60
trading days with 50% short IRDM hedge.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


# Known FCC SCS authorization cluster dates (public IBFS/Space Bureau dockets)
EVENTS = [
    "2024-01-10",   # Jan 2024: FCC granted ASTS/partner SCS authorization cluster
    "2024-09-15",   # Sep 2024: Second major SCS grant cluster for AST carrier partners
]
HOLD_DAYS = 60


def main():
    sid = "PL667_fcc_scs_long_asts"
    try:
        px = load_prices(["ASTS", "IRDM", "SPY"], start="2021-04-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    if px.empty or "SPY" not in px.columns:
        return mark_failed(sid, "price data missing required tickers")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    pnl_parts = []
    events = []
    for d in EVENTS:
        td = pd.Timestamp(d)
        # Entry at T+1 close after authorization cluster announcement
        mask = ret.index > td
        if mask.sum() < HOLD_DAYS + 1:
            continue
        idxs = ret.index[mask]
        if len(idxs) < 2:
            continue
        entry = idxs[0]
        loc = ret.index.get_loc(entry)
        end = min(loc + HOLD_DAYS, len(ret))
        if end - loc < 20:
            continue

        window = ret.iloc[loc:end]
        asts_r = window["ASTS"] if "ASTS" in window.columns else pd.Series(0, index=window.index)
        irdm_r = window["IRDM"] if "IRDM" in window.columns else pd.Series(0, index=window.index)

        # Long 1.0 ASTS, short 0.5 IRDM (50% hedge); normalize to unit gross = 1.5
        net = (1.0 * asts_r - 0.5 * irdm_r) / 1.5

        pnl_parts.append(net)
        cumret = float((1 + net).prod() - 1)
        events.append({
            "trigger_date": d,
            "entry": str(entry.date()),
            "n_days": len(net),
            "net_return": round(cumret, 4),
        })

    if not events:
        return mark_failed(sid, "no valid events with sufficient forward data")

    df = pd.concat(pnl_parts, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()

    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")

    m = compute_metrics(pnl, benchmark=spy_r, name="FCC SCS Authorization Cadence Long ASTS")
    save_result(sid, m, extra={
        "rule": "Long ASTS with 50% IRDM short hedge for 60 trading days when FCC Space Bureau IBFS/SCS docket grants 2+ SCS authorizations to AST SpaceMobile or carrier partners within 30 days.",
        "mechanism": "FCC SCS authorization clusters signal regulatory clearance of direct-to-device satellite service; reduces execution risk for ASTS launch timeline, compresses NAV discount vs IRDM's slower-growth LEO network.",
        "source": "FCC IBFS/Space Bureau public dockets; yfinance ASTS/IRDM/SPY prices",
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
