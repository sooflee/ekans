"""
F13 EU AI Act phased entry-into-force dates.

The EU AI Act (Regulation 2024/1689) enters into force in phases:
  - 2025-02-02: Prohibited AI practices apply
  - 2025-08-02: GPAI (general-purpose AI) obligations apply
  - 2026-08-02: Most rules apply (high-risk Annex III)
  - 2027-08-02: Full application (incl. legacy high-risk systems)

Strategy: short SAP.DE + DASTY + ASML.AS 10 trading days before each phase
start.

Only 1-2 realized events; tiny N.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from harness import compute_metrics, print_metrics, save_result, mark_failed

PHASES = [
    "2025-02-02",
    "2025-08-02",
    "2026-08-02",
    "2027-08-02",
]


def main():
    sid = "F13_eu_ai_act"
    try:
        tickers = ["SAP.DE", "DASTY", "ASML.AS"]
        import yfinance as yf
        df = yf.download(tickers + ["SPY"], start="2024-01-01", end="2026-12-31",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            px = df["Close"]
        else:
            px = df[["Close"]]; px.columns = tickers + ["SPY"]
        px = px.dropna(how="all").sort_index()
        loaded = [t for t in tickers if t in px.columns and px[t].notna().sum() > 30]
        if len(loaded) < 2:
            return mark_failed(sid, f"Too few EU tickers loaded ({loaded})")

        rets = px.pct_change()
        idx = rets.index

        pos = pd.DataFrame(0.0, index=idx, columns=loaded)
        used = []
        for d in PHASES:
            Dts = pd.Timestamp(d)
            loc = idx.searchsorted(Dts, side="left")
            if loc >= len(idx):
                continue
            start = max(loc - 10, 0)
            end = loc
            for t in loaded:
                pos.iloc[start:end, pos.columns.get_loc(t)] = -1.0 / len(loaded)
            used.append(d)

        port = (pos.shift(1) * rets[loaded]).sum(axis=1)
        active = (pos.abs().sum(axis=1) > 0)
        port_active = port[active.shift(1).fillna(False)].dropna()
        if len(port_active) < 3:
            return mark_failed(sid, f"Too few active days ({len(port_active)}); "
                                    f"phases realized: {used}")

        m_full = compute_metrics(port.fillna(0).dropna(),
                                 benchmark=rets["SPY"].reindex(port.index),
                                 name="F13 EU AI Act pre-phase short EU tech")
        if "error" in m_full:
            import numpy as _np
            mean_r = float(port_active.mean()); std_r = float(port_active.std())
            m = {
                "name": "F13 EU AI Act pre-phase short EU tech",
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
            "status": "ok_tiny_n",
            "rule": "10 trading days before each EU AI Act phase-start date, short equal-weight "
                    "SAP.DE+DASTY+ASML.AS.",
            "mechanism": "Compliance cost overhang on EU enterprise software and AI suppliers.",
            "universe": str(loaded),
            "n_events": len(used),
            "events": used,
            "data_caveat": "Tiny N (1-2 phases realized); interpret cautiously.",
            "source": "EU Regulation 2024/1689 (AI Act) phase dates",
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return mark_failed(sid, f"unhandled exception: {e}")


if __name__ == "__main__":
    main()
