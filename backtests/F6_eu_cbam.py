"""
F6 EU CBAM quarterly reporting deadlines.

Carbon Border Adjustment Mechanism transitional reporting deadlines are
Jan 31 / Apr 30 / Jul 31 / Oct 31 since Oct 2023.

Short an equal-weight basket of EU carbon-intensive industrial names
(ThyssenKrupp TKA.DE, SSAB-B.ST, HeidelbergMaterials HEI.DE, Holcim HOLN.SW)
in the 15 calendar days before each deadline.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed

# Transitional reporting deadlines (Oct 2023 - 2026).
DEADLINES = [
    "2024-01-31",
    "2024-04-30",
    "2024-07-31",
    "2024-10-31",
    "2025-01-31",
    "2025-04-30",
    "2025-07-31",
    "2025-10-31",
    "2026-01-31",
    "2026-04-30",
]


def main():
    sid = "F6_eu_cbam"
    try:
        tickers = ["TKA.DE", "SSAB-B.ST", "HEI.DE", "HOLN.SW"]
        # Use SPY for benchmark approximation (or EZU). Use EZU as Euro proxy.
        bench_t = "EZU"
        all_t = tickers + [bench_t]
        import yfinance as yf
        df = yf.download(all_t, start="2022-06-01", end="2026-12-31",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            px = df["Close"]
        else:
            px = df[["Close"]]; px.columns = all_t
        px = px.dropna(how="all").sort_index()
        # Only use columns that loaded
        loaded = [t for t in tickers if t in px.columns and px[t].notna().sum() > 50]
        if len(loaded) < 2:
            return mark_failed(sid, f"Too few EU tickers loaded (got {loaded})")

        rets = px.pct_change()
        idx = rets.index

        pos = pd.DataFrame(0.0, index=idx, columns=loaded)
        used = []
        for d in DEADLINES:
            Dts = pd.Timestamp(d)
            loc = idx.searchsorted(Dts, side="left")
            if loc >= len(idx):
                continue
            start = max(loc - 11, 0)  # ~15 calendar days ≈ 11 trading days
            end = loc  # up to deadline session
            for t in loaded:
                pos.iloc[start:end, pos.columns.get_loc(t)] = -1.0 / len(loaded)
            used.append(d)

        port = (pos.shift(1) * rets[loaded]).sum(axis=1)
        active = (pos.abs().sum(axis=1) > 0)
        port_active = port[active.shift(1).fillna(False)].dropna()
        if len(port_active) < 5:
            return mark_failed(sid, "Too few active days")

        bench = rets[bench_t] if bench_t in rets.columns else None
        m = compute_metrics(port_active,
                            benchmark=(bench.reindex(port_active.index) if bench is not None else None),
                            name="F6 EU CBAM pre-deadline short carbon")
        print_metrics(m)
        save_result(sid, m, extra={
            "status": "ok",
            "rule": "Short equal-weight TKA.DE+SSAB-B.ST+HEI.DE+HOLN.SW from ~T-15c "
                    "(=~T-11 trading days) up to each CBAM deadline.",
            "mechanism": "Carbon liability scrutiny / disclosure risk into reporting deadlines.",
            "universe": str(loaded),
            "n_events": len(used),
            "events": used,
            "source": "EU CBAM Regulation 2023/956 deadlines; tickers via yfinance",
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return mark_failed(sid, f"unhandled exception: {e}")


if __name__ == "__main__":
    main()
