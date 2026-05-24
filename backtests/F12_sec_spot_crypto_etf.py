"""
F12 SEC 19b-4 spot crypto ETF decision deadlines.

The SEC has rule-change decision deadlines under 19b-4 for proposed spot
crypto ETF listings. Approval-track deadlines historically produced strong
run-ups into the decision date.

Strategy: long the underlying spot crypto (BTC-USD or ETH-USD) 30 trading
days ending on the decision deadline.

Known major events:
  - BTC spot ETF final 19b-4 decision: 2024-01-10 (approved)
  - ETH spot ETF final 19b-4 decision: 2024-05-23 (approved)
  Optional/pending: SOL, XRP, LTC under 2025 review windows.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed

EVENTS = [
    ("2024-01-10", "BTC-USD"),
    ("2024-05-23", "ETH-USD"),
    # Pending / approved later — used if data available.
    ("2025-10-16", "SOL-USD"),  # SOL spot ETF final deadline (approx)
    ("2025-11-14", "XRP-USD"),  # XRP spot ETF deadline (approx)
]


def main():
    sid = "F12_sec_spot_crypto_etf"
    try:
        # Reduce to available pairs.
        unique = sorted({t for _, t in EVENTS})
        all_t = unique + ["SPY"]
        import yfinance as yf
        df = yf.download(all_t, start="2022-06-01", end="2026-12-31",
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            px = df["Close"]
        else:
            px = df[["Close"]]; px.columns = all_t
        px = px.dropna(how="all").sort_index()
        loaded = [t for t in unique if t in px.columns and px[t].notna().sum() > 20]

        rets = px.pct_change()
        idx = rets.index

        pos = pd.DataFrame(0.0, index=idx, columns=loaded)
        used = []
        for d, tk in EVENTS:
            if tk not in pos.columns:
                continue
            Dts = pd.Timestamp(d)
            loc = idx.searchsorted(Dts, side="left")
            if loc >= len(idx):
                continue  # future
            start = max(loc - 30, 0)
            end = loc  # inclusive of deadline session
            pos.iloc[start:end + 1, pos.columns.get_loc(tk)] = 1.0
            used.append({"deadline": d, "ticker": tk})

        port = (pos.shift(1) * rets[loaded]).sum(axis=1)
        # If multiple overlap, sum positions (here events don't overlap, so OK).
        active = (pos.abs().sum(axis=1) > 0)
        port_active = port[active.shift(1).fillna(False)].dropna()
        if len(port_active) < 5:
            return mark_failed(sid, f"Too few active days (got {len(port_active)})")

        m = compute_metrics(port_active, benchmark=rets["SPY"].reindex(port_active.index),
                            name="F12 SEC spot crypto ETF pre-deadline")
        print_metrics(m)
        save_result(sid, m, extra={
            "status": "ok_tiny_n",
            "rule": "30 trading days ending on each SEC 19b-4 spot crypto ETF final decision "
                    "deadline, long the underlying spot crypto.",
            "mechanism": "Approval-track run-up: market prices in approval probability; "
                         "ETF flows accumulate near deadline.",
            "universe": str(loaded),
            "n_events": len(used),
            "events": used,
            "data_caveat": "Tiny N; signal heavily concentrated in BTC Jan 2024 and ETH May 2024.",
            "source": "SEC EDGAR 19b-4 filings; major events publicly known",
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return mark_failed(sid, f"unhandled exception: {e}")


if __name__ == "__main__":
    main()
