"""PL705 — EU DMA Article 8 Non-Compliance -> Short Named Gatekeeper"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# Curated EC DMA enforcement announcements
# Format: (date, ticker, description, action_type)
# DMA entered into force March 7, 2024; most actions from mid-2024 onward
# Also including precursor DSA/antitrust fines and DMA preliminary findings
DMA_EVENTS = [
    # EU Commission DMA investigation opened against AAPL (interoperability)
    ("2024-06-24", "AAPL", "EC opens DMA Article 5(7) non-compliance proceeding vs Apple", "preliminary_finding"),
    # EC DMA non-compliance finding against META (pay or consent)
    ("2024-07-01", "META", "EC opens DMA non-compliance proceeding vs Meta (pay-or-consent)", "preliminary_finding"),
    # EC DMA non-compliance against GOOGL (search self-preferencing)
    ("2024-06-24", "GOOGL", "EC opens DMA Article 6(5) non-compliance proceeding vs Google", "preliminary_finding"),
    # Apple DMA fine: Interoperability feature blocking (Sep 2024)
    ("2024-09-19", "AAPL", "EC confirms DMA non-compliance; Apple faces potential fine", "finding_confirmed"),
    # Meta fine announced / confirmed
    ("2024-11-14", "META", "EC DMA fine Meta: pay-or-consent model non-compliant", "fine"),
    # GOOGL - search / shopping DMA non-compliance confirmed
    ("2025-02-19", "GOOGL", "EC DMA non-compliance finding vs Google Shopping", "finding_confirmed"),
    # Apple DMA fine formally issued
    ("2025-03-19", "AAPL", "EC issues formal DMA fine against Apple (streaming rules)", "fine"),
    # Meta large DMA fine Q1 2025
    ("2025-04-23", "META", "EC issues DMA fine Meta (Marketplace data practices)", "fine"),
]


def main():
    sid = "PL705_eu_dma_noncompliance_short_megacap"

    try:
        px = load_prices(["AAPL", "META", "GOOGL", "SPY"], start="2024-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    required = ["AAPL", "META", "GOOGL", "SPY"]
    missing = [t for t in required if t not in px.columns or px[t].dropna().empty]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    px = px.ffill().dropna(subset=required)
    ret = daily_returns(px)
    spy_r = ret["SPY"]

    # Build PnL: short named gatekeeper for 10 trading days per event
    hold = 10
    pnl = pd.Series(0.0, index=ret.index)
    events = []
    last_per_ticker = {}  # debounce per ticker (no two events same ticker within 20 days)

    for date_str, ticker, desc, action_type in DMA_EVENTS:
        event_dt = pd.Timestamp(date_str)
        candidates = ret.index[ret.index >= event_dt]
        if candidates.empty:
            continue
        entry_date = candidates[0]

        # Debounce per ticker: min 20 days between events for same ticker
        last = last_per_ticker.get(ticker, pd.Timestamp("2000-01-01"))
        if (entry_date - last).days < 20:
            continue

        if ticker not in ret.columns:
            continue
        ticker_r = ret[ticker]

        p = ret.index.get_loc(entry_date)
        ep = min(p + hold, len(ret))

        chunk = ticker_r.iloc[p:ep]
        # Short the gatekeeper
        chunk_pnl = -1.0 * chunk
        n = len(chunk_pnl)

        pnl.iloc[p:p + n] += chunk_pnl.values[:n]
        last_per_ticker[ticker] = entry_date

        gatekeeper_ret = float((1 + chunk).prod() - 1)
        short_ret = -gatekeeper_ret
        sp_chunk = spy_r.iloc[p:ep]
        sp_ret = float((1 + sp_chunk).prod() - 1) if len(sp_chunk) > 0 else None

        events.append({
            "entry_date": str(entry_date.date()),
            "ticker": ticker,
            "action_type": action_type,
            "description": desc,
            "gatekeeper_return": round(gatekeeper_ret, 4),
            "short_return": round(short_ret, 4),
            "spy_return": round(sp_ret, 4) if sp_ret is not None else None,
        })

    print(f"DMA enforcement events: {len(events)}")

    active = pnl[pnl != 0]
    print(f"Active PnL days: {len(active)}")
    if len(active) < 20:
        return mark_failed(sid, f"insufficient active days ({len(active)}) — DMA enforcement era began March 2024, too short for robust backtest")

    m = compute_metrics(active, benchmark=spy_r, name="EU DMA Non-Compliance Short Named Gatekeeper")

    rets = [e["short_return"] for e in events]
    save_result(sid, m, extra={
        "rule": "Short named gatekeeper (AAPL/META/GOOGL) for 10 trading days on EC DMA Article 8 non-compliance preliminary finding or formal fine announcement",
        "mechanism": "EU DMA enforcement creates direct fine liability, negative regulatory optics, forced product changes, and potential market access risk for EU operations; short-term sentiment headwind",
        "source": "yfinance AAPL, META, GOOGL; EC DMA enforcement announcements (curated from EC press releases 2024-2025)",
        "n_events": len(events),
        "avg_event_return": round(float(np.mean(rets)), 4) if rets else None,
        "event_win_rate": round(float(np.mean([r > 0 for r in rets])), 4) if rets else None,
        "events": events,
        "caveats": "DMA enforcement only began March 2024; very limited history (~8 events); markets may have already priced in regulatory risk; fine amounts small relative to megacap market caps",
    })
    print(f"Done: {len(events)} events, Sharpe={m.get('sharpe', 'N/A'):.2f}, CAGR={m.get('cagr', 0)*100:.1f}%")


if __name__ == "__main__":
    main()
