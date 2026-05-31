"""PL578_sp500_inclusion_announce_to_effective_drift - S&P 500 Inclusion Announce -> Effective Drift
Long T_ann+1 open to T_eff close on each addition.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL578_sp500_inclusion_announce_to_effective_drift"
    # Curated S&P 500 additions (announce, effective, ticker)
    events_raw = [
        ("2023-12-01", "2023-12-15", "UBER"),
        ("2024-09-06", "2024-09-20", "PLTR"),
        ("2024-09-06", "2024-09-20", "DELL"),
        ("2024-09-06", "2024-09-20", "ERIE"),
        ("2024-12-06", "2024-12-20", "APO"),
        ("2024-12-06", "2024-12-20", "WDAY"),
        ("2024-12-06", "2024-12-20", "LII"),
        ("2025-03-07", "2025-03-21", "DASH"),
        ("2025-03-07", "2025-03-21", "WSM"),
        ("2025-03-07", "2025-03-21", "TKO"),
        # Older
        ("2022-06-03", "2022-06-21", "VICI"),
        ("2022-12-02", "2022-12-19", "STLD"),
        ("2023-03-03", "2023-03-20", "BLDR"),
        ("2023-09-01", "2023-09-18", "LULU"),
    ]
    tickers = sorted({t for _, _, t in events_raw} | {"SPY"})
    try:
        px = load_prices(tickers, start="2022-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    ret = daily_returns(px)
    spy_r = ret["SPY"]

    legs = []
    event_results = []
    for ann, eff, tkr in events_raw:
        if tkr not in ret.columns:
            continue
        a, e = pd.Timestamp(ann), pd.Timestamp(eff)
        mask = (ret.index > a) & (ret.index <= e)
        win = ret[tkr][mask]
        if len(win) < 2:
            continue
        legs.append(win)
        cum = float((1 + win).prod() - 1)
        event_results.append({"ticker": tkr, "ann": ann, "eff": eff, "ret": round(cum, 4)})

    if not legs:
        return mark_failed(sid, "no valid events")
    df = pd.concat(legs, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")
    m = compute_metrics(pnl, benchmark=spy_r, name="SP500 Inclusion Drift")
    save_result(sid, m, extra={
        "rule": "Long addition ticker from T_ann+1 open to T_eff close.",
        "mechanism": "Index fund forced buying on effective date -> drift premium",
        "source": "S&P 500 add announcements + yfinance",
        "n_events": len(event_results),
        "events": event_results,
    })
    print(f"Done {sid}: events={len(event_results)} Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
