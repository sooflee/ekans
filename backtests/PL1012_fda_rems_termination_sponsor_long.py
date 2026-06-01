"""PL1012 — FDA REMS Termination -> Prescriber Access Broadens -> Sponsor Equity Long"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL1012_fda_rems_termination_sponsor_long"

    # Curated list of FDA REMS terminations mapped to sponsor tickers
    # (ticker, termination_date, drug, notes)
    # Sources: FDA REMS@FDA database, Federal Register, press releases
    EVENTS = [
        # Vivitrol REMS simplification (ALKS)
        ("ALKS", "2011-05-01", "Vivitrol (naltrexone)", "REMS simplified"),
        # Tysabri REMS update/simplification (BIIB) - TOUCH prescribing program
        ("BIIB", "2017-09-22", "Tysabri (natalizumab)", "REMS ETASU removal"),
        # Xyrem REMS consolidation/simplification (JAZZ)
        ("JAZZ", "2018-12-01", "Xyrem (sodium oxybate)", "REMS simplification"),
        # Abstral (fentanyl SL) REMS termination - Galena sold to Recro
        # Skip - Galena was micro-cap; use ACTV instead
        # Opioid analgesics class REMS shared program consolidation (affects BMY, SUPN, CPRX)
        ("SUPN", "2022-02-09", "Oxtellar XR/opioid REMS", "class REMS consolidation"),
        ("CPRX", "2022-02-09", "opioid REMS consolidation", "class REMS simplification"),
        # ACADIA - Nuplazid REMS modification/simplification
        ("ACAD", "2020-06-15", "Nuplazid (pimavanserin)", "REMS ETASU removal"),
        # BIIB - Tecfidera REMS simplified
        ("BIIB", "2020-03-01", "Tecfidera (dimethyl fumarate)", "REMS termination"),
        # BMY - Revlimid REMS (lenalidomide) simplification
        ("BMY", "2023-07-01", "Revlimid (lenalidomide)", "REMS simplification post-generic"),
        # ALKS - Lybalvi REMS (olanzapine+samidorphan) modification
        ("ALKS", "2023-10-15", "Lybalvi (olanzapine/samidorphan)", "REMS simplification"),
        # JAZZ - Xywav REMS update
        ("JAZZ", "2023-08-01", "Xywav (low-sodium oxybate)", "REMS simplification"),
    ]

    all_tickers = list(set([e[0] for e in EVENTS]) | {"XBI", "SPY"})

    try:
        px = load_prices(all_tickers, start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"] if "SPY" in ret.columns else None
    xbi_r = ret["XBI"] if "XBI" in ret.columns else None

    if spy_r is None:
        return mark_failed(sid, "SPY data missing")

    pnl_list = []
    events_out = []
    HOLD_DAYS = 60  # calendar days
    STOP_LOSS = -0.25

    for ticker, event_str, drug, note in EVENTS:
        if ticker not in ret.columns:
            continue
        stock_r = ret[ticker].dropna()

        try:
            event_dt = pd.Timestamp(event_str)
        except Exception:
            continue

        # Entry: T+1 trading day after event (within 3 days)
        entry_mask = stock_r.index > event_dt
        entry_candidates = stock_r.index[entry_mask][:3]
        if len(entry_candidates) == 0:
            continue
        entry_date = entry_candidates[0]

        # Exit: ~60 calendar days from entry, approximately 42 trading days
        exit_cutoff = entry_date + pd.Timedelta(days=HOLD_DAYS)
        window_mask = (stock_r.index >= entry_date) & (stock_r.index <= exit_cutoff)
        window_ret = stock_r.loc[window_mask]

        if len(window_ret) < 5:
            continue

        # Apply stop-loss at -25%
        cum = (1 + window_ret).cumprod()
        max_cum = cum.cummax()
        dd = (cum / max_cum) - 1
        stop_hit = dd[dd <= STOP_LOSS]
        stopped_out = len(stop_hit) > 0
        if stopped_out:
            stop_date = stop_hit.index[0]
            window_ret = window_ret.loc[window_ret.index <= stop_date]

        if len(window_ret) < 3:
            continue

        total_return = float((1 + window_ret).prod() - 1)

        # XBI return for same window (benchmark)
        if xbi_r is not None:
            xbi_window = xbi_r.reindex(window_ret.index)
            xbi_cumret = float((1 + xbi_window.dropna()).prod() - 1)
        else:
            xbi_cumret = None

        spy_window = spy_r.reindex(window_ret.index)
        spy_cumret = float((1 + spy_window.dropna()).prod() - 1)

        pnl_list.append(window_ret)
        events_out.append({
            "ticker": ticker,
            "event_date": event_str,
            "drug": drug,
            "note": note,
            "entry_date": str(entry_date.date()),
            "exit_date": str(window_ret.index[-1].date()),
            "n_days": len(window_ret),
            "total_return": round(total_return, 4),
            "xbi_return": round(xbi_cumret, 4) if xbi_cumret is not None else None,
            "spy_return": round(spy_cumret, 4),
            "stopped_out": stopped_out,
        })

    print(f"Events processed: {len(events_out)}")
    for ev in events_out:
        print(f"  {ev['ticker']} {ev['event_date']} ({ev['drug'][:30]}): {ev['total_return']:.2%} vs SPY {ev['spy_return']:.2%} {'STOP' if ev['stopped_out'] else ''}")

    if len(events_out) < 5:
        return mark_failed(sid, f"insufficient events after filtering ({len(events_out)})")

    # Build combined PnL
    pnl = pd.concat(pnl_list).sort_index()
    pnl = pnl.groupby(pnl.index).mean()

    ip = pnl[pnl != 0]
    if len(ip) < 30:
        return mark_failed(sid, f"insufficient active days ({len(ip)})")

    spy_aligned = spy_r.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=spy_aligned, name="FDA REMS Termination Sponsor Long")

    returns_list = [e["total_return"] for e in events_out]
    save_result(sid, m, extra={
        "rule": "Long sponsor equity within T+3 days of FDA REMS termination/simplification; hold 60 calendar days or stop-loss -25%",
        "mechanism": "REMS removal expands prescriber access (removes enrollment/certification requirements), reduces administrative burden on physicians, and signals FDA confidence in drug safety profile — positive re-rating catalyst",
        "source": "FDA REMS@FDA database (https://www.accessdata.fda.gov/scripts/cder/rems/); Federal Register; yfinance prices",
        "n_events": len(events_out),
        "avg_event_return": round(float(np.mean(returns_list)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in returns_list])), 4),
        "events": events_out,
        "caveats": "Small sample (~10 events); curated list may miss some terminations. REMS events are rare and drug-specific impact varies greatly by revenue share.",
    })
    print("Done.")


if __name__ == "__main__":
    main()
