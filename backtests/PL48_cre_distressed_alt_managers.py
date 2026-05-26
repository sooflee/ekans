"""PL48_cre_distressed_alt_managers — CRE Transaction Volume Trough -> Long Alt Managers
Entry 12mo after CRE volume trough. Long BX+ARES+KKR equal-weight 12mo.
Events: entry 2010-06 (GFC trough), entry 2024-03 (rate-hike trough).
ARES IPO 2014, KKR IPO 2010 — use available tickers per event.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL48_cre_distressed_alt_managers"
    try:
        px = load_prices(["BX", "ARES", "KKR", "SPY"], start="2008-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    # CRE trough events — entry is 12 months AFTER the trough
    # Event 1: GFC trough Jun 2009 -> entry Jun 2010.  BX IPO'd Jun 2007, KKR IPO'd Jul 2010, ARES IPO'd May 2014
    #   So for 2010-06: BX only (KKR just IPO'd, use BX as sole ticker)
    # Event 2: Rate-hike trough Mar 2023 -> entry Mar 2024.  All three available
    trough_events = [
        ("2010-06-01", ["BX"], "GFC CRE trough Jun 2009 +12mo; BX only (KKR just IPO'd, ARES not yet)"),
        ("2024-03-01", ["BX", "ARES", "KKR"], "Rate-hike CRE trough Mar 2023 +12mo; all three available"),
    ]

    events = []
    pnl_parts = []
    hold_days = 252  # 12 months

    for date_str, tickers, desc in trough_events:
        entry_date = pd.Timestamp(date_str)
        entry_mask = ret.index >= entry_date
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = ret.index[entry_mask][0]
        entry_loc = ret.index.get_loc(entry_idx)
        exit_loc = min(entry_loc + hold_days, len(ret.index) - 1)

        window = slice(entry_loc, exit_loc)

        # Use only tickers available in this period
        available = []
        for t in tickers:
            if t in ret.columns and not ret[t].iloc[window].isna().all():
                available.append(t)
        if not available:
            continue

        basket_r = ret[available].iloc[window].mean(axis=1)
        spy_window = spy_r.iloc[window]
        pnl_parts.append(basket_r)

        basket_cum = float((1 + basket_r).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)

        per_ticker = {}
        for t in available:
            t_cum = float((1 + ret[t].iloc[window]).prod() - 1)
            per_ticker[t] = round(t_cum, 4)

        events.append({
            "entry_date": date_str,
            "description": desc,
            "tickers_used": available,
            "basket_12m_return": round(basket_cum, 4),
            "spy_12m_return": round(spy_cum, 4),
            "excess": round(basket_cum - spy_cum, 4),
            "per_ticker": per_ticker,
        })

    if not events:
        return mark_failed(sid, "No CRE trough events with alt manager data")

    all_pnl = pd.concat(pnl_parts)
    m = compute_metrics(all_pnl, benchmark=spy_r.reindex(all_pnl.index).dropna(),
                        name="CRE Trough -> Long Alt Managers")

    avg_basket = np.mean([e["basket_12m_return"] for e in events])
    avg_excess = np.mean([e["excess"] for e in events])
    win_count = sum(1 for e in events if e["basket_12m_return"] > 0)

    save_result(sid, m, extra={
        "rule": "Long BX+ARES+KKR equal-weight 12mo, entry 12mo after CRE volume trough",
        "mechanism": "CRE distress cycle trough -> alt managers deploy dry powder at discounted prices -> AUM/fee re-rating -> stock re-rates",
        "source": "MSCI RCA CRE volume (hand-coded trough dates); yfinance BX, ARES, KKR",
        "n_events": len(events),
        "avg_basket_return": round(avg_basket, 4),
        "avg_excess_vs_spy": round(avg_excess, 4),
        "win_rate": f"{win_count}/{len(events)}",
        "events": events,
    })
    sharpe = m.get('sharpe', 0)
    cagr = m.get('cagr', 0)
    print(f"Done: {len(events)} events, Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")
    print(f"  Avg basket: {avg_basket*100:.1f}%  Avg excess: {avg_excess*100:.1f}%  Win: {win_count}/{len(events)}")
    for e in events:
        flag = "+" if e["basket_12m_return"] > 0 else "-"
        print(f"  {flag} {e['entry_date']} ({', '.join(e['tickers_used'])}): basket {e['basket_12m_return']*100:+.1f}%, spy {e['spy_12m_return']*100:+.1f}%, excess {e['excess']*100:+.1f}%")
    if sharpe > 0.5 and cagr > 0.10:
        print(f"*** WINNER FOUND: {sid} -- Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")


if __name__ == "__main__":
    main()
