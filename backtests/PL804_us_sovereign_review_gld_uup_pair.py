"""PL804_us_sovereign_review_gld_uup_pair — US Sovereign Rating Review/Downgrade -> Long GLD Short UUP Pair"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL804_us_sovereign_review_gld_uup_pair"

    # Known US sovereign rating events:
    # 2011-07-15: S&P places US on CreditWatch Negative (pre-downgrade)
    # 2011-08-05: S&P downgrades US from AAA to AA+
    # 2013-10-01: US government shutdown (near-default context, debt ceiling)
    # 2023-07-19: Fitch places US on Negative Watch (pre-downgrade)
    # 2023-08-01: Fitch downgrades US from AAA to AA+
    # 2023-11-10: Moody's changes US outlook to Negative from Stable
    # Also include 2025-05-16: Moody's downgrades US from Aaa to Aa1
    events = [
        ("2011-07-15", "S&P CreditWatch Negative placement"),
        ("2011-08-05", "S&P US downgrade AAA -> AA+"),
        ("2013-10-01", "US gov shutdown / near-default"),
        ("2023-07-19", "Fitch Negative Watch placement"),
        ("2023-08-01", "Fitch US downgrade AAA -> AA+"),
        ("2023-11-10", "Moody's US outlook to Negative"),
        ("2025-05-16", "Moody's US downgrade Aaa -> Aa1"),
    ]

    tickers = ["GLD", "UUP", "SPY"]
    try:
        px = load_prices(tickers, start="2011-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if "GLD" not in px.columns or "UUP" not in px.columns:
        return mark_failed(sid, "GLD or UUP not available")

    ret = daily_returns(px)
    gld_r = ret["GLD"]
    uup_r = ret["UUP"]
    spy_r = ret["SPY"]

    hold = 30  # 6 weeks = ~30 trading days
    event_results = []
    pnl_parts = []
    seen_windows = []  # track windows to avoid overlap between close events

    for event_date_str, desc in events:
        event_date = pd.Timestamp(event_date_str)

        # Skip if too close to a previous event (within 30 days)
        too_close = any(abs((event_date - prev).days) < 25 for prev in seen_windows)
        if too_close:
            continue

        # Entry on next trading day after event
        future_mask = gld_r.index > event_date
        if future_mask.sum() < hold // 2:
            continue

        entry_idx = gld_r.index[future_mask][0]
        pos = gld_r.index.get_loc(entry_idx)
        end_pos = min(pos + hold, len(gld_r))

        gld_window = gld_r.iloc[pos:end_pos]
        uup_window = uup_r.reindex(gld_window.index).fillna(0)
        spy_window = spy_r.reindex(gld_window.index).fillna(0)

        # Pair: long GLD, short UUP (dollar-neutral)
        trade_pnl = gld_window - uup_window

        pnl_parts.append(trade_pnl)
        seen_windows.append(event_date)

        gld_car = float((1 + gld_window).prod() - 1)
        uup_car = float((1 + uup_window).prod() - 1)
        pair_car = float((1 + trade_pnl).prod() - 1)
        spy_car = float((1 + spy_window).prod() - 1)

        event_results.append({
            "event_date": event_date_str,
            "entry_date": str(entry_idx.date()),
            "description": desc,
            "hold_days": len(gld_window),
            "gld_return": round(gld_car, 4),
            "uup_return": round(uup_car, 4),
            "pair_return": round(pair_car, 4),
            "spy_return": round(spy_car, 4),
        })

    if not event_results:
        return mark_failed(sid, "no valid events found")

    if len(pnl_parts) < 3:
        return mark_failed(sid, f"insufficient events: only {len(pnl_parts)} valid trades")

    # Combine all trade PnL
    all_pnl = pd.concat(pnl_parts).sort_index()
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]

    if len(all_pnl) < 30:
        return mark_failed(sid, f"insufficient trading days: {len(all_pnl)}")

    m = compute_metrics(all_pnl, benchmark=spy_r, name="US Sovereign Review -> Long GLD Short UUP")

    save_result(sid, m, extra={
        "rule": "Long GLD / Short UUP on US sovereign rating review/downgrade (within 90d of X-date); hold 6 weeks or until watch removed",
        "mechanism": "US sovereign credit stress triggers safe-haven gold bid and USD weakness as reserve-currency confidence declines; pair is dollar-neutral isolating gold premium over USD",
        "source": "S&P/Fitch/Moody's rating event dates (manual); yfinance prices",
        "n_events": len(event_results),
        "avg_pair_return": round(float(np.mean([e["pair_return"] for e in event_results])), 4),
        "win_rate": round(float(np.mean([e["pair_return"] > 0 for e in event_results])), 4),
        "events": event_results,
        "caveat": "Only 4-5 independent sovereign stress events since 2011; 2011 S&P downgrade is the highest-quality analog; 2023 events partially anticipated by market",
    })

    print(f"Done: {len(event_results)} events")
    for e in event_results:
        print(f"  {e['event_date']} ({e['description'][:45]}): GLD={e['gld_return']:.3f}, UUP={e['uup_return']:.3f}, pair={e['pair_return']:.3f}")


if __name__ == "__main__":
    main()
