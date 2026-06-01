"""PL1043_doe_grip_awards_long_pwr_myrg
DOE GRIP Grant Award Cluster -> Long PWR / MYRG Transmission Contractors

Event-study: when DOE announces a GRIP award cluster >= $500M (or ARRA smart grid grants),
go long equal-weight PWR + MYRG for 63 trading days starting next trading day after announcement.
Known events: 2009-10-27 (ARRA), 2023-10-18 (GRIP Round 1), 2024-03-07 (GRIP Round 2), 2024-09-19.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL1043_doe_grip_awards_long_pwr_myrg"
    tickers = ["PWR", "MYRG", "XLI", "SPY"]

    try:
        px = load_prices(tickers, start="2008-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if px is None or len(px) == 0:
        try:
            px = load_prices(tickers, start="2008-01-01", cache=False)
        except Exception as e:
            return mark_failed(sid, f"data load retry: {e}")

    px = px.sort_index().ffill(limit=3)

    missing = [t for t in ["PWR", "MYRG", "SPY"] if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()
    idx = ret.index

    # Known DOE GRIP / ARRA smart grid major award cluster dates
    # Signal date = announcement date; entry = next trading day
    signal_dates_raw = [
        "2009-10-27",   # ARRA Smart Grid Investment Grant $3.4B
        "2023-10-18",   # DOE GRIP Round 1 $3.46B
        "2024-03-07",   # DOE GRIP Round 2 ~$1.5B
        "2024-09-19",   # DOE GRIP additional ~$1.7B
    ]

    hold_days = 63
    stop_loss = -0.10       # basket cumulative -10%
    take_profit = 0.20      # basket cumulative +20%

    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)
    events = []

    for sd_str in signal_dates_raw:
        sd = pd.Timestamp(sd_str)
        # Find first trading day strictly after signal date
        future_days = idx[idx > sd]
        if len(future_days) == 0:
            continue
        entry_date = future_days[0]
        entry_pos = idx.get_loc(entry_date)

        # Check MYRG available at entry
        if pd.isna(px["MYRG"].get(entry_date, np.nan)) or pd.isna(px["PWR"].get(entry_date, np.nan)):
            events.append({"signal_date": sd_str, "skipped": "missing_price_at_entry"})
            continue

        entry_pwr = px["PWR"].loc[entry_date]
        entry_myrg = px["MYRG"].loc[entry_date]

        end_pos = min(entry_pos + hold_days, len(idx))
        exit_reason = "max_hold"
        exit_pos = end_pos - 1
        cum_basket = 0.0

        for j in range(entry_pos, end_pos):
            day = idx[j]
            r_pwr = ret["PWR"].get(day, 0.0) if "PWR" in ret.columns else 0.0
            r_myrg = ret["MYRG"].get(day, 0.0) if "MYRG" in ret.columns else 0.0
            if pd.isna(r_pwr):
                r_pwr = 0.0
            if pd.isna(r_myrg):
                r_myrg = 0.0

            # Equal-weight long: 0.5 PWR + 0.5 MYRG
            day_pnl = 0.5 * r_pwr + 0.5 * r_myrg
            pnl.iloc[j] += day_pnl
            positions.iloc[j] = 1.0
            cum_basket += day_pnl

            # Stop loss or take profit check
            if cum_basket <= stop_loss:
                exit_reason = "stop_loss"
                exit_pos = j
                break
            if cum_basket >= take_profit:
                exit_reason = "take_profit"
                exit_pos = j
                break

        exit_date = idx[exit_pos]
        event_ret = float((1 + pnl.loc[entry_date:exit_date]).prod() - 1)

        events.append({
            "signal_date": sd_str,
            "entry_date": str(entry_date.date()),
            "exit_date": str(exit_date.date()),
            "exit_reason": exit_reason,
            "event_return": round(event_ret, 4),
        })

    in_pos_days = (pnl != 0).sum()
    if len(pnl.dropna()) < 30:
        return mark_failed(sid, f"insufficient data: {len(pnl.dropna())} days")

    pnl_clean = pnl.dropna()

    m = compute_metrics(
        pnl_clean,
        benchmark=spy_r,
        name="DOE GRIP Awards Long PWR/MYRG",
    )

    n_events_clean = len([e for e in events if not e.get("skipped")])
    ev_returns = [e["event_return"] for e in events if not e.get("skipped")]
    win_rate = float(np.mean([r > 0 for r in ev_returns])) if ev_returns else None
    avg_event = float(np.mean(ev_returns)) if ev_returns else None

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "When DOE announces GRIP/ARRA smart grid award cluster >= $500M, "
                "go long equal-weight PWR + MYRG for 63 trading days from next trading day. "
                "Stop loss: basket -10% from entry. Take profit: basket +20%."
            ),
            "mechanism": (
                "DOE GRIP grants create binding utility project obligations with "
                "18-24 month commencement milestones. PWR (Quanta Services) and MYRG "
                "(MYR Group) are the largest US T&D EPC contractors. Grant awards create "
                "clear backlog inflection 2-4 quarters ahead, visible in 10-Q filings. "
                "Market typically prices this in over 3-12 months post announcement."
            ),
            "source": (
                "DOE GRIP program (https://www.energy.gov/ooe/grip-program); "
                "ARRA SGIG 2009; yfinance adjusted closes PWR/MYRG/XLI/SPY"
            ),
            "tickers": ["PWR", "MYRG"],
            "known_signal_dates": signal_dates_raw,
            "n_events": n_events_clean,
            "event_win_rate": round(win_rate, 4) if win_rate is not None else None,
            "avg_event_return": round(avg_event, 4) if avg_event is not None else None,
            "events": events,
            "caveats": (
                "Very small sample N=4 events (3 GRIP + 1 ARRA). ARRA 2009 and GRIP 2023-24 "
                "are different policy regimes. PWR and MYRG are highly correlated. "
                "Strategy relies on public DOE announcements which may leak earlier. "
                "Hold period overlaps may occur if multiple awards announced close together."
            ),
        },
        pnl=pnl_clean,
    )

    print(f"Done: {sid}")
    print(f"  n_events: {n_events_clean}, win_rate: {win_rate}, avg_event_return: {avg_event}")
    print(
        f"  Sharpe: {m.get('sharpe', 0):.2f}  CAGR: {m.get('cagr', 0)*100:.2f}%  "
        f"MaxDD: {m.get('max_dd', 0)*100:.2f}%  t-stat: {m.get('t_stat', 0):.2f}"
    )
    if "oos_sharpe" in m:
        print(f"  OOS Sharpe: {m['oos_sharpe']:.2f}")


if __name__ == "__main__":
    main()
