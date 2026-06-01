"""PL733_basel_endgame_short_capmkt_long_deposit
Basel III Endgame -> Short GS/MS Long USB/PNC (Counter)

On Fed/OCC/FDIC publication of Basel III Endgame proposals or final rules,
short equal-weight GS+MS (capital markets banks, most affected) and long
equal-weight USB+PNC (regional deposit banks, less affected) for 45 days.

Known Basel III Endgame milestones (hand-coded from Federal Register):
  - 2023-07-27: Fed/OCC/FDIC publish Basel III Endgame NPR (proposed rule)
  - 2024-09-09: US agencies announce intention to re-propose revised Endgame
  - 2025-01-xx: Further discussions/revisions under new administration
  - 2023-10-27: Comment period deadline (regulatory uncertainty peak)

Note: Only 2 clear publication events. Strategy bt_feasibility=3.
We include the comment deadline and the re-proposal date as additional events.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL733_basel_endgame_short_capmkt_long_deposit"
    tickers = ["GS", "MS", "USB", "PNC", "SPY"]

    try:
        px = load_prices(tickers, start="2022-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    # Hand-coded Basel III Endgame milestones
    # Source: Federal Register, Federal Reserve press releases
    event_signal_dates = [
        "2023-07-27",  # Fed/OCC/FDIC publish Basel III Endgame NPR
        "2023-10-27",  # Comment period deadline (uncertainty peak)
        "2024-09-09",  # US agencies announce intention to re-propose
    ]

    hold_days = 45
    events = []
    pnl_parts = []
    positions_parts = []

    idx = ret.index

    for sig_str in event_signal_dates:
        sig_date = pd.Timestamp(sig_str)
        future_idx = idx[idx > sig_date]
        if len(future_idx) < hold_days:
            continue
        entry_date = future_idx[0]
        entry_loc = idx.get_loc(entry_date)
        exit_loc = min(entry_loc + hold_days, len(idx))

        window = slice(entry_loc, exit_loc)

        # Short GS + MS (equal weight), Long USB + PNC (equal weight)
        # Spread pnl = (USB+PNC)/2 - (GS+MS)/2
        gs_r = ret["GS"].iloc[window].fillna(0)
        ms_r = ret["MS"].iloc[window].fillna(0)
        usb_r = ret["USB"].iloc[window].fillna(0)
        pnc_r = ret["PNC"].iloc[window].fillna(0)
        spy_w = spy_r.reindex(gs_r.index).fillna(0)

        if gs_r.isna().all():
            continue

        # Long deposit banks, short capital markets banks
        short_leg = -(gs_r + ms_r) / 2
        long_leg = (usb_r + pnc_r) / 2
        pnl_window = (short_leg + long_leg) / 2  # half-sized each side
        pos_window = pd.Series(0.5, index=gs_r.index)  # net half position

        gs_cum = float((1 + gs_r).prod() - 1)
        ms_cum = float((1 + ms_r).prod() - 1)
        usb_cum = float((1 + usb_r).prod() - 1)
        pnc_cum = float((1 + pnc_r).prod() - 1)
        spy_cum = float((1 + spy_w).prod() - 1)
        spread = (usb_cum + pnc_cum) / 2 - (gs_cum + ms_cum) / 2

        pnl_parts.append(pnl_window)
        positions_parts.append(pos_window)

        events.append({
            "signal_date": sig_str,
            "entry_date": str(entry_date.date()),
            "gs_45d": round(gs_cum, 4),
            "ms_45d": round(ms_cum, 4),
            "usb_45d": round(usb_cum, 4),
            "pnc_45d": round(pnc_cum, 4),
            "spy_45d": round(spy_cum, 4),
            "spread_pnl": round(spread, 4),
        })

    if len(events) < 3:
        return mark_failed(
            sid,
            f"insufficient events: {len(events)} (Basel III Endgame milestones; only ~3 known)",
        )

    all_pnl = pd.concat(pnl_parts)
    all_pos = pd.concat(positions_parts)

    m = compute_metrics(
        all_pnl,
        benchmark=spy_r.reindex(all_pnl.index).fillna(0),
        name="Basel III Endgame -> Short GSIB / Long Regional",
        positions=all_pos,
        cost_bps=15,
    )

    n_events = len(events)
    spreads = [e["spread_pnl"] for e in events]
    win_rate = float(np.mean([r > 0 for r in spreads]))
    avg_spread = float(np.mean(spreads))

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "On Fed/OCC/FDIC Basel III Endgame publication/milestone, "
                "short equal-weight GS+MS and long equal-weight USB+PNC "
                "for 45 trading days."
            ),
            "mechanism": (
                "Basel III Endgame imposes disproportionate capital requirements on "
                "capital-markets-heavy GSIBs (GS, MS) vs deposit-funded regionals "
                "(USB, PNC). Publication triggers sell-side repricing of GS/MS "
                "ROE targets and re-rating of regionals as relatively advantaged."
            ),
            "source": (
                "yfinance GS, MS, USB, PNC, SPY; event dates from Federal Register "
                "and Federal Reserve press releases 2023-2024"
            ),
            "tickers": tickers,
            "n_events": n_events,
            "event_win_rate": round(win_rate, 4),
            "avg_spread_pnl": round(avg_spread, 4),
            "events": events,
            "caveats": (
                "Very small sample (3 events). Basel III Endgame was ultimately "
                "substantially watered down under the Trump administration (2025), "
                "which would have made long GS/MS more profitable ex-post. "
                "bt_feasibility=3. Regulatory outcome is highly path-dependent."
            ),
        },
        pnl=all_pnl,
    )

    print(f"Done: {sid}")
    print(f"  n_events: {n_events}, win_rate: {win_rate:.0%}, avg_spread: {avg_spread:.4f}")
    sharpe = m.get("sharpe", 0)
    cagr = m.get("cagr", 0)
    print(
        f"  Sharpe: {sharpe:.2f}  CAGR: {cagr*100:.2f}%  "
        f"MaxDD: {m.get('max_dd', 0)*100:.2f}%  t-stat: {m.get('t_stat', 0):.2f}"
    )
    for e in events:
        flag = "+" if e["spread_pnl"] > 0 else "-"
        print(f"  {flag} {e['signal_date']}: spread={e['spread_pnl']*100:+.1f}%, SPY={e['spy_45d']*100:+.1f}%")


if __name__ == "__main__":
    main()
