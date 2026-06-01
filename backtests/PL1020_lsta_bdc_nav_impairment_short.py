"""PL1020_lsta_bdc_nav_impairment_short — NFCI Credit Surge → BDC NAV Impairment Lag → Short ARCC/GBDC"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL1020_lsta_bdc_nav_impairment_short"

    try:
        fred = load_fred(["NFCICREDIT"], start="2004-01-01")
        nfci = fred["NFCICREDIT"].dropna()
    except Exception as e:
        return mark_failed(sid, f"FRED data load: {e}")

    try:
        px = load_prices(["ARCC", "GBDC", "SPY"], start="2004-01-01")
    except Exception as e:
        return mark_failed(sid, f"price data load: {e}")

    for col in ["ARCC", "SPY"]:
        if col not in px.columns:
            return mark_failed(sid, f"{col} not in price data")

    ret = daily_returns(px)
    spy_r = ret["SPY"]
    arcc_r = ret["ARCC"]
    gbdc_r = ret.get("GBDC", pd.Series(0.0, index=ret.index))

    # 4-week NFCICREDIT surge: rolling 20-trading-day change
    nfci_weekly = nfci.resample("W-FRI").last().dropna()
    nfci_4wk_change = nfci_weekly - nfci_weekly.shift(4)

    # Entry signal: 4-week change > 0.30 AND still rising (not reversing)
    entry_signal = nfci_4wk_change[nfci_4wk_change > 0.30]

    print(f"NFCICREDIT stats: min={nfci.min():.3f}, max={nfci.max():.3f}")
    print(f"4-week surge >0.30 weeks: {len(entry_signal)}")

    hold_calendar = 40  # days
    stop_loss = 0.15    # 15% combined position loss
    revert_thresh = 0.05  # NFCI 4wk change below this = exit
    cooldown = 30       # days between entries

    pnl = pd.Series(0.0, index=ret.index)
    event_records = []
    last_exit = None
    in_trade = False

    for dt, surge in nfci_4wk_change.items():
        if surge <= 0.30:
            continue
        if pd.isna(surge):
            continue

        # Entry: next equity open after NFCI publication (weekly, Friday)
        # Use Monday after the signal week
        entry_date = dt + pd.Timedelta(days=3)  # Friday + 3 = Monday

        if last_exit is not None and (entry_date - last_exit).days < cooldown:
            continue

        # Find actual trading day
        future_idx = arcc_r.index[arcc_r.index >= entry_date]
        if len(future_idx) < 3:
            continue

        entry_idx = future_idx[0]
        if last_exit is not None and (entry_idx - last_exit).days < cooldown:
            continue

        # Build trade
        exit_date = entry_idx + pd.Timedelta(days=hold_calendar)
        trade_dates = arcc_r.index[(arcc_r.index > entry_idx) & (arcc_r.index <= exit_date)]
        if len(trade_dates) == 0:
            continue

        # Use ARCC alone for pre-GBDC period (before 2010-04); combined after
        gbdc_available = "GBDC" in ret.columns

        exit_reason = "time"
        active_dates = []
        cum_arcc = 1.0
        cum_gbdc = 1.0

        for td in trade_dates:
            active_dates.append(td)
            a_r = arcc_r.get(td, 0.0)
            g_r = gbdc_r.get(td, 0.0) if gbdc_available and not pd.isna(gbdc_r.get(td, np.nan)) else 0.0
            cum_arcc *= (1 + a_r)
            cum_gbdc *= (1 + g_r)

            # Use equal-weight if GBDC available and has data
            if gbdc_available and gbdc_r.get(td, 0) != 0:
                combined_loss = 0.5 * (cum_arcc - 1) + 0.5 * (cum_gbdc - 1)
            else:
                combined_loss = cum_arcc - 1

            # Stop: short book gains 15% for us = we lose 15%
            if combined_loss > stop_loss:
                exit_reason = "stop_loss"
                break

            # Check NFCI reversion
            recent_nfci4 = nfci_4wk_change[nfci_4wk_change.index <= td]
            if len(recent_nfci4) > 0:
                latest_surge = recent_nfci4.iloc[-1]
                if latest_surge < revert_thresh:
                    exit_reason = "nfci_reversion"
                    break

        if not active_dates:
            continue

        last_exit = active_dates[-1]

        # Compute trade PnL (short both)
        arcc_cum = float((1 + arcc_r.loc[active_dates]).prod() - 1)
        if gbdc_available:
            gbdc_vals = gbdc_r.reindex(active_dates).fillna(0)
            gbdc_cum = float((1 + gbdc_vals).prod() - 1)
            trade_cum = -(0.5 * arcc_cum + 0.5 * gbdc_cum)
        else:
            gbdc_cum = 0.0
            trade_cum = -arcc_cum

        spy_cum = float((1 + spy_r.reindex(active_dates).fillna(0)).prod() - 1)

        event_records.append({
            "entry": str(entry_idx.date()),
            "exit": str(active_dates[-1].date()),
            "n_days": len(active_dates),
            "nfci_surge": round(float(surge), 3),
            "arcc_cum": round(arcc_cum, 4),
            "gbdc_cum": round(gbdc_cum, 4),
            "trade_pnl": round(trade_cum, 4),
            "spy_cum": round(spy_cum, 4),
            "exit_reason": exit_reason,
        })

        for td in active_dates:
            if td in pnl.index:
                a_r = arcc_r.get(td, 0.0)
                g_r = gbdc_r.get(td, 0.0) if gbdc_available and not pd.isna(gbdc_r.get(td, np.nan)) else 0.0
                if gbdc_available and g_r != 0:
                    pnl[td] -= (0.5 * a_r + 0.5 * g_r)
                else:
                    pnl[td] -= a_r

    print(f"\nTrade records ({len(event_records)}):")
    for e in event_records:
        print(f"  {e['entry']} -> {e['exit']} ({e['n_days']}d, {e['exit_reason']}, surge={e['nfci_surge']:.3f}): "
              f"ARCC={e['arcc_cum']:.2%}, GBDC={e['gbdc_cum']:.2%}, pnl={e['trade_pnl']:.2%}, SPY={e['spy_cum']:.2%}")

    active_pnl = pnl[pnl != 0]
    print(f"Active pnl days: {len(active_pnl)}")

    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active pnl days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="NFCI Credit Surge Short ARCC/GBDC")

    save_result(sid, m, extra={
        "rule": "Short equal-weight ARCC+GBDC when FRED NFCICREDIT 4-week change >0.30 (credit tightening). Exit at NFCI reversion <0.05, 40 days, or 15% stop.",
        "mechanism": "BDC loan portfolios face NAV impairment with a lag when credit markets tighten; NFCICREDIT captures leveraged loan stress before it flows through quarterly BDC NAV marks.",
        "source": "FRED NFCICREDIT; ARCC, GBDC via yfinance.",
        "n_events": len(event_records),
        "events": event_records,
    })


if __name__ == "__main__":
    main()
