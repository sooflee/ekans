"""PL892 — DOL FUTA Credit-Reduction State Cluster -> Short Regional Banks vs Long KBE"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics, save_result, mark_failed, daily_returns)


def main():
    sid = "PL892_futa_credit_reduction_regional_bank_short"

    # DOL FUTA credit-reduction preliminary list release dates (August each year)
    # These mark years when DOL announced >=3 states would face FUTA credit reduction
    # 2010 list -> 2011 credit reduction year (first major post-GFC cluster)
    # 2011 list -> 2012 credit reduction (CA, IN, MI, MN, MO, NY, NC, OH, WI, + more)
    # 2012 list -> 2013 (same cluster, some additions)
    # 2013 list -> 2014 (last major year of the cycle)
    SIGNAL_DATES = [
        pd.Timestamp("2010-08-16"),  # 2010 DOL preliminary list
        pd.Timestamp("2011-08-15"),  # 2011 DOL preliminary list
        pd.Timestamp("2012-08-20"),  # 2012 DOL preliminary list
        pd.Timestamp("2013-08-19"),  # 2013 DOL preliminary list
    ]
    HOLD = 80  # ~16 weeks in trading days

    try:
        # KBE available from ~2005; KEY, MTB longer history
        px = load_prices(["KEY", "MTB", "KBE", "KRE", "SPY"], start="2009-01-01")
        spy_r = daily_returns(px[["SPY"]]).iloc[:, 0]
        key_r = daily_returns(px[["KEY"]]).iloc[:, 0]
        mtb_r = daily_returns(px[["MTB"]]).iloc[:, 0]
        kbe_r = daily_returns(px[["KBE"]]).iloc[:, 0]
    except Exception as e:
        try:
            px = load_prices(["KEY", "MTB", "KBE", "KRE", "SPY"], start="2009-01-01", cache=False)
            spy_r = daily_returns(px[["SPY"]]).iloc[:, 0]
            key_r = daily_returns(px[["KEY"]]).iloc[:, 0]
            mtb_r = daily_returns(px[["MTB"]]).iloc[:, 0]
            kbe_r = daily_returns(px[["KBE"]]).iloc[:, 0]
        except Exception as e2:
            return mark_failed(sid, f"price load: {e2}")

    # Short basket: KEY 50% + MTB 50% (COLB not used for pre-2023 events)
    bank_short_r = 0.5 * key_r + 0.5 * mtb_r

    events = []
    pnl_dates = []
    pnl_rets = []

    for sig_date in SIGNAL_DATES:
        # Entry: next trading day after DOL announcement
        future_idx = bank_short_r.index[bank_short_r.index >= sig_date]
        if len(future_idx) < HOLD:
            print(f"  Skipping {sig_date.date()}: insufficient forward data")
            continue
        entry_idx = future_idx[0]
        pos = bank_short_r.index.get_loc(entry_idx)
        end_pos = min(pos + HOLD, len(bank_short_r))

        short_window = bank_short_r.iloc[pos:end_pos]
        kbe_window = kbe_r.reindex(short_window.index).fillna(0)
        spy_window = spy_r.reindex(short_window.index).fillna(0)

        # Pair PnL: long KBE, short KEY/MTB basket (dollar-neutral)
        pair_r = kbe_window - short_window

        short_cum = float((1 + short_window).prod() - 1)
        kbe_cum = float((1 + kbe_window).prod() - 1)
        pair_cum = float((1 + pair_r).prod() - 1)

        events.append({
            "signal_date": str(sig_date.date()),
            "entry_date": str(entry_idx.date()),
            "hold_days": end_pos - pos,
            "bank_return_16w": round(short_cum, 4),
            "kbe_return_16w": round(kbe_cum, 4),
            "pair_return_16w": round(pair_cum, 4),
        })
        pnl_dates.extend(pair_r.index.tolist())
        pnl_rets.extend(pair_r.values.tolist())
        print(f"  {sig_date.date()}: short(KEY/MTB)={short_cum:.1%}, KBE={kbe_cum:.1%}, pair={pair_cum:.1%}")

    if not events:
        return mark_failed(sid, "no valid events after price join")

    combined_pnl = pd.Series(pnl_rets, index=pd.DatetimeIndex(pnl_dates)).sort_index()
    print(f"Events: {len(events)}, Combined active days: {len(combined_pnl)}")

    if len(combined_pnl) < 30:
        return mark_failed(sid, f"insufficient active days: {len(combined_pnl)}")

    m = compute_metrics(combined_pnl, benchmark=spy_r, name="FUTA Credit-Reduction → Short KEY/MTB vs Long KBE")
    win_rates = [1 if e["pair_return_16w"] > 0 else 0 for e in events]

    save_result(sid, m, extra={
        "rule": "Short KEY+MTB (50/50) vs Long KBE for ~16 weeks when DOL August preliminary FUTA credit-reduction list includes >=3 states",
        "mechanism": "FUTA credit-reduction raises effective payroll tax cost for businesses in affected states → increases commercial loan credit risk in regional banks with concentrated OH/NY/NJ exposure (KEY/MTB) vs broader bank benchmark (KBE)",
        "source": "DOL FUTA credit-reduction annual preliminary list (known events 2010-2013); yfinance KEY/MTB/KBE/SPY",
        "n_events": len(events),
        "event_win_rate": round(float(np.mean(win_rates)), 4),
        "events": events,
        "caveats": "Only 4 events (2010-2013 cycle); cycle ended when states replenished UI trust funds; no recurrence since 2014; COLB not used for these historical events",
    })
    print(f"Done: {len(events)} events, Sharpe={m.get('sharpe', 'N/A'):.3f}, CAGR={m.get('cagr', 0)*100:.1f}%")


if __name__ == "__main__":
    main()
