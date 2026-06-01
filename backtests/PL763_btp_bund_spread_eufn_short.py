"""PL763_btp_bund_spread_eufn_short — BTP-Bund Spread Stress (>250bps) -> Short EUFN"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL763_btp_bund_spread_eufn_short"

    # Load FRED monthly yields: Italy 10y and Germany 10y
    try:
        italy = load_fred("IRLTLT01ITM156N", start="2005-01-01")
        germany = load_fred("IRLTLT01DEM156N", start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED data load: {e}")

    # Compute BTP-Bund spread in basis points (yields are in percent, *100 = bps)
    italy_y = italy.squeeze()
    germany_y = germany.squeeze()

    # Align on common index
    spread = (italy_y - germany_y) * 100  # in basis points
    spread = spread.dropna().sort_index()

    if spread.empty:
        return mark_failed(sid, "empty BTP-Bund spread series")

    print(f"Spread series: {spread.index[0].date()} to {spread.index[-1].date()}, "
          f"n={len(spread)}")
    print(f"Max spread: {spread.max():.0f}bps, current: {spread.iloc[-1]:.0f}bps")

    # Load price data
    try:
        px = load_prices(["EUFN", "KBE", "SPY"], start="2009-01-01")
        ret = daily_returns(px)
        eufn_r = ret["EUFN"]
        kbe_r = ret["KBE"]
        spy_r = ret["SPY"]
    except Exception as e:
        return mark_failed(sid, f"price data load: {e}")

    # --- Signal generation ---
    # Monthly signal: spread > 250bps AND 2-month widening >= 30bps
    hold = 30          # trading days
    threshold = 250    # bps
    widening_req = 30  # bps over prior 2 months
    cooldown_days = 90  # min calendar days between entries
    spread_exit = 200   # bps (early exit if stress resolves)

    # Identify trigger months
    # spread is monthly: resample to month-end
    spread_monthly = spread.resample("ME").last()

    triggers = []
    last_exit_date = pd.Timestamp("1900-01-01")

    for i in range(2, len(spread_monthly)):
        date = spread_monthly.index[i]
        val = spread_monthly.iloc[i]
        val_2m_ago = spread_monthly.iloc[i - 2]

        if pd.isna(val) or pd.isna(val_2m_ago):
            continue

        widening = val - val_2m_ago

        if val > threshold and widening >= widening_req:
            # Check cooldown
            if (date - last_exit_date).days >= cooldown_days:
                # Entry = first trading day of following month
                entry_month_start = date + pd.DateOffset(months=1)
                entry_month_start = entry_month_start.replace(day=1)
                triggers.append({
                    "trigger_month": str(date.date()),
                    "spread_bps": round(float(val), 1),
                    "2m_widening_bps": round(float(widening), 1),
                    "entry_target": str(entry_month_start.date()),
                })

    print(f"Raw triggers: {len(triggers)}")
    for t in triggers:
        print(f"  {t['trigger_month']}: spread={t['spread_bps']:.0f}bps, "
              f"widening={t['2m_widening_bps']:.0f}bps")

    if not triggers:
        return mark_failed(sid, "no BTP-Bund stress triggers found")

    # --- Execute trades ---
    pnl = pd.Series(0.0, index=spy_r.index)
    trade_records = []
    last_exit_date = pd.Timestamp("1900-01-01")

    for trig in triggers:
        entry_target = pd.Timestamp(trig["entry_target"])

        # Check cooldown (recheck with actual exit dates)
        if (entry_target - last_exit_date).days < cooldown_days:
            print(f"  Skip {trig['trigger_month']}: cooldown")
            continue

        # Find first trading day on or after entry_target
        spy_mask = spy_r.index >= entry_target
        if spy_mask.sum() < hold:
            print(f"  Skip {trig['trigger_month']}: insufficient future data")
            continue

        entry_idx = spy_r.index[spy_mask][0]
        p_spy = spy_r.index.get_loc(entry_idx)

        # Check if EUFN has data at entry
        if entry_idx not in eufn_r.index:
            # Find closest date
            eufn_mask = eufn_r.index >= entry_idx
            if eufn_mask.sum() < hold:
                print(f"  Skip {trig['trigger_month']}: EUFN data unavailable")
                continue
            entry_idx = eufn_r.index[eufn_mask][0]
            p_spy = spy_r.index.get_loc(entry_idx) if entry_idx in spy_r.index else None
            if p_spy is None:
                continue

        p_eufn = eufn_r.index.get_loc(entry_idx)
        p_kbe = kbe_r.index.get_loc(entry_idx) if entry_idx in kbe_r.index else None

        # Build position PnL: short 2 EUFN + long 1 KBE (2:1 notional ratio)
        # Normalized: 2/3 weight short EUFN, 1/3 weight long KBE
        eufn_end = min(p_eufn + hold, len(eufn_r))
        eufn_window = eufn_r.iloc[p_eufn:eufn_end]
        short_eufn = -eufn_window

        kbe_window = None
        if p_kbe is not None:
            kbe_end = min(p_kbe + hold, len(kbe_r))
            kbe_window = kbe_r.iloc[p_kbe:kbe_end]

        min_len = len(short_eufn)
        if kbe_window is not None:
            min_len = min(min_len, len(kbe_window))

        # Check early exit: BTP-Bund spread tightens below 200bps
        # Monthly spread — we check the spread reading for each month in the hold period
        actual_min_len = min_len
        for day_idx in range(min_len):
            current_date = eufn_r.index[p_eufn + day_idx]
            # Find last monthly spread reading on or before this date
            spread_mask = spread_monthly.index <= current_date
            if spread_mask.sum() > 0:
                current_spread = spread_monthly[spread_mask].iloc[-1]
                if current_spread < spread_exit:
                    actual_min_len = day_idx + 1
                    print(f"  Early exit at day {day_idx+1}: spread tightened to "
                          f"{current_spread:.0f}bps")
                    break

        # Combine position: 2/3 short EUFN + 1/3 long KBE
        short_e = short_eufn.iloc[:actual_min_len].values
        if kbe_window is not None and len(kbe_window) >= actual_min_len:
            long_k = kbe_window.iloc[:actual_min_len].values
            combined = (2.0 / 3) * short_e + (1.0 / 3) * long_k
        else:
            combined = short_e  # fallback: just short EUFN

        trade_dates = eufn_r.index[p_eufn: p_eufn + actual_min_len]
        pnl.loc[trade_dates] = combined[:len(trade_dates)]

        exit_date = trade_dates[-1]
        last_exit_date = exit_date

        cum_short_eufn = float((1 + pd.Series(short_e)).prod() - 1)
        cum_long_kbe = (float((1 + pd.Series(long_k[:actual_min_len])).prod() - 1)
                        if kbe_window is not None and len(kbe_window) >= actual_min_len
                        else None)
        cum_combined = float((1 + pd.Series(combined)).prod() - 1)

        p_spy_entry = spy_r.index.get_loc(entry_idx) if entry_idx in spy_r.index else None
        cum_spy = None
        if p_spy_entry is not None:
            sp_end = min(p_spy_entry + actual_min_len, len(spy_r))
            cum_spy = float((1 + spy_r.iloc[p_spy_entry:sp_end]).prod() - 1)

        trade_records.append({
            "trigger_month": trig["trigger_month"],
            "spread_bps": trig["spread_bps"],
            "2m_widening_bps": trig["2m_widening_bps"],
            "entry_date": str(entry_idx.date()),
            "exit_date": str(exit_date.date()),
            "hold_days": actual_min_len,
            "short_eufn_return": round(cum_short_eufn, 4),
            "long_kbe_return": round(cum_long_kbe, 4) if cum_long_kbe is not None else None,
            "combined_return": round(cum_combined, 4),
            "spy_return": round(cum_spy, 4) if cum_spy is not None else None,
        })
        print(f"  Trade {trig['trigger_month']}: combined {cum_combined*100:.1f}% "
              f"vs SPY {cum_spy*100:.1f}%"
              if cum_spy is not None
              else f"  Trade {trig['trigger_month']}: combined {cum_combined*100:.1f}%")

    print(f"Total executed trades: {len(trade_records)}")

    if len(trade_records) == 0:
        return mark_failed(sid, "no trades executed after cooldown filtering")

    active_pnl = pnl[pnl != 0]
    print(f"Active PnL days: {len(active_pnl)}")

    if len(active_pnl) < 30:
        return mark_failed(sid, f"insufficient active PnL days ({len(active_pnl)})")

    m = compute_metrics(active_pnl, benchmark=spy_r,
                        name="BTP-Bund Spread Stress -> Short EUFN")
    save_result(sid, m, extra={
        "rule": "Short 2/3 EUFN + long 1/3 KBE for 30 trading days when BTP-Bund spread "
                ">250bps AND has widened >=30bps over prior 2 months (monthly FRED data)",
        "mechanism": "BTP-Bund spread widening signals Italian sovereign stress, which "
                     "propagates to European bank funding costs via sovereign-bank nexus; "
                     "EUFN (EU financials) underperforms while US banks (KBE) hedge "
                     "global financial contagion exposure.",
        "source": "FRED IRLTLT01ITM156N (Italy 10y), IRLTLT01DEM156N (Germany 10y); "
                  "yfinance EUFN, KBE, SPY",
        "n_events": len(trade_records),
        "trades": trade_records,
        "caveats": "Monthly FRED data creates 1-5 week lag from spread breach to entry. "
                   "EUFN only available from Feb 2010, missing 2008-2009 stress. "
                   "ECB TPI tool (2022+) may limit future spread widening duration.",
    })
    print(f"Done: {len(trade_records)} trades, Sharpe={m.get('sharpe', 'N/A'):.2f}")


if __name__ == "__main__":
    main()
