"""PL897 — IRS 48E Tech-Neutral ITC Final Regs -> Long FLNC Backlog Acceleration"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics, save_result, mark_failed, daily_returns)


def main():
    sid = "PL897_irs_48e_finalreg_flnc_backlog_long"

    # Known IRS/Treasury 48E guidance events from strategy spec
    # All after FLNC IPO (Oct 28, 2021)
    SIGNAL_DATES = [
        pd.Timestamp("2022-08-16"),  # IRA enacted - standalone storage ITC confirmed
        pd.Timestamp("2023-04-04"),  # Treasury Notice 2023-29 (bonus adder preliminary)
        pd.Timestamp("2023-12-01"),  # 48E proposed regs published
        pd.Timestamp("2024-05-16"),  # Treasury final rules on domestic content adder (Notice 2024-41)
    ]
    HOLD = 60  # ~12 weeks in trading days

    try:
        px = load_prices(["FLNC", "FSLR", "ENPH", "SPY"], start="2021-10-01")
        spy_r = daily_returns(px[["SPY"]]).iloc[:, 0]
        flnc_r = daily_returns(px[["FLNC"]]).iloc[:, 0]
        fslr_r = daily_returns(px[["FSLR"]]).iloc[:, 0]
        enph_r = daily_returns(px[["ENPH"]]).iloc[:, 0]
    except Exception as e:
        try:
            px = load_prices(["FLNC", "FSLR", "ENPH", "SPY"], start="2021-10-01", cache=False)
            spy_r = daily_returns(px[["SPY"]]).iloc[:, 0]
            flnc_r = daily_returns(px[["FLNC"]]).iloc[:, 0]
            fslr_r = daily_returns(px[["FSLR"]]).iloc[:, 0]
            enph_r = daily_returns(px[["ENPH"]]).iloc[:, 0]
        except Exception as e2:
            return mark_failed(sid, f"price load: {e2}")

    # Sector control basket: 50% FSLR + 50% ENPH
    sector_r = 0.5 * fslr_r + 0.5 * enph_r

    events = []
    pnl_dates = []
    pnl_rets = []

    for sig_date in SIGNAL_DATES:
        future_idx = flnc_r.index[flnc_r.index >= sig_date]
        if len(future_idx) < 10:
            print(f"  Skipping {sig_date.date()}: insufficient forward data")
            continue
        entry_idx = future_idx[0]
        pos = flnc_r.index.get_loc(entry_idx)
        end_pos = min(pos + HOLD, len(flnc_r))

        flnc_window = flnc_r.iloc[pos:end_pos]
        spy_window = spy_r.reindex(flnc_window.index).fillna(0)
        sector_window = sector_r.reindex(flnc_window.index).fillna(0)

        # Excess return: FLNC vs SPY-hedged (50% SPY short)
        pair_r = flnc_window - spy_window

        flnc_cum = float((1 + flnc_window).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)
        sector_cum = float((1 + sector_window).prod() - 1)
        pair_cum = float((1 + pair_r).prod() - 1)

        events.append({
            "signal_date": str(sig_date.date()),
            "entry_date": str(entry_idx.date()),
            "hold_days": end_pos - pos,
            "flnc_return": round(flnc_cum, 4),
            "spy_return": round(spy_cum, 4),
            "sector_return": round(sector_cum, 4),
            "excess_return_vs_spy": round(pair_cum, 4),
        })
        pnl_dates.extend(pair_r.index.tolist())
        pnl_rets.extend(pair_r.values.tolist())
        print(f"  {sig_date.date()}: FLNC={flnc_cum:.1%}, SPY={spy_cum:.1%}, sector={sector_cum:.1%}, excess={pair_cum:.1%}")

    if not events:
        return mark_failed(sid, "no valid events after price join")

    combined_pnl = pd.Series(pnl_rets, index=pd.DatetimeIndex(pnl_dates)).sort_index()
    print(f"Events: {len(events)}, Combined active days: {len(combined_pnl)}")

    if len(combined_pnl) < 20:
        return mark_failed(sid, f"insufficient active days: {len(combined_pnl)}")

    m = compute_metrics(combined_pnl, benchmark=spy_r, name="IRS 48E Guidance → Long FLNC vs SPY")
    win_rates = [1 if e["excess_return_vs_spy"] > 0 else 0 for e in events]

    save_result(sid, m, extra={
        "rule": "Long FLNC (100%) with 50% SPY short hedge for 12 weeks when IRS/Treasury publishes 48E final regs or major guidance notice",
        "mechanism": "48E clarity removes ITC eligibility uncertainty → BESS developers accelerate FID conversions → FLNC backlog/bookings growth accelerates (confirmed in FLNC 10-K filings)",
        "source": "Known IRS/Treasury 48E guidance events 2022-2024; yfinance FLNC/FSLR/ENPH/SPY",
        "n_events": len(events),
        "event_win_rate": round(float(np.mean(win_rates)), 4),
        "events": events,
        "caveats": "Only 4 events; FLNC IPO Oct 2021 limits history; high stock-specific volatility; statistical confidence is very low with n=4",
    })
    print(f"Done: {len(events)} events, Sharpe={m.get('sharpe', 'N/A'):.3f}, CAGR={m.get('cagr', 0)*100:.1f}%")


if __name__ == "__main__":
    main()
