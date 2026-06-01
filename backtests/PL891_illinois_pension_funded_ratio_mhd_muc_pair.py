"""PL891 — Illinois Pension Funded Ratio Drift Below 45% -> Short MHD / Long MUC Muni CEF Pair"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics, save_result, mark_failed, daily_returns)


def main():
    sid = "PL891_illinois_pension_funded_ratio_mhd_muc_pair"

    # Illinois pension/credit stress event study.
    # Short MHD (national muni CEF, ~12-15% IL exposure), Long MUC (CA-concentrated).
    # Hold up to 12 calendar weeks (~60 trading days).
    HOLD = 60  # trading days (~12 weeks)

    # Known Illinois pension/credit stress trigger dates
    KNOWN_EVENTS = [
        pd.Timestamp("2016-05-18"),   # Moody's downgrade to Baa3
        pd.Timestamp("2017-06-01"),   # S&P downgrade to BBB- (736-day budget impasse)
        pd.Timestamp("2020-04-01"),   # COVID crash: pension funded ratio ~38-40%
        pd.Timestamp("2022-09-01"),   # Rising rates hit pension assets
    ]

    try:
        px = load_prices(["MHD", "MUC", "HYD", "MUB", "SPY"], start="2013-01-01")
    except Exception as e:
        try:
            px = load_prices(["MHD", "MUC", "HYD", "MUB", "SPY"], start="2013-01-01", cache=False)
        except Exception as e2:
            return mark_failed(sid, f"price load: {e2}")

    # Verify key tickers loaded
    available = list(px.columns)
    print(f"Available tickers: {available}")

    if "MHD" not in available:
        return mark_failed(sid, f"MHD not available in price data; available: {available}")
    if "MUC" not in available:
        return mark_failed(sid, f"MUC not available in price data; available: {available}")

    spy_r = daily_returns(px[["SPY"]]).iloc[:, 0]
    mhd_r = daily_returns(px[["MHD"]]).iloc[:, 0]
    muc_r = daily_returns(px[["MUC"]]).iloc[:, 0]

    # Pair return: short MHD, long MUC
    # positive when MUC outperforms MHD (IL stress hurts MHD)
    pair_r = muc_r.subtract(mhd_r, fill_value=0).dropna()

    pnl = pd.Series(0.0, index=pair_r.index)
    events = []
    last_trade_end = pd.Timestamp("1900-01-01")

    for event_date in sorted(KNOWN_EVENTS):
        if event_date <= last_trade_end:
            continue

        future_idx = pair_r.index[pair_r.index >= event_date]
        if len(future_idx) < HOLD + 1:
            continue

        entry_idx = future_idx[0]

        # Stop-loss: close if MHD outperforms MUC by >4% from entry
        mhd_price = px["MHD"].dropna()
        muc_price = px["MUC"].dropna()

        pos = pair_r.index.get_loc(entry_idx)
        end_pos = min(pos + HOLD, len(pair_r))

        entry_mhd = mhd_price.get(entry_idx, None)
        entry_muc = muc_price.get(entry_idx, None)
        if entry_mhd is not None and entry_muc is not None and entry_mhd > 0 and entry_muc > 0:
            mhd_w = mhd_price.iloc[pos:end_pos]
            muc_w = muc_price.iloc[pos:end_pos]
            mhd_vs_muc = (mhd_w / entry_mhd) / (muc_w / entry_muc) - 1
            stop_hit = (mhd_vs_muc > 0.04)
            if stop_hit.any():
                stop_loc = mhd_price.index.get_loc(stop_hit.idxmax())
                end_pos = min(stop_loc + 1, end_pos)

        window = pair_r.iloc[pos:end_pos]
        mhd_window_r = mhd_r.reindex(window.index).fillna(0)
        muc_window_r = muc_r.reindex(window.index).fillna(0)

        pnl.iloc[pos:end_pos] += window.values[:end_pos - pos]
        last_trade_end = pair_r.index[end_pos - 1]

        pair_cum = float((1 + window).prod() - 1)
        mhd_cum = float((1 + mhd_window_r).prod() - 1)
        muc_cum = float((1 + muc_window_r).prod() - 1)

        events.append({
            "event_date": str(event_date.date()),
            "entry_date": str(entry_idx.date()),
            "hold_days": end_pos - pos,
            "pair_return": round(pair_cum, 4),
            "mhd_return": round(mhd_cum, 4),
            "muc_return": round(muc_cum, 4),
        })

    if not events:
        return mark_failed(sid, "no valid events found")

    active_pnl = pnl[pnl != 0]
    print(f"Events: {len(events)}, Active trading days: {len(active_pnl)}")
    for e in events:
        print(f"  {e['event_date']}: pair={e['pair_return']:.1%}, MHD={e['mhd_return']:.1%}, MUC={e['muc_return']:.1%}")

    if len(active_pnl) < 30:
        return mark_failed(sid, f"insufficient active days: {len(active_pnl)} ({len(events)} events)")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="Illinois Pension Stress: Short MHD / Long MUC")
    win_rates = [1 if e["pair_return"] > 0 else 0 for e in events]

    save_result(sid, m, extra={
        "rule": "Short MHD (national muni CEF ~15% IL) / Long MUC (CA muni CEF) for up to 12 weeks following Illinois GO downgrade or pension funded ratio drop below 45%",
        "mechanism": "Illinois pension underfunding (45% funded ratio) triggers credit spread widening for IL GO bonds; MHD with IL exposure underperforms CA-focused MUC; CEF pair isolates the state-specific credit risk",
        "source": "Illinois TRS/SERS CAFR filings; Moody's/S&P credit event dates; yfinance MHD/MUC/HYD/MUB/SPY",
        "n_events": len(events),
        "event_win_rate": round(float(np.mean(win_rates)), 4) if win_rates else 0,
        "events": events,
        "caveats": "Very low event count (4 events); MHD has ~12-15% IL exposure only (diluted signal); CEF discount-to-NAV dynamics can dominate; pension funded ratio requires manual extraction from CAFR PDFs",
    })
    print(f"Done: Sharpe={m.get('sharpe', 'N/A'):.3f}, CAGR={m.get('cagr', 0)*100:.1f}%")


if __name__ == "__main__":
    main()
