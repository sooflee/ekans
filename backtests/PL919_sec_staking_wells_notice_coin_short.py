"""PL919 — SEC Staking-as-Security Wells Notice / Enforcement Action -> Short COIN Event Derate"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL919_sec_staking_wells_notice_coin_short"

    try:
        px = load_prices(["COIN", "HOOD", "IBIT", "SPY"], start="2021-08-01")
        spy_r = daily_returns(px[["SPY"]]).iloc[:, 0].dropna()
        coin_r = daily_returns(px[["COIN"]]).iloc[:, 0].dropna()
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    # HOOD IPO'd Aug 2021
    hood_r = None
    if "HOOD" in px.columns and px["HOOD"].dropna().shape[0] > 50:
        hood_r = daily_returns(px[["HOOD"]]).iloc[:, 0].dropna()

    # IBIT (BlackRock Bitcoin ETF) for additional context
    ibit_r = None
    if "IBIT" in px.columns and px["IBIT"].dropna().shape[0] > 50:
        ibit_r = daily_returns(px[["IBIT"]]).iloc[:, 0].dropna()

    # Known SEC enforcement events against crypto exchange staking programs
    # These are hard event dates from SEC filings / press releases
    enforcement_events = [
        {
            "date": "2023-02-09",
            "name": "SEC-Kraken settlement: $30M + US staking shutdown",
            "type": "settlement",
        },
        {
            "date": "2023-06-06",
            "name": "SEC formal complaint against Coinbase (staking + exchange)",
            "type": "complaint",
        },
        {
            "date": "2023-06-12",
            "name": "SEC complaint against Binance.US (secondary enforcement pressure)",
            "type": "complaint_secondary",
        },
        {
            "date": "2022-09-08",
            "name": "Ethereum Merge / SEC Gensler staking-as-security speech",
            "type": "speech_signal",
        },
        {
            "date": "2022-02-14",
            "name": "BlockFi SEC settlement - first staking/lending enforcement action",
            "type": "settlement_precursor",
        },
    ]

    # Build PnL: short COIN, long HOOD (or IBIT as partial hedge), hold 15 trading days
    # HOOD is the long leg (less staking exposure, crypto-adjacent fintech)
    # If HOOD unavailable for early events, use IBIT or just pure COIN short vs SPY

    hold_days = 15
    borrow_cost_annual = 0.02  # 2% annual borrow cost on COIN short

    # Align series
    if hood_r is not None:
        common_idx = coin_r.index.intersection(hood_r.index).intersection(spy_r.index)
    else:
        common_idx = coin_r.index.intersection(spy_r.index)

    coin_r = coin_r.reindex(common_idx)
    spy_r_c = spy_r.reindex(common_idx)
    if hood_r is not None:
        hood_r = hood_r.reindex(common_idx)

    events = []

    for ev in enforcement_events:
        ev_date = pd.Timestamp(ev["date"])
        future = common_idx[common_idx > ev_date]
        if len(future) < 2:
            continue

        entry_date = future[0]  # T+1

        entry_loc = common_idx.get_loc(entry_date)
        exit_loc = min(entry_loc + hold_days, len(common_idx) - 1)

        slice_idx = common_idx[entry_loc:exit_loc + 1]
        if len(slice_idx) < 3:
            continue

        coin_slice = coin_r.reindex(slice_idx).fillna(0)
        spy_slice = spy_r_c.reindex(slice_idx).fillna(0)

        # Short COIN, Long HOOD (equal notional)
        # Add borrow cost to short leg
        daily_borrow = borrow_cost_annual / 252
        if hood_r is not None and not hood_r.reindex(slice_idx).isna().all():
            hood_slice = hood_r.reindex(slice_idx).fillna(0)
            # Net PnL = HOOD return - COIN return - borrow cost
            trade_r = hood_slice - coin_slice - daily_borrow
        else:
            # If HOOD unavailable, use SPY long as hedge
            trade_r = spy_slice - coin_slice - daily_borrow

        # Stop loss: -8% on spread
        cum = trade_r.cumsum()
        stop = cum < -0.08
        if stop.any():
            stop_idx = stop.idxmax()
            trade_r = trade_r.loc[:stop_idx]
            spy_slice = spy_slice.loc[:stop_idx]

        # Profit target: +15%
        cum2 = trade_r.cumsum()
        tp = cum2 > 0.15
        if tp.any():
            tp_idx = tp.idxmax()
            trade_r = trade_r.loc[:tp_idx]
            spy_slice = spy_slice.loc[:tp_idx]

        cum_ret = float((1 + trade_r).prod() - 1)
        cum_spy = float((1 + spy_slice.fillna(0)).prod() - 1)

        events.append({
            "event_date": ev["date"],
            "event_name": ev["name"],
            "event_type": ev["type"],
            "entry_date": str(entry_date.date()),
            "exit_date": str(trade_r.index[-1].date()),
            "hold_days": len(trade_r),
            "used_hood": hood_r is not None,
            "trade_return": round(cum_ret, 4),
            "spy_return": round(cum_spy, 4),
            "alpha": round(cum_ret - cum_spy, 4),
        })

    print(f"Found {len(events)} enforcement events")

    if len(events) < 3:
        return mark_failed(sid, f"too few events ({len(events)}) — need at least 3 historical enforcement actions")

    # Build full PnL series
    pnl = pd.Series(0.0, index=common_idx)
    daily_borrow = borrow_cost_annual / 252

    for ev in events:
        entry = pd.Timestamp(ev["entry_date"])
        exit_d = pd.Timestamp(ev["exit_date"])

        coin_slice = coin_r.loc[entry:exit_d].fillna(0)
        if hood_r is not None:
            hood_slice = hood_r.loc[entry:exit_d].fillna(0)
            trade_slice = hood_slice - coin_slice - daily_borrow
        else:
            spy_slice = spy_r_c.loc[entry:exit_d].fillna(0)
            trade_slice = spy_slice - coin_slice - daily_borrow

        pnl.loc[entry:exit_d] += trade_slice

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r_c,
                        name="SEC Staking Enforcement: Short COIN / Long HOOD")
    m["n_events"] = len(events)

    save_result(sid, m, extra={
        "rule": "T+1 after SEC Wells notice / enforcement action against US crypto exchange staking: short COIN, long HOOD (equal notional), hold 15 trading days. Stop -8%, target +15%. 2% annual borrow cost on COIN short.",
        "mechanism": "SEC enforcement actions against crypto exchange staking programs trigger immediate COIN re-rating as regulatory overhang materializes. HOOD has lower staking revenue exposure, providing a cleaner long hedge. The COIN/HOOD spread reverts after enforcement certainty is priced in.",
        "source": "yfinance (COIN, HOOD, IBIT, SPY); SEC EDGAR 8-K filings; SEC press releases (justice.gov/rss)",
        "n_events": len(events),
        "events": events,
        "borrow_cost_bps_annual": int(borrow_cost_annual * 10000),
    })

    avg_alpha = float(np.mean([e["alpha"] for e in events]))
    win_rate = float(np.mean([1 if e["trade_return"] > 0 else 0 for e in events]))
    print(f"Sharpe={m.get('sharpe','N/A'):.2f}, CAGR={m.get('cagr','N/A')*100:.1f}%, events={len(events)}, win_rate={win_rate:.0%}, avg_alpha={avg_alpha:.3f}")


if __name__ == "__main__":
    main()
