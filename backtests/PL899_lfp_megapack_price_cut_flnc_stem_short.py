"""PL899 — LFP Cell Spot Price Crash + Tesla Megapack Price Cut -> Short FLNC/STEM"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics, save_result, mark_failed, daily_returns)


def main():
    sid = "PL899_lfp_megapack_price_cut_flnc_stem_short"

    # Known LFP cell price decline / Tesla Megapack price-cut events
    # From strategy spec:
    # 1. 2022-10-01: Tesla posted Megapack prices publicly; LFP cell costs declining in H2 2022
    # 2. 2024-01-15: Tesla Megapack price removed from public listing; BNEF battery price -20% YoY
    # 3. 2024-07-01: Q3 2024 BNEF pack cost decline confirmation
    # 4. 2025-01-01: Further competitive ASP compression
    SIGNAL_DATES = [
        pd.Timestamp("2022-10-01"),
        pd.Timestamp("2024-01-15"),
        pd.Timestamp("2024-07-01"),
        pd.Timestamp("2025-01-01"),
    ]
    HOLD = 40  # ~8 weeks in trading days

    try:
        px = load_prices(["FLNC", "STEM", "SHLS", "SPY"], start="2021-10-01")
        spy_r = daily_returns(px[["SPY"]]).iloc[:, 0]
        flnc_r = daily_returns(px[["FLNC"]]).iloc[:, 0]
        stem_r = daily_returns(px[["STEM"]]).iloc[:, 0]
        shls_r = daily_returns(px[["SHLS"]]).iloc[:, 0]
    except Exception as e:
        try:
            px = load_prices(["FLNC", "STEM", "SHLS", "SPY"], start="2021-10-01", cache=False)
            spy_r = daily_returns(px[["SPY"]]).iloc[:, 0]
            flnc_r = daily_returns(px[["FLNC"]]).iloc[:, 0]
            stem_r = daily_returns(px[["STEM"]]).iloc[:, 0]
            shls_r = daily_returns(px[["SHLS"]]).iloc[:, 0]
        except Exception as e2:
            return mark_failed(sid, f"price load: {e2}")

    # Short basket: 50% FLNC + 50% STEM
    short_basket_r = 0.5 * flnc_r + 0.5 * stem_r

    events = []
    pnl_dates = []
    pnl_rets = []

    for sig_date in SIGNAL_DATES:
        future_idx = short_basket_r.index[short_basket_r.index >= sig_date]
        if len(future_idx) < 10:
            print(f"  Skipping {sig_date.date()}: insufficient forward data")
            continue
        entry_idx = future_idx[0]
        pos = short_basket_r.index.get_loc(entry_idx)
        end_pos = min(pos + HOLD, len(short_basket_r))

        short_window = short_basket_r.iloc[pos:end_pos]
        spy_window = spy_r.reindex(short_window.index).fillna(0)
        shls_window = shls_r.reindex(short_window.index).fillna(0)

        # PnL: short FLNC/STEM basket vs long SPY
        pair_r = spy_window - short_window

        short_cum = float((1 + short_window).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)
        pair_cum = float((1 + pair_r).prod() - 1)

        events.append({
            "signal_date": str(sig_date.date()),
            "entry_date": str(entry_idx.date()),
            "hold_days": end_pos - pos,
            "flnc_stem_return_8w": round(short_cum, 4),
            "spy_return_8w": round(spy_cum, 4),
            "pair_return_8w": round(pair_cum, 4),
        })
        pnl_dates.extend(pair_r.index.tolist())
        pnl_rets.extend(pair_r.values.tolist())
        print(f"  {sig_date.date()}: short(FLNC/STEM)={short_cum:.1%}, SPY={spy_cum:.1%}, pair(long SPY/short basket)={pair_cum:.1%}")

    if not events:
        return mark_failed(sid, "no valid events after price join")

    combined_pnl = pd.Series(pnl_rets, index=pd.DatetimeIndex(pnl_dates)).sort_index()
    print(f"Events: {len(events)}, Combined active days: {len(combined_pnl)}")

    if len(combined_pnl) < 20:
        return mark_failed(sid, f"insufficient active days: {len(combined_pnl)}")

    m = compute_metrics(combined_pnl, benchmark=spy_r, name="LFP Price Crash → Short FLNC/STEM vs Long SPY")
    win_rates = [1 if e["pair_return_8w"] > 0 else 0 for e in events]

    save_result(sid, m, extra={
        "rule": "Short FLNC+STEM (50/50) vs Long SPY for ~8 weeks when LFP cell spot prices drop >15% QoQ AND Tesla Megapack price-list shows cut/removal",
        "mechanism": "Rapid LFP ASP compression forces FLNC/STEM to reprice in-flight project bids; gross margin compression arrives in next 2-3 quarters; stock reprices ahead of earnings",
        "source": "Tesla Megapack price-cut events (Wayback Machine documented); BNEF battery price survey headlines; yfinance FLNC/STEM/SPY",
        "n_events": len(events),
        "event_win_rate": round(float(np.mean(win_rates)), 4),
        "events": events,
        "caveats": "Only 4 events; FLNC/STEM very volatile; proxy events are approximate; STEM delisted risk noted; statistical power extremely low",
    })
    print(f"Done: {len(events)} events, Sharpe={m.get('sharpe', 'N/A'):.3f}, CAGR={m.get('cagr', 0)*100:.1f}%")


if __name__ == "__main__":
    main()
