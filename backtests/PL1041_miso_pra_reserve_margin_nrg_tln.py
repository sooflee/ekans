"""PL1041_miso_pra_reserve_margin_nrg_tln
MISO Planning Resource Auction Reserve Margin Tightening → Long NRG / TLN Merchant Generation

Entry: When MISO PRA results show reserve margin stress (Zone 4 or MISO-wide),
go long NRG + TLN (or VST as TLN proxy pre-2023) equal-weight basket.
Hold 20 trading days. Stop: SPY drawdown >8%, or individual leg up >15%.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


# MISO PRA publication dates where reserve margin tightened or clearing prices spiked
# Sources: MISO press releases, annual PRA results PDFs
MISO_PRA_EVENTS = [
    "2019-05-15",   # Zone 4 tightening (Illinois/Indiana reserve stress)
    "2022-04-29",   # MISO-wide reserve margin warning (post-coal retirement)
    "2023-05-11",   # Reserve margin stress continued; MISO emergency procurement discussions
    "2024-04-30",   # Zone 4 emergency; MISO issued capacity shortfall warning
]


def main():
    sid = "PL1041_miso_pra_reserve_margin_nrg_tln"

    # TLN only since Oct 2023; use VST as MISO merchant proxy for pre-2023
    tickers = ["NRG", "TLN", "VST", "CEG", "SPY"]

    try:
        px = load_prices(tickers, start="2018-01-01")
    except Exception as e:
        try:
            px = load_prices(tickers, start="2018-01-01", cache=False)
        except Exception as e2:
            return mark_failed(sid, f"data load prices: {e2}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in ["NRG", "SPY"] if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing required tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()
    trading_dates = px.index

    hold_days = 20          # 4 weeks = 20 trading days
    spy_stop_loss = 0.08    # SPY drawdown >8% from entry
    leg_profit_take = 0.15  # individual leg up >15%

    positions_nrg = pd.Series(0.0, index=trading_dates)
    positions_vst = pd.Series(0.0, index=trading_dates)
    events = []
    open_until_date = pd.Timestamp("1900-01-01")

    spy_prices = px["SPY"]
    nrg_prices = px["NRG"]
    vst_prices = px["VST"] if "VST" in px.columns else None

    for event_str in MISO_PRA_EVENTS:
        event_date = pd.Timestamp(event_str)

        # Entry: first trading day on or after event date (within 2 days)
        after = trading_dates[(trading_dates >= event_date) &
                               (trading_dates <= event_date + pd.Timedelta(days=3))]
        if len(after) == 0:
            print(f"  Skipping {event_str}: no trading day within 2 days")
            continue
        entry_date = after[0]

        if entry_date <= open_until_date:
            print(f"  Skipping {entry_date.date()}: overlaps with prior trade")
            continue

        entry_idx = trading_dates.get_loc(entry_date)
        nrg_entry = nrg_prices.get(entry_date, np.nan)
        spy_entry = spy_prices.get(entry_date, np.nan)

        if np.isnan(nrg_entry) or np.isnan(spy_entry):
            print(f"  Skipping {entry_date.date()}: missing price data")
            continue

        # Determine which proxy to use for TLN slot
        # TLN has data after ~Oct 2023
        use_tln = False
        if "TLN" in px.columns:
            tln_entry = px["TLN"].get(entry_date, np.nan)
            if not np.isnan(tln_entry):
                use_tln = True

        proxy_ticker = "TLN" if use_tln else ("VST" if vst_prices is not None else None)
        proxy_prices = px["TLN"] if use_tln else vst_prices
        proxy_entry = proxy_prices.get(entry_date, np.nan) if proxy_prices is not None else np.nan

        # Walk forward
        trade_window = trading_dates[
            (trading_dates >= entry_date) &
            (trading_dates <= trading_dates[min(entry_idx + hold_days, len(trading_dates)-1)])
        ]
        trade_window = trade_window[:hold_days + 1]
        if len(trade_window) == 0:
            continue

        actual_exit_idx = len(trade_window) - 1
        exit_reason = "20_day_hold"

        for j, td in enumerate(trade_window):
            cur_spy = spy_prices.get(td, np.nan)
            cur_nrg = nrg_prices.get(td, np.nan)
            cur_proxy = proxy_prices.get(td, np.nan) if proxy_prices is not None else np.nan

            if not np.isnan(cur_spy) and not np.isnan(spy_entry):
                spy_dd = (cur_spy - spy_entry) / spy_entry
                if spy_dd <= -spy_stop_loss:
                    actual_exit_idx = j
                    exit_reason = "spy_8pct_drawdown"
                    break

            if not np.isnan(cur_nrg) and not np.isnan(nrg_entry):
                nrg_chg = (cur_nrg - nrg_entry) / nrg_entry
                if nrg_chg >= leg_profit_take:
                    actual_exit_idx = j
                    exit_reason = "nrg_15pct_profit"
                    break

            if not np.isnan(cur_proxy) and not np.isnan(proxy_entry):
                proxy_chg = (cur_proxy - proxy_entry) / proxy_entry
                if proxy_chg >= leg_profit_take:
                    actual_exit_idx = j
                    exit_reason = f"{proxy_ticker.lower()}_15pct_profit"
                    break

        actual_exit_date = trade_window[actual_exit_idx]
        open_until_date = actual_exit_date

        trade_slice = trade_window[:actual_exit_idx + 1]

        # Equal-weight long basket: NRG + proxy (VST or TLN)
        # Each leg gets 0.5 weight
        positions_nrg.loc[trade_slice] = 0.5

        if proxy_prices is not None and not np.isnan(proxy_entry):
            positions_vst.loc[trade_slice] = 0.5

        nrg_exit = nrg_prices.get(actual_exit_date, np.nan)
        nrg_ret = (nrg_exit - nrg_entry) / nrg_entry if not np.isnan(nrg_exit) else np.nan
        proxy_exit = proxy_prices.get(actual_exit_date, np.nan) if proxy_prices is not None else np.nan
        proxy_ret = (proxy_exit - proxy_entry) / proxy_entry if not np.isnan(proxy_exit) and not np.isnan(proxy_entry) else np.nan

        basket_ret = np.nanmean([r for r in [nrg_ret, proxy_ret] if not np.isnan(r)])

        events.append({
            "miso_event": event_str,
            "entry_date": str(entry_date.date()),
            "exit_date": str(actual_exit_date.date()),
            "exit_reason": exit_reason,
            "proxy_used": proxy_ticker,
            "nrg_return": round(float(nrg_ret), 4) if not np.isnan(nrg_ret) else None,
            "proxy_return": round(float(proxy_ret), 4) if not np.isnan(proxy_ret) else None,
            "basket_return": round(float(basket_ret), 4) if not np.isnan(basket_ret) else None,
        })
        print(f"  Trade {entry_date.date()} → {actual_exit_date.date()} ({exit_reason}), NRG={nrg_ret:.2%}, {proxy_ticker}={proxy_ret:.2%}, basket={basket_ret:.2%}")

    print(f"  Total events: {len(events)}")
    if len(events) < 2:
        return mark_failed(sid, f"insufficient events: {len(events)}")

    # Combined PnL: weight * return for each leg
    pos_nrg_shifted = positions_nrg.shift(1).fillna(0)
    pos_vst_shifted = positions_vst.shift(1).fillna(0)

    nrg_r_aligned = ret["NRG"].reindex(trading_dates).fillna(0)

    # Use whichever of TLN or VST we have
    if "TLN" in ret.columns:
        tln_r = ret["TLN"].reindex(trading_dates).fillna(0)
        vst_r = ret["VST"].reindex(trading_dates).fillna(0) if "VST" in ret.columns else pd.Series(0.0, index=trading_dates)
        # For post-TLN events, positions_vst holds TLN weight
        # For pre-TLN events, positions_vst holds VST weight
        # Since we used VST for pre-2023 and TLN for 2024 event:
        # The positions_vst series will have 0.5 weight where applicable
        # Use VST returns for pre-TLN, TLN returns for post-TLN
        # Find the TLN listing date
        tln_prices = px["TLN"]
        tln_first = tln_prices.dropna().index[0] if len(tln_prices.dropna()) > 0 else pd.Timestamp("2099-01-01")
        proxy_r = pd.Series(0.0, index=trading_dates)
        proxy_r[trading_dates >= tln_first] = tln_r[trading_dates >= tln_first]
        proxy_r[trading_dates < tln_first] = vst_r[trading_dates < tln_first]
    elif "VST" in ret.columns:
        proxy_r = ret["VST"].reindex(trading_dates).fillna(0)
    else:
        proxy_r = pd.Series(0.0, index=trading_dates)

    pnl_raw = pos_nrg_shifted * nrg_r_aligned + pos_vst_shifted * proxy_r

    pnl = pnl_raw.reindex(spy_r.index).dropna()

    in_pos_days = (pnl != 0).sum()
    print(f"  In-position days: {in_pos_days}")
    if in_pos_days < 10:
        return mark_failed(sid, f"insufficient in-position days: {in_pos_days}")

    total_pos = (abs(pos_nrg_shifted) + abs(pos_vst_shifted)).reindex(pnl.index).fillna(0)

    m = compute_metrics(
        pnl,
        benchmark=spy_r,
        name="MISO PRA Reserve Stress Long NRG+VST",
        positions=total_pos,
        cost_bps=10,
    )

    ev_returns = [e["basket_return"] for e in events if e.get("basket_return") is not None]
    win_rate = float(np.mean([r > 0 for r in ev_returns])) if ev_returns else None
    avg_event = float(np.mean(ev_returns)) if ev_returns else None

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "Long NRG + TLN (VST pre-2023) equal-weight basket when MISO annual PRA "
                "results show Zone 4 or MISO-wide reserve margin stress "
                "(clearing prices spike or margin falls below 15.8% LOLE threshold). "
                "Entry within 2 trading days of PRA publication. "
                "Hold 20 trading days. Stop: SPY -8% from entry or either leg +15%."
            ),
            "mechanism": (
                "MISO merchant generators NRG and TLN benefit directly from capacity "
                "scarcity: lower reserve margins push capacity clearing prices higher "
                "in MISO Zone 4, boosting unhedged capacity revenue. "
                "The PRA publication crystallizes investor expectations about annual "
                "capacity revenue, driving near-term re-rating of merchant generators."
            ),
            "source": "MISO PRA press releases/PDFs; yfinance NRG/TLN/VST/CEG/SPY",
            "tickers": ["NRG", "TLN", "VST", "SPY"],
            "n_events": len(events),
            "event_win_rate": round(win_rate, 4) if win_rate is not None else None,
            "avg_event_return": round(avg_event, 4) if avg_event is not None else None,
            "events": events,
            "miso_pra_dates": MISO_PRA_EVENTS,
            "caveats": (
                "Very small sample: only ~4 tightening events in 2019-2024. "
                "TLN only listed since Oct 2023 (insufficient history for primary backtest). "
                "VST is a proxy but has different geographic/asset mix than TLN. "
                "MISO PRA reserve margin results are from manually sourced event dates. "
                "Annual signal frequency limits portfolio utility."
            ),
        },
        pnl=pnl,
    )

    print(f"Done: {sid}")
    print(f"  n_events: {len(events)}, win_rate: {win_rate}, avg_event_return: {avg_event}")
    print(
        f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
        f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
    )
    if "oos_sharpe" in m:
        print(f"  OOS Sharpe: {m.get('oos_sharpe'):.2f}  IS Sharpe: {m.get('is_sharpe'):.2f}")


if __name__ == "__main__":
    main()
