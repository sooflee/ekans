"""PL471_doe_arpae_program_cleantech_thematic
DOE ARPA-E Program Announcement -> Technology Focus -> Clean Energy Thematic Long

When DOE ARPA-E announces a major new program (>$100M), long the most relevant public company.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns

import numpy as np
import pandas as pd


def main():
    sid = "PL471_doe_arpae_program_cleantech_thematic"

    # Hand-coded ARPA-E major program announcements with >$100M funding
    # and the most relevant public company ticker
    events = [
        {"date": "2016-02-08", "label": "ARPA-E GENSETS (gas turbine generators)", "ticker": "GE", "funding_m": 130},
        {"date": "2018-01-29", "label": "ARPA-E MEITNER (nuclear fission)", "ticker": "BWXT", "funding_m": 120},
        {"date": "2019-04-15", "label": "ARPA-E BREAKERS (grid equipment)", "ticker": "ETN", "funding_m": 100},
        {"date": "2020-10-06", "label": "ARPA-E LIFTOFF (hydrogen aviation)", "ticker": "PLUG", "funding_m": 135},
        {"date": "2021-09-08", "label": "ARPA-E SCALEUP (commercialization bridge)", "ticker": "BE", "funding_m": 175},
        {"date": "2022-03-22", "label": "ARPA-E HESTIA (nuclear heating)", "ticker": "BWXT", "funding_m": 120},
        {"date": "2022-09-20", "label": "ARPA-E hydrogen programs cluster", "ticker": "PLUG", "funding_m": 200},
        {"date": "2023-03-14", "label": "ARPA-E long-duration storage program", "ticker": "FSLR", "funding_m": 175},
        {"date": "2023-09-19", "label": "ARPA-E nuclear microreactor programs", "ticker": "BWXT", "funding_m": 150},
        {"date": "2024-03-18", "label": "ARPA-E grid modernization focus", "ticker": "ETN", "funding_m": 200},
    ]

    # Load all unique tickers
    all_tickers = list(set([ev["ticker"] for ev in events] + ["SPY"]))
    try:
        px = load_prices(all_tickers, start="2015-01-01")
    except Exception as e:
        return mark_failed(sid, f"price data: {e}")

    ret = daily_returns(px)
    spy_ret = ret["SPY"].dropna()

    hold_days = 30
    pnl_series = pd.Series(0.0, index=spy_ret.index)
    positions = pd.Series(0.0, index=spy_ret.index)
    event_results = []

    for ev in events:
        entry_date = pd.Timestamp(ev["date"])
        ticker = ev["ticker"]

        if ticker not in ret.columns:
            event_results.append({**ev, "status": "skipped", "reason": f"{ticker} not in data"})
            continue

        ticker_ret = ret[ticker].dropna()
        mask = ticker_ret.index > entry_date
        if mask.sum() < hold_days + 1:
            event_results.append({**ev, "status": "skipped", "reason": "insufficient future data"})
            continue

        start_idx = ticker_ret.index[mask][0]
        pos = ticker_ret.index.get_loc(start_idx)
        end_pos = min(pos + hold_days, len(ticker_ret))

        event_rets = ticker_ret.iloc[pos:end_pos]
        cumret = float((1 + event_rets).prod() - 1)

        spy_mask = spy_ret.index >= start_idx
        if spy_mask.sum() >= hold_days:
            spy_start = spy_ret.index[spy_mask][0]
            spy_pos = spy_ret.index.get_loc(spy_start)
            spy_event = spy_ret.iloc[spy_pos:spy_pos + hold_days]
            spy_cumret = float((1 + spy_event).prod() - 1)
        else:
            spy_cumret = None

        for i in range(pos, end_pos):
            idx = ticker_ret.index[i]
            if idx in pnl_series.index:
                pnl_series.loc[idx] = ticker_ret.iloc[i]
                positions.loc[idx] = 1.0

        event_results.append({
            **ev,
            "status": "ok",
            "stock_30d_return": round(cumret, 4),
            "spy_30d_return": round(spy_cumret, 4) if spy_cumret is not None else None,
            "excess_return": round(cumret - spy_cumret, 4) if spy_cumret is not None else None,
        })

    ok_events = [e for e in event_results if e["status"] == "ok"]
    if len(ok_events) < 2:
        return mark_failed(sid, f"Only {len(ok_events)} valid events")

    stock_rets = np.array([e["stock_30d_return"] for e in ok_events])
    excess_arr = np.array([e["excess_return"] for e in ok_events if e.get("excess_return") is not None])
    avg_ret = float(stock_rets.mean())
    avg_excess = float(excess_arr.mean()) if len(excess_arr) > 0 else None
    win_rate = float((stock_rets > 0).mean())

    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) >= 30:
        m = compute_metrics(in_pos, benchmark=spy_ret, name="ARPA-E Program -> Clean Tech Long", positions=positions[positions != 0])
    else:
        m = {
            "name": "ARPA-E Program -> Clean Tech Long",
            "n_days": len(in_pos),
            "sharpe": float(in_pos.mean() / in_pos.std() * np.sqrt(252)) if len(in_pos) > 1 and in_pos.std() > 0 else 0,
            "cagr": avg_ret,
            "max_dd": float((((1 + in_pos).cumprod() / (1 + in_pos).cumprod().cummax()) - 1).min()) if len(in_pos) > 0 else 0,
            "t_stat": float(in_pos.mean() / (in_pos.std() / np.sqrt(len(in_pos)))) if len(in_pos) > 1 and in_pos.std() > 0 else 0,
        }

    save_result(sid, m, extra={
        "status": "ok",
        "rule": "When DOE ARPA-E announces >$100M program in specific clean energy tech, long most relevant public company 30d",
        "mechanism": "ARPA-E funding validates technology -> investor attention -> stock rerate",
        "source": "DOE ARPA-E announcements + yfinance",
        "events": event_results,
        "n_events": len(ok_events),
        "avg_event_return": round(avg_ret, 4),
        "avg_excess_return": round(avg_excess, 4) if avg_excess is not None else None,
        "event_win_rate": round(win_rate, 4),
    }, pnl=in_pos if len(in_pos) >= 30 else None)

    print(f"Done: {len(ok_events)} events, avg={avg_ret*100:.2f}%, win={win_rate*100:.0f}%")
    print(f"  Sharpe: {m.get('sharpe', 'N/A')}")
    for e in event_results:
        if e["status"] == "ok":
            print(f"  {e['date']} {e['ticker']}: {e['stock_30d_return']*100:+.1f}%, excess={e.get('excess_return','N/A')}")


if __name__ == "__main__":
    main()
