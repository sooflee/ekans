"""PL478_jolts_industry_divergence_sector_rotation_long
BLS JOLTS Industry Openings Divergence -> Long Outperforming Sector ETF

When JOLTS shows a specific industry's job openings surging >20% YoY while total
nonfarm is flat/declining, long the corresponding sector ETF.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns

import numpy as np
import pandas as pd


def main():
    sid = "PL478_jolts_industry_divergence_sector_rotation_long"

    # JOLTS data on FRED: JTS000000000000000JOL (total nonfarm openings)
    # Industry mapping:
    # Healthcare: JTS6200JOL -> XLV
    # Construction: JTS2300JOL -> XHB
    # Manufacturing: JTS3000JOL -> XLI
    # Prof/Business: JTS540099JOL -> XLK (proxy)

    try:
        jolts = load_fred(["JTSJOL", "JTS6200JOL", "JTS2300JOL", "JTS3000JOL"], start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED JOLTS data: {e}")

    if jolts.empty or "JTSJOL" not in jolts.columns:
        return mark_failed(sid, "JOLTS data unavailable or insufficient")

    # Compute YoY changes
    total_yoy = jolts["JTSJOL"].pct_change(12)

    # Industry-to-ETF mapping
    industry_map = {
        "JTS6200JOL": "XLV",  # Healthcare
        "JTS2300JOL": "XHB",  # Construction -> Homebuilders
        "JTS3000JOL": "XLI",  # Manufacturing -> Industrials
    }

    # Load sector ETF prices
    etf_tickers = list(set(industry_map.values())) + ["SPY"]
    try:
        px = load_prices(etf_tickers, start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"price data: {e}")

    ret = daily_returns(px)
    spy_ret = ret["SPY"].dropna()

    # Identify divergence events: industry YoY > 20% AND total YoY <= 0
    hold_days = 42
    min_gap = 180  # days
    all_events = []

    for fred_series, etf_ticker in industry_map.items():
        if fred_series not in jolts.columns:
            continue
        ind_yoy = jolts[fred_series].pct_change(12)
        # Find months where industry surging while total flat/declining
        divergence = (ind_yoy > 0.20) & (total_yoy <= 0)
        signal_dates = divergence[divergence].index

        last_signal = None
        for date in signal_dates:
            if last_signal is not None and (date - last_signal).days < min_gap:
                continue
            all_events.append({
                "date": str(date.date()),
                "industry": fred_series,
                "etf": etf_ticker,
                "ind_yoy": float(ind_yoy.loc[date]),
                "total_yoy": float(total_yoy.loc[date]),
            })
            last_signal = date

    print(f"Found {len(all_events)} divergence events")

    if len(all_events) < 2:
        return mark_failed(sid, f"Only {len(all_events)} divergence events found")

    pnl_series = pd.Series(0.0, index=spy_ret.index)
    positions = pd.Series(0.0, index=spy_ret.index)
    event_results = []

    for ev in all_events:
        entry_date = pd.Timestamp(ev["date"])
        etf = ev["etf"]

        if etf not in ret.columns:
            event_results.append({**ev, "status": "skipped", "reason": f"{etf} not in data"})
            continue

        etf_ret = ret[etf].dropna()
        mask = etf_ret.index > entry_date
        if mask.sum() < hold_days + 1:
            event_results.append({**ev, "status": "skipped", "reason": "insufficient future data"})
            continue

        start_idx = etf_ret.index[mask][0]
        pos = etf_ret.index.get_loc(start_idx)
        end_pos = min(pos + hold_days, len(etf_ret))

        event_rets = etf_ret.iloc[pos:end_pos]
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
            idx = etf_ret.index[i]
            if idx in pnl_series.index:
                pnl_series.loc[idx] = etf_ret.iloc[i]
                positions.loc[idx] = 1.0

        event_results.append({
            **ev,
            "status": "ok",
            "etf_42d_return": round(cumret, 4),
            "spy_42d_return": round(spy_cumret, 4) if spy_cumret is not None else None,
            "excess_return": round(cumret - spy_cumret, 4) if spy_cumret is not None else None,
        })

    ok_events = [e for e in event_results if e["status"] == "ok"]
    if len(ok_events) < 2:
        return mark_failed(sid, f"Only {len(ok_events)} valid events after filtering")

    etf_rets = np.array([e["etf_42d_return"] for e in ok_events])
    excess_arr = np.array([e["excess_return"] for e in ok_events if e.get("excess_return") is not None])
    avg_ret = float(etf_rets.mean())
    avg_excess = float(excess_arr.mean()) if len(excess_arr) > 0 else None
    win_rate = float((etf_rets > 0).mean())

    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) >= 30:
        m = compute_metrics(in_pos, benchmark=spy_ret, name="JOLTS Divergence -> Sector Rotation", positions=positions[positions != 0])
    else:
        m = {
            "name": "JOLTS Divergence -> Sector Rotation",
            "n_days": len(in_pos),
            "sharpe": float(in_pos.mean() / in_pos.std() * np.sqrt(252)) if len(in_pos) > 1 and in_pos.std() > 0 else 0,
            "cagr": avg_ret,
            "max_dd": float((((1 + in_pos).cumprod() / (1 + in_pos).cumprod().cummax()) - 1).min()) if len(in_pos) > 0 else 0,
            "t_stat": float(in_pos.mean() / (in_pos.std() / np.sqrt(len(in_pos)))) if len(in_pos) > 1 and in_pos.std() > 0 else 0,
        }

    save_result(sid, m, extra={
        "status": "ok",
        "rule": "When JOLTS industry openings surge >20% YoY while total nonfarm flat/declining, long sector ETF 42d",
        "mechanism": "Labor demand divergence signals sector-specific strength that flows through to revenue",
        "source": "FRED JOLTS + yfinance",
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
            print(f"  {e['date']} {e['etf']} ({e['industry']}): {e['etf_42d_return']*100:+.1f}%, excess={e.get('excess_return','N/A')}")


if __name__ == "__main__":
    main()
