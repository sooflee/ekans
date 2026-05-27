"""PL460_census_housing_completions_hvac_long
Census Housing Completions Surge -> Long Residential HVAC (CARR/LII)

When FRED COMPUTSA 3-month MA exceeds 1.4M SAAR, long CARR+LII for 63 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns

import numpy as np
import pandas as pd


def main():
    sid = "PL460_census_housing_completions_hvac_long"

    # Load FRED data
    try:
        completions = load_fred("COMPUTSA", start="2004-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED data: {e}")

    if completions.empty or len(completions) < 60:
        return mark_failed(sid, f"Insufficient COMPUTSA data: {len(completions)} rows")

    comp = completions["COMPUTSA"].dropna()
    comp_3m = comp.rolling(3).mean()

    # Identify months where 3m MA > 1400 (1.4M SAAR)
    threshold = 1400
    above = comp_3m[comp_3m > threshold]
    print(f"Months with 3m MA > {threshold}K SAAR: {len(above)}")

    # Group into events with 180-day minimum gap
    events = []
    last_signal = None
    for date, val in above.items():
        if last_signal is not None and (date - last_signal).days < 180:
            continue
        events.append({"date": str(date.date()), "completions_3m_ma": float(val)})
        last_signal = date

    print(f"Signal events (180-day gap): {len(events)}")

    # Load prices: CARR only since Apr 2020, use LII for full history
    try:
        px = load_prices(["LII", "SPY"], start="2004-01-01")
        px_carr = load_prices(["CARR"], start="2020-04-01")
    except Exception as e:
        return mark_failed(sid, f"price data: {e}")

    if "LII" not in px.columns or px["LII"].dropna().shape[0] < 200:
        return mark_failed(sid, "LII price data insufficient")

    ret = daily_returns(px)
    ret_carr = daily_returns(px_carr) if "CARR" in px_carr.columns else None

    lii_ret = ret["LII"].dropna()
    spy_ret = ret["SPY"].dropna()

    hold_days = 63
    pnl_series = pd.Series(0.0, index=lii_ret.index)
    positions = pd.Series(0.0, index=lii_ret.index)
    event_results = []

    for ev in events:
        entry_date = pd.Timestamp(ev["date"])

        # Use CARR+LII avg if post Apr 2020, else LII only
        if ret_carr is not None and entry_date >= pd.Timestamp("2020-05-01"):
            carr_ret = ret_carr["CARR"].dropna()
            # Align on common dates
            common_idx = lii_ret.index.intersection(carr_ret.index)
            combo = (lii_ret.reindex(common_idx) + carr_ret.reindex(common_idx)) / 2
        else:
            combo = lii_ret

        mask = combo.index > entry_date
        if mask.sum() < hold_days + 1:
            event_results.append({**ev, "status": "skipped", "reason": "insufficient future data"})
            continue

        start_idx = combo.index[mask][0]
        pos = combo.index.get_loc(start_idx)
        end_pos = min(pos + hold_days, len(combo))

        event_rets = combo.iloc[pos:end_pos]
        cumret = float((1 + event_rets).prod() - 1)

        # SPY
        spy_mask = spy_ret.index >= start_idx
        if spy_mask.sum() >= hold_days:
            spy_start = spy_ret.index[spy_mask][0]
            spy_pos = spy_ret.index.get_loc(spy_start)
            spy_event = spy_ret.iloc[spy_pos:spy_pos + hold_days]
            spy_cumret = float((1 + spy_event).prod() - 1)
        else:
            spy_cumret = None

        # Fill daily series (using LII index for alignment)
        for i in range(pos, end_pos):
            idx = combo.index[i]
            if idx in pnl_series.index:
                pnl_series.loc[idx] = combo.iloc[i]
                positions.loc[idx] = 1.0

        event_results.append({
            **ev,
            "status": "ok",
            "hvac_63d_return": round(cumret, 4),
            "spy_63d_return": round(spy_cumret, 4) if spy_cumret is not None else None,
            "excess_return": round(cumret - spy_cumret, 4) if spy_cumret is not None else None,
        })

    ok_events = [e for e in event_results if e["status"] == "ok"]
    if len(ok_events) < 2:
        return mark_failed(sid, f"Only {len(ok_events)} valid events")

    hvac_rets = np.array([e["hvac_63d_return"] for e in ok_events])
    excess_arr = np.array([e["excess_return"] for e in ok_events if e.get("excess_return") is not None])
    avg_ret = float(hvac_rets.mean())
    avg_excess = float(excess_arr.mean()) if len(excess_arr) > 0 else None
    win_rate = float((hvac_rets > 0).mean())

    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) >= 30:
        m = compute_metrics(in_pos, benchmark=spy_ret, name="Housing Completions -> HVAC Long", positions=positions[positions != 0])
    else:
        m = {
            "name": "Housing Completions -> HVAC Long",
            "n_days": len(in_pos),
            "sharpe": float(in_pos.mean() / in_pos.std() * np.sqrt(252)) if len(in_pos) > 1 and in_pos.std() > 0 else 0,
            "cagr": avg_ret,
            "max_dd": float((((1 + in_pos).cumprod() / (1 + in_pos).cumprod().cummax()) - 1).min()) if len(in_pos) > 0 else 0,
            "t_stat": float(in_pos.mean() / (in_pos.std() / np.sqrt(len(in_pos)))) if len(in_pos) > 1 and in_pos.std() > 0 else 0,
        }

    save_result(sid, m, extra={
        "status": "ok",
        "rule": "When FRED COMPUTSA 3m MA > 1.4M SAAR, long CARR+LII (or LII pre-2020) for 63 trading days. 180-day gap.",
        "mechanism": "Housing completions surge -> HVAC install demand with 1-2 quarter lag -> CARR/LII revenue",
        "source": "FRED COMPUTSA + yfinance",
        "events": event_results,
        "n_events": len(ok_events),
        "avg_event_return": round(avg_ret, 4),
        "avg_excess_return": round(avg_excess, 4) if avg_excess is not None else None,
        "event_win_rate": round(win_rate, 4),
    }, pnl=in_pos if len(in_pos) >= 30 else None)

    print(f"\nDone: {len(ok_events)} events, avg 63d return={avg_ret*100:.2f}%, win={win_rate*100:.0f}%")
    print(f"  Sharpe: {m.get('sharpe', 'N/A')}")
    for e in event_results:
        if e["status"] == "ok":
            print(f"  {e['date']} (comp 3mMA={e['completions_3m_ma']:.0f}K): HVAC={e['hvac_63d_return']*100:+.1f}%, excess={e.get('excess_return','N/A')}")


if __name__ == "__main__":
    main()
