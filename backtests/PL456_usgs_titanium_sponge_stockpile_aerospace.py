"""PL456_usgs_titanium_sponge_stockpile_aerospace
USGS Titanium Sponge Stockpile Drawdown -> Aerospace Supply Tightening -> Specialty Metals Long

Annual event study: when titanium supply is tight + aerospace demand rising, long CRS+ATI.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns

import numpy as np
import pandas as pd


def main():
    sid = "PL456_usgs_titanium_sponge_stockpile_aerospace"

    # USGS MCS titanium data (annual, published late January)
    # Hand-coded from USGS Mineral Commodity Summaries annual reports.
    # Signal years: titanium sponge apparent consumption/imports rising
    # + Boeing/Airbus deliveries trending up = supply tightening
    #
    # USGS MCS publication dates are typically late January.
    # Signal fires when: titanium supply indicators show tightening + aerospace production rising.

    events = [
        {"date": "2011-01-31", "label": "2011 MCS: Ti sponge imports up, 787 production ramp", "year": 2011},
        {"date": "2014-01-31", "label": "2014 MCS: Aerospace build rates at record, Ti tight", "year": 2014},
        {"date": "2015-01-30", "label": "2015 MCS: Continued aerospace ramp, Ti stockpile draw", "year": 2015},
        {"date": "2018-01-31", "label": "2018 MCS: 737 MAX ramp, Ti demand surge", "year": 2018},
        {"date": "2023-01-31", "label": "2023 MCS: Post-COVID aerospace recovery, Russia Ti sanctions risk", "year": 2023},
        {"date": "2024-01-31", "label": "2024 MCS: Continued aerospace recovery, VSMPO-AVISMA sanctions", "year": 2024},
        {"date": "2025-01-31", "label": "2025 MCS: Boeing 737/787 production increases, Ti supply constrained", "year": 2025},
    ]

    # Load prices
    try:
        px = load_prices(["CRS", "ATI", "SPY"], start="2008-01-01")
    except Exception as e:
        return mark_failed(sid, f"price data: {e}")

    for t in ["CRS", "ATI"]:
        if t not in px.columns or px[t].dropna().shape[0] < 100:
            return mark_failed(sid, f"{t} price data insufficient")

    ret = daily_returns(px)
    metals_ret = ret[["CRS", "ATI"]].mean(axis=1).dropna()
    spy_ret = ret["SPY"].dropna()

    hold_days = 40
    pnl_series = pd.Series(0.0, index=metals_ret.index)
    positions = pd.Series(0.0, index=metals_ret.index)
    event_results = []

    for ev in events:
        entry_date = pd.Timestamp(ev["date"])

        mask = metals_ret.index > entry_date
        if mask.sum() < hold_days + 1:
            event_results.append({**ev, "status": "skipped", "reason": "insufficient future data"})
            continue

        start_idx = metals_ret.index[mask][0]
        pos = metals_ret.index.get_loc(start_idx)
        end_pos = min(pos + hold_days, len(metals_ret))

        event_rets = metals_ret.iloc[pos:end_pos]
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

        for i in range(pos, end_pos):
            idx = metals_ret.index[i]
            if idx in pnl_series.index:
                pnl_series.loc[idx] = metals_ret.iloc[i]
                positions.loc[idx] = 1.0

        event_results.append({
            **ev,
            "status": "ok",
            "metals_40d_return": round(cumret, 4),
            "spy_40d_return": round(spy_cumret, 4) if spy_cumret is not None else None,
            "excess_return": round(cumret - spy_cumret, 4) if spy_cumret is not None else None,
        })

    ok_events = [e for e in event_results if e["status"] == "ok"]
    if len(ok_events) < 2:
        return mark_failed(sid, f"Only {len(ok_events)} valid events")

    metals_rets = np.array([e["metals_40d_return"] for e in ok_events])
    excess_arr = np.array([e["excess_return"] for e in ok_events if e.get("excess_return") is not None])
    avg_ret = float(metals_rets.mean())
    avg_excess = float(excess_arr.mean()) if len(excess_arr) > 0 else None
    win_rate = float((metals_rets > 0).mean())

    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) >= 30:
        m = compute_metrics(in_pos, benchmark=spy_ret, name="Ti Sponge Tight -> CRS+ATI Long", positions=positions[positions != 0])
    else:
        m = {
            "name": "Ti Sponge Tight -> CRS+ATI Long",
            "n_days": len(in_pos),
            "sharpe": float(in_pos.mean() / in_pos.std() * np.sqrt(252)) if len(in_pos) > 1 and in_pos.std() > 0 else 0,
            "cagr": avg_ret,
            "max_dd": float((((1 + in_pos).cumprod() / (1 + in_pos).cumprod().cummax()) - 1).min()) if len(in_pos) > 0 else 0,
            "t_stat": float(in_pos.mean() / (in_pos.std() / np.sqrt(len(in_pos)))) if len(in_pos) > 1 and in_pos.std() > 0 else 0,
        }

    save_result(sid, m, extra={
        "status": "ok",
        "rule": "When USGS MCS shows titanium supply tightening + aerospace production rising, long CRS+ATI 40d post-MCS publication",
        "mechanism": "Ti sponge stockpile drawdown + aerospace ramp -> specialty metals pricing power -> CRS/ATI revenue",
        "source": "USGS MCS Titanium + yfinance",
        "events": event_results,
        "n_events": len(ok_events),
        "avg_event_return": round(avg_ret, 4),
        "avg_excess_return": round(avg_excess, 4) if avg_excess is not None else None,
        "event_win_rate": round(win_rate, 4),
    }, pnl=in_pos if len(in_pos) >= 30 else None)

    print(f"Done: {len(ok_events)} events, avg return={avg_ret*100:.2f}%, win={win_rate*100:.0f}%")
    print(f"  Sharpe: {m.get('sharpe', 'N/A')}")
    for e in event_results:
        if e["status"] == "ok":
            print(f"  {e['date']}: CRS+ATI={e['metals_40d_return']*100:+.1f}%, excess={e.get('excess_return','N/A')}")


if __name__ == "__main__":
    main()
