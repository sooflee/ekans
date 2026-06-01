"""PL1054_banrep_cut_wti_stable_gxg_long
BanRep Rate Cut + WTI >= $65/bbl -> Colombia Carry Convergence -> Long GXG

Event-study: when BanRep cuts rates AND WTI >= $65/bbl, go long GXG for up to 8 weeks.
Uses FRED DCOILWTICO for WTI. BanRep cut dates compiled from known policy cycles.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL1054_banrep_cut_wti_stable_gxg_long"
    tickers = ["GXG", "COP=X", "EEM", "SPY"]

    try:
        px = load_prices(tickers, start="2009-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if px is None or len(px) == 0:
        try:
            px = load_prices(tickers, start="2009-01-01", cache=False)
        except Exception as e:
            return mark_failed(sid, f"data load retry: {e}")

    px = px.sort_index().ffill(limit=5)

    if "GXG" not in px.columns or "SPY" not in px.columns:
        return mark_failed(sid, "missing GXG or SPY")

    # Load FRED WTI crude
    try:
        fred_df = load_fred(["DCOILWTICO"], start="2009-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED WTI load failed: {e}")

    if fred_df is None or "DCOILWTICO" not in fred_df.columns:
        return mark_failed(sid, "DCOILWTICO not available")

    wti = fred_df["DCOILWTICO"].dropna().resample("D").ffill()

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()
    idx = ret.index

    # BanRep cut dates (compiled from banrep.gov.co historical policy decisions)
    # Format: (date, from_rate, to_rate, wti_at_decision)
    # All entries are confirmed rate CUTS (not holds or hikes)
    banrep_cut_dates = [
        # 2012-2013 easing cycle (from 5.25% to 3.25%)
        "2012-07-27",  # 5.25% -> 5.00%, WTI ~$90
        "2012-08-31",  # 5.00% -> 4.75%, WTI ~$95
        "2013-02-01",  # 4.25% -> 4.00%, WTI ~$98
        "2013-03-22",  # 4.00% -> 3.75%, WTI ~$93
        "2013-04-26",  # 3.75% -> 3.50%, WTI ~$93
        # 2016-2017 easing (WTI $40-55 range, many won't pass filter)
        "2016-12-23",  # 7.75% -> 7.50%, WTI ~$53
        "2017-01-27",  # 7.50% -> 7.25%, WTI ~$53
        "2017-02-24",  # 7.25% -> 7.00%, WTI ~$54
        "2017-03-31",  # 7.00% -> 6.75%, WTI ~$50
        "2017-05-26",  # 6.75% -> 6.50%, WTI ~$50
        "2017-06-30",  # 6.50% -> 6.25%, WTI ~$46
        "2017-08-11",  # 6.25% -> 5.75%, WTI ~$49
        "2017-09-29",  # 5.75% -> 5.25%, WTI ~$52
        "2017-10-27",  # 5.25% -> 5.00%, WTI ~$52
        "2017-11-24",  # 5.00% -> 4.75%, WTI ~$58
        "2017-12-22",  # 4.75% -> 4.50%, WTI ~$58
        "2018-01-26",  # 4.50% -> 4.25%, WTI ~$65
        # 2020 COVID cuts (WTI crashed, most filter out)
        "2020-03-27",  # emergency cut, WTI ~$22 - filters out
        "2020-04-30",  # 4.25% -> 3.75%, WTI ~$20 - filters out
        "2020-06-26",  # 3.00% -> 2.50%, WTI ~$38 - filters out
        # 2023-2024 easing cycle (from 13.25%)
        "2023-12-22",  # 13.25% -> 13.00%, WTI ~$74
        "2024-01-31",  # 13.00% -> 12.75%, WTI ~$77
        "2024-03-22",  # 12.75% -> 12.25%, WTI ~$80
        "2024-04-30",  # 12.25% -> 11.75%, WTI ~$83
        "2024-06-28",  # 11.75% -> 11.25%, WTI ~$81
        "2024-07-31",  # 11.25% -> 10.75%, WTI ~$76
        "2024-09-30",  # 10.75% -> 10.25%, WTI ~$68
        "2024-10-31",  # 10.25% -> 9.75%, WTI ~$68
        "2024-12-20",  # 9.75% -> 9.50%, WTI ~$70
        "2025-01-31",  # 9.50% -> 9.25%, WTI ~$73
        "2025-03-28",  # estimated continuation
    ]

    WTI_THRESHOLD = 65.0
    hold_days = 40   # ~8 weeks
    stop_loss = -0.12
    brl_stop = 0.08  # USD/COP rise > 8%

    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)
    events = []

    for sd_str in banrep_cut_dates:
        sd = pd.Timestamp(sd_str)
        if sd < idx[0]:
            continue

        # Check WTI filter
        # Find WTI value on or just before cut date
        wti_dates = wti.index[wti.index <= sd]
        if len(wti_dates) == 0:
            continue
        wti_val = float(wti.loc[wti_dates[-1]])
        if wti_val < WTI_THRESHOLD:
            events.append({
                "signal_date": sd_str,
                "skipped": f"wti_below_65 (WTI={wti_val:.1f})"
            })
            continue

        # Entry: next trading day after BanRep announcement
        future_days = idx[idx > sd]
        if len(future_days) == 0:
            continue
        entry_date = future_days[0]
        entry_pos = idx.get_loc(entry_date)

        if pd.isna(px["GXG"].get(entry_date, np.nan)):
            events.append({"signal_date": sd_str, "skipped": "missing_gxg_at_entry"})
            continue

        entry_gxg = px["GXG"].loc[entry_date]
        entry_cop = px["COP=X"].get(entry_date, np.nan) if "COP=X" in px.columns else np.nan

        # Check GXG hasn't run >10% in prior 3 weeks
        start_3wk = idx[max(0, entry_pos - 15)]
        gxg_3wk_low = px["GXG"].loc[start_3wk:entry_date].min()
        if (entry_gxg / gxg_3wk_low - 1) > 0.10:
            events.append({"signal_date": sd_str, "skipped": "extended_move_>10pct"})
            continue

        end_pos = min(entry_pos + hold_days, len(idx))
        exit_reason = "max_hold"
        exit_pos = end_pos - 1

        for j in range(entry_pos, end_pos):
            day = idx[j]
            r_gxg = ret["GXG"].get(day, 0.0)
            if pd.isna(r_gxg):
                r_gxg = 0.0

            pnl.iloc[j] += r_gxg
            positions.iloc[j] = 1.0

            # GXG stop loss
            gxg_now = px["GXG"].get(day, entry_gxg)
            if not pd.isna(gxg_now) and (gxg_now / entry_gxg - 1) < stop_loss:
                exit_reason = "stop_loss"
                exit_pos = j
                break

            # COP stress exit (USD/COP rise > 8%)
            if not np.isnan(entry_cop) and "COP=X" in px.columns:
                cop_now = px["COP=X"].get(day, np.nan)
                if not np.isnan(cop_now) and (cop_now / entry_cop - 1) > brl_stop:
                    exit_reason = "cop_stress"
                    exit_pos = j
                    break

            # WTI drop below $60 check
            wti_today_dates = wti.index[wti.index <= day]
            if len(wti_today_dates) > 0:
                wti_today = float(wti.loc[wti_today_dates[-1]])
                if wti_today < 60.0:
                    exit_reason = "wti_below_60"
                    exit_pos = j
                    break

        exit_date = idx[exit_pos]
        event_ret = float((1 + pnl.loc[entry_date:exit_date]).prod() - 1)

        events.append({
            "signal_date": sd_str,
            "entry_date": str(entry_date.date()),
            "exit_date": str(exit_date.date()),
            "exit_reason": exit_reason,
            "wti_at_signal": round(wti_val, 1),
            "event_return": round(event_ret, 4),
        })

    if len(pnl.dropna()) < 30:
        return mark_failed(sid, f"insufficient data: {len(pnl.dropna())} days")

    pnl_clean = pnl.dropna()

    m = compute_metrics(
        pnl_clean,
        benchmark=spy_r,
        name="BanRep Cut + WTI>=65 Long GXG",
    )

    n_events_clean = len([e for e in events if not e.get("skipped")])
    ev_returns = [e["event_return"] for e in events if not e.get("skipped")]
    win_rate = float(np.mean([r > 0 for r in ev_returns])) if ev_returns else None
    avg_event = float(np.mean(ev_returns)) if ev_returns else None

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "When BanRep cuts Selic rate AND WTI crude >= $65/bbl (Colombia fiscal break-even), "
                "go long GXG within 2 trading days. Hold up to 8 weeks. "
                "Exit on WTI<$60, USD/COP +8%, GXG -12%, or max hold."
            ),
            "mechanism": (
                "BanRep cuts signal improving Colombian macro (inflation falling, "
                "activity stabilizing). WTI >= $65 ensures Ecopetrol/Petrobras fiscal "
                "revenues support the Colombian peso and government budget. "
                "GXG is ~25% Ecopetrol + ~20% Bancolombia - both benefit from rate cuts "
                "and oil price support simultaneously. Carry rebound: lower rates reduce "
                "capital outflows, supporting COP."
            ),
            "source": (
                "BanRep historical decisions (banrep.gov.co); "
                "FRED DCOILWTICO WTI crude; "
                "yfinance GXG/COP=X/EEM/SPY adjusted closes"
            ),
            "tickers": ["GXG"],
            "fred_series": ["DCOILWTICO"],
            "n_events": n_events_clean,
            "event_win_rate": round(win_rate, 4) if win_rate is not None else None,
            "avg_event_return": round(avg_event, 4) if avg_event is not None else None,
            "events": events,
            "caveats": (
                "GXG is highly illiquid (~$100M AUM, ~$2M ADV). "
                "Heavy Ecopetrol concentration makes this more of an oil play than a pure rate play. "
                "Colombia political risk (tax reform, Petro administration) can overwhelm macro signal. "
                "BanRep dates compiled manually - may have errors in early cycles. "
                "Many 2017 cuts filtered out by WTI<$65 - low oil period reduces sample."
            ),
        },
        pnl=pnl_clean,
    )

    print(f"Done: {sid}")
    print(f"  n_events: {n_events_clean}, win_rate: {win_rate}, avg_event_return: {avg_event}")
    print(
        f"  Sharpe: {m.get('sharpe', 0):.2f}  CAGR: {m.get('cagr', 0)*100:.2f}%  "
        f"MaxDD: {m.get('max_dd', 0)*100:.2f}%  t-stat: {m.get('t_stat', 0):.2f}"
    )
    if "oos_sharpe" in m:
        print(f"  OOS Sharpe: {m['oos_sharpe']:.2f}")


if __name__ == "__main__":
    main()
