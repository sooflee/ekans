"""
PL633 — Offshore Driller Relative Strength Proxy -> Long OIH

Entry:
  - RIG, VAL, NE all close above their respective 200d SMA
  - (RIG+VAL+NE) equal-weight basket 60d return >= +15%
  - OIH close <= 1.20 * OIH 200d SMA (not chasing)
Hold up to 100 trading days. Exit:
  - OIH +15% from entry (profit)
  - any of RIG/VAL/NE closes < 200d SMA for 10 consecutive days (regime break)
  - OIH -10% from entry (stop)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics, save_result,
                     mark_failed, daily_returns)


SIGNAL_ID = "PL633_gom_offshore_inflection_long_oih"
NAME = "Offshore Drillers Joint Breakout -> Long OIH (100d)"


def main():
    sid = SIGNAL_ID
    try:
        px = load_prices(["OIH", "RIG", "VAL", "NE", "SPY"], start="2018-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")
    for t in ["OIH", "RIG", "VAL", "NE", "SPY"]:
        if t not in px.columns:
            return mark_failed(sid, f"missing ticker {t}")

    px = px.sort_index().ffill(limit=3)
    ret = daily_returns(px)
    spy_r = ret["SPY"]
    trading_idx = ret.index

    def sma(s, n):
        return s.rolling(n, min_periods=max(20, n // 4)).mean()

    oih_sma200 = sma(px["OIH"], 200)
    rig_sma200 = sma(px["RIG"], 200)
    val_sma200 = sma(px["VAL"], 200)
    ne_sma200 = sma(px["NE"], 200)

    cond_rig = px["RIG"] > rig_sma200
    cond_val = px["VAL"] > val_sma200
    cond_ne = px["NE"] > ne_sma200
    basket = (px["RIG"] / px["RIG"].shift(1) - 1.0)  # placeholder unused
    # 60d return of basket: use equal-weighted price index
    basket_px = (px["RIG"] / px["RIG"].iloc[0] +
                 px["VAL"] / px["VAL"].dropna().iloc[0] if not px["VAL"].dropna().empty else px["VAL"]) / 2
    # Simpler: arithmetic mean of three trailing 60d returns
    rig_60 = px["RIG"] / px["RIG"].shift(60) - 1.0
    val_60 = px["VAL"] / px["VAL"].shift(60) - 1.0
    ne_60 = px["NE"] / px["NE"].shift(60) - 1.0
    basket_60 = (rig_60 + val_60 + ne_60) / 3.0

    oih_over_sma = px["OIH"] / oih_sma200

    cond_basket_60 = basket_60 >= 0.15
    cond_chase = oih_over_sma <= 1.20

    entry_signal = (cond_rig & cond_val & cond_ne & cond_basket_60 & cond_chase).fillna(False)

    oih_r = ret["OIH"]
    oih_r_d = oih_r.reindex(trading_idx).fillna(0.0)

    hold_days = 100
    take_profit = 0.15
    stop_loss = -0.10

    positions = pd.Series(0.0, index=trading_idx)
    events = []
    last_exit_loc = -1

    for i, dt in enumerate(trading_idx):
        if i <= last_exit_loc:
            continue
        if not entry_signal.iloc[i]:
            continue
        entry_loc = i + 1
        if entry_loc >= len(trading_idx) - 1:
            continue
        if pd.isna(px["OIH"].iloc[entry_loc]):
            continue
        end_loc = min(entry_loc + hold_days, len(trading_idx) - 1)
        entry_price = px["OIH"].iloc[entry_loc]
        oih_win = px["OIH"].iloc[entry_loc:end_loc + 1]
        cumret = oih_win / entry_price - 1.0
        exit_loc = end_loc
        exit_reason = "max_hold"
        below_count = 0
        for k, val in enumerate(cumret.values):
            cur_i = entry_loc + k
            if pd.notna(val) and val >= take_profit:
                exit_loc = cur_i
                exit_reason = "profit_target"
                break
            if pd.notna(val) and val <= stop_loss:
                exit_loc = cur_i
                exit_reason = "stop_loss"
                break
            # any of RIG/VAL/NE < its 200d SMA today
            below_today = False
            for name in ["RIG", "VAL", "NE"]:
                psma = {"RIG": rig_sma200, "VAL": val_sma200, "NE": ne_sma200}[name]
                if cur_i < len(px[name]) and pd.notna(px[name].iloc[cur_i]) and pd.notna(psma.iloc[cur_i]):
                    if px[name].iloc[cur_i] < psma.iloc[cur_i]:
                        below_today = True
                        break
            if below_today:
                below_count += 1
                if below_count >= 10:
                    exit_loc = cur_i
                    exit_reason = "regime_break"
                    break
            else:
                below_count = 0

        positions.iloc[entry_loc:exit_loc + 1] = 1.0
        last_exit_loc = exit_loc
        events.append({
            "trigger_date": str(dt.date()),
            "entry_date": str(trading_idx[entry_loc].date()),
            "exit_date": str(trading_idx[exit_loc].date()),
            "exit_reason": exit_reason,
            "oih_return": float(cumret.iloc[min(exit_loc - entry_loc, len(cumret) - 1)]),
            "basket_60d_at_trigger": float(basket_60.iloc[i]) if pd.notna(basket_60.iloc[i]) else None,
        })

    if not events:
        return mark_failed(
            sid,
            "no joint offshore-driller breakout events fired",
            extra={
                "rule": "Long OIH 100d on RIG/VAL/NE joint breakout + basket 60d +15%",
                "mechanism": "Offshore-pure-play breakout signals deepwater dayrate visibility",
                "source": "PL633 catalog",
            },
        )

    pnl = positions.shift(1).fillna(0.0) * oih_r_d
    pnl = pnl.dropna()
    first_entry = pd.Timestamp(events[0]["entry_date"])
    pnl = pnl.loc[pnl.index >= first_entry]
    active_pos = positions.loc[positions.index >= first_entry]

    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient PnL: {len(pnl)} days")

    m = compute_metrics(
        pnl,
        benchmark=spy_r.reindex(pnl.index),
        name=NAME,
        positions=active_pos,
        cost_bps=10,
    )
    m["n_events"] = len(events)
    m["pct_in_market"] = float(active_pos.mean())
    m["events"] = events
    m["status"] = "ok"

    save_result(
        sid,
        m,
        extra={
            "rule": "Go long OIH at next-day close when RIG, VAL, NE all close above their 200d SMA AND the equal-weight 60-day return of (RIG+VAL+NE) >= +15% AND OIH close <= 1.20 * OIH 200d SMA. Hold up to 100 trading days; exit on +15% (profit), -10% (stop), or one of RIG/VAL/NE closes below its 200d SMA for 10 consecutive days (regime break).",
            "mechanism": "Joint breakout of offshore-pure-play drillers (Transocean / Valaris / Noble) above 200d SMA with strong 60d basket return signals an inflection in BSEE GOM rig demand / dayrate visibility. OIH (oilfield services ETF) re-rates over multi-quarter horizons as the same fundamentals flow through to broader OFS revenues.",
            "source": "PL633 idea catalog; yfinance",
            "caveats": "BSEE OGOR data not in FRED — driller relative strength is the proxy. VAL/NE post-2021-BK so effective sample is 2021+ which is small. Single-direction long-bias strategy.",
        },
        pnl=pnl,
    )
    print(f"Saved {sid}: n_events={len(events)}  Sharpe={m.get('sharpe',0):.2f}  CAGR={m.get('cagr',0)*100:.2f}%")


if __name__ == "__main__":
    main()
