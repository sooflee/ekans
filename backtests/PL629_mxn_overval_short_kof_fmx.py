"""
PL629 — MXN Overvaluation + USD/MXN Reversal Proxy -> Short KOF/FMX

Entry:
  - DEXMXUS z-score over trailing 1260 trading days <= -1.0 (MXN richly valued)
  - DEXMXUS 20-day change >= +0.02 (early reversal: USD/MXN rising)
Position: short equal-weight 0.5*KOF + 0.5*FMX.
Exit (earliest of):
  - 50 trading days
  - basket cumret <= -8% (profit on short)
  - DEXMXUS z-score crosses above 0 (regime exit)
  - basket cumret >= +6% (stop)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


SIGNAL_ID = "PL629_mxn_overval_short_kof_fmx"
NAME = "DEXMXUS z<=-1 + 20d rev >=2% -> Short KOF/FMX (50d basket)"


def main():
    sid = SIGNAL_ID
    try:
        px = load_prices(["KOF", "FMX", "SPY"], start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load (prices): {e}")
    for t in ["KOF", "FMX", "SPY"]:
        if t not in px.columns:
            return mark_failed(sid, f"missing ticker {t}")

    try:
        fr = load_fred(["DEXMXUS"], start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load (FRED): {e}")
    if "DEXMXUS" not in fr.columns:
        return mark_failed(sid, "DEXMXUS not returned")

    px = px.sort_index().ffill(limit=3)
    ret = daily_returns(px)
    spy_r = ret["SPY"]
    trading_idx = ret.index

    fx = fr["DEXMXUS"].dropna()
    fx_sma = fx.rolling(1260, min_periods=252).mean()
    fx_std = fx.rolling(1260, min_periods=252).std()
    z = (fx - fx_sma) / fx_std
    fx_20chg = fx / fx.shift(20) - 1.0

    z_d = z.reindex(trading_idx, method="ffill")
    fx_d = fx.reindex(trading_idx, method="ffill")
    fx_20chg_d = fx_20chg.reindex(trading_idx, method="ffill")

    basket_r = (0.5 * ret["KOF"] + 0.5 * ret["FMX"]).dropna()
    basket_r_d = basket_r.reindex(trading_idx).fillna(0.0)

    cond_z = z_d <= -1.0
    cond_rev = fx_20chg_d >= 0.02
    entry_signal = (cond_z & cond_rev).fillna(False)

    hold_days = 50
    take_profit = -0.08
    stop_loss = 0.06

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
        if pd.isna(px["KOF"].iloc[entry_loc]) or pd.isna(px["FMX"].iloc[entry_loc]):
            continue
        end_loc = min(entry_loc + hold_days, len(trading_idx) - 1)
        bw = basket_r_d.iloc[entry_loc:end_loc + 1]
        cumret = (1 + bw).cumprod() - 1
        exit_loc = end_loc
        exit_reason = "max_hold"
        for k, val in enumerate(cumret.values):
            cur_i = entry_loc + k
            if pd.notna(val) and val <= take_profit:
                exit_loc = cur_i
                exit_reason = "profit_target"
                break
            if pd.notna(val) and val >= stop_loss:
                exit_loc = cur_i
                exit_reason = "stop_loss"
                break
            if cur_i < len(z_d) and pd.notna(z_d.iloc[cur_i]) and z_d.iloc[cur_i] > 0:
                exit_loc = cur_i
                exit_reason = "regime_exit"
                break
        positions.iloc[entry_loc:exit_loc + 1] = -1.0
        last_exit_loc = exit_loc
        events.append({
            "trigger_date": str(dt.date()),
            "entry_date": str(trading_idx[entry_loc].date()),
            "exit_date": str(trading_idx[exit_loc].date()),
            "exit_reason": exit_reason,
            "basket_return": float(cumret.iloc[min(exit_loc - entry_loc, len(cumret) - 1)]),
            "z_at_trigger": float(z_d.iloc[i]) if pd.notna(z_d.iloc[i]) else None,
            "fx_20d_at_trigger": float(fx_20chg_d.iloc[i]) if pd.notna(fx_20chg_d.iloc[i]) else None,
        })

    if not events:
        return mark_failed(
            sid,
            "no MXN overvaluation + reversal events fired",
            extra={
                "rule": "Short KOF/FMX 50d on DEXMXUS z<=-1 + 20d rev >=2%",
                "mechanism": "MXN overvaluation + early reversal pressures peso-denominated EBITDA",
                "source": "PL629 catalog",
            },
        )

    pnl = positions.shift(1).fillna(0.0) * basket_r_d
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
        positions=active_pos.abs(),
        cost_bps=15,
    )
    m["n_events"] = len(events)
    m["pct_in_market"] = float(active_pos.abs().mean())
    m["events"] = events[:30]
    m["status"] = "ok"

    save_result(
        sid,
        m,
        extra={
            "rule": "Short equal-weight (0.5*KOF + 0.5*FMX) at next-day close when DEXMXUS z-score over trailing 1260 trading days (5y) <= -1.0 AND DEXMXUS 20-day change >= +0.02 (early reversal). Hold up to 50 trading days; exit on -8% basket (profit), +6% basket (stop), or z-score crosses above 0 (regime exit).",
            "mechanism": "MXN strongly overvalued vs USD (DEXMXUS far below 5y mean) maps to Banxico CR99 REER above 5y mean (proxy). Once USD/MXN starts to rise (early-reversal trigger), the peso weakens, compressing peso-denominated EBITDA of KOF (Coca-Cola FEMSA) and FMX (FEMSA holding) when translated to USD ADR equity. Specific-name basket targets the FX-translation channel without broad EWW country beta.",
            "source": "PL629 idea catalog; FRED DEXMXUS",
            "caveats": "Banxico CR99 REER not in FRED — DEXMXUS z-score substitutes. Specific-name basket carries idiosyncratic risk (e.g. KOF beverage volume).",
        },
        pnl=pnl,
    )
    print(f"Saved {sid}: n_events={len(events)}  Sharpe={m.get('sharpe',0):.2f}  CAGR={m.get('cagr',0)*100:.2f}%")


if __name__ == "__main__":
    main()
