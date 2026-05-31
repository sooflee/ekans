"""
PL627 — USD/JPY > 158 + JGB 10Y > 1.7% -> Short QQQ (BOJ Intervention Risk)

Entry:
  - DEXJPUS > 158
  - IRLTLT01JPM156N latest monthly > 1.70
  - DEXJPUS 20-day change >= +0.03 (3% USD/JPY rise)
Hold 10 trading days. Exit:
  - QQQ down >=4% (profit on short)
  - DEXJPUS drops 3% from in-trade peak (intervention)
  - QQQ up >=3% (stop)

Counter-signal against long_QQQ / long_semis.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


SIGNAL_ID = "PL627_mof_boj_intervention_short_qqq"
NAME = "USD/JPY>158 + JGB>1.7 + JPY-weak-20d -> Short QQQ (10d)"


def main():
    sid = SIGNAL_ID
    try:
        px = load_prices(["QQQ", "SPY"], start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load (prices): {e}")
    for t in ["QQQ", "SPY"]:
        if t not in px.columns:
            return mark_failed(sid, f"missing ticker {t}")

    try:
        fr = load_fred(["DEXJPUS", "IRLTLT01JPM156N"], start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load (FRED): {e}")

    for s in ["DEXJPUS", "IRLTLT01JPM156N"]:
        if s not in fr.columns:
            return mark_failed(sid, f"missing FRED series {s}")

    px = px.sort_index().ffill(limit=3)
    ret = daily_returns(px)
    qqq_r = ret["QQQ"]
    spy_r = ret["SPY"]
    trading_idx = ret.index

    fx = fr["DEXJPUS"].dropna()
    fx_20chg = fx / fx.shift(20) - 1.0
    fx_d = fx.reindex(trading_idx, method="ffill")
    fx_20chg_d = fx_20chg.reindex(trading_idx, method="ffill")

    jgb = fr["IRLTLT01JPM156N"].dropna()
    jgb_d = jgb.reindex(trading_idx, method="ffill")

    cond_fx = fx_d > 158
    cond_jgb = jgb_d > 1.70
    cond_carry = fx_20chg_d >= 0.03
    entry_signal = (cond_fx & cond_jgb & cond_carry).fillna(False)

    hold_days = 10
    take_profit = -0.04  # QQQ down -> short wins
    stop_loss = 0.03

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
        if pd.isna(px["QQQ"].iloc[entry_loc]):
            continue
        end_loc = min(entry_loc + hold_days, len(trading_idx) - 1)
        entry_price = px["QQQ"].iloc[entry_loc]
        qqq_win = px["QQQ"].iloc[entry_loc:end_loc + 1]
        cumret = qqq_win / entry_price - 1.0
        exit_loc = end_loc
        exit_reason = "max_hold"
        peak_fx = -np.inf
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
            cur_fx = fx_d.iloc[cur_i] if cur_i < len(fx_d) else np.nan
            if pd.notna(cur_fx):
                if cur_fx > peak_fx:
                    peak_fx = cur_fx
                if peak_fx > 0 and (peak_fx - cur_fx) / peak_fx >= 0.03:
                    exit_loc = cur_i
                    exit_reason = "intervention_proxy"
                    break
        positions.iloc[entry_loc:exit_loc + 1] = -1.0
        last_exit_loc = exit_loc
        events.append({
            "trigger_date": str(dt.date()),
            "entry_date": str(trading_idx[entry_loc].date()),
            "exit_date": str(trading_idx[exit_loc].date()),
            "exit_reason": exit_reason,
            "qqq_return": float(cumret.iloc[min(exit_loc - entry_loc, len(cumret) - 1)]),
            "dexjpus_at_trigger": float(fx_d.iloc[i]),
            "fx_20d_at_trigger": float(fx_20chg_d.iloc[i]),
            "jgb_at_trigger": float(jgb_d.iloc[i]),
        })

    if not events:
        return mark_failed(
            sid,
            "no USD/JPY>158 + JGB>1.7 events fired",
            extra={
                "rule": "Short QQQ 10d on USD/JPY>158 + JGB>1.7 + 20d carry rally",
                "mechanism": "Intervention risk + JGB selloff unwinds yen-carry-funded long tech",
                "source": "PL627 catalog",
            },
        )

    pnl = positions.shift(1).fillna(0.0) * qqq_r.reindex(positions.index).fillna(0.0)
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
        cost_bps=8,
    )
    m["n_events"] = len(events)
    m["pct_in_market"] = float(active_pos.abs().mean())
    m["events"] = events
    m["status"] = "ok"

    save_result(
        sid,
        m,
        extra={
            "rule": "Short QQQ at next-day close when DEXJPUS > 158 AND IRLTLT01JPM156N (JP 10y yield) > 1.70 AND DEXJPUS up >=3% over trailing 20 trading days. Hold up to 10 trading days; exit on QQQ -4% (profit), QQQ +3% (stop), or DEXJPUS drops 3% from in-trade peak (intervention proxy).",
            "mechanism": "USD/JPY > 158 with JGB yields rising and carry trade still expanding marks the point at which MOF/BOJ intervention risk peaks. Historical episodes (2022-10, 2024-04, 2024-07) show that intervention coincides with sharp yen-carry unwind which slams long-tech/QQQ via cross-asset deleveraging. Counter-signal against long_QQQ / long_semis defaults.",
            "source": "PL627 idea catalog; FRED DEXJPUS + IRLTLT01JPM156N",
            "caveats": "USD/JPY > 158 is rare so small N. Monthly JGB yield series is forward-filled. Counter-signal against widely-held long-tech bias.",
        },
        pnl=pnl,
    )
    print(f"Saved {sid}: n_events={len(events)}  Sharpe={m.get('sharpe',0):.2f}  CAGR={m.get('cagr',0)*100:.2f}%")


if __name__ == "__main__":
    main()
