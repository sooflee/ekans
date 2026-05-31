"""
PL625 — Credit Stress + Bank Deposit Outflow Proxy -> Long KBE / Short KRE Pair

Entry:
  - BAMLH0A0HYM2 20d change >= +0.75 (HY OAS widening 75bps+)
  - H8B1058NCBCMG (Fed H.8 small-bank deposits, weekly) 4-week change < 0
  - KRE 30d - KBE 30d >= -0.02 (divergence not yet priced)
Hold 50 trading days. Exit:
  - +5% pair (profit)
  - HY OAS down >50bps from peak during trade (regime exit)
  - -3% pair (stop)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


SIGNAL_ID = "PL625_credit_stress_long_kbe_short_kre"
NAME = "HY +75bps + H8-Deposits Out -> Long KBE / Short KRE (50d pair)"


def main():
    sid = SIGNAL_ID
    try:
        px = load_prices(["KBE", "KRE", "SPY"], start="2007-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load (prices): {e}")
    for t in ["KBE", "KRE", "SPY"]:
        if t not in px.columns:
            return mark_failed(sid, f"missing ticker {t}")

    try:
        fr = load_fred(["BAMLH0A0HYM2", "H8B1058NCBCMG"], start="2007-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load (FRED): {e}")

    for s in ["BAMLH0A0HYM2", "H8B1058NCBCMG"]:
        if s not in fr.columns:
            return mark_failed(sid, f"missing FRED series {s}")

    px = px.sort_index().ffill(limit=3)
    ret = daily_returns(px)
    spy_r = ret["SPY"]
    trading_idx = ret.index

    hy = fr["BAMLH0A0HYM2"].dropna()
    hy_20chg = hy - hy.shift(20)

    h8 = fr["H8B1058NCBCMG"].dropna()
    # H.8 weekly, take 4-week change
    h8_4w = h8.diff(4)

    hy_d = hy.reindex(trading_idx, method="ffill")
    hy_20chg_d = hy_20chg.reindex(trading_idx, method="ffill")
    h8_4w_d = h8_4w.reindex(trading_idx, method="ffill")

    kbe_30 = px["KBE"] / px["KBE"].shift(30) - 1.0
    kre_30 = px["KRE"] / px["KRE"].shift(30) - 1.0
    div_30 = kre_30 - kbe_30  # we want this >= -0.02

    cond_hy_widen = hy_20chg_d >= 0.75
    cond_h8 = h8_4w_d < 0
    cond_div = div_30 >= -0.02
    entry_signal = (cond_hy_widen & cond_h8 & cond_div).fillna(False)

    pair_r = (ret["KBE"] - ret["KRE"]).dropna()
    pair_r_d = pair_r.reindex(trading_idx).fillna(0.0)

    hold_days = 50
    take_profit = 0.05
    stop_loss = -0.03

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
        if pd.isna(px["KBE"].iloc[entry_loc]) or pd.isna(px["KRE"].iloc[entry_loc]):
            continue
        end_loc = min(entry_loc + hold_days, len(trading_idx) - 1)
        bw = pair_r_d.iloc[entry_loc:end_loc + 1]
        cumret = (1 + bw).cumprod() - 1
        exit_loc = end_loc
        exit_reason = "max_hold"
        hy_window = hy_d.iloc[entry_loc:end_loc + 1]
        peak_hy = -np.inf
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
            if cur_i < len(hy_d) and pd.notna(hy_d.iloc[cur_i]):
                if hy_d.iloc[cur_i] > peak_hy:
                    peak_hy = hy_d.iloc[cur_i]
                if peak_hy - hy_d.iloc[cur_i] >= 0.50:
                    exit_loc = cur_i
                    exit_reason = "regime_exit"
                    break
        positions.iloc[entry_loc:exit_loc + 1] = 1.0
        last_exit_loc = exit_loc
        events.append({
            "trigger_date": str(dt.date()),
            "entry_date": str(trading_idx[entry_loc].date()),
            "exit_date": str(trading_idx[exit_loc].date()),
            "exit_reason": exit_reason,
            "pair_return": float(cumret.iloc[min(exit_loc - entry_loc, len(cumret) - 1)]),
            "hy_20d_chg_at_trigger": float(hy_20chg_d.iloc[i]) if pd.notna(hy_20chg_d.iloc[i]) else None,
            "h8_4w_at_trigger": float(h8_4w_d.iloc[i]) if pd.notna(h8_4w_d.iloc[i]) else None,
        })

    if not events:
        return mark_failed(
            sid,
            "no joint HY-widen + H8-out events fired",
            extra={
                "rule": "Pair long KBE / short KRE 50d on credit-stress + deposit-outflow regime",
                "mechanism": "Regional bank deposit flight + HY widening favors money-center banks vs regionals",
                "source": "PL625 catalog",
            },
        )

    pnl = positions.shift(1).fillna(0.0) * pair_r_d
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
    m["events"] = events[:30]
    m["status"] = "ok"

    save_result(
        sid,
        m,
        extra={
            "rule": "Open long KBE / short KRE pair at next-day close when (a) FRED BAMLH0A0HYM2 has widened by >=75bps over trailing 20 trading days, (b) FRED H8B1058NCBCMG (small commercial bank deposits) 4-week change < 0, and (c) KRE 30d - KBE 30d >= -0.02 (entry before divergence is fully priced). Hold up to 50 trading days; exit on +5% pair (profit), -3% pair (stop), or HY OAS down >=50bps from in-trade peak (regime exit).",
            "mechanism": "Credit-stress widening (BAMLH0A0HYM2 +75bps) combined with small-bank deposit outflows (H.8 4-week negative) replicates the dynamic of 2008Q4 and 2023-Q1 (SVB/FRC) when regional banks underperformed center banks structurally due to deposit-flight and unrealized loss concerns. KBE includes more money-center / custody-bank exposure (BK, STT, JPM-tier) while KRE is heavily regional. Pair captures relative regional underperformance.",
            "source": "PL625 idea catalog; FRED BAMLH0A0HYM2 + H8B1058NCBCMG",
            "caveats": "FDIC Call Report uninsured-deposit-share signal substituted with H.8 small-bank deposit aggregate. Counter-signal against long_XLF / long_KRE / long_SPY_financials defaults. Small N around 2008Q4 and 2023-Q1 anchors.",
        },
        pnl=pnl,
    )
    print(f"Saved {sid}: n_events={len(events)}  Sharpe={m.get('sharpe',0):.2f}  CAGR={m.get('cagr',0)*100:.2f}%")


if __name__ == "__main__":
    main()
