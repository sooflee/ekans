"""PL769_opec_cut_stng_long_fro_short — OPEC+ Surprise Cut -> Long STNG / Short FRO Product-vs-Crude Tanker Pair"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL769_opec_cut_stng_long_fro_short"

    # Hand-coded OPEC+ surprise cut announcement dates
    # (actual cut > 200 kb/d above consensus)
    # Entry is T+1 after announcement
    announcement_dates = [
        ("2020-04-12", "OPEC+ historic 9.7mb/d cut agreement"),
        ("2020-06-06", "OPEC+ 2020 cut extension agreement"),
        ("2022-10-05", "OPEC+ 2mb/d cut announcement"),
        ("2023-04-02", "OPEC+ voluntary surprise cut 1.16mb/d"),
        ("2023-11-26", "OPEC+ Nov 2023 additional voluntary cuts"),
        ("2024-06-02", "OPEC+ Jun 2024 meeting voluntary cut extension"),
        ("2025-05-04", "OPEC+ May 2025 production adjustment"),
    ]
    # Entry at T+1 (next trading day after announcement)

    try:
        px = load_prices(["STNG", "FRO", "INSW", "SPY"], start="2018-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if px.empty or "STNG" not in px.columns:
        return mark_failed(sid, "STNG price data unavailable")

    # Check if FRO is available; fall back to INSW if needed
    if "FRO" not in px.columns or px["FRO"].dropna().empty:
        if "INSW" in px.columns and not px["INSW"].dropna().empty:
            print("FRO unavailable, using INSW as crude tanker proxy")
            px["FRO"] = px["INSW"]
        else:
            return mark_failed(sid, "Both FRO and INSW price data unavailable")

    ret = daily_returns(px)
    stng_r = ret["STNG"].dropna()
    fro_r = ret["FRO"].dropna()
    spy_r = ret["SPY"].dropna()

    # Align STNG and FRO to common index
    common_idx = stng_r.index.intersection(fro_r.index)
    stng_r = stng_r.reindex(common_idx)
    fro_r = fro_r.reindex(common_idx)

    if len(common_idx) < 60:
        return mark_failed(sid, f"insufficient common data: {len(common_idx)} days")

    HOLD_CAL = 30  # 30 calendar days
    STOP_LOSS = -0.08  # -8% pair spread triggers stop
    PROFIT_TARGET = 0.15  # +15% pair spread profit take

    pnl = pd.Series(0.0, index=common_idx)
    evts = []

    for announce_str, label in announcement_dates:
        announce_date = pd.Timestamp(announce_str)

        # Entry: first trading day AFTER announcement date
        after_announce = common_idx[common_idx > announce_date]
        if len(after_announce) == 0:
            continue

        entry_idx = after_announce[0]
        entry_pos = common_idx.get_loc(entry_idx)

        # Exit: 30 calendar days after entry
        exit_date = entry_idx + pd.Timedelta(days=HOLD_CAL)
        exit_candidates = common_idx[common_idx >= exit_date]
        if len(exit_candidates) == 0:
            exit_pos = len(common_idx)
        else:
            exit_pos = common_idx.get_loc(exit_candidates[0])

        if exit_pos <= entry_pos:
            continue

        # Compute pair return: long STNG, short FRO
        stng_window = stng_r.iloc[entry_pos:exit_pos]
        fro_window = fro_r.iloc[entry_pos:exit_pos]

        if len(stng_window) < 3:
            continue

        # Equal-notional pair: (stng - fro) / 2
        pair_daily = (stng_window - fro_window) / 2

        # Apply stop-loss and profit target
        cum_spread = (1 + pair_daily).cumprod() - 1
        stop_idx = None
        profit_idx = None
        for k, cs in enumerate(cum_spread):
            if cs <= STOP_LOSS:
                stop_idx = k + 1
                break
            if cs >= PROFIT_TARGET:
                profit_idx = k + 1
                break

        exit_reason = "hold_complete"
        if stop_idx is not None:
            pair_daily = pair_daily.iloc[:stop_idx]
            exit_reason = "stop_loss"
        elif profit_idx is not None:
            pair_daily = pair_daily.iloc[:profit_idx]
            exit_reason = "profit_target"

        actual_exit_pos = entry_pos + len(pair_daily)
        pnl.iloc[entry_pos:actual_exit_pos] = pair_daily.values

        stng_cum = float((1 + stng_window.iloc[:len(pair_daily)]).prod() - 1)
        fro_cum = float((1 + fro_window.iloc[:len(pair_daily)]).prod() - 1)
        pair_cum = float((1 + pair_daily).prod() - 1)
        spy_w = spy_r.reindex(pair_daily.index)
        spy_cum = float((1 + spy_w).prod() - 1) if len(spy_w) > 0 else None

        evts.append({
            "announcement": announce_str,
            "label": label,
            "entry_date": str(entry_idx.date()),
            "n_days": len(pair_daily),
            "exit_reason": exit_reason,
            "stng_return": round(stng_cum, 4),
            "fro_return": round(fro_cum, 4),
            "pair_return": round(pair_cum, 4),
            "spy_return": round(spy_cum, 4) if spy_cum is not None else None,
        })

    print(f"Events: {len(evts)}")
    if not evts:
        return mark_failed(sid, "no valid events found")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 30:
        return mark_failed(sid, f"insufficient active days ({len(active_pnl)})")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="OPEC+ Surprise Cut -> Long STNG / Short FRO")
    returns_list = [e["pair_return"] for e in evts]

    save_result(sid, m, extra={
        "rule": "On OPEC+ surprise cut >200kb/d: enter long STNG / short FRO equal-notional at T+1, hold 30 calendar days or stop/profit",
        "mechanism": "OPEC cuts reduce crude supply -> crude tanker (VLCC) rates fall while product tanker demand holds; product tankers decouple from crude shipping",
        "source": "OPEC press releases (opec.org); yfinance STNG, FRO, SPY",
        "n_events": len(evts),
        "avg_event_return": round(float(np.mean(returns_list)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in returns_list])), 4),
        "events": evts,
    })

    print(f"Done: {len(evts)} events, Sharpe={m.get('sharpe', 'N/A'):.2f}, CAGR={m.get('cagr', 0)*100:.1f}%")
    for e in evts:
        print(f"  {e['announcement']} ({e['label'][:30]}): pair={e['pair_return']:.1%} ({e['exit_reason']})")


if __name__ == "__main__":
    main()
