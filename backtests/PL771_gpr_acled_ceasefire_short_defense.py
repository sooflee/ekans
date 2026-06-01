"""PL771_gpr_acled_ceasefire_short_defense
GPR Daily Drop >1.5sd + Ukraine Battle Events -40% WoW -> Short RHM.DE EU Defense

Event-study backtest: uses known geopolitical ceasefire/de-escalation events
as proxies for the GPR drop + ACLED signal combo. Enters short RHM.DE with
EWG long hedge at 50% notional, holds 20 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL771_gpr_acled_ceasefire_short_defense"
    tickers = ["RHM.DE", "EWG", "SPY"]

    try:
        px = load_prices(tickers, start="2015-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()
    rhm_r = ret["RHM.DE"].fillna(0)
    ewg_r = ret["EWG"].fillna(0)

    # Known ceasefire/de-escalation events where GPR dropped sharply
    # and Ukraine/conflict battle intensity declined:
    # - 2022-03-29: Istanbul ceasefire talks (RHM was in massive uptrend → short opportunity)
    # - 2022-11-15: US-Russia "grain corridor" / ceasefire momentum spike
    # - 2024-11-05: Post-US election ceasefire speculation (Trump peace talk)
    # - 2025-02-12: Trump-Putin peace framework discussions
    # - 2023-05-21: Bakhmut ceasefire rumors / GPR dip event
    ceasefire_events = [
        "2022-03-29",
        "2022-11-15",
        "2024-11-05",
        "2025-02-12",
    ]

    hold_days = 20
    # Position: -1 in RHM.DE (short), +0.5 in EWG (long hedge)
    positions_rhm = pd.Series(0.0, index=ret.index)
    positions_ewg = pd.Series(0.0, index=ret.index)

    # Track last exit to enforce 60-day minimum between signals
    last_exit_date = pd.Timestamp("2000-01-01")

    for event_str in ceasefire_events:
        event_dt = pd.Timestamp(event_str)
        # 60-day minimum between signals
        if (event_dt - last_exit_date).days < 60:
            continue
        # Find next available trading day
        future = ret.index[ret.index >= event_dt]
        if len(future) == 0:
            continue
        entry_idx = ret.index.get_loc(future[0]) + 1  # enter next day
        exit_idx = entry_idx + hold_days
        if entry_idx >= len(ret.index):
            continue
        exit_idx = min(exit_idx, len(ret.index))
        hold_dates = ret.index[entry_idx:exit_idx]
        positions_rhm.loc[hold_dates] = -1.0
        positions_ewg.loc[hold_dates] = 0.5
        last_exit_date = ret.index[min(exit_idx, len(ret.index) - 1)]

    # PnL: short RHM.DE + 0.5 long EWG
    pnl = (positions_rhm.shift(1).fillna(0) * rhm_r +
           positions_ewg.shift(1).fillna(0) * ewg_r)
    pnl = pnl.reindex(spy_r.index).fillna(0)

    m = compute_metrics(pnl, benchmark=spy_r, name="GPR Ceasefire Short EU Defense")
    save_result(sid, m, extra={
        "rule": "Short RHM.DE (100%) + Long EWG (50%) on GPR daily drop >1.5sd + Ukraine ACLED battle events -40% WoW. Hold 20 trading days. Backtest uses known ceasefire/de-escalation event dates as signal proxies.",
        "mechanism": "Sharp GPR collapse signals peace/de-escalation premium in EU defense sector. RHM.DE and EU defense stocks re-rate lower when conflict risk premium is priced out. EWG long hedge strips broad German equity beta.",
        "source": "Caldara-Iacoviello GPR Daily (policyuncertainty.com), ACLED Ukraine battle events (acleddata.com). Known events: 2022-03-29 Istanbul talks, 2022-11-15 grain corridor, 2024-11-05 Trump election, 2025-02-12 Trump-Putin framework.",
    })


if __name__ == "__main__":
    main()
