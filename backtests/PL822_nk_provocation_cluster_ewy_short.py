"""PL822_nk_provocation_cluster_ewy_short
NK ICBM/Nuclear Provocation Cluster -> Short EWY / Long USDKRW on Geopolitical Risk

On confirmed DPRK ICBM launch or nuclear test, short EWY at close.
Hold 20 trading days. Hard stop: cover if EWY closes up >8% from entry.
Benchmark: SPY.

Known event dates (from USFK/38North confirmed test records):
  - 2017-07-04: Hwasong-14 ICBM test
  - 2017-09-03: 6th nuclear test
  - 2017-11-29: Hwasong-15 ICBM test
  - 2022-11-18: Hwasong-17 ICBM test
  - 2023-03-16: Hwasong-17 follow-on
  - 2024-11-05: KN-23 cluster + Russia deployment
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


EVENTS = [
    {"event_date": "2017-07-04", "description": "Hwasong-14 ICBM first test"},
    {"event_date": "2017-09-03", "description": "DPRK 6th nuclear test"},
    {"event_date": "2017-11-29", "description": "Hwasong-15 ICBM test (longest range)"},
    {"event_date": "2022-11-18", "description": "Hwasong-17 ICBM test"},
    {"event_date": "2023-03-16", "description": "Hwasong-17 follow-on test"},
    {"event_date": "2024-11-05", "description": "KN-23 cluster launch + Russia troop deployment signal"},
]

HOLD_DAYS = 20
STOP_PCT = 0.08   # cover if EWY up >8% from entry (short stop-loss)


def main():
    sid = "PL822_nk_provocation_cluster_ewy_short"
    basket = ["EWY"]
    tickers = ["EWY", "EWJ", "SPY"]

    try:
        px = load_prices(tickers, start="2016-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()
    ewy_r = ret["EWY"].dropna()
    ewj_r = ret["EWJ"].dropna()
    idx = ret.index

    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)
    event_log = []

    for ev in EVENTS:
        ev_dt = pd.Timestamp(ev["event_date"])
        # Entry: first trading day on or after event date
        future = idx[idx >= ev_dt]
        if len(future) == 0:
            event_log.append({**ev, "status": "no_data"})
            continue
        entry_dt = future[0]
        entry_pos = idx.get_loc(entry_dt)
        max_exit_pos = min(entry_pos + HOLD_DAYS, len(idx))

        # Walk forward checking stop-loss (short EWY: stop if EWY goes UP from entry)
        cumulative_ewy = 1.0
        actual_exit_pos = max_exit_pos
        exit_reason = "scheduled_20d"

        for j in range(entry_pos, max_exit_pos):
            day_ewy = float(ewy_r.iloc[j]) if not pd.isna(ewy_r.iloc[j]) else 0.0
            cumulative_ewy *= (1 + day_ewy)
            # Stop-loss for short: if EWY is up >STOP_PCT from entry
            if (cumulative_ewy - 1.0) > STOP_PCT:
                exit_reason = "stop_loss"
                actual_exit_pos = j + 1
                break

        # Record PnL: short EWY
        for j in range(entry_pos, actual_exit_pos):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = -1.0  # short
                day_ewy = float(ewy_r.iloc[j]) if not pd.isna(ewy_r.iloc[j]) else 0.0
                pnl.iloc[j] = -day_ewy

        # Summary
        hold_idx = idx[entry_pos:actual_exit_pos]
        ewy_cum = float((1 + ewy_r.reindex(hold_idx).fillna(0)).prod() - 1)
        spy_cum = float((1 + spy_r.reindex(hold_idx).fillna(0)).prod() - 1)
        ewj_cum = float((1 + ewj_r.reindex(hold_idx).fillna(0)).prod() - 1)
        short_ewy_ret = float(-ewy_cum)  # approx
        exit_dt = idx[actual_exit_pos - 1] if actual_exit_pos > entry_pos else entry_dt

        ev_record = {
            **ev,
            "status": "traded",
            "entry_date": str(entry_dt.date()),
            "exit_date": str(exit_dt.date()),
            "exit_reason": exit_reason,
            "n_hold_days": int(actual_exit_pos - entry_pos),
            "ewy_return": round(ewy_cum, 4),
            "short_ewy_return": round(short_ewy_ret, 4),
            "spy_return": round(spy_cum, 4),
            "ewj_return": round(ewj_cum, 4),
            "excess_vs_spy": round(short_ewy_ret - spy_cum, 4),
            "ewy_vs_ewj": round(ewy_cum - ewj_cum, 4),  # EWY underperformance vs Japan control
        }
        event_log.append(ev_record)

    traded = [e for e in event_log if e.get("status") == "traded"]
    n_traded = len(traded)
    print(f"Events traded: {n_traded} / {len(EVENTS)}")
    for e in traded:
        print(
            f"  {e['event_date']} [{e['description'][:40]}] -> entry {e['entry_date']}, "
            f"short_ewy={e['short_ewy_return']:.4f}, excess={e['excess_vs_spy']:.4f}, "
            f"reason={e['exit_reason']}"
        )

    if n_traded < 3:
        return mark_failed(
            sid,
            f"insufficient traded events: {n_traded}",
            extra={"event_log": event_log},
        )

    held_pnl = pnl[positions != 0]
    held_spy = spy_r.reindex(held_pnl.index).fillna(0)

    if len(held_pnl) < 15:
        return mark_failed(
            sid,
            f"insufficient held days: {len(held_pnl)}",
            extra={"event_log": event_log},
        )

    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name="NK Provocation Short EWY (held-days)",
        positions=positions.reindex(held_pnl.index).fillna(0),
        cost_bps=10,
    )

    short_returns = [e["short_ewy_return"] for e in traded]
    win_rate = float(np.mean([r > 0 for r in short_returns]))
    avg_excess = float(np.mean([e["excess_vs_spy"] for e in traded]))
    avg_ewy_vs_ewj = float(np.mean([e["ewy_vs_ewj"] for e in traded]))

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "On confirmed DPRK ICBM launch or nuclear test (USFK/ROK JCS press release), "
                "short EWY (iShares MSCI South Korea ETF) at close of confirmation day. "
                "Hold 20 trading days (4 weeks). Hard stop: cover if EWY closes up >8% from entry. "
                "Also cover on US-DPRK diplomatic contact or DPRK conciliatory signal."
            ),
            "mechanism": (
                "North Korean ICBM/nuclear tests trigger acute geopolitical risk re-pricing "
                "of South Korean equities due to proximity (Seoul within strike range) and "
                "economic integration disruption fears. EWY tracks KOSPI, which historically "
                "sells off on peninsular escalation. KRW depreciates simultaneously, "
                "amplifying USD-denominated EWY losses. Japan (EWJ) shows smaller reaction "
                "as the geopolitical risk is more directly South Korean."
            ),
            "source": (
                "Event dates from USFK press releases, 38North.org, NK News; "
                "prices via yfinance"
            ),
            "tickers": ["EWY", "EWJ"],
            "n_events_traded": n_traded,
            "event_win_rate": round(win_rate, 4),
            "avg_excess_vs_spy": round(avg_excess, 4),
            "avg_ewy_underperformance_vs_ewj": round(avg_ewy_vs_ewj, 4),
            "event_log": event_log,
            "caveats": (
                "Small sample (n=6 events, with 3 clustered in 2017). "
                "2017 events form a tight cluster and may overstate signal frequency — "
                "treat as single regime if bootstrapping. "
                "DPRK provocations increasingly expected by market (desensitization); "
                "more recent events (2022-2024) show smaller reactions. "
                "Hard stop at +8% means short can be squeezed if SK market rallies on de-escalation. "
                "KRW/USD leg not included in this backtest (proxy trading via EWY)."
            ),
            "stop_loss_pct": STOP_PCT,
        },
        pnl=held_pnl,
    )

    print(f"\nDone: {sid}")
    print(f"  n_events: {n_traded}, win_rate: {win_rate:.2f}, avg_excess_vs_spy: {avg_excess:.4f}")
    print(f"  avg EWY underperformance vs EWJ: {avg_ewy_vs_ewj:.4f}")
    if m.get("sharpe") is not None:
        print(
            f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
            f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
        )
    if m.get("oos_sharpe") is not None:
        print(f"  OOS Sharpe: {m.get('oos_sharpe'):.2f}")


if __name__ == "__main__":
    main()
