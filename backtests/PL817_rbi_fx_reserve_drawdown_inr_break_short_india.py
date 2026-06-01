"""PL817_rbi_fx_reserve_drawdown_inr_break_short_india
RBI FX Reserve Drawdown + INR Break -> Short INDA/IBN vs Long GLD Counter-Signal

Trigger: USD/INR closes above 20-week (100-session) rolling high (proxy for FX
reserve stress since FRED DEXINUS is automatable; RBI WSS requires manual scrape).
Entry: Short INDA (75%) + Short IBN (25%) + Long GLD (100%) at next open.
Hold 30 trading days. Stop if PnL < -6% from entry.
Benchmark: SPY.

Known approximate event windows:
  - 2013-06-12: taper tantrum; INR -20%
  - 2018-04-23: EM selloff; INR -14%
  - 2022-01-25: USD strength; INR -8%
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


# Notional weights: short INDA -75%, short IBN -25%, long GLD +100%
WEIGHTS = {"INDA": -0.75, "IBN": -0.25, "GLD": 1.0}
HOLD_DAYS = 30
STOP_LOSS = 0.06   # -6% combined PnL from entry
INR_LOOKBACK = 100  # sessions (~20 weeks)


def find_triggers_from_inr(inr: pd.Series) -> pd.DatetimeIndex:
    """
    Trigger: USD/INR closes above its 100-session rolling max (previous 100 days,
    exclusive of today). Use FRED DEXINUS for this.
    Returns dates of new-high breakouts; de-duplicate so no two triggers are
    within 60 days of each other (avoid re-entering an existing trend).
    """
    rolling_max = inr.shift(1).rolling(INR_LOOKBACK).max()
    breakout = inr > rolling_max  # True on days INR makes new 100-day high

    # Only keep the FIRST breakout after a 60-day cooldown
    triggers = []
    last = pd.Timestamp("2000-01-01")
    for dt in inr.index[breakout.fillna(False)]:
        if (dt - last).days >= 60:
            triggers.append(dt)
            last = dt
    return pd.DatetimeIndex(triggers)


def main():
    sid = "PL817_rbi_fx_reserve_drawdown_inr_break_short_india"
    basket_tickers = ["INDA", "IBN", "GLD"]
    all_tickers = basket_tickers + ["SPY"]

    # Load prices
    try:
        px = load_prices(all_tickers, start="2012-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    # Load DEXINUS (USD/INR)
    try:
        inr_raw = load_fred("DEXINUS", start="2012-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED DEXINUS load: {e}")

    px = px.sort_index().ffill(limit=3)
    missing = [t for t in all_tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    # Align INR to trading calendar
    inr = inr_raw.squeeze().reindex(ret.index).ffill(limit=5).dropna()

    # Find triggers
    triggers = find_triggers_from_inr(inr)
    print(f"Total INR breakout triggers: {len(triggers)}")
    for t in triggers:
        print(f"  {t.date()}: DEXINUS={inr.get(t, float('nan')):.4f}")

    if len(triggers) < 3:
        return mark_failed(
            sid,
            f"too few triggers: {len(triggers)} (need >=3)",
            extra={"triggers": [str(d.date()) for d in triggers]},
        )

    # Event study
    idx = ret.index
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)
    event_log = []

    for trig_dt in triggers:
        # Entry: next trading day after trigger
        future = idx[idx > trig_dt]
        if len(future) == 0:
            event_log.append({"trigger": str(trig_dt.date()), "status": "no_future_data"})
            continue
        entry_dt = future[0]
        entry_pos = idx.get_loc(entry_dt)
        max_exit_pos = min(entry_pos + HOLD_DAYS, len(idx))

        # Compute combined daily PnL with position weights
        cumulative_pnl = 0.0
        actual_exit_pos = max_exit_pos
        exit_reason = "scheduled_30d"

        for j in range(entry_pos, max_exit_pos):
            day_pnl = sum(
                WEIGHTS[t] * float(ret[t].iloc[j]) if t in ret.columns and not pd.isna(ret[t].iloc[j])
                else 0.0
                for t in basket_tickers
            )
            cumulative_pnl += day_pnl  # approximate (not compounded for stop check)
            if cumulative_pnl < -STOP_LOSS:
                exit_reason = "stop_loss"
                actual_exit_pos = j + 1
                break

        # Record PnL
        for j in range(entry_pos, actual_exit_pos):
            if positions.iloc[j] == 0.0:
                day_pnl = sum(
                    WEIGHTS[t] * float(ret[t].iloc[j]) if t in ret.columns and not pd.isna(ret[t].iloc[j])
                    else 0.0
                    for t in basket_tickers
                )
                positions.iloc[j] = 1.0  # "invested"
                pnl.iloc[j] = day_pnl

        # Event summary
        exit_dt = idx[actual_exit_pos - 1] if actual_exit_pos > entry_pos else entry_dt
        event_pnl = float(pnl.iloc[entry_pos:actual_exit_pos].sum())
        spy_cum = float((1 + spy_r.reindex(idx[entry_pos:actual_exit_pos]).fillna(0)).prod() - 1)

        ev = {
            "trigger_date": str(trig_dt.date()),
            "entry_date": str(entry_dt.date()),
            "exit_date": str(exit_dt.date()),
            "exit_reason": exit_reason,
            "n_hold_days": int(actual_exit_pos - entry_pos),
            "strategy_return": round(event_pnl, 4),
            "spy_return": round(spy_cum, 4),
            "excess_return": round(event_pnl - spy_cum, 4),
        }
        event_log.append(ev)

    traded = [e for e in event_log if "strategy_return" in e]
    n_traded = len(traded)
    print(f"\nEvents traded: {n_traded}")
    for e in traded:
        print(
            f"  trigger={e['trigger_date']} entry={e['entry_date']} "
            f"strat={e['strategy_return']:.4f} excess={e['excess_return']:.4f} "
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

    if len(held_pnl) < 20:
        return mark_failed(
            sid,
            f"insufficient held days: {len(held_pnl)}",
            extra={"event_log": event_log},
        )

    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name="RBI INR-Break Short INDA+IBN / Long GLD (held-days)",
        positions=positions.reindex(held_pnl.index).fillna(0),
        cost_bps=10,
    )

    strat_returns = [e["strategy_return"] for e in traded]
    win_rate = float(np.mean([r > 0 for r in strat_returns]))
    avg_excess = float(np.mean([e["excess_return"] for e in traded]))

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "When USD/INR (FRED DEXINUS) closes above its 100-session rolling high "
                "(proxy for RBI FX reserve stress / capital outflow episode), "
                "enter next day: Short INDA (-75% notional) + Short IBN (-25%) + Long GLD (+100%). "
                "Hold 30 trading days. Hard stop if combined PnL falls -6% from entry."
            ),
            "mechanism": (
                "INR 100-day breakout signals acute EM capital outflow and FX reserve drawdown "
                "pressure. INDA and IBN are directly exposed to INR depreciation (USD-priced ETF/ADR). "
                "GLD long captures safe-haven demand and domestic gold-price appreciation in INR terms. "
                "Short INR-sensitive names + long GLD profits from the flight-to-quality spread."
            ),
            "source": (
                "FRED DEXINUS (USD/INR daily, Federal Reserve); "
                "prices via yfinance; RBI WSS (not used — DEXINUS is the automatable proxy)"
            ),
            "tickers": basket_tickers,
            "position_weights": WEIGHTS,
            "inr_lookback_sessions": INR_LOOKBACK,
            "n_triggers": len(triggers),
            "n_traded": n_traded,
            "event_win_rate": round(win_rate, 4),
            "avg_excess_vs_spy": round(avg_excess, 4),
            "event_log": event_log,
            "caveats": (
                "Trigger based solely on DEXINUS 100-day breakout (omits RBI WSS reserve data "
                "which requires manual scraping). Many breakout episodes may be gradual trend "
                "continuations rather than acute stress events. INDA inception 2012 limits "
                "history. The short-INDA + long-GLD combo has mixed carry (GLD earns nothing). "
                "n events depends on calibration; adjust lookback to tune frequency."
            ),
        },
        pnl=held_pnl,
    )

    print(f"\nDone: {sid}")
    print(f"  n_triggers: {len(triggers)}, n_traded: {n_traded}, win_rate: {win_rate:.2f}, avg_excess: {avg_excess:.4f}")
    if m.get("sharpe") is not None:
        print(
            f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
            f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
        )
    if m.get("oos_sharpe") is not None:
        print(f"  OOS Sharpe: {m.get('oos_sharpe'):.2f}")


if __name__ == "__main__":
    main()
