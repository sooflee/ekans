"""PL730_em_maturity_short_pbr_ibn
EM Maturity Wall Coverage <1.5x -> Short PBR/IBN (Counter)

Original spec: When trailing-12m EM eurobond maturity-wall vs FX-reserves
coverage ratio for Brazil (PBR) or India (IBN) falls below 1.5x, short the
named-country equity proxy for 60 trading days.

Implementation note: IMF SDDS / World Bank quarterly maturity-wall and FX-reserve
data is not available programmatically in this environment. We use a price-based
currency-stress proxy:
  - Brazil trigger: BRL/USD 63-day (≈1 quarter) depreciation crosses >15%
    (significant FX weakness signals deteriorating external coverage position)
    → short PBR for 60 trading days.
  - India trigger: INR/USD 63-day depreciation crosses >6% (INR is more managed;
    6% = ~2-sigma move that reflects RBI reserve burn)
    → short IBN for 60 trading days.

Each country signal is independent. Cooldown of 90 days prevents restacking.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


# FRED daily FX series (USD per foreign unit => higher = stronger foreign currency)
BRAZIL_FX_SERIES = "DEXBZUS"  # BRL per USD; higher = weaker BRL
INDIA_FX_SERIES  = "DEXINUS"  # INR per USD; higher = weaker INR

HOLD_DAYS = 60
FX_LOOKBACK = 63   # ~1 quarter
BRL_THRESHOLD = 0.15   # BRL weakens 15% over 63d => stress signal
INR_THRESHOLD = 0.06   # INR weakens 6% over 63d => stress signal
MIN_COOLDOWN = 90      # days between triggers per country


def build_fx_signals(fx_series_daily, threshold, lookback=63, cooldown=90):
    """
    Given a daily FX series (higher = weaker domestic currency, i.e. more USD per unit),
    return trigger dates where 63-day depreciation > threshold.
    """
    fx = fx_series_daily.dropna()
    # Depreciation = (fx_today / fx_63d_ago) - 1 => positive = weaker domestic ccy
    depr = fx / fx.shift(lookback) - 1
    depr = depr.dropna()

    # Crosses from below to above threshold
    prev    = depr.shift(1)
    raw_sig = (prev <= threshold) & (depr > threshold)

    triggers = []
    last_idx = -999
    dates_list = list(depr.index)
    for i, (dt, is_sig) in enumerate(raw_sig.items()):
        if is_sig and not pd.isna(is_sig):
            if (i - last_idx) >= cooldown:
                triggers.append(dt)
                last_idx = i

    return triggers, depr


def run_event_study(triggers, equity_ret, spy_r, depr_at_trigger, hold_days, country):
    idx       = equity_ret.index
    pnl       = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)
    event_log = []

    for trig_dt in triggers:
        future = idx[idx > trig_dt]
        if len(future) == 0:
            event_log.append({"trigger_date": str(trig_dt.date()), "country": country, "status": "no_data"})
            continue
        entry_dt  = future[0]
        entry_pos = idx.get_loc(entry_dt)
        exit_pos  = min(entry_pos + hold_days, len(idx))
        exit_dt   = idx[exit_pos - 1] if exit_pos > entry_pos else entry_dt

        slice_eq  = equity_ret.iloc[entry_pos:exit_pos]
        slice_spy = spy_r.iloc[entry_pos:exit_pos]
        gross_ev  = float((1 + (-slice_eq)).prod() - 1)
        spy_ev    = float((1 + slice_spy).prod() - 1)

        depr_val  = float(depr_at_trigger.get(trig_dt, np.nan))

        event_log.append({
            "country":          country,
            "trigger_date":     str(trig_dt.date()),
            "fx_depr_63d":      round(depr_val, 4),
            "entry_date":       str(entry_dt.date()),
            "exit_date":        str(exit_dt.date()),
            "n_hold_days":      int(exit_pos - entry_pos),
            "short_eq_return":  round(gross_ev, 4),
            "spy_return":       round(spy_ev, 4),
            "excess_vs_spy":    round(gross_ev - spy_ev, 4),
        })

        for j in range(entry_pos, exit_pos):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = -1.0
                pnl.iloc[j]       = -equity_ret.iloc[j]

    return pnl, positions, event_log


def main():
    sid = "PL730_em_maturity_short_pbr_ibn"
    equity_tickers = ["PBR", "IBN", "SPY"]

    try:
        px = load_prices(equity_tickers, start="2014-01-01")
    except Exception as e:
        return mark_failed(sid, f"equity data load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in equity_tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing equity tickers: {missing}")

    try:
        brl_usd = load_fred(BRAZIL_FX_SERIES, start="2014-01-01").squeeze().dropna()
        inr_usd = load_fred(INDIA_FX_SERIES,  start="2014-01-01").squeeze().dropna()
    except Exception as e:
        return mark_failed(sid, f"FRED FX data load: {e}")

    ret   = daily_returns(px)
    pbr_r = ret["PBR"].fillna(0)
    ibn_r = ret["IBN"].fillna(0)
    spy_r = ret["SPY"].fillna(0)

    # Build triggers for Brazil (BRL weakness → short PBR)
    brl_triggers, brl_depr = build_fx_signals(
        brl_usd, threshold=BRL_THRESHOLD, lookback=FX_LOOKBACK, cooldown=MIN_COOLDOWN
    )
    # Build triggers for India (INR weakness → short IBN)
    inr_triggers, inr_depr = build_fx_signals(
        inr_usd, threshold=INR_THRESHOLD, lookback=FX_LOOKBACK, cooldown=MIN_COOLDOWN
    )

    print(f"Brazil triggers: {len(brl_triggers)}")
    print(f"India triggers:  {len(inr_triggers)}")

    total_triggers = len(brl_triggers) + len(inr_triggers)
    if total_triggers < 3:
        return mark_failed(sid, f"too few triggers: Brazil={len(brl_triggers)}, India={len(inr_triggers)}")

    # Run event studies for each country
    brl_pnl, brl_pos, brl_events = run_event_study(
        brl_triggers, pbr_r, spy_r, brl_depr, HOLD_DAYS, "Brazil"
    )
    inr_pnl, inr_pos, inr_events = run_event_study(
        inr_triggers, ibn_r, spy_r, inr_depr, HOLD_DAYS, "India"
    )

    # Combine both country PnL series onto common index
    combined_idx = spy_r.index
    pnl       = pd.Series(0.0, index=combined_idx)
    positions = pd.Series(0.0, index=combined_idx)

    for base_pnl, base_pos in [(brl_pnl, brl_pos), (inr_pnl, inr_pos)]:
        for dt in base_pos.index:
            if dt not in combined_idx:
                continue
            if base_pos.loc[dt] != 0.0 and positions.loc[dt] == 0.0:
                positions.loc[dt] = base_pos.loc[dt]
                pnl.loc[dt]       = base_pnl.loc[dt]

    all_events = brl_events + inr_events
    all_events.sort(key=lambda e: e.get("trigger_date", ""))

    n_events = len([e for e in all_events if e.get("entry_date")])
    if n_events == 0:
        return mark_failed(sid, "no valid events with entry_date")

    held_pnl = pnl[positions != 0]
    held_spy = spy_r.reindex(held_pnl.index).fillna(0)

    if len(held_pnl) < 20:
        return mark_failed(sid, f"insufficient held days: {len(held_pnl)}")

    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name="EM FX Stress (BRL/INR) → Short PBR/IBN (held-days)",
    )

    brl_ev_rets = [e["short_eq_return"] for e in brl_events if e.get("short_eq_return") is not None]
    inr_ev_rets = [e["short_eq_return"] for e in inr_events if e.get("short_eq_return") is not None]
    all_ev_rets = brl_ev_rets + inr_ev_rets

    summary = {
        "n_events_total":      n_events,
        "n_events_brazil":     len(brl_ev_rets),
        "n_events_india":      len(inr_ev_rets),
        "avg_short_return":    round(float(np.mean(all_ev_rets)), 4) if all_ev_rets else None,
        "win_rate":            round(float(np.mean([r > 0 for r in all_ev_rets])), 4) if all_ev_rets else None,
    }

    save_result(
        sid, m,
        extra={
            "status": "ok",
            "rule": (
                f"Brazil: When BRL/USD 63-day depreciation > {BRL_THRESHOLD*100:.0f}%, "
                f"short PBR for {HOLD_DAYS} trading days. "
                f"India: When INR/USD 63-day depreciation > {INR_THRESHOLD*100:.0f}%, "
                f"short IBN for {HOLD_DAYS} trading days. Cooldown {MIN_COOLDOWN} days per country."
            ),
            "mechanism": (
                "A sharp domestic currency depreciation over one quarter signals FX reserve "
                "drawdown and deteriorating coverage of near-term external debt maturities "
                "(maturity wall). For Brazil (PBR is state-owned oil co tightly linked to "
                "sovereign credit quality) and India (IBN/ICICI Bank exposed to EM credit "
                "spreads), equity prices reprice to reflect the higher cost of FX-denominated "
                "debt rollover within 60 trading days of the FX stress trigger."
            ),
            "source": (
                "FRED DEXBZUS (BRL/USD daily) and DEXINUS (INR/USD daily). "
                "Equity prices via yfinance PBR, IBN, SPY. Original spec uses IMF SDDS "
                "quarterly maturity-wall + FX reserves coverage; FX depreciation proxy used "
                "here as a feasibility approximation."
            ),
            "caveats": (
                "BRL depreciation threshold of 15% is high and may fire infrequently. "
                "INR is heavily managed by RBI making clean signals harder. PBR is exposed "
                "to oil price and Petrobras-specific governance, not just Brazil sovereign "
                "quality. IBN/ICICI is not a pure EM-external-debt proxy. The two-country "
                "combined PnL series does not stack positions when both fire simultaneously."
            ),
            "tickers": equity_tickers,
            "fx_series": [BRAZIL_FX_SERIES, INDIA_FX_SERIES],
            "hold_days": HOLD_DAYS,
            "n_events": n_events,
            "events": all_events,
            "summary": summary,
        },
        pnl=pnl[positions != 0],
    )

    print(f"Done: {sid}")
    print(f"  n_events: {n_events}")
    print(f"  summary:  {summary}")
    if "error" not in m:
        print(
            f"  Sharpe: {m.get('sharpe', float('nan')):.2f}  "
            f"CAGR: {m.get('cagr', float('nan'))*100:.2f}%  "
            f"MaxDD: {m.get('max_dd', float('nan'))*100:.2f}%  "
            f"t-stat: {m.get('t_stat', float('nan')):.2f}"
        )


if __name__ == "__main__":
    main()
