"""PL1061_banxico_remittance_decel_mxn — Banxico Remittance YoY Deceleration
-> Short EWW / Long USDMXN.

Monthly: YoY growth of Banxico family remittances (SIE SE27803, USD mm), 3-month
moving average. ENTER short EWW when the 3m-avg YoY crosses below 0% from above
AND the 3m-avg was >= +5% at some point in the prior 6 monthly observations
(regime-break filter). EXIT when the 3m-avg YoY recovers above +2%.
Execution: close of first trading day on or after the 1st of month M+2 relative
to data month M (publication lag ~5 weeks; M+2 is conservative, no lookahead).
Headline = short EWW (flat out of regime). Variant = 50/50 short-EWW + long
USD/MXN (FRED DEXMXUS). Benchmark: SPY buy-and-hold.
"""
import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics, save_result,
                     mark_failed, daily_returns, print_metrics, DATA)

SID = "PL1061_banxico_remittance_decel_mxn"
BANXICO_URL = ("https://www.banxico.org.mx/SieInternet/consultasieiqy?"
               "series=SE27803&locale=en&formatoCSV.x=1"
               "&anoInicial=1995&anoFinal=2026")
ENTRY_CROSS = 0.0       # 3mma YoY crosses below this from above
REGIME_FILTER = 0.05    # 3mma YoY >= +5% somewhere in prior 6 obs
EXIT_LEVEL = 0.02       # 3mma YoY recovers above +2%
PUB_LAG_MONTHS = 2      # data month M tradable first business day of M+2


def fetch_remittances():
    """Banxico SIE SE27803 (total family remittances, monthly, USD mm).
    Token-free HTML endpoint, parsed with pd.read_html. Cached to data/."""
    cache = DATA / "PL1061_banxico_se27803.csv"
    if cache.exists():
        s = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
        return s.sort_index()

    import requests
    r = requests.get(BANXICO_URL, timeout=60,
                     headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    t = pd.read_html(io.StringIO(r.text))[-1]
    # first row is header ['DATE', 'SE27803']
    t = t.iloc[1:].copy()
    dates = pd.to_datetime(t.iloc[:, 0], format="%m/%Y")
    vals = pd.to_numeric(t.iloc[:, 1], errors="coerce")
    s = pd.Series(vals.values, index=dates, name="SE27803").dropna().sort_index()
    s.to_frame().to_csv(cache)
    return s


def main():
    try:
        rem = fetch_remittances()
    except Exception as e:
        return mark_failed(SID, f"Banxico SE27803 fetch: {e}")
    if len(rem) < 30:
        return mark_failed(SID, f"Banxico SE27803 too short: {len(rem)} rows")

    try:
        px = load_prices(["EWW", "SPY"], start="1996-01-01")
        usdmxn = load_fred("DEXMXUS", start="1995-01-01")["DEXMXUS"]
    except Exception as e:
        return mark_failed(SID, f"price load: {e}")

    px = px.sort_index().ffill(limit=5)
    ret = daily_returns(px)
    eww_r = ret["EWW"].dropna()
    spy_r = ret["SPY"].dropna()
    idx = eww_r.index

    # Long-USD leg: USD/MXN daily % change, aligned to EWW trading days
    usd_px = usdmxn.dropna().reindex(idx, method="ffill")
    usd_r = usd_px.pct_change().fillna(0.0)

    # --- Monthly signal series (YoY removes seasonality; never use MoM) ---
    yoy = rem.pct_change(12)
    mma = yoy.rolling(3).mean().dropna()

    # State machine over monthly observations, mapped to M+2 effective dates
    changes = []   # (effective_trading_day, new_state, data_month, mma_value)
    state = 0
    for i in range(1, len(mma)):
        d, cur, prev = mma.index[i], mma.iloc[i], mma.iloc[i - 1]
        new_state = state
        if state == 0:
            crossed = (cur < ENTRY_CROSS) and (prev >= ENTRY_CROSS)
            prior6 = mma.iloc[max(0, i - 6):i]
            regime_ok = len(prior6) > 0 and prior6.max() >= REGIME_FILTER
            if crossed and regime_ok:
                new_state = 1
        else:
            if cur > EXIT_LEVEL:
                new_state = 0
        if new_state != state:
            eff_month = d + pd.DateOffset(months=PUB_LAG_MONTHS)
            eff_first = eff_month.replace(day=1)
            future = idx[idx >= eff_first]
            if len(future) == 0:
                # signal known but not yet tradable (beyond price history)
                state = new_state
                continue
            changes.append((future[0], new_state, str(d.date()), float(cur)))
            state = new_state

    if not changes:
        return mark_failed(SID, "no regime entries triggered")

    # Regime indicator on trading days: set at the close of the effective day
    regime = pd.Series(np.nan, index=idx)
    for eff, st, _, _ in changes:
        regime.loc[eff] = float(st)
    regime = regime.ffill().fillna(0.0)

    # Position set at close of effective day -> PnL accrues from next day
    pos_short_eww = -regime                       # -1 short EWW in regime
    pnl = pos_short_eww.shift(1) * eww_r
    pnl = pnl.fillna(0.0)

    # 50/50 variant: short EWW + long USD/MXN
    pnl_combo = regime.shift(1) * (0.5 * (-eww_r) + 0.5 * usd_r)
    pnl_combo = pnl_combo.fillna(0.0)

    # --- Event log ---
    events = []
    open_ev = None
    for eff, st, data_month, val in changes:
        if st == 1:
            open_ev = {"entry_date": str(eff.date()), "entry_data_month": data_month,
                       "entry_3mma_yoy": round(val, 4)}
        elif open_ev is not None:
            seg = pnl.loc[pd.Timestamp(open_ev["entry_date"]):eff]
            open_ev.update({
                "exit_date": str(eff.date()), "exit_data_month": data_month,
                "exit_3mma_yoy": round(val, 4),
                "event_return": round(float((1 + seg).prod() - 1), 4),
            })
            events.append(open_ev)
            open_ev = None
    if open_ev is not None:
        seg = pnl.loc[pd.Timestamp(open_ev["entry_date"]):]
        open_ev.update({
            "exit_date": None, "exit_data_month": "OPEN",
            "event_return": round(float((1 + seg).prod() - 1), 4),
        })
        events.append(open_ev)

    for e in events:
        print(f"  entry {e['entry_date']} (data {e['entry_data_month']}, "
              f"3mma {e['entry_3mma_yoy']*100:+.1f}%) -> "
              f"exit {e.get('exit_date')} (data {e.get('exit_data_month')}): "
              f"short-EWW ret {e['event_return']*100:+.1f}%")

    days_in_regime = int((regime > 0).sum())
    print(f"\nRegimes: {len(events)}  days in regime: {days_in_regime} "
          f"of {len(idx)}")

    m = compute_metrics(pnl, benchmark=spy_r, positions=pos_short_eww,
                        name="Banxico Remittance YoY Decel -> Short EWW")
    m_combo = compute_metrics(pnl_combo, benchmark=spy_r, positions=regime,
                              name="50/50 short EWW + long USDMXN variant")
    m_active = compute_metrics(pnl[regime.shift(1) > 0], benchmark=spy_r,
                               name="Short EWW, in-regime days only")

    save_result(SID, m, extra={
        "status": "ok",
        "rule": ("3m-avg YoY of Banxico family remittances (SE27803) crosses "
                 "below 0% from above with 3mma >= +5% in prior 6 months -> "
                 "short EWW at close of first trading day of data-month M+2; "
                 "exit when 3mma YoY recovers above +2% (same M+2 timing). "
                 "Flat out of regime."),
        "mechanism": ("Remittances are ~4% of Mexican GDP and the marginal "
                      "driver of household consumption and USD supply into MXN; "
                      "a regime break from strong growth to contraction leads "
                      "peso weakness and domestic-demand earnings downgrades, "
                      "which unhedged-USD EWW compounds (equity down + MXN "
                      "down)."),
        "source": ("Banxico SIE SE27803 token-free CSV endpoint; FRED DEXMXUS; "
                   "yfinance EWW/SPY adjusted closes"),
        "tickers": ["EWW"],
        "n_events": len(events),
        "days_in_regime": days_in_regime,
        "events": events,
        "combo_variant": {k: m_combo.get(k) for k in
                          ("sharpe", "cagr", "max_dd", "t_stat",
                           "is_sharpe", "oos_sharpe", "net_sharpe")},
        "active_only": {k: m_active.get(k) for k in
                        ("sharpe", "cagr", "max_dd", "t_stat", "n_days")},
        "caveats": ("Few regime episodes (GFC, 2025) dominate; full-window "
                    "series is mostly flat zeros which dilutes Sharpe; Banxico "
                    "revises remittance prints modestly; M+2 execution is "
                    "conservative vs actual ~5-week lag; last regime may still "
                    "be open."),
    }, pnl=pnl)

    print_metrics(m)
    print()
    print_metrics(m_combo)
    if "oos_sharpe" in m:
        print(f"\n  IS Sharpe: {m['is_sharpe']:.2f}  OOS Sharpe: {m['oos_sharpe']:.2f}")


if __name__ == "__main__":
    main()
