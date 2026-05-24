"""
AC-3 UK Autumn Statement Gilt Pre-Drift Fade.

Rule (simplified for free-data):
  - Hand-coded Autumn Statement / Autumn Budget dates 2017-2025.
  - On each event: check if IGLT.L (UK gilt ETF) total return is < -1.5% over the
    prior 10 trading days (proxy for yield rising materially i.e. gilt selloff).
  - If yes: long IGLT.L at close on T-1 (close before statement) and exit T+3.
  - Build event-return series, compute metrics, compare to same-window IGLT
    buy-and-hold benchmark.

Mechanism: Post-Truss trauma → pre-statement gilt shorts; OBR-blessed package
typically fiscally measured → shorts cover post-event.

Source: HM Treasury calendar (hand-coded), yfinance IGLT.L.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (
    load_prices, daily_returns,
    compute_metrics, print_metrics, save_result, mark_failed,
)


# UK Autumn Statement / Autumn Budget dates
AUTUMN_DATES = [
    "2016-11-23",  # Hammond Autumn Statement
    "2017-11-22",  # Hammond Autumn Budget
    "2018-10-29",  # Hammond Budget (oct)
    "2021-10-27",  # Sunak Autumn Budget
    "2022-11-17",  # Hunt Autumn Statement (post-Truss)
    "2023-11-22",  # Hunt Autumn Statement
    "2024-10-30",  # Reeves Autumn Budget
    "2025-11-26",  # Reeves Autumn Budget
]

GATE_BPS_EQUIVALENT = -0.015  # 1.5% IGLT decline ~ ≥15bp yield rise
HOLD_DAYS = 3
LOOKBACK = 10


def main():
    try:
        # IGLT.L = iShares Core UK Gilts; BUKL.L is an alternative.
        px = load_prices(["IGLT.L"], start="2010-01-01")
    except Exception as e:
        return mark_failed("AC-3", f"data load failed: {e}")
    px = px.dropna()
    if "IGLT.L" not in px.columns or len(px) < 200:
        return mark_failed("AC-3", "IGLT.L unavailable or too short")

    s = px["IGLT.L"]
    rets = s.pct_change()

    rows = []
    pnl_list = []
    bench_list = []
    for d in AUTUMN_DATES:
        ts = pd.Timestamp(d)
        # Find last trading day on/before statement date
        idx_pos = s.index.searchsorted(ts, side="right") - 1
        if idx_pos < LOOKBACK + 1 or idx_pos + HOLD_DAYS >= len(s):
            print(f"Skipping {d}: insufficient surrounding data")
            continue
        # Statement date itself is treated as event day; if event day is a
        # trading day, the close BEFORE statement is s.index[idx_pos-1] only
        # when the search lands on the statement date exactly. Use the trading
        # day that is the statement date itself OR the prior trading day as T-1.
        # Convention: "T-1 close" = close on last trading day STRICTLY before
        # statement date.
        stmt_pos = s.index.searchsorted(ts, side="left")  # first idx >= ts
        t_minus_1 = stmt_pos - 1
        if t_minus_1 < LOOKBACK or t_minus_1 + HOLD_DAYS >= len(s):
            print(f"Skipping {d}: bounds")
            continue
        # Prior 10-day return (T-11 → T-1)
        prior_ret = s.iloc[t_minus_1] / s.iloc[t_minus_1 - LOOKBACK] - 1
        # Hold: T-1 close → T+HOLD_DAYS close
        hold_ret = s.iloc[t_minus_1 + HOLD_DAYS] / s.iloc[t_minus_1] - 1
        gated = prior_ret < GATE_BPS_EQUIVALENT
        rows.append({
            "date": d,
            "t_minus_1": s.index[t_minus_1].date(),
            "prior_10d_ret": float(prior_ret),
            "hold_ret_T-1_to_T+3": float(hold_ret),
            "gated": bool(gated),
        })
        if gated:
            pnl_list.append(hold_ret)
            bench_list.append(hold_ret)  # benchmark = same buy-and-hold
        # also report unconditional for context

    df = pd.DataFrame(rows)
    print("\nEvent table:")
    print(df.to_string(index=False))

    # Unconditional event stats
    uncond_rets = df["hold_ret_T-1_to_T+3"].values
    print(f"\nUnconditional: N={len(uncond_rets)}, mean={np.mean(uncond_rets)*100:.2f}%, "
          f"hit={np.mean(uncond_rets > 0)*100:.1f}%")

    gated_rets = df[df["gated"]]["hold_ret_T-1_to_T+3"].values
    n_gated = len(gated_rets)
    print(f"Gated (prior 10d < -1.5%): N={n_gated}")
    if n_gated > 0:
        print(f"  mean={np.mean(gated_rets)*100:.2f}%, "
              f"hit={np.mean(gated_rets > 0)*100:.1f}%")

    if n_gated < 3:
        # Very small N — report unconditional too as the "honest" cut.
        print(f"\nGated N={n_gated} too small; reporting UNCONDITIONAL series.")
        use_rets = uncond_rets
        cut = "unconditional"
    else:
        use_rets = gated_rets
        cut = "gated_prior_10d_lt_-1.5%"

    if len(use_rets) < 3:
        return mark_failed("AC-3", f"too few events ({len(use_rets)})")

    # Build a synthetic daily series so compute_metrics works:
    # Each event contributes HOLD_DAYS days of daily returns from IGLT around the event.
    pnl_daily = pd.Series(0.0, index=s.index)
    bench_daily = pd.Series(0.0, index=s.index)
    use_dates = df[df["gated"]]["date"].tolist() if n_gated >= 3 \
        else df["date"].tolist()
    for d in use_dates:
        ts = pd.Timestamp(d)
        stmt_pos = s.index.searchsorted(ts, side="left")
        t_minus_1 = stmt_pos - 1
        if t_minus_1 + HOLD_DAYS >= len(s):
            continue
        win = s.index[t_minus_1 + 1 : t_minus_1 + 1 + HOLD_DAYS]
        r = rets.reindex(win).fillna(0.0)
        pnl_daily.loc[win] = r.values
        bench_daily.loc[win] = r.values

    in_pos = (pnl_daily != 0.0)
    pnl_e = pnl_daily[in_pos]
    bench_e = bench_daily[in_pos]
    # Event-level stats (preferred for low-N calendar events)
    mean_evt = float(np.mean(use_rets))
    n_evt = len(use_rets)
    std_evt = float(np.std(use_rets, ddof=1)) if n_evt > 1 else 0.0
    t_evt = float(mean_evt / (std_evt / np.sqrt(n_evt))) \
        if n_evt > 1 and std_evt > 0 else 0.0
    # Annualised "CAGR" from per-event compound × ~252/HOLD events per year
    # but with only ~1 event/year, just report cumulative compound return
    compound = float(np.prod(1.0 + use_rets) - 1.0)
    # crude annualization: span years from first to last event
    years_span = max(1.0, (pd.Timestamp(df["date"].max()) -
                           pd.Timestamp(df["date"].min())).days / 365.25)
    cagr_evt = float((1 + compound) ** (1 / years_span) - 1)
    m = {
        "name": "AC-3 UK Autumn Statement gilt fade",
        "start": str(df["date"].min()),
        "end": str(df["date"].max()),
        "n_days": int(len(pnl_e)),
        "n_events": int(n_evt),
        "cagr": cagr_evt,
        "ann_vol": float(std_evt * np.sqrt(252.0 / HOLD_DAYS)),
        "sharpe": float(mean_evt / std_evt * np.sqrt(252.0 / HOLD_DAYS)) if std_evt > 0 else 0.0,
        "max_dd": float(min(np.minimum.accumulate(np.cumprod(1 + use_rets)) /
                            np.maximum.accumulate(np.cumprod(1 + use_rets)) - 1)) if n_evt > 0 else 0.0,
        "calmar": None,
        "hit_rate": float(np.mean(use_rets > 0)) if n_evt > 0 else 0.0,
        "t_stat": t_evt,
    }
    print_metrics(m)

    save_result("AC-3", m, extra={
        "status": "ok",
        "rule": "Long IGLT.L T-1 close to T+3 close around UK Autumn Statement, "
                "gated by prior 10-day return < -1.5%.",
        "mechanism": "Post-Truss reflexive pre-statement short → OBR-blessed package → cover.",
        "source": "HM Treasury calendar (hand-coded); yfinance IGLT.L.",
        "cut": cut,
        "n_events_gated": int(n_gated),
        "n_events_total": int(len(df)),
        "mean_event_ret": mean_evt,
        "t_stat_event": t_evt,
        "events": df.assign(date=df["date"].astype(str),
                            t_minus_1=df["t_minus_1"].astype(str))
                     .to_dict(orient="records"),
    })


if __name__ == "__main__":
    main()
