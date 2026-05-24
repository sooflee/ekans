"""
AC-12 December Tax-Loss-Harvest Year-Trailing-Loser Bounce (coarse version).

Coarse rule (per spec): On Dec 20, long IWM (Russell 2000 proxy), exit Jan 31,
GATED by SPX YTD > -10% (i.e. avoid broad bear-market years like 2008, 2022).

Caveat: Full version would screen Russell 3000 constituents with YTD < -30%
and mcap > $1B (cleaner construction). Using IWM is a blunt proxy.

Mechanism: Forced tax-loss selling peaks mid-Dec, reverses once wash-sale
window + new tax year begin; effect largest in small-caps.

Source: yfinance IWM + SPY.
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


def main():
    try:
        px = load_prices(["IWM", "SPY"], start="2002-01-01")
    except Exception as e:
        return mark_failed("AC-12", f"data load failed: {e}")

    iwm = px["IWM"].dropna()
    spy = px["SPY"].dropna()
    df = pd.concat([iwm.rename("IWM"), spy.rename("SPY")], axis=1).dropna()

    years = sorted(set(df.index.year))
    # We want events Dec 20 of year y → Jan 31 of year y+1
    # First eligible: 2002. Last: y such that y+1 has Jan 31 available.

    rows = []
    pnl_daily = pd.Series(0.0, index=df.index)
    bench_daily = pd.Series(0.0, index=df.index)

    for y in years:
        # YTD SPY by Dec 20 (using prior year-end close to Dec 20)
        # Find first trading day of year y and trading day on/before Dec 20 of year y
        yr_idx = df.index[df.index.year == y]
        if len(yr_idx) < 200:
            continue
        # Use prior year-end close as base for YTD
        prior = df.index[df.index.year == (y - 1)]
        if len(prior) == 0:
            continue
        base_close = float(df.loc[prior[-1], "SPY"])
        # Dec 20 trading day
        dec20 = pd.Timestamp(f"{y}-12-20")
        i_dec20 = df.index.searchsorted(dec20, side="right") - 1
        if i_dec20 < 0 or df.index[i_dec20].year != y:
            continue
        spy_ytd = float(df.iloc[i_dec20]["SPY"] / base_close - 1)
        # Exit date: Jan 31 of year y+1
        jan31 = pd.Timestamp(f"{y+1}-01-31")
        i_jan31 = df.index.searchsorted(jan31, side="right") - 1
        if i_jan31 <= i_dec20:
            continue
        win = df.index[i_dec20 : i_jan31 + 1]
        iwm_ret = float(df.iloc[i_jan31]["IWM"] / df.iloc[i_dec20]["IWM"] - 1)
        spy_ret = float(df.iloc[i_jan31]["SPY"] / df.iloc[i_dec20]["SPY"] - 1)
        gated = spy_ytd > -0.10
        rows.append({
            "year": y,
            "dec20": df.index[i_dec20].date(),
            "spy_ytd_at_dec20": spy_ytd,
            "iwm_ret_dec20_jan31": iwm_ret,
            "spy_ret_dec20_jan31": spy_ret,
            "gated": gated,
        })
        if gated:
            r_iwm = df["IWM"].reindex(win).pct_change().fillna(0.0)
            r_spy = df["SPY"].reindex(win).pct_change().fillna(0.0)
            pnl_daily.loc[win] = r_iwm.values
            bench_daily.loc[win] = r_spy.values

    df_evt = pd.DataFrame(rows)
    print("\nEvent table:")
    print(df_evt.to_string(index=False))

    triggered = df_evt[df_evt["gated"]]["iwm_ret_dec20_jan31"].values
    uncond_iwm = df_evt["iwm_ret_dec20_jan31"].values
    uncond_spy = df_evt["spy_ret_dec20_jan31"].values
    triggered_spy = df_evt[df_evt["gated"]]["spy_ret_dec20_jan31"].values
    print(f"\nTriggered (SPY YTD>-10%): N={len(triggered)}, "
          f"IWM_mean={np.mean(triggered)*100:.2f}%, SPY_mean="
          f"{np.mean(triggered_spy)*100:.2f}%, "
          f"excess={np.mean(triggered - triggered_spy)*100:.2f}%, "
          f"hit_IWM={np.mean(triggered>0)*100:.1f}%, "
          f"hit_vs_SPY={np.mean(triggered > triggered_spy)*100:.1f}%")
    print(f"Unconditional: N={len(uncond_iwm)}, "
          f"IWM_mean={np.mean(uncond_iwm)*100:.2f}%, "
          f"SPY_mean={np.mean(uncond_spy)*100:.2f}%")

    use = triggered
    use_spy = triggered_spy
    n = len(use)
    if n < 3:
        return mark_failed("AC-12", f"too few gated events ({n})")

    mean_iwm = float(np.mean(use))
    mean_spy = float(np.mean(use_spy))
    excess = use - use_spy
    mean_excess = float(np.mean(excess))
    std_excess = float(np.std(excess, ddof=1)) if n > 1 else 0.0
    t_excess = float(mean_excess / (std_excess / np.sqrt(n))) if std_excess > 0 else 0.0
    std_iwm = float(np.std(use, ddof=1)) if n > 1 else 0.0
    t_iwm = float(mean_iwm / (std_iwm / np.sqrt(n))) if std_iwm > 0 else 0.0

    in_pos = (pnl_daily != 0.0) | (bench_daily != 0.0)
    pnl_e = pnl_daily[in_pos]
    bench_e = bench_daily[in_pos]
    m = compute_metrics(pnl_e, benchmark=bench_e,
                        name="AC-12 December tax-loss bounce (IWM proxy)")
    print_metrics(m)
    print(f"\nEvent-level: t_iwm={t_iwm:.2f}, t_excess_vs_SPY={t_excess:.2f}")

    save_result("AC-12", m, extra={
        "status": "ok",
        "rule": "Long IWM Dec 20 → Jan 31; gated by SPY YTD-at-Dec-20 > -10%.",
        "mechanism": "Tax-loss-harvest forced selling peaks mid-Dec; reverses on "
                     "wash-sale + new-tax-year; small-caps most affected.",
        "source": "yfinance IWM (Russell 2000 proxy) + SPY (gating). "
                  "Caveat: coarse proxy; cleaner version would screen R3000 names "
                  "with YTD < -30%.",
        "n_events_total": int(len(df_evt)),
        "n_events_gated": int(n),
        "mean_iwm_ret": mean_iwm,
        "mean_spy_ret": mean_spy,
        "mean_excess_ret": mean_excess,
        "t_stat_iwm": t_iwm,
        "t_stat_excess_vs_spy": t_excess,
        "events": df_evt.assign(dec20=df_evt["dec20"].astype(str))
                        .to_dict(orient="records"),
    })


if __name__ == "__main__":
    main()
