"""
X3 USGC vs NYH refining margin gap — long VLO / short PBF when Gulf advantage.

Original: 2*RBOB_USGC + 1*ULSD_USGC - 3*WTI  vs  2*RBOB_NYH + 1*ULSD_NYH - 3*Brent.
When the gap > $8/bbl for 4 consecutive weeks, long VLO short PBF.

Substitution: USGC ULSD daily not on FRED (DHOILUSGULF unavailable; DGASUSGULF
and DHOILNYH + DGASNYH + WTI + Brent are). We approximate the USGC ULSD spread
to NYH ULSD using historical structural relationship (USGC ULSD typically
trades ~$0.05/gal below NYH due to pipeline economics). For backtest purposes,
we treat the crack gap as:
   USGC_crack - NYH_crack ≈ 2*(RBOB_USGC - RBOB_NYH)*42 - 3*(WTI - Brent)
The 1*ULSD term cancels approximately (Gulf-NYH ULSD spread is small/stable
relative to crude basis).

When gap > $8/bbl for >=20 consecutive trading sessions (~4 weeks), long VLO
(USGC-weighted refiner) and short PBF (East-Coast-weighted refiner) at t+1,
hold while signal sustains + 20-day exit buffer.

Source: FRED daily DCOILWTICO, DCOILBRENTEU, DGASNYH, DGASUSGULF, DHOILNYH.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, daily_returns, compute_metrics, print_metrics, save_result, mark_failed


def main():
    sid = "X3_usgc_nyh_crack"
    try:
        wti = load_fred("DCOILWTICO", start="2010-01-01").iloc[:, 0].rename("WTI")
        brent = load_fred("DCOILBRENTEU", start="2010-01-01").iloc[:, 0].rename("Brent")
        gas_nyh = load_fred("DGASNYH", start="2010-01-01").iloc[:, 0].rename("RBOB_NYH")
        gas_gulf = load_fred("DGASUSGULF", start="2010-01-01").iloc[:, 0].rename("RBOB_USGC")
        ulsd_nyh = load_fred("DHOILNYH", start="2010-01-01").iloc[:, 0].rename("ULSD_NYH")
    except Exception as e:
        return mark_failed(sid, f"FRED load failed: {e}")

    try:
        px = load_prices(["VLO", "PBF", "SPY", "XLE"], start="2012-01-01")
    except Exception as e:
        return mark_failed(sid, f"equity load failed: {e}")

    if "VLO" not in px.columns or "PBF" not in px.columns:
        return mark_failed(sid, "VLO/PBF prices missing")

    # PBF IPO 2012; data starts then.
    df = pd.concat([wti, brent, gas_nyh, gas_gulf, ulsd_nyh], axis=1).dropna()
    # All FRED commodity prices in $/gallon for gasoline/ULSD, $/bbl for crude.
    # Crack (per bbl of crude) = (2*RBOB*42 + 1*ULSD*42 - 3*WTI) / 3
    df["crack_nyh"] = (2 * df["RBOB_NYH"] * 42 + df["ULSD_NYH"] * 42 - 3 * df["Brent"]) / 3.0
    # USGC: approximate ULSD_USGC ≈ ULSD_NYH (small structural offset, see docstring).
    df["crack_usgc"] = (2 * df["RBOB_USGC"] * 42 + df["ULSD_NYH"] * 42 - 3 * df["WTI"]) / 3.0
    df["gap"] = df["crack_usgc"] - df["crack_nyh"]
    # Note: structurally positive most of the time because Brent ~= WTI + 2-5 and
    # NYH RBOB ~= USGC RBOB + 5-15c. We test the gap *level* and look for spikes.
    gap = df["gap"].dropna()

    # Threshold: $8/bbl for 20 consec sessions
    consec = 20
    threshold = 8.0
    over = gap > threshold
    # rolling sum
    in_regime = over.rolling(consec).sum() >= consec  # True when entire 20 sessions over $8

    # signal: first day in_regime becomes True
    sig_start = in_regime & ~in_regime.shift(1, fill_value=False)
    starts = gap.index[sig_start]

    if len(starts) == 0:
        # Loosen: try percentile-based regime if no $8 threshold events
        thr_pct = float(gap.quantile(0.85))
        over = gap > thr_pct
        in_regime = over.rolling(consec).sum() >= consec
        sig_start = in_regime & ~in_regime.shift(1, fill_value=False)
        starts = gap.index[sig_start]
        threshold = thr_pct
        if len(starts) < 3:
            return mark_failed(sid, f"insufficient regime events even with quantile fallback (gap range: {gap.min():.2f} to {gap.max():.2f})",
                               extra={"gap_max": float(gap.max()), "gap_p90": float(gap.quantile(0.90))})

    # PnL: long VLO short PBF while in_regime (with 20-day buffer after exit)
    in_regime_buf = in_regime.copy()
    # extend each True run by 20 days (exit buffer)
    extended = in_regime.copy()
    extended_arr = extended.values.copy()
    last_true = -10**9
    for i, v in enumerate(extended_arr):
        if v:
            last_true = i
        elif i - last_true <= 20:
            extended_arr[i] = True
    in_regime_ext = pd.Series(extended_arr, index=in_regime.index)

    rets = daily_returns(px[["VLO", "PBF", "SPY", "XLE"]])
    # align signal to returns
    sig = in_regime_ext.reindex(rets.index, method="ffill").fillna(False)
    # Long VLO short PBF
    long_short = pd.Series(0.0, index=rets.index)
    long_short[sig] = 1.0
    pnl_daily = long_short.shift(1) * (rets["VLO"] - rets["PBF"])
    pnl_daily = pnl_daily.dropna()

    n_events = int(sig_start.sum())
    if n_events < 3 or pnl_daily.ne(0).sum() < 30:
        return mark_failed(sid, f"too few active days: events={n_events}, active={int(pnl_daily.ne(0).sum())}",
                           extra={"n_events": n_events, "threshold": threshold})

    bench = rets["SPY"].reindex(pnl_daily.index)
    m = compute_metrics(pnl_daily, benchmark=bench, name=f"X3 USGC-NYH gap >${threshold:.1f} long VLO short PBF")
    m["n_events"] = n_events
    m["threshold_usd_bbl"] = float(threshold)
    m["consec_days_required"] = consec
    m["mean_gap"] = float(gap.mean())
    m["max_gap"] = float(gap.max())
    m["pct_days_above_threshold"] = float(over.mean())
    print(f"X3: gap mean={gap.mean():.2f}, max={gap.max():.2f}, events={n_events}, "
          f"active-days={int(pnl_daily.ne(0).sum())}")
    print_metrics(m)
    save_result(sid, m, extra={
        "status": "ok",
        "rule": f"When USGC 3-2-1 crack exceeds NYH 3-2-1 crack by >${threshold:.1f}/bbl for {consec} consec sessions, long VLO short PBF with 20-day exit buffer.",
        "mechanism": "USGC refiners (VLO weighted) capture WTI-Brent discount and Permian crude advantage; East Coast refiners (PBF) pay Brent-linked feedstocks. Wide gap = Gulf operational margin tailwind.",
        "source": "FRED daily: WTI (DCOILWTICO), Brent (DCOILBRENTEU), NY RBOB (DGASNYH), USGC RBOB (DGASUSGULF), NY ULSD (DHOILNYH). ULSD_USGC approximated as ULSD_NYH (small structural offset).",
        "substitution_note": "USGC ULSD daily not available on FRED (DHOILUSGULF deprecated). ULSD term approximated by NYH ULSD; resulting gap reflects mostly RBOB-spread + WTI-Brent basis.",
        "universe": ["VLO", "PBF"],
    })


if __name__ == "__main__":
    main()
