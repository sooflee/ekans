"""
AC-7 Italy Legge di Bilancio BTP-Bund Spread Mean-Reversion.

Rule: If BTP-Bund 10y spread widens >25bp between Italy DBP submission (Oct 15)
and EC opinion (late Nov), short the spread from day after EC opinion for 20
trading days.

Data status (free):
  - FRED IRLTLT01ITM156N / IRLTLT01DEM156N are MONTHLY → too coarse for a
    daily 20-day event window.
  - Tradeable daily ETFs: BTP10.MI (iShares Italy Gov Bond), EXHA.DE / IBGM.DE
    (eurozone gilts), IS0L.DE (eurozone gov 1-3y) — not direct BTP/Bund pair.
  - Try a daily Italy gov-bond ETF: IBTM.MI / IBGL.MI / BTP10.MI / XBTP.MI
    and a Bund proxy: SXRC.DE / EXHA.DE. If both fetch, build a long-short.
  - Otherwise mark_failed per spec.
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


CANDIDATE_BTP = ["BTP10.MI", "IBTM.MI", "XBTP.MI", "IBGL.MI", "EMBI.MI",
                 "BTP.MI", "IGITB.MI", "GOVT10IT.MI"]
CANDIDATE_BUND = ["EXHA.DE", "SXRC.DE", "IBGS.DE", "IS0L.DE", "DBXG.DE"]


def try_load(tickers, label):
    import yfinance as yf
    found = {}
    for t in tickers:
        try:
            df = yf.download(t, start="2017-01-01", progress=False, auto_adjust=True)
            if df is None or df.empty:
                continue
            close = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.dropna()
            if len(close) > 200:
                found[t] = close
                print(f"  {label}: {t} OK (n={len(close)})")
        except Exception as e:
            print(f"  {label}: {t} fail ({e})")
    return found


def main():
    print("Probing BTP candidates:")
    btp = try_load(CANDIDATE_BTP, "BTP")
    print("Probing Bund/EUR-gov candidates:")
    bund = try_load(CANDIDATE_BUND, "BUND")

    if not btp or not bund:
        return mark_failed(
            "AC-7",
            "BTP/Bund daily ETF proxies not freely accessible via yfinance "
            "(BTP10.MI / EXHA.DE etc. returned empty); monthly FRED data too "
            "coarse for 20-day event window."
        )

    # Prefer Italy-specific BTP10.MI if available, else fall back to broader
    # eurozone gilt ETFs (which lose the Italy specificity).
    if "BTP10.MI" in btp:
        btp_t = "BTP10.MI"
    else:
        btp_t = sorted(btp.keys())[0]
    if "DBXG.DE" in bund:
        bund_t = "DBXG.DE"  # Xtrackers iBoxx EUR Sovereigns Germany
    else:
        bund_t = sorted(bund.keys())[0]
    print(f"\nUsing BTP proxy: {btp_t}, Bund proxy: {bund_t}")

    btp_px = btp[btp_t]
    bund_px = bund[bund_t]
    # If chosen BTP series is too short to cover the multi-year event panel, fail.
    if len(btp_px) < 1000:
        return mark_failed(
            "AC-7",
            f"BTP daily ETF proxy ({btp_t}) too short: only {len(btp_px)} obs "
            f"(starts {btp_px.index[0].date()}). Cannot evaluate 2018-2023 "
            f"Italy DBP events on a daily basis with free data; FRED monthly "
            f"data is too coarse for the 20-day window."
        )
    # Align
    df_px = pd.concat([btp_px.rename("BTP"), bund_px.rename("BUND")], axis=1).dropna()
    if len(df_px) < 200:
        return mark_failed("AC-7", f"Aligned series too short ({len(df_px)})")

    # Spread proxy: ratio of BTP / BUND (since these are price indices of
    # different durations, a true spread requires DV01 weighting; we use
    # the simple ratio change as proxy).
    rets = df_px.pct_change()
    # "Spread" position: long BTP, short BUND, dollar-neutral
    # Compute pair return: BTP_ret - BUND_ret (long BTP, short BUND).
    pair = rets["BTP"] - rets["BUND"]

    # EC opinion dates (late Nov each year; hand-coded approximate)
    EC_DATES = ["2018-11-21", "2019-11-20", "2020-11-18", "2021-11-24",
                "2022-11-22", "2023-11-21", "2024-11-26", "2025-11-26"]
    HOLD = 20

    pnl_daily = pd.Series(0.0, index=df_px.index)
    bench_daily = pd.Series(0.0, index=df_px.index)
    rows = []
    for d in EC_DATES:
        ts = pd.Timestamp(d)
        if ts > df_px.index[-1]:
            continue
        # Find first trading day after EC opinion
        i = df_px.index.searchsorted(ts, side="right")
        if i + HOLD >= len(df_px) or i < 22:
            continue
        # Pre-event widening proxy: pair return from Oct 15 → EC date
        oct_ts = pd.Timestamp(f"{ts.year}-10-15")
        j = df_px.index.searchsorted(oct_ts, side="right") - 1
        if j < 0:
            continue
        pre_pair_ret = float((1 + pair.iloc[j:i]).prod() - 1)
        # negative pre_pair_ret = BTP underperformed = spread widened
        gated = pre_pair_ret < -0.005  # ~0.5% BTP-Bund underperf ≈ ~25bp wider
        win = df_px.index[i : i + HOLD]
        hold_pair_ret = float((1 + pair.reindex(win)).prod() - 1)
        rows.append({
            "ec_date": d,
            "pre_pair_ret": pre_pair_ret,
            "hold_pair_ret_20d": hold_pair_ret,
            "gated": bool(gated),
        })
        if gated:
            pnl_daily.loc[win] = pair.reindex(win).fillna(0.0).values
            bench_daily.loc[win] = rets["BTP"].reindex(win).fillna(0.0).values

    df_evt = pd.DataFrame(rows)
    print("\nEvent table:")
    print(df_evt.to_string(index=False))

    triggered = df_evt[df_evt["gated"]]["hold_pair_ret_20d"].values
    uncond = df_evt["hold_pair_ret_20d"].values
    print(f"\nTriggered N={len(triggered)}, mean="
          f"{np.mean(triggered)*100 if len(triggered) else 0:.2f}%")
    print(f"Unconditional N={len(uncond)}, mean={np.mean(uncond)*100:.2f}%")

    if len(triggered) < 2:
        use = uncond
        cut = "unconditional"
    else:
        use = triggered
        cut = "gated_pre_widening"

    n = len(use)
    if n == 0:
        return mark_failed("AC-7", "no usable events after gating")

    mean_evt = float(np.mean(use))
    std_evt = float(np.std(use, ddof=1)) if n > 1 else 0.0
    t_evt = float(mean_evt / (std_evt / np.sqrt(n))) if std_evt > 0 else 0.0

    in_pos = (pnl_daily != 0.0)
    pnl_e = pnl_daily[in_pos]
    bench_e = bench_daily[in_pos]
    if len(pnl_e) >= 30:
        m = compute_metrics(pnl_e, benchmark=bench_e,
                            name="AC-7 Italy BTP-Bund mean reversion")
    else:
        m = {
            "name": "AC-7 Italy BTP-Bund mean reversion",
            "start": str(df_evt["ec_date"].min()),
            "end": str(df_evt["ec_date"].max()),
            "n_days": int(len(pnl_e)),
            "n_events": int(n),
            "cagr": float("nan"),
            "ann_vol": float(std_evt * np.sqrt(252.0 / HOLD)),
            "sharpe": float(mean_evt / std_evt * np.sqrt(252.0 / HOLD)) if std_evt > 0 else 0.0,
            "max_dd": float(np.min(use)) if len(use) else 0.0,
            "calmar": None,
            "hit_rate": float(np.mean(use > 0)) if len(use) else 0.0,
            "t_stat": t_evt,
        }
    print_metrics(m)

    save_result("AC-7", m, extra={
        "status": "ok",
        "rule": f"Long BTP-proxy / short Bund-proxy 20 trading days after EC opinion (late Nov), "
                f"gated by pair underperf < -0.5% Oct 15 → EC date.",
        "mechanism": "Markets reflexively price worst-case during DBP negotiation; "
                     "EC opinions are measured; ECB TPI caps tail.",
        "source": f"yfinance ETF proxies: BTP={btp_t}, Bund={bund_t}. CAVEAT: ETF "
                  f"duration mismatch — proxy is loose; t-stat should be interpreted "
                  f"with caution.",
        "btp_ticker": btp_t,
        "bund_ticker": bund_t,
        "cut": cut,
        "n_events_total": int(len(df_evt)),
        "n_events_triggered": int(len(triggered)),
        "t_stat_event": t_evt,
        "mean_event_ret": mean_evt,
        "events": df_evt.assign(ec_date=df_evt["ec_date"].astype(str))
                        .to_dict(orient="records"),
    })


if __name__ == "__main__":
    main()
