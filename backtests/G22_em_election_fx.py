"""
G-22 EM election surprise FX fade.

Curated EM election dates: Mexico (AMLO 2018, Sheinbaum 2024), Brazil (Bolsonaro 2018, Lula 2022),
Turkey (Erdogan 2023), India (Modi 2024), South Africa (ANC 2019, 2024), Argentina (Milei 2023),
Indonesia (Prabowo 2024).

Mapping: each election has a USD/EM FX pair on yfinance:
  Mexico   -> MXN=X  (USDMXN)
  Brazil   -> BRL=X
  Turkey   -> TRY=X
  India    -> INR=X
  S.Africa -> ZAR=X
  Argentina-> ARS=X (note: official rate often pegged; results may be noisy/unusable)
  Indonesia-> IDR=X

Rule:
- Define T0 = election date (or first trading day on/after).
- T0_to_T1 return = pct change in FX pair from T0 close to T+1 close.
- If |T0_to_T1| > 2%: place a FADE position at T+2 open in OPPOSITE direction.
  - If pair gapped up (USD strong vs EM) -> short USD/EM (= long EM FX) at T+2.
  - If pair gapped down -> long USD/EM at T+2.
- Exit at T+20 close. Track per-event return.
- PnL series records the per-trade return on its exit date.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed

EVENTS = [
    ("Mexico AMLO",     "2018-07-01", "MXN=X"),
    ("Brazil Bolsonaro","2018-10-28", "BRL=X"),
    ("S.Africa ANC2019","2019-05-08", "ZAR=X"),
    ("Argentina Milei", "2023-11-19", "ARS=X"),
    ("Brazil Lula",     "2022-10-30", "BRL=X"),
    ("Turkey Erdogan",  "2023-05-28", "TRY=X"),
    ("Indonesia Prabowo","2024-02-14", "IDR=X"),
    ("India Modi",      "2024-06-04", "INR=X"),
    ("Mexico Sheinbaum","2024-06-02", "MXN=X"),
    ("S.Africa ANC2024","2024-05-29", "ZAR=X"),
]


def main():
    tickers = sorted({t for _, _, t in EVENTS})
    pxs = {}
    for t in tickers:
        try:
            df = load_prices([t], start="2015-01-01")
            pxs[t] = df.iloc[:, 0]
        except Exception:
            pxs[t] = None

    trades = []
    for name, d, t in EVENTS:
        s = pxs.get(t)
        if s is None or s.dropna().empty:
            trades.append({"event": name, "ticker": t, "ret": np.nan, "note": "no FX data"})
            continue
        s = s.dropna()
        D = pd.Timestamp(d)
        idx = s.index
        if D < idx[0] or D > idx[-1]:
            trades.append({"event": name, "ticker": t, "ret": np.nan, "note": "out of FX range"})
            continue
        loc = idx.searchsorted(D)
        T0 = min(loc, len(idx) - 1)
        # Move ahead in trading days
        T1 = min(T0 + 1, len(idx) - 1)
        T2 = min(T0 + 2, len(idx) - 1)
        T20 = min(T0 + 22, len(idx) - 1)  # ~T+20 from entry (T+2)
        if T20 <= T2:
            trades.append({"event": name, "ticker": t, "ret": np.nan, "note": "insufficient post window"})
            continue
        # Gap return T0 close to T+1 close
        gap = s.iloc[T1] / s.iloc[T0] - 1.0
        if abs(gap) < 0.02:
            trades.append({"event": name, "ticker": t, "ret": 0.0, "gap": float(gap), "note": "no >2% gap; no trade"})
            continue
        # Fade direction
        entry_px = s.iloc[T2]
        exit_px = s.iloc[T20]
        if gap > 0:
            # USD/EM gapped UP -> fade by shorting (long EM FX): pos on USDEM = -1
            ret = (entry_px - exit_px) / entry_px
            side = "short USDEM (long EM FX)"
        else:
            ret = (exit_px - entry_px) / entry_px
            side = "long USDEM"
        trades.append({"event": name, "ticker": t, "entry": float(entry_px),
                       "exit": float(exit_px), "gap": float(gap),
                       "ret": float(ret), "side": side,
                       "entry_date": str(idx[T2].date()),
                       "exit_date": str(idx[T20].date())})

    tr = pd.DataFrame(trades)
    valid = tr.dropna(subset=["ret"])
    faded = valid[valid.get("side").notna()] if "side" in valid.columns else pd.DataFrame()
    if faded.empty:
        return mark_failed("G22_em_election_fx", "no events triggered (no gap > 2%)",
                           extra={"events_examined": tr.to_dict(orient="records")})

    # Daily PnL = each trade's return on its exit date. To compute Sharpe coherently we put zeros
    # between events on a synthetic daily calendar built from any FX series. Use MXN=X if present
    # else first ticker.
    base_t = "MXN=X" if "MXN=X" in pxs and pxs["MXN=X"] is not None else tickers[0]
    base = pxs[base_t].dropna()
    idx_all = base.index
    pnl = pd.Series(0.0, index=idx_all)
    for _, r in faded.iterrows():
        d = pd.Timestamp(r["exit_date"])
        loc = idx_all.searchsorted(d, side="right") - 1
        if 0 <= loc < len(idx_all):
            pnl.iloc[loc] += float(r["ret"])

    rets = faded["ret"].astype(float).values
    mean = float(rets.mean())
    std = float(rets.std(ddof=1)) if len(rets) > 1 else float("nan")
    tstat = mean / (std / np.sqrt(len(rets))) if std and std > 0 else 0.0
    hit = float((rets > 0).mean())

    m = compute_metrics(pnl.dropna(), name="G22 EM election FX fade")
    print_metrics(m)
    save_result("G22_em_election_fx", m, extra={
        "status": "ok",
        "rule": "If USD/EM pair moves >2% T0->T+1 after election, fade at T+2 open opposite direction; "
                "exit T+20 close.",
        "universe": "Multiple EM FX (MXN, BRL, TRY, INR, ZAR, IDR, ARS)",
        "n_events_examined": int(len(tr)),
        "n_trades_taken": int(len(faded)),
        "per_trade_mean_ret": mean,
        "per_trade_std": std,
        "per_trade_tstat": tstat,
        "per_trade_hit": hit,
        "events": tr.to_dict(orient="records"),
        "source": "Curated EM election dates",
        "notes": "Tiny N; Argentina ARS = official rate (pegged); excluded if no gap.",
    })


if __name__ == "__main__":
    main()
