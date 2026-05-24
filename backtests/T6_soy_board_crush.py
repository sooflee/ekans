"""
T6 Soybean board crush.

Rule: When CBOT board crush < $0.40/bu for 3 consecutive days, go long ZS=F for 6 weeks.

Conversion (industry-standard "board crush"):
  Crush ($/bu soy) = ZL_price * 0.11 * 60   # oil yield 11 lb/bu, ZL quoted in c/lb -> /100
                   + ZM_price * 0.022       # meal yield ~48 lb (0.022 short tons) per bu; ZM in $/short ton
                   - ZS_price               # ZS in c/bu, also /100
Notes:
  yfinance ZS=F / ZM=F / ZL=F are quoted in cents-per-bushel, $/short-ton, cents-per-lb
  respectively. We normalise:
      ZS_d = ZS/100   ($/bu)
      ZM_d = ZM       ($/short ton)
      ZL_d = ZL/100   ($/lb)
  Crush ($/bu) = ZL_d * 11 + ZM_d * 0.022 - ZS_d
  (1 bu soy ~ 11 lb oil + 48 lb meal; 48 lb = 0.024 ST; using 0.022 is a
  conservative widely-cited shortcut accounting for spec/holdback.)

Long position is unhedged ZS=F (we don't model spread legs).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
from harness import (
    load_prices, daily_returns, compute_metrics, print_metrics,
    save_result, mark_failed,
)


def main():
    try:
        px = load_prices(["ZS=F", "ZM=F", "ZL=F"], start="2008-01-01")
    except Exception as e:
        return mark_failed("T6_soy_board_crush", f"price load failed: {e}")

    px = px.dropna(how="any")
    if px.empty or len(px) < 252:
        return mark_failed("T6_soy_board_crush", "insufficient price data")

    zs_d = px["ZS=F"] / 100.0
    zm_d = px["ZM=F"]
    zl_d = px["ZL=F"] / 100.0
    crush = zl_d * 11.0 + zm_d * 0.022 - zs_d
    crush.name = "crush"

    # Signal: crush < $0.40 for 3 consecutive days
    cond = crush < 0.40
    sig_raw = cond & cond.shift(1) & cond.shift(2)

    # Build daily position from triggers, hold 6 weeks (~30 trading days)
    ret = px["ZS=F"].pct_change()
    pos = pd.Series(0.0, index=px.index)
    HOLD = 30
    for d in sig_raw[sig_raw].index:
        i = px.index.searchsorted(d) + 1
        if i >= len(px.index):
            continue
        end = min(i + HOLD, len(px.index))
        pos.iloc[i:end] = np.maximum(pos.iloc[i:end].values, 1.0)  # OR'd holds

    pnl = (pos.shift(1) * ret).dropna()
    bench = ret.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="T6 Soy board crush -> long ZS=F")
    print_metrics(m)
    n_trigger = int(sig_raw.sum())
    n_days = int((pos > 0).sum())
    print(f"triggers (3-day low crush): {n_trigger}, in-position days: {n_days}")
    print(f"crush range: {crush.min():.2f} to {crush.max():.2f}, mean {crush.mean():.2f}")

    save_result("T6_soy_board_crush", m, extra={
        "status": "ok",
        "rule": ("Board crush = ZL*11 + ZM*0.022 - ZS (with ZS,ZL in $). "
                 "When crush < $0.40 for 3 consecutive days, long ZS=F 6 weeks."),
        "mechanism": ("Negative crush => processors slow down => soy beans accumulate, "
                      "but historically crushers normalize and beans bid up as meal/oil "
                      "demand recovers. Contrarian play."),
        "source": "yfinance ZS=F, ZM=F, ZL=F (CBOT)",
        "n_triggers": n_trigger,
        "n_inposition_days": n_days,
        "crush_mean": float(crush.mean()),
        "caveats": ("Conversion factor 0.022 vs 0.024 is a simplification; "
                    "real crush spread uses month-specific contracts (Jul/Nov etc.)."),
    })


if __name__ == "__main__":
    main()
