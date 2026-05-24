"""
R-N3 AAR intermodal vs carloads (retry).

Original spec: weekly AAR intermodal vs carloads from aar.org (PDF-only).
Substitution: monthly U.S. rail freight intermodal vs carloads from FRED
(RAILFRTINTERMODALD11 and RAILFRTCARLOADSD11). Run a monthly rebalance
instead of weekly (still preserves the economic signal).

Rule: When intermodal YoY % > carload YoY % by >= 5 pp at month-end,
go long XTN / short IYT for the following month.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import save_result, mark_failed, compute_metrics, load_prices, load_fred


SIGNAL_ID = "R-N3_aar_intermodal"


def main():
    try:
        carloads = load_fred("RAILFRTCARLOADSD11", start="2011-01-01")
        intermodal = load_fred("RAILFRTINTERMODALD11", start="2011-01-01")
    except Exception as e:
        return mark_failed(SIGNAL_ID, f"FRED load failed: {e}")

    carloads.columns = ["carloads"]
    intermodal.columns = ["intermodal"]
    rail = pd.concat([carloads, intermodal], axis=1).dropna()
    print(f"  rail months: {len(rail)}")

    rail["car_yoy"] = rail["carloads"].pct_change(12) * 100
    rail["int_yoy"] = rail["intermodal"].pct_change(12) * 100
    rail["spread"] = rail["int_yoy"] - rail["car_yoy"]
    rail = rail.dropna()
    print(f"  with YoY: {len(rail)} months, spread range [{rail['spread'].min():.1f}, {rail['spread'].max():.1f}] pp")

    # Get ETFs
    px = load_prices(["XTN", "IYT"], start="2012-01-01")
    px = px.dropna()
    print(f"  XTN/IYT: {len(px)} days, {px.index[0].date()} to {px.index[-1].date()}")
    ret = px.pct_change()

    # Resample to monthly returns at month-end
    monthly = (1 + ret).resample("M").prod() - 1
    spread_m = rail["spread"].copy()
    # FRED dates are typically month-start; align to month-end for "as of"
    spread_m.index = spread_m.index + pd.offsets.MonthEnd(0)

    # Signal at end of month t -> position for month t+1
    threshold_pp = 5.0
    signal = (spread_m > threshold_pp).astype(int) - (spread_m < -threshold_pp).astype(int)
    # rule says "long XTN / short IYT" only when spread > 5 (and originally "2 consecutive months").
    consec = (spread_m > threshold_pp) & (spread_m.shift(1) > threshold_pp)
    pos_long = consec.astype(float)  # 1 when go long XTN, short IYT
    n_events = int(pos_long.sum())
    print(f"  positive 2-consec months: {n_events}")

    # Compute the strategy's monthly PnL: long XTN, short IYT for month t+1 if pos_long[t]==1
    monthly_long_short = monthly["XTN"] - monthly["IYT"]
    pos_shifted = pos_long.reindex(monthly.index).shift(1).fillna(0.0)
    pnl_m = pos_shifted * monthly_long_short

    # Convert monthly PnL to daily by spreading: assume holding flat over month.
    # For simplicity, we treat each month's PnL as a single observation; compute_metrics treats
    # as daily so we need to upsample. Instead, build a daily-equivalent series by holding pos.
    daily_pos_long = pos_long.reindex(ret.index, method="ffill").fillna(0.0).shift(1).fillna(0.0)
    # daily_pos_long is 1 during months following a signal; "long XTN short IYT" each day
    pnl_d = daily_pos_long * (ret["XTN"] - ret["IYT"])

    metrics = compute_metrics(pnl_d.dropna(), benchmark=ret["XTN"], name="AAR intermodal vs carload spread > 5pp")
    print("Metrics:", metrics)

    extra = {
        "rule": "When intermodal-YoY > carload-YoY by >= 5pp for 2 consecutive months, long XTN / short IYT next month.",
        "mechanism": "Intermodal outpacing carloads signals consumer/containerized demand strength; transports stock segment underperforms heavy rail when containers dominate.",
        "source": "FRED RAILFRTINTERMODALD11 + RAILFRTCARLOADSD11 (monthly); XTN/IYT from yfinance.",
        "n_events": n_events,
        "data_substitution": "Original AAR weekly PDFs unavailable; substituted FRED monthly U.S. rail freight intermodal/carloads. Rule changed from 2-week to 2-month confirmation.",
        "status": "ok",
    }
    save_result(SIGNAL_ID, metrics, extra=extra)
    return metrics


if __name__ == "__main__":
    main()
