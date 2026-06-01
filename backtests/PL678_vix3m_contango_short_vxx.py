"""PL678 — VIX3M/VIX Steep Contango Persistence - Short VXX"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL678_vix3m_contango_short_vxx"

    try:
        px = load_prices(["^VIX", "^VIX3M", "VXX", "VIXY", "SPY"], start="2012-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    # VXX and VIXY are the actual ETF prices (adjusted)
    if "^VIX" not in px.columns or "^VIX3M" not in px.columns:
        return mark_failed(sid, "VIX indices not available")
    if "VXX" not in px.columns:
        return mark_failed(sid, "VXX not available")

    vix = px["^VIX"]
    vix3m = px["^VIX3M"]
    vxx = px["VXX"]
    vixy = px["VIXY"] if "VIXY" in px.columns else None
    spy = px["SPY"]

    # Compute daily returns for ETFs (from adjusted prices)
    ret = daily_returns(px)
    vxx_r = ret["VXX"]
    spy_r = ret["SPY"]
    vixy_r = ret["VIXY"] if "VIXY" in ret.columns else None

    # Align all series to VXX trading dates (ETF is investable proxy)
    vix_a = vix.reindex(vxx_r.index, method="ffill")
    vix3m_a = vix3m.reindex(vxx_r.index, method="ffill")
    spy_r_a = spy_r.reindex(vxx_r.index, method="ffill")

    # Compute VIX3M/VIX ratio
    ratio = vix3m_a / vix_a.replace(0, np.nan)

    # Realized 20-day SPY vol (annualized)
    spy_ret_full = ret["SPY"].reindex(vxx_r.index, method="ffill")
    realized_vol = spy_ret_full.rolling(20).std() * np.sqrt(252)

    # Consecutive days with ratio > 1.15
    above_ratio = (ratio > 1.15).astype(int)
    consec = above_ratio.copy()
    for i in range(1, len(consec)):
        if above_ratio.iloc[i] == 1:
            consec.iloc[i] = consec.iloc[i - 1] + 1
        else:
            consec.iloc[i] = 0

    # Trigger: ratio > 1.15 for 10+ consecutive days AND realized vol < 12%
    trigger = (consec >= 10) & (realized_vol < 0.12)

    hold = 20
    pnl = pd.Series(0.0, index=vxx_r.index)
    evts = []
    last_entry = None

    for i in range(len(trigger)):
        if not trigger.iloc[i]:
            continue
        dt = trigger.index[i]
        if last_entry is not None and (dt.to_pydatetime() - last_entry).days < hold * 1.5:
            continue  # avoid overlapping windows
        last_entry = dt.to_pydatetime()

        p = vxx_r.index.get_loc(dt)
        ep = min(p + hold, len(vxx_r))

        vxx_window = vxx_r.iloc[p:ep]
        if vixy_r is not None and dt in vixy_r.index:
            vixy_p = vixy_r.index.get_loc(dt)
            vixy_window = vixy_r.iloc[vixy_p:min(vixy_p + hold, len(vixy_r))]
        else:
            vixy_window = None

        n = len(vxx_window)
        # Short VXX + 25% long VIXY as gamma hedge
        if vixy_window is not None and len(vixy_window) >= n:
            ls_daily = -vxx_window.values[:n] + 0.25 * vixy_window.values[:n]
        else:
            ls_daily = -vxx_window.values[:n]

        pnl.iloc[p:p + n] += ls_daily

        cum_ls = float((1 + pd.Series(ls_daily)).prod() - 1)
        cum_vxx = float((1 + vxx_window).prod() - 1)

        evts.append({
            "trigger_date": str(dt.date()),
            "ratio_at_entry": round(float(ratio.iloc[i]), 3),
            "consec_days": int(consec.iloc[i]),
            "realized_vol": round(float(realized_vol.iloc[i]), 4),
            "vxx_return": round(cum_vxx, 4),
            "ls_return": round(cum_ls, 4),
        })

    print(f"Events: {len(evts)}")
    if not evts:
        return mark_failed(sid, "no trigger events")

    active = pnl[pnl != 0]
    if len(active) < 30:
        return mark_failed(sid, f"insufficient active days ({len(active)})")

    m = compute_metrics(active, benchmark=spy_r, name="VIX3M/VIX Steep Contango → Short VXX")
    avg_ret = float(np.mean([e["ls_return"] for e in evts]))
    win_rate = float(np.mean([e["ls_return"] > 0 for e in evts]))

    save_result(sid, m, extra={
        "rule": "When VIX3M/VIX ratio >1.15 for 10+ consecutive days and realized 20d SPY vol <12%, short VXX for 20 trading days, hedge 25% long VIXY",
        "mechanism": "Persistent steep contango in VIX futures curve means VXX continuously bleeds via roll cost; low realized vol environment reduces jump risk, making VXX short favorable",
        "source": "yfinance ^VIX ^VIX3M VXX VIXY SPY",
        "n_events": len(evts),
        "avg_ls_return": round(avg_ret, 4),
        "event_win_rate": round(win_rate, 4),
        "events": evts,
    })
    print(f"Done: Sharpe={m.get('sharpe'):.2f}, CAGR={m.get('cagr'):.2%}, Events={len(evts)}")


if __name__ == "__main__":
    main()
