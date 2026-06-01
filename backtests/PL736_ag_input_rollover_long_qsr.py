"""PL736 — Ag-Input PPI Rollover -> Long QSR (MCD/YUM/JACK)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns)


def main():
    sid = "PL736_ag_input_rollover_long_qsr"
    try:
        # Load FRED PPI series for agricultural inputs
        # WPU0141 = Livestock & poultry (meat inputs proxy)
        # WPS0122 = Farm products - grains (wheat proxy)
        # WPU022 = Dairy & related products PPI
        meat = load_fred("WPU0141", start="2010-01-01").squeeze()
        grain = load_fred("WPS0122", start="2010-01-01").squeeze()
        dairy = load_fred("WPU022", start="2010-01-01").squeeze()
    except Exception as e:
        return mark_failed(sid, f"FRED data load: {e}")

    try:
        px = load_prices(["MCD", "YUM", "JACK", "SPY"], start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"price data load: {e}")

    # Align all FRED series to common monthly index
    fred_df = pd.DataFrame({"meat": meat, "grain": grain, "dairy": dairy})
    fred_df = fred_df.dropna(how="all").sort_index()

    # Forward fill within month for alignment
    fred_monthly = fred_df.resample("ME").last().ffill()

    # Build equal-weight composite (normalize to index = 100 at start)
    composite = fred_monthly.mean(axis=1)
    composite = composite.dropna()

    # Compute trailing-3m rolling mean
    trail_3m = composite.rolling(3).mean()
    prior_3m = trail_3m.shift(3)

    # Signal: trailing 3m is >=5% below the prior 3m average
    signal = ((trail_3m - prior_3m) / prior_3m) <= -0.05

    # Convert signal to monthly trigger dates, then map to trading days with ~21-day release lag
    events = []
    last_trigger = pd.Timestamp("1900-01-01")

    for dt, sig in signal.items():
        if not sig:
            continue
        if (dt - last_trigger).days < 90:  # de-duplicate within 3 months
            continue
        # Release lag: PPI is released ~2-3 weeks after month-end
        release_date = dt + pd.Timedelta(days=21)
        events.append({
            "signal_month": dt,
            "release_date": release_date,
            "composite_val": float(composite.loc[dt]),
            "trail_3m_pct_vs_prior": float((trail_3m.loc[dt] - prior_3m.loc[dt]) / prior_3m.loc[dt]) if prior_3m.loc[dt] != 0 else None,
        })
        last_trigger = dt

    print(f"Events identified: {len(events)}")
    if not events:
        return mark_failed(sid, "no qualifying ag-input rollover events")

    # Build daily PnL
    ret = daily_returns(px)
    spy_r = ret["SPY"]
    mcd_r = ret["MCD"]
    yum_r = ret["YUM"]
    jack_r = ret["JACK"]

    hold_days = 60
    pnl = pd.Series(0.0, index=ret.index)
    valid_events = []

    for ev in events:
        rd = ev["release_date"]
        future_idx = ret.index[ret.index >= rd]
        if len(future_idx) < hold_days:
            continue
        entry_date = future_idx[0]
        ep = ret.index.get_loc(entry_date)
        ex = min(ep + hold_days, len(ret))

        # Equal-weight QSR basket: MCD + YUM + JACK
        m_r = mcd_r.iloc[ep:ex]
        y_r = yum_r.iloc[ep:ex]
        j_r = jack_r.iloc[ep:ex]
        basket_r = (m_r.values[:ex-ep] + y_r.values[:ex-ep] + j_r.values[:ex-ep]) / 3

        # Check overlap
        trade_slice = pnl.iloc[ep:ex]
        if (trade_slice != 0).sum() > hold_days * 0.5:
            continue

        pnl.iloc[ep:ex] = pnl.iloc[ep:ex] + basket_r

        mcd_ret = float((1 + m_r).prod() - 1)
        yum_ret = float((1 + y_r).prod() - 1)
        jack_ret = float((1 + j_r).prod() - 1)
        basket_ret = (mcd_ret + yum_ret + jack_ret) / 3
        spy_e = spy_r.iloc[ep:ex]
        spy_ret = float((1 + spy_e).prod() - 1)

        valid_events.append({
            "release_date": str(rd.date()),
            "signal_month": str(ev["signal_month"].date()),
            "pct_change_vs_prior": round(float(ev["trail_3m_pct_vs_prior"]), 4) if ev["trail_3m_pct_vs_prior"] is not None else None,
            "basket_return": round(basket_ret, 4),
            "mcd_return": round(mcd_ret, 4),
            "yum_return": round(yum_ret, 4),
            "jack_return": round(jack_ret, 4),
            "spy_return": round(spy_ret, 4),
        })

    print(f"Valid events: {len(valid_events)}")
    if not valid_events:
        return mark_failed(sid, "no valid events in tradeable window")

    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 30:
        return mark_failed(sid, f"insufficient active days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="Ag-Input PPI Rollover Long QSR")
    basket_rets = [ev["basket_return"] for ev in valid_events]
    save_result(sid, m, extra={
        "rule": "When trailing-3m ag-input PPI composite (beef+wheat+dairy) is >=5% below prior 3m, long MCD+YUM+JACK for 60 trading days.",
        "mechanism": "QSR operators face ~25-35% food cost margins; ag-input deflation flows directly to restaurant COGS with ~2-quarter lag, expanding margins and driving earnings beats.",
        "source": "FRED WPU0141 (livestock), WPS0122 (grain), WPU022 (dairy); yfinance MCD/YUM/JACK/SPY",
        "n_events": len(valid_events),
        "avg_basket_return": round(float(np.mean(basket_rets)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in basket_rets])), 4),
        "events": valid_events,
    })
    print(f"Done: {len(valid_events)} events, Sharpe={m.get('sharpe', 'N/A'):.2f}, CAGR={m.get('cagr', 0)*100:.1f}%")


if __name__ == "__main__":
    main()
