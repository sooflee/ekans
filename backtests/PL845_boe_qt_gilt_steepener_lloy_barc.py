"""PL845_boe_qt_gilt_steepener_lloy_barc — BoE QT Pace Acceleration + DMO Long-End Issuance: Long LLOY.L / Short BARC.L Gilt Steepener Pair"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL845_boe_qt_gilt_steepener_lloy_barc"
    HOLD_DAYS = 20
    STEEP_DAYS = 10          # Rolling window for steepening signal
    STEEP_THRESHOLD = 0.15   # 15bp steepening over 10 trading days
    MIN_GAP_DAYS = 30        # Min days between entries

    # FRED series: UK 10y gilt yield (monthly) and UK 3m interbank rate (monthly)
    try:
        uk10y = load_fred("IRLTLT01GBM156N", start="2008-01-01")
        uk3m = load_fred("IR3TIB01GBM156N", start="2008-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED data load: {e}")

    # Interpolate monthly series to business daily
    uk10y = uk10y.squeeze()
    uk3m = uk3m.squeeze()

    try:
        px = load_prices(["LLOY.L", "BARC.L", "ISF.L", "SPY"], start="2009-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    ret = daily_returns(px)

    # Check we have UK tickers
    if "LLOY.L" not in ret.columns or "BARC.L" not in ret.columns:
        return mark_failed(sid, "LLOY.L or BARC.L not available in price data")

    lloy_r = ret["LLOY.L"]
    barc_r = ret["BARC.L"]
    spy_r = ret["SPY"]

    # Build daily index aligned to UK equity trading days
    daily_idx = lloy_r.index

    # Interpolate FRED monthly yield curves to daily
    uk10y_daily = uk10y.reindex(daily_idx, method="ffill")
    uk3m_daily = uk3m.reindex(daily_idx, method="ffill")
    curve_2s10s = uk10y_daily - uk3m_daily  # UK 2s10s proxy (10y - 3m)

    # 10-day change in the curve
    curve_change_10d = curve_2s10s.diff(STEEP_DAYS)

    pnl = pd.Series(0.0, index=daily_idx)
    evts = []
    last_entry_date = None

    for i, dt in enumerate(daily_idx):
        if i < STEEP_DAYS + HOLD_DAYS:
            continue

        # Enforce gap between entries
        if last_entry_date is not None:
            if (dt - last_entry_date).days < MIN_GAP_DAYS:
                continue

        # Entry condition 1: curve steepened >= 15bp over 10 trading days
        steep = curve_change_10d.loc[dt]
        if np.isnan(steep) or steep < STEEP_THRESHOLD:
            continue

        # Entry condition 2: LLOY.L 10-day return > BARC.L 10-day return by 1pp
        lloy_10d = float((1 + lloy_r.iloc[max(0, i - STEEP_DAYS):i]).prod() - 1)
        barc_10d = float((1 + barc_r.iloc[max(0, i - STEEP_DAYS):i]).prod() - 1)
        if lloy_10d - barc_10d < 0.01:
            continue

        # ISF.L above 200-day MA check
        if "ISF.L" in ret.columns:
            isf_px = px["ISF.L"]
            if dt in isf_px.index:
                isf_loc = isf_px.index.get_loc(dt)
                if isf_loc >= 200:
                    isf_ma200 = float(isf_px.iloc[max(0, isf_loc - 200):isf_loc].mean())
                    isf_cur = float(isf_px.iloc[isf_loc])
                    if isf_cur < isf_ma200:
                        continue

        # Enter long LLOY.L / short BARC.L pair
        p = daily_idx.get_loc(dt)
        ep = min(p + HOLD_DAYS, len(daily_idx))

        lloy_seg = lloy_r.iloc[p:ep]
        barc_seg = barc_r.iloc[p:ep]

        # Pair PnL: Long LLOY.L, Short BARC.L (dollar-neutral)
        pair_r = lloy_seg.values - barc_seg.values
        pnl.iloc[p:ep] = pair_r

        lloy_cum = float((1 + lloy_seg).prod() - 1)
        barc_cum = float((1 + barc_seg).prod() - 1)

        spy_cum = None
        if dt in spy_r.index:
            sp_loc = spy_r.index.get_loc(dt)
            spy_cum = float((1 + spy_r.iloc[sp_loc:min(sp_loc + HOLD_DAYS, len(spy_r))]).prod() - 1)

        last_entry_date = dt
        evts.append({
            "entry_date": str(dt.date()),
            "curve_10d_steepening_bp": round(float(steep * 100), 1),
            "lloy_10d_ret": round(lloy_10d, 4),
            "barc_10d_ret": round(barc_10d, 4),
            "n_days": ep - p,
            "lloy_return": round(lloy_cum, 4),
            "barc_return": round(barc_cum, 4),
            "pair_excess": round(float(lloy_cum - barc_cum), 4),
            "spy_return": round(spy_cum, 4) if spy_cum is not None else None,
        })

    print(f"Events found: {len(evts)}")
    if not evts:
        return mark_failed(sid, "no signal events found — check FRED data and thresholds")

    ip = pnl[pnl != 0]
    print(f"Active PnL days: {len(ip)}")

    if len(ip) < 30:
        return mark_failed(sid, f"insufficient active days ({len(ip)})")

    m = compute_metrics(ip, benchmark=spy_r, name="BoE QT Gilt Steepener → Long LLOY / Short BARC")
    m["n_events"] = len(evts)

    save_result(sid, m, extra={
        "rule": "Long LLOY.L + Short BARC.L (dollar-neutral) when UK 2s10s gilt curve steepens >=15bp over 10 trading days AND LLOY.L 10d return > BARC.L by 1pp AND ISF.L above 200-day MA. Hold 20 trading days.",
        "mechanism": "UK gilt curve steepening directly expands net interest margin (NIM) for LLOY.L (UK mortgage bank with liability-sensitive balance sheet) while BARC.L (significant US/IB exposure) does not benefit proportionally. BoE QT and DMO long-end issuance acceleration are the structural drivers.",
        "source": "FRED IRLTLT01GBM156N (UK 10y), IR3TIB01GBM156N (UK 3m); LLOY.L, BARC.L, ISF.L, SPY via yfinance",
        "n_events": len(evts),
        "events": evts,
    })

    print(f"Done: {len(evts)} events")
    for e in evts:
        spy_str = f", spy={e['spy_return']:.1%}" if e['spy_return'] is not None else ""
        print(f"  {e['entry_date']}: pair={e['pair_excess']:.1%} (lloy={e['lloy_return']:.1%} - barc={e['barc_return']:.1%}), curve_steep={e['curve_10d_steepening_bp']}bp{spy_str}")


if __name__ == "__main__":
    main()
