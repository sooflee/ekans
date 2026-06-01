"""PL842_uranium_smr_euphoria_fade — Uranium Spot-Term Premium Blowoff + SMR Retail Euphoria Fade"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL842_uranium_smr_euphoria_fade"
    HOLD_DAYS = 20
    LOOKBACK = 30  # trading days for rolling return
    EUPHORIA_THRESHOLD = 0.60  # 60% 30-day return
    DIVERGENCE_RATIO = 0.50   # CCJ must be < 50% of OKLO/NNE avg gain

    try:
        px = load_prices(["OKLO", "NNE", "SMR", "CCJ", "SPY"], start="2023-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    # Build rolling 30-day returns for available tickers
    cum_ret = {}
    for ticker in ["OKLO", "NNE", "SMR", "CCJ"]:
        if ticker in ret.columns:
            # Rolling cumulative return over LOOKBACK days
            cr = (1 + ret[ticker]).rolling(LOOKBACK).apply(lambda x: x.prod() - 1, raw=True)
            cum_ret[ticker] = cr

    if "OKLO" not in cum_ret:
        return mark_failed(sid, "OKLO data not available")

    # Build a combined index from first date all tickers overlap
    # NNE starts May 2024, so we need to handle that
    common_idx = ret.index

    pnl = pd.Series(0.0, index=common_idx)
    evts = []
    last_entry = None

    for i, dt in enumerate(common_idx):
        if i < LOOKBACK + HOLD_DAYS:
            continue

        # Skip if still in a holding period
        if last_entry is not None and (dt - last_entry).days < HOLD_DAYS * 1.5:
            continue

        # Get available SMR names for euphoria measurement
        available_euphoria = []
        if "OKLO" in cum_ret and dt in cum_ret["OKLO"].index:
            oklo_ret = cum_ret["OKLO"].loc[dt]
            if not np.isnan(oklo_ret):
                available_euphoria.append(oklo_ret)

        if "NNE" in cum_ret and dt in cum_ret["NNE"].index:
            nne_ret = cum_ret["NNE"].loc[dt]
            if not np.isnan(nne_ret):
                available_euphoria.append(nne_ret)

        if not available_euphoria:
            continue

        avg_smr_ret = float(np.mean(available_euphoria))

        # Check CCJ divergence
        if "CCJ" not in cum_ret or dt not in cum_ret["CCJ"].index:
            continue
        ccj_ret = float(cum_ret["CCJ"].loc[dt])
        if np.isnan(ccj_ret):
            continue

        # Entry condition: euphoria in pre-revenue SMR names, CCJ lagging
        if avg_smr_ret < EUPHORIA_THRESHOLD:
            continue
        if ccj_ret >= DIVERGENCE_RATIO * avg_smr_ret:
            continue

        # Signal triggered: Short equal-weight (OKLO+NNE+SMR) basket, Long CCJ
        p = common_idx.get_loc(dt)
        ep = min(p + HOLD_DAYS, len(common_idx))

        # Build short basket returns
        short_tickers = [t for t in ["OKLO", "NNE", "SMR"] if t in ret.columns]
        short_r = ret[short_tickers].iloc[p:ep].mean(axis=1)  # equal-weight average
        long_r = ret["CCJ"].iloc[p:ep]

        # Pair PnL: long CCJ, short basket
        pair_r = long_r.values - short_r.values

        pnl.iloc[p:ep] = pair_r

        basket_cum = float((1 + short_r).prod() - 1)
        ccj_cum = float((1 + long_r).prod() - 1)
        pair_cum = float(np.sum(pair_r))

        spy_cum = None
        if dt in spy_r.index:
            sp_idx = spy_r.index.get_loc(dt)
            spy_cum = float((1 + spy_r.iloc[sp_idx:min(sp_idx + HOLD_DAYS, len(spy_r))]).prod() - 1)

        last_entry = dt
        evts.append({
            "entry_date": str(dt.date()),
            "avg_smr_30d_ret": round(avg_smr_ret, 4),
            "ccj_30d_ret": round(ccj_ret, 4),
            "n_days": ep - p,
            "ccj_return": round(ccj_cum, 4),
            "smr_basket_return": round(basket_cum, 4),
            "pair_excess": round(pair_cum, 4),
            "spy_return": round(spy_cum, 4) if spy_cum is not None else None,
        })

    print(f"Events found: {len(evts)}")
    if not evts:
        return mark_failed(sid, "no signal events found in scan period")

    ip = pnl[pnl != 0]
    print(f"Active PnL days: {len(ip)}")

    if len(ip) < 10:
        return mark_failed(sid, f"insufficient active days ({len(ip)})")

    m = compute_metrics(ip, benchmark=spy_r, name="SMR Euphoria Fade → Short OKLO/NNE/SMR, Long CCJ")
    m["n_events"] = len(evts)

    save_result(sid, m, extra={
        "rule": "Short equal-weight (OKLO+NNE+SMR) basket + Long CCJ when avg 30d return of OKLO/NNE >= 60% AND CCJ 30d return < 50% of that avg. Hold 20 trading days.",
        "mechanism": "Pre-revenue SMR equities (OKLO, NNE, SMR) experience retail-driven momentum spikes disconnected from cash-flow fundamentals. When they diverge sharply from cash-flowing uranium producers (CCJ), mean reversion into the spread is likely as retail enthusiasm fades.",
        "source": "yfinance daily prices; OKLO/NNE/SMR/CCJ/SPY",
        "n_events": len(evts),
        "events": evts,
    })

    print(f"Done: {len(evts)} events")
    for e in evts:
        spy_str = f", spy={e['spy_return']:.1%}" if e['spy_return'] is not None else ""
        print(f"  {e['entry_date']}: pair={e['pair_excess']:.1%} (ccj={e['ccj_return']:.1%} - smr_basket={e['smr_basket_return']:.1%}){spy_str}")


if __name__ == "__main__":
    main()
