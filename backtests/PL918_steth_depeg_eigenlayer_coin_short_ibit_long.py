"""PL918 — ETH/BTC Ratio Breakdown → Short COIN / Long IBIT (BTC-Dominance Rotation)"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns

HOLD_DAYS = 10
MIN_COOLDOWN = 15  # trading days between entries to avoid churning

def main():
    sid = "PL918_steth_depeg_eigenlayer_coin_short_ibit_long"
    try:
        # Load crypto prices and equity prices separately
        crypto = load_prices(["ETH-USD", "BTC-USD"], start="2020-01-01")
        eq = load_prices(["COIN", "IBIT", "SPY"], start="2020-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    # Merge on dates
    px = pd.concat([crypto, eq], axis=1).dropna(how="all")
    if "COIN" not in px.columns:
        return mark_failed(sid, "COIN data missing")

    ret = daily_returns(px)
    eth_r  = ret["ETH-USD"] if "ETH-USD" in ret.columns else None
    btc_r  = ret["BTC-USD"] if "BTC-USD" in ret.columns else None
    coin_r = ret["COIN"]
    spy_r  = ret["SPY"]

    if eth_r is None or btc_r is None:
        return mark_failed(sid, "ETH-USD or BTC-USD missing")

    # Use IBIT where available (launched 2024-01-11), BTC-USD as proxy before that
    ibit_available = "IBIT" in ret.columns
    if ibit_available:
        long_r = ret["IBIT"].combine_first(btc_r)
    else:
        long_r = btc_r

    # Compute signals
    # 1) ETH 5d return minus BTC 5d return < -5%
    eth_5d  = (1 + eth_r).rolling(5).apply(lambda x: x.prod(), raw=True) - 1
    btc_5d  = (1 + btc_r).rolling(5).apply(lambda x: x.prod(), raw=True) - 1
    eth_lag = eth_5d - btc_5d  # negative = ETH underperforming BTC

    # 2) ETH/BTC ratio below its 50-day SMA
    eth_btc_ratio = px["ETH-USD"] / px["BTC-USD"]
    eth_btc_sma50 = eth_btc_ratio.rolling(50).mean()
    ratio_below_sma = eth_btc_ratio < eth_btc_sma50

    # 3) COIN 60d return > 0 (crowded long condition)
    coin_60d = (1 + coin_r).rolling(60).apply(lambda x: x.prod(), raw=True) - 1

    # Align all signals on shared index
    common = coin_r.index
    eth_lag    = eth_lag.reindex(common)
    ratio_below = ratio_below_sma.reindex(common)
    coin_60d   = coin_60d.reindex(common)

    signal = (eth_lag < -0.05) & ratio_below & (coin_60d > 0)

    # Build signal entries with cooldown
    strat_r = -coin_r + long_r.reindex(common)  # short COIN + long BTC/IBIT
    pnl = pd.Series(0.0, index=strat_r.index)
    evts = []
    last_entry = -MIN_COOLDOWN - 1

    for i, (dt, sig) in enumerate(signal.items()):
        if not sig or np.isnan(sig):
            continue
        if i - last_entry < MIN_COOLDOWN:
            continue
        p = strat_r.index.get_loc(dt)
        ep = min(p + 1 + HOLD_DAYS, len(strat_r))
        seg = strat_r.iloc[p+1:ep]  # enter T+1
        if len(seg) < 5:
            continue

        for idx, v in seg.items():
            if pnl[idx] == 0.0:
                pnl[idx] = v

        cum_strat = float((1 + seg).prod() - 1)
        cum_coin  = float((1 + coin_r.iloc[p+1:ep]).prod() - 1)
        cum_long  = float((1 + long_r.reindex(common).iloc[p+1:ep]).prod() - 1)
        cum_spy   = float((1 + spy_r.reindex(common).iloc[p+1:ep]).prod() - 1)

        evts.append({
            "signal_date": str(dt.date()),
            "entry_date": str(strat_r.index[p+1].date()) if p+1 < len(strat_r) else None,
            "strat_return": round(cum_strat, 4),
            "coin_return": round(cum_coin, 4),
            "long_return": round(cum_long, 4),
            "spy_return": round(cum_spy, 4),
            "eth_btc_5d_diff": round(float(eth_lag.iloc[p]), 4),
            "using_ibit": bool(ibit_available and strat_r.index[p+1] >= pd.Timestamp("2024-01-11")) if p+1 < len(strat_r) else False,
        })
        last_entry = i

    if not evts:
        return mark_failed(sid, "no signal triggers found")

    ip = pnl[pnl != 0]
    if len(ip) < 20:
        return mark_failed(sid, f"insufficient active days ({len(ip)})")

    m = compute_metrics(ip, benchmark=spy_r.reindex(common), name="ETH/BTC Breakdown: Short COIN / Long IBIT")
    sr = [e["strat_return"] for e in evts]
    save_result(sid, m, extra={
        "rule": "Short COIN / long IBIT (BTC-USD pre-2024) when ETH 5d underperforms BTC by >5% AND ETH/BTC ratio below 50-DMA AND COIN 60d return>0; hold 10 days",
        "mechanism": "stETH/ETH depeg and ETH underperformance episodes reduce speculative appetite for crypto equities (COIN); BTC sees flight-to-quality within crypto",
        "source": "yfinance ETH-USD, BTC-USD, COIN, IBIT, SPY; DeFiLlama EigenLayer TVL (context only)",
        "n_events": len(evts),
        "avg_strat_return": round(float(np.mean(sr)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in sr])), 4),
        "hold_days": HOLD_DAYS,
        "events": evts[:20],  # cap for readability
        "caveats": "IBIT only since Jan 2024; pre-2024 uses BTC-USD as long proxy. COIN IPO April 2021 limits history. stETH/Curve depeg not directly measured — proxied by ETH/BTC ratio.",
    })
    print(f"Done: {len(evts)} events, avg strat return {np.mean(sr):.2%}")

if __name__ == "__main__":
    main()
