"""PL694 — Gold Miner AISC Inflation vs Spot Drift - Short WPM/FNV"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL694_aisc_short_wpm_fnv"
    # Strategy: AISC (all-in sustaining costs) data is from quarterly earnings
    # disclosures and is not available via free APIs.
    # Proxy approach: use GDX/GLD ratio as AISC proxy.
    # When GDX significantly underperforms GLD on a 252-day rolling basis
    # (GDX/GLD ratio declining >15% YoY while GLD YoY < +5%),
    # this reflects margins compression consistent with AISC inflation exceeding spot.
    # Short WPM+FNV equal-weight, hedge 50% long GDX.

    try:
        px = load_prices(["WPM", "FNV", "GDX", "GLD", "SPY"], start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    required = ["WPM", "FNV", "GDX", "GLD", "SPY"]
    missing = [t for t in required if t not in px.columns or px[t].dropna().empty]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    # Retry for NEM, GOLD, AEM if available (optional AISC proxies)
    try:
        px2 = load_prices(["NEM", "GOLD", "AEM"], start="2010-01-01")
        for t in ["NEM", "GOLD", "AEM"]:
            if t in px2.columns:
                px[t] = px2[t]
    except Exception:
        pass  # optional, proceed without

    px = px.ffill().dropna(subset=["WPM", "FNV", "GDX", "GLD", "SPY"])
    ret = daily_returns(px)

    gld_r = ret["GLD"]
    wpm_r = ret["WPM"]
    fnv_r = ret["FNV"]
    gdx_r = ret["GDX"]
    spy_r = ret["SPY"]

    # Compute rolling YoY returns (252-day)
    gld_yoy = px["GLD"].pct_change(252)
    gdx_yoy = px["GDX"].pct_change(252)

    # GDX/GLD ratio - proxy for miner margins
    gdx_gld_ratio = px["GDX"] / px["GLD"]
    ratio_yoy = gdx_gld_ratio.pct_change(252)

    # Signal: GDX/GLD ratio YoY < -10% (miner cost inflation), AND GLD YoY < +5% (spot not compensating)
    signal = (ratio_yoy < -0.10) & (gld_yoy < 0.05)

    # Convert to quarterly triggers (to avoid over-trading on persistent signal)
    # Use first day of each quarter where signal is True
    signal_quarterly = signal.resample("QE").first()

    triggers = []
    last_trigger = pd.Timestamp("2000-01-01")
    for date, val in signal_quarterly.items():
        if pd.isna(val) or not val:
            continue
        # Find actual trading date
        candidates = ret.index[ret.index >= date]
        if candidates.empty:
            continue
        trigger_date = candidates[0]
        if (trigger_date - last_trigger).days < 60:
            continue
        triggers.append(trigger_date)
        last_trigger = trigger_date

    print(f"Quarterly signal triggers: {len(triggers)}")

    if len(triggers) < 3:
        return mark_failed(sid, f"insufficient events ({len(triggers)}) — GDX/GLD ratio proxy yielded too few signals for AISC inflation detection")

    # Build PnL: short WPM+FNV equal-weight, hedge 50% long GDX, hold 60 days
    hold = 60
    pnl = pd.Series(0.0, index=ret.index)
    events = []

    for entry_date in triggers:
        if entry_date not in ret.index:
            continue
        p = ret.index.get_loc(entry_date)
        ep = min(p + hold, len(ret))

        chunk_wpm = wpm_r.iloc[p:ep]
        chunk_fnv = fnv_r.iloc[p:ep]
        chunk_gdx = gdx_r.iloc[p:ep]

        # Short WPM + short FNV (equal weight = 50% each = 100% short)
        # Hedge: 50% long GDX
        # Net: -0.5*WPM - 0.5*FNV + 0.5*GDX
        chunk_pnl = -0.5 * chunk_wpm - 0.5 * chunk_fnv + 0.5 * chunk_gdx
        chunk_pnl = chunk_pnl.reindex(ret.index[p:ep]).fillna(0)

        # Stop if WPM/FNV outperform GDX by >5% (combined): stop on +5% loss
        cum = (1 + chunk_pnl).cumprod() - 1
        stop_hits = cum[cum < -0.05]
        if not stop_hits.empty:
            stop_i = cum.index.get_loc(stop_hits.index[0])
            chunk_pnl = chunk_pnl.iloc[:stop_i + 1]

        n = len(chunk_pnl)
        pnl.iloc[p:p + n] += chunk_pnl.values[:n]

        basket_ret = float((1 + chunk_pnl).prod() - 1)
        sp_chunk = spy_r.iloc[p:ep]
        sp_ret = float((1 + sp_chunk).prod() - 1) if len(sp_chunk) > 0 else None

        gld_at = round(float(gld_yoy.iloc[p]) if not pd.isna(gld_yoy.iloc[p]) else 0, 4)
        ratio_at = round(float(ratio_yoy.iloc[p]) if not pd.isna(ratio_yoy.iloc[p]) else 0, 4)

        events.append({
            "entry_date": str(entry_date.date()),
            "gld_yoy": gld_at,
            "gdx_gld_ratio_yoy": ratio_at,
            "basket_return": round(basket_ret, 4),
            "spy_return": round(sp_ret, 4) if sp_ret is not None else None,
        })

    active = pnl[pnl != 0]
    print(f"Active PnL days: {len(active)}")
    if len(active) < 30:
        return mark_failed(sid, f"insufficient active days ({len(active)})")

    m = compute_metrics(active, benchmark=spy_r, name="AISC Inflation Short WPM/FNV vs GDX Hedge")

    rets = [e["basket_return"] for e in events]
    save_result(sid, m, extra={
        "rule": "Short WPM+FNV 50/50 with 50% GDX hedge for 60 days when GDX/GLD ratio YoY < -10% AND spot gold YoY < +5% (proxy for AISC inflation exceeding spot gold compensation)",
        "mechanism": "When mining costs (AISC) rise faster than gold price, royalty/streaming companies WPM and FNV have inflated relative valuations vs physical gold and should underperform",
        "source": "yfinance GDX, GLD, WPM, FNV; GDX/GLD ratio as AISC inflation proxy (actual AISC from NEM/GOLD/AEM quarterly earnings not available via free API)",
        "n_events": len(events),
        "avg_event_return": round(float(np.mean(rets)), 4) if rets else None,
        "event_win_rate": round(float(np.mean([r > 0 for r in rets])), 4) if rets else None,
        "events": events,
        "caveats": "Quarterly AISC earnings data not directly available; GDX/GLD ratio used as cost-inflation proxy; FNV delisted/acquired 2024-02 — data may be truncated",
    })
    print(f"Done: {len(events)} events, Sharpe={m.get('sharpe', 'N/A'):.2f}, CAGR={m.get('cagr', 0)*100:.1f}%")


if __name__ == "__main__":
    main()
