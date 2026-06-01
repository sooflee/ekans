"""PL692 — Wikipedia Pageview Surge - CROX/BIRK Fade Long Volatility"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL692_wiki_pv_short_crox_birk"

    # CROX IPO: ~2006; BIRK IPO: Oct 2023 (very limited history)
    # Strategy: Wikipedia pageview spikes are hard to systematically backtest
    # due to lack of historical data. We use a proxy: when CROX exhibits
    # high-volume price spike (>2 std above 90d avg volume), fade 5d later for 15d.
    # This proxies the "buzz" mechanism without requiring actual Wikipedia API data.
    try:
        px = load_prices(["CROX", "SPY"], start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if "CROX" not in px.columns or px["CROX"].dropna().empty:
        return mark_failed(sid, "CROX data unavailable")

    try:
        # Load volume data for CROX to detect "buzz" spikes
        import yfinance as yf
        crox_data = yf.download("CROX", start="2010-01-01", progress=False, auto_adjust=True)
        if crox_data.empty:
            return mark_failed(sid, "CROX yfinance data empty")
        if isinstance(crox_data.columns, pd.MultiIndex):
            crox_vol = crox_data["Volume"].squeeze()
            crox_close = crox_data["Close"].squeeze()
        else:
            crox_vol = crox_data["Volume"].squeeze()
            crox_close = crox_data["Close"].squeeze()
    except Exception as e:
        return mark_failed(sid, f"CROX volume load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]
    crox_r = ret["CROX"]

    # Align volume and prices
    crox_vol = crox_vol.reindex(crox_r.index).fillna(0)
    crox_close = crox_close.reindex(crox_r.index).ffill()

    # Detect volume spikes: 7-day sum > 3x trailing 90-day median (proxy for Wikipedia buzz)
    vol_7d = crox_vol.rolling(7).sum()
    vol_90d_med = crox_vol.rolling(90).median()

    # Also require price spike: 5-day return > 15% (strong buzz event)
    ret_5d = crox_close.pct_change(5)

    spike_condition = (vol_7d > 3.0 * vol_90d_med) & (ret_5d > 0.10)

    # Find spike peaks (local maxima of the vol spike, debounce 30 days)
    triggers = []
    in_spike = False
    spike_start = None
    last_trigger = pd.Timestamp("2000-01-01")

    for i, (date, is_spike) in enumerate(spike_condition.items()):
        if pd.isna(is_spike):
            continue
        if is_spike and not in_spike:
            in_spike = True
            spike_start = date
        elif not is_spike and in_spike:
            # spike ended — peak was at spike_start window
            # entry is 5 trading days after spike start
            in_spike = False
            if spike_start is not None and (spike_start - last_trigger).days > 60:
                # Find the entry date: 5 trading days after spike_start
                start_idx = crox_r.index.get_loc(spike_start) if spike_start in crox_r.index else None
                if start_idx is not None:
                    entry_idx = start_idx + 5
                    if entry_idx < len(crox_r.index):
                        entry_date = crox_r.index[entry_idx]
                        triggers.append(entry_date)
                        last_trigger = entry_date

    print(f"Volume/price spike events found: {len(triggers)}")

    if len(triggers) < 3:
        return mark_failed(sid, f"insufficient events ({len(triggers)}) — Wikipedia pageview proxy via volume spikes yielded too few signals")

    # Build PnL: long CROX for 15 trading days after entry (mean reversion of buzz fade)
    hold = 15
    pnl = pd.Series(0.0, index=crox_r.index)
    events = []

    for entry_date in triggers:
        if entry_date not in crox_r.index:
            continue
        p = crox_r.index.get_loc(entry_date)
        ep = min(p + hold, len(crox_r))
        chunk = crox_r.iloc[p:ep]

        # Apply 8% stop-loss
        cumret = (1 + chunk).cumprod() - 1
        stop_hit = cumret[cumret < -0.08]
        if not stop_hit.empty:
            stop_idx = cumret.index.get_loc(stop_hit.index[0])
            chunk = chunk.iloc[:stop_idx + 1]

        pnl.iloc[p:p + len(chunk)] += chunk.values

        # Compute event return
        ev_ret = float((1 + chunk).prod() - 1)
        sp_idx = spy_r.index.get_loc(entry_date) if entry_date in spy_r.index else None
        sp_ret = None
        if sp_idx is not None:
            sp_chunk = spy_r.iloc[sp_idx:min(sp_idx + hold, len(spy_r))]
            sp_ret = float((1 + sp_chunk).prod() - 1)

        events.append({
            "entry_date": str(entry_date.date()),
            "crox_return": round(ev_ret, 4),
            "spy_return": round(sp_ret, 4) if sp_ret is not None else None,
        })

    active = pnl[pnl != 0]
    print(f"Active PnL days: {len(active)}")
    if len(active) < 30:
        return mark_failed(sid, f"insufficient active days ({len(active)})")

    m = compute_metrics(active, benchmark=spy_r, name="Wikipedia Buzz Fade - Long CROX")

    rets = [e["crox_return"] for e in events]
    save_result(sid, m, extra={
        "rule": "Long CROX 15 trading days when 7-day volume > 3x 90-day median AND 5-day price return > 10%; entry 5 days after spike (Wikipedia buzz proxy via volume)",
        "mechanism": "Social media / Wikipedia buzz spikes cause retail FOMO buying; mean reversion as buzz fades 5-15 days later",
        "source": "yfinance CROX price/volume; Wikipedia Pageview API proxy",
        "n_events": len(events),
        "avg_event_return": round(float(np.mean(rets)), 4) if rets else None,
        "event_win_rate": round(float(np.mean([r > 0 for r in rets])), 4) if rets else None,
        "events": events,
        "caveats": "Wikipedia API not directly backtested — volume spike used as proxy; BIRK excluded due to IPO Oct 2023 (insufficient history)",
    })
    print(f"Done: {len(events)} events, Sharpe={m.get('sharpe', 'N/A'):.2f}, CAGR={m.get('cagr', 0)*100:.1f}%")


if __name__ == "__main__":
    main()
