"""PL1030 — RRP Sudden Re-Accumulation ($50B→$300B) → T-Bill Scarcity → Long IEF Belly"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL1030_rrp_reaccumulation_ief_belly"
    try:
        # Load ON RRP balance and yield curve data from FRED
        rrp = load_fred("RRPONTSYD", start="2010-01-01").squeeze()
        dgs2 = load_fred("DGS2", start="2010-01-01").squeeze()
        dgs10 = load_fred("DGS10", start="2010-01-01").squeeze()
    except Exception as e:
        return mark_failed(sid, f"FRED load: {e}")

    if rrp.empty:
        return mark_failed(sid, "RRPONTSYD empty")

    # Fill forward for daily (FRED publishes business days; RRP is daily)
    rrp = rrp.ffill()
    dgs2 = dgs2.ffill()
    dgs10 = dgs10.ffill()

    # Identify signal dates:
    # - 30-day rolling minimum of RRPONTSYD < 50 (near-zero floor)
    # - Current RRPONTSYD > 300 (re-accumulation threshold)
    # - This is the first crossing above 300 from that low (to avoid re-triggering)
    # - Yield curve inverted: DGS2 > DGS10 at signal date
    rrp_30min = rrp.rolling(window=30, min_periods=20).min()

    # Build a common index across all series
    common = rrp.index.intersection(dgs2.index).intersection(dgs10.index)
    rrp_c = rrp.reindex(common).ffill()
    rrp_30min_c = rrp_30min.reindex(common).ffill()
    dgs2_c = dgs2.reindex(common).ffill()
    dgs10_c = dgs10.reindex(common).ffill()

    triggers = []
    in_signal_window = False  # prevent re-triggering while already in a high-RRP regime
    cooldown = 0  # trading days since last signal (prevent overlap)

    for i in range(1, len(common)):
        d = common[i]
        rrp_val = float(rrp_c.iloc[i])
        min30 = float(rrp_30min_c.iloc[i])
        prev_rrp = float(rrp_c.iloc[i - 1])

        if np.isnan(rrp_val) or np.isnan(min30):
            continue

        # Reset flag when RRP drops back below 50B (regime resets)
        if rrp_val < 50:
            in_signal_window = False
            cooldown = max(0, cooldown - 1)
            continue

        cooldown = max(0, cooldown - 1)

        # Signal: crossed above 300 from a 30-day min < 50, yield curve inverted
        if (not in_signal_window and
                prev_rrp <= 300 and rrp_val > 300 and
                min30 < 50 and cooldown == 0):
            # Check yield curve inversion
            d2 = float(dgs2_c.iloc[i]) if not np.isnan(float(dgs2_c.iloc[i])) else None
            d10 = float(dgs10_c.iloc[i]) if not np.isnan(float(dgs10_c.iloc[i])) else None
            if d2 is not None and d10 is not None and d2 > d10:
                triggers.append(d)
                in_signal_window = True
                cooldown = 60  # no new signal for 60 trading days
        elif rrp_val < 200:
            # RRP has retreated significantly; allow new signal in future
            in_signal_window = False

    # Also add known events that meet conditions but may not have strict inversion
    # The 2021 event: May 2021, yield curve was NOT inverted (2Y < 10Y)
    # so we also try a version WITHOUT the inversion filter to capture more events
    triggers_noinv = []
    in_signal_window2 = False
    cooldown2 = 0

    for i in range(1, len(common)):
        rrp_val = float(rrp_c.iloc[i])
        min30 = float(rrp_30min_c.iloc[i])
        prev_rrp = float(rrp_c.iloc[i - 1])

        if np.isnan(rrp_val) or np.isnan(min30):
            continue

        if rrp_val < 50:
            in_signal_window2 = False
            cooldown2 = max(0, cooldown2 - 1)
            continue

        cooldown2 = max(0, cooldown2 - 1)

        if (not in_signal_window2 and
                prev_rrp <= 300 and rrp_val > 300 and
                min30 < 50 and cooldown2 == 0):
            triggers_noinv.append(common[i])
            in_signal_window2 = True
            cooldown2 = 60
        elif rrp_val < 200:
            in_signal_window2 = False

    print(f"Signals (with inversion filter): {len(triggers)}")
    print(f"Signals (no inversion filter): {len(triggers_noinv)}")
    print(f"  Triggers (no-inv): {[str(t.date()) for t in triggers_noinv]}")

    # Use no-inversion version to have more events (more robust)
    use_triggers = triggers_noinv if len(triggers_noinv) >= len(triggers) else triggers

    if not use_triggers:
        return mark_failed(sid, "no RRP re-accumulation events found in data")

    try:
        px = load_prices(["IEF", "SHY", "SPY"], start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    ret = daily_returns(px)
    ief_r = ret["IEF"]
    spy_r = ret["SPY"]
    hold_days = 30  # calendar days ≈ ~21 trading days

    # Build a daily PnL series aligned to IEF return index
    pnl = pd.Series(0.0, index=ief_r.index)
    events = []

    for sig_date in use_triggers:
        # Find next trading day at or after signal date
        future_mask = ief_r.index >= sig_date
        if future_mask.sum() < 5:
            continue
        entry_idx = ief_r.index[future_mask][0]
        entry_pos = ief_r.index.get_loc(entry_idx)

        # Approx 30 calendar days ≈ 21 trading days; use ~22 to be safe
        approx_td = 22
        exit_pos = min(entry_pos + approx_td, len(ief_r))
        window = ief_r.iloc[entry_pos:exit_pos]

        # Compute cumulative for event stats
        cum_ret = float((1 + window).prod() - 1)

        # Check stop-loss: IEF closes > 3% below entry
        entry_price_idx = entry_pos
        cum_price = 1.0
        stop_triggered = False
        actual_exit = exit_pos
        for j, r in enumerate(window):
            cum_price *= (1 + r)
            if cum_price < 0.97:  # 3% stop
                stop_triggered = True
                actual_exit = entry_pos + j + 1
                break

        window_actual = ief_r.iloc[entry_pos:actual_exit]
        cum_ret_actual = float((1 + window_actual).prod() - 1)

        # Mark PnL (no overlap handling for now — single event)
        pnl.iloc[entry_pos:actual_exit] = window_actual.values[: actual_exit - entry_pos]

        # SPY over same window
        spy_mask = spy_r.index >= sig_date
        spy_cum = None
        if spy_mask.sum() >= 5:
            sp_entry = spy_r.index[spy_mask][0]
            sp_pos = spy_r.index.get_loc(sp_entry)
            sp_win = spy_r.iloc[sp_pos: sp_pos + approx_td]
            spy_cum = float((1 + sp_win).prod() - 1)

        events.append({
            "signal_date": str(sig_date.date()),
            "entry_date": str(entry_idx.date()),
            "ief_return": round(cum_ret_actual, 4),
            "spy_return": round(spy_cum, 4) if spy_cum is not None else None,
            "stop_triggered": stop_triggered,
        })

    if not events:
        return mark_failed(sid, "no valid events after filtering")

    active = pnl[pnl != 0]
    print(f"Active trading days: {len(active)}")
    print(f"Events: {events}")

    if len(active) < 10:
        return mark_failed(sid, f"too few active trading days ({len(active)}) — only {len(events)} event(s)")

    m = compute_metrics(active, benchmark=spy_r, name="RRP Re-Accumulation → Long IEF Belly")
    ief_returns = [e["ief_return"] for e in events]
    save_result(sid, m, extra={
        "rule": "Long IEF for ~30 calendar days when ON RRP crosses $300B from a 30-day min <$50B",
        "mechanism": "RRP surge signals T-bill scarcity, driving demand into 7-10Y belly; Fed QE pace outstrips T-bill issuance",
        "source": "FRED RRPONTSYD, DGS2, DGS10; yfinance IEF",
        "n_events": len(events),
        "avg_event_return": round(float(np.mean(ief_returns)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in ief_returns])), 4),
        "events": events,
    })
    print(f"Done: {len(events)} events, Sharpe={m.get('sharpe', 'N/A')}")


if __name__ == "__main__":
    main()
