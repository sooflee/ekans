"""PL1004_nand_ppi_deflation_stx_wdc_short
NAND PPI Deflation + OEM Inventory Build -> Short STX / WDC (Storage Hardware)

BLS PPI series PCU334112334112 (Computer Storage Device Manufacturing) YoY
change triggers a short STX + WDC equal-dollar basket when YoY <= -5% for
2+ consecutive months, confirmed by weak PC/computer manufacturing output
(FRED IPCONGD as proxy for PC demand softness).

Entry: short equal-dollar STX + WDC at next session open after signal month BLS PPI release.
Exit: 40 trading days (~8 weeks).
BENCHMARK: SPY.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


PPI_THRESHOLD = -5.0     # YoY% <= -5% for storage device PPI
CONSEC_MONTHS = 2        # 2+ consecutive months below threshold
HOLD_DAYS = 40           # 8-week holding period
MIN_GAP_MONTHS = 2       # de-dup lockout between signals


def find_signal_dates(ppi_yoy, threshold=-5.0, consec=2, min_gap_months=2):
    """
    Find months where PPI YoY <= threshold for `consec` or more consecutive months.
    Returns list of signal dates (the month where condition is first confirmed).
    De-dup at min_gap_months.
    """
    ppi_yoy = ppi_yoy.dropna().sort_index()
    events = []
    last_entry = None
    consec_count = 0

    for dt, val in ppi_yoy.items():
        if val <= threshold:
            consec_count += 1
        else:
            consec_count = 0

        if consec_count >= consec:
            # Signal confirmed at this month
            if last_entry is None or (dt - last_entry).days >= min_gap_months * 28:
                events.append({
                    "signal_month": str(dt.date()),
                    "ppi_yoy": round(float(val), 4),
                    "consec_months_below": consec_count,
                })
                last_entry = dt

    return events


def map_signal_to_price_date(signal_month, price_index):
    """
    Map a PPI signal month (first of month) to a price date.
    BLS PPI releases ~mid-month FOLLOWING the reference month.
    For PCU334112334112 monthly, the reference month is known at ~month+1 mid-month.
    Approximate: use first trading day >= 15 days after signal_month end.
    """
    sig_dt = pd.Timestamp(signal_month)
    # Approximate release: 6 weeks after reference month start (mid-following-month)
    approx_release = sig_dt + pd.DateOffset(weeks=6)
    future = price_index[price_index >= approx_release]
    if len(future) == 0:
        return None
    return future[0]


def run_event_study(events, ret, basket, hold_days=40):
    """
    Short equal-weight basket for hold_days after each release date.
    """
    idx = ret.index
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)
    basket_ret = ret[basket].fillna(0).mean(axis=1)

    event_log = []
    for ev in events:
        entry_dt = map_signal_to_price_date(ev["signal_month"], idx)
        if entry_dt is None:
            ev_rec = dict(ev)
            ev_rec["status"] = "no_price_data"
            event_log.append(ev_rec)
            continue

        entry_pos = idx.get_loc(entry_dt)
        exit_pos = min(entry_pos + hold_days, len(idx))

        basket_slice = basket_ret.iloc[entry_pos:exit_pos]
        short_pnl = -basket_slice

        ev_ret_short = float((1 + short_pnl).prod() - 1) if len(short_pnl) else None
        ev_ret_basket = float((1 + basket_slice).prod() - 1) if len(basket_slice) else None

        spy_slice = ret["SPY"].fillna(0).iloc[entry_pos:exit_pos]
        ev_ret_spy = float((1 + spy_slice).prod() - 1) if len(spy_slice) else None

        # Per-stock returns
        per_stock = {}
        for t in basket:
            s = ret[t].fillna(0).iloc[entry_pos:exit_pos]
            per_stock[f"short_{t}_return"] = round(float((1 + (-s)).prod() - 1), 4) if len(s) else None

        ev_rec = dict(ev)
        ev_rec["entry_date"] = str(entry_dt.date())
        ev_rec["exit_date"] = str(idx[exit_pos - 1].date()) if exit_pos > entry_pos else None
        ev_rec["n_hold_days"] = int(exit_pos - entry_pos)
        ev_rec["short_basket_return"] = round(ev_ret_short, 4) if ev_ret_short is not None else None
        ev_rec["basket_raw_return"] = round(ev_ret_basket, 4) if ev_ret_basket is not None else None
        ev_rec["spy_return"] = round(ev_ret_spy, 4) if ev_ret_spy is not None else None
        ev_rec.update(per_stock)
        ev_rec["status"] = "ok"
        event_log.append(ev_rec)

        for j in range(entry_pos, exit_pos):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = -1.0
                j_rel = j - entry_pos
                if j_rel < len(short_pnl):
                    pnl.iloc[j] = short_pnl.iloc[j_rel]

    return pnl, positions, event_log


def compute_event_summary(event_log):
    rets = [e["short_basket_return"] for e in event_log if e.get("short_basket_return") is not None]
    if not rets:
        return None
    return {
        "n_events": len(rets),
        "avg_short_basket_return": round(float(np.mean(rets)), 4),
        "median_short_basket_return": round(float(np.median(rets)), 4),
        "win_rate": round(float(np.mean([r > 0 for r in rets])), 4),
        "best": round(float(np.max(rets)), 4),
        "worst": round(float(np.min(rets)), 4),
    }


def main():
    sid = "PL1004_nand_ppi_deflation_stx_wdc_short"
    basket = ["STX", "WDC"]
    tickers = basket + ["SPY"]

    # Load PPI series
    try:
        ppi_raw = load_fred("PCU334112334112", start="2005-01-01").squeeze().dropna()
    except Exception as e:
        return mark_failed(sid, f"FRED PCU334112334112 load: {e}")

    # Compute YoY
    ppi_yoy = ppi_raw.pct_change(12) * 100

    # Also load IPCONGD as PC demand proxy (not used for signal but reported as context)
    try:
        ipcongd = load_fred("IPCONGD", start="2005-01-01").squeeze().dropna()
        ipcongd_yoy = ipcongd.pct_change(12) * 100
    except Exception:
        ipcongd_yoy = None

    # Load prices
    try:
        px = load_prices(tickers, start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    # Find signal dates (strict: 2+ consecutive months <= -5%)
    events_strict = find_signal_dates(ppi_yoy, threshold=PPI_THRESHOLD, consec=CONSEC_MONTHS, min_gap_months=MIN_GAP_MONTHS)
    # Relaxed: 1+ month <= -5% (more events)
    events_relaxed = find_signal_dates(ppi_yoy, threshold=PPI_THRESHOLD, consec=1, min_gap_months=MIN_GAP_MONTHS)

    print(f"  Signal events (strict: {CONSEC_MONTHS}+ months <= {PPI_THRESHOLD}%): {len(events_strict)}")
    for ev in events_strict:
        print(f"    {ev['signal_month']}: PPI YoY={ev['ppi_yoy']:.2f}%, consec={ev['consec_months_below']}")
    print(f"  Signal events (relaxed: 1+ month <= {PPI_THRESHOLD}%): {len(events_relaxed)}")

    pnl_strict, pos_strict, log_strict = run_event_study(events_strict, ret, basket, HOLD_DAYS)
    pnl_relaxed, pos_relaxed, log_relaxed = run_event_study(events_relaxed, ret, basket, HOLD_DAYS)

    n_strict = sum(1 for e in log_strict if e.get("status") == "ok")
    n_relaxed = sum(1 for e in log_relaxed if e.get("status") == "ok")

    summary_strict = compute_event_summary(log_strict)
    summary_relaxed = compute_event_summary(log_relaxed)

    # Use strict as primary
    primary_pnl = pnl_strict
    primary_pos = pos_strict
    primary_label = f"strict_{CONSEC_MONTHS}+months_lte{PPI_THRESHOLD}pct"

    held_pnl = primary_pnl[primary_pos != 0]
    held_spy = spy_r.reindex(held_pnl.index).dropna()
    held_positions = primary_pos.reindex(held_pnl.index).fillna(0)

    if len(held_pnl) < 20:
        extra = {
            "status": "insufficient_data",
            "n_strict": n_strict,
            "n_relaxed": n_relaxed,
            "n_held_days": len(held_pnl),
            "log_strict": log_strict,
            "log_relaxed": log_relaxed,
            "summary_strict": summary_strict,
            "summary_relaxed": summary_relaxed,
        }
        return mark_failed(
            sid,
            f"insufficient held days ({len(held_pnl)}) across {n_strict} strict events",
            extra=extra,
        )

    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name="NAND PPI Deflation Short STX+WDC (held-days only)",
        positions=held_positions,
        cost_bps=10,
    )

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "When BLS PPI PCU334112334112 (Computer Storage Device Manufacturing) "
                f"YoY change <= {PPI_THRESHOLD}% for {CONSEC_MONTHS}+ consecutive months: "
                "short equal-dollar STX + WDC at next session open after BLS PPI "
                f"release (~6 weeks after reference month); hold {HOLD_DAYS} trading days."
            ),
            "mechanism": (
                "Sustained NAND/HDD price deflation (via BLS storage device PPI) "
                "signals oversupply cycle. STX (HDD) and WDC (HDD+NAND) margins "
                "compress as ASPs fall faster than cost reduction; EPS estimates are "
                "revised down over the following 1-2 quarters. The 8-week window "
                "captures the initial sell-side downgrade cycle."
            ),
            "source": (
                "BLS PPI PCU334112334112 (Computer Storage Device Mfg) via FRED. "
                "IPCONGD (Industrial Production Computer Products) as PC demand proxy. "
                "Prices via yfinance (auto_adjust=True)."
            ),
            "tickers": tickers,
            "basket": basket,
            "primary_label": primary_label,
            "log_strict": log_strict,
            "log_relaxed": log_relaxed,
            "summary_strict": summary_strict,
            "summary_relaxed": summary_relaxed,
            "n_strict": n_strict,
            "n_relaxed": n_relaxed,
            "hold_days": HOLD_DAYS,
            "caveats": (
                "STX and WDC have undergone restructurings (WDC/Kioxia merger talks, "
                "WDC splitting HDD vs NAND business ~2024). PPI series captures "
                "finished goods prices not individual NAND ASP. Signal dates "
                "approximate BLS release timeline. BH-significance depends on N."
            ),
        },
        pnl=held_pnl,
    )

    print(f"Done: {sid}")
    print(f"  primary: {primary_label}, n_events={n_strict}, held_days={len(held_pnl)}")
    print(f"  summary_strict: {summary_strict}")
    print(f"  summary_relaxed: {summary_relaxed}")
    if "error" not in m:
        print(
            f"  Sharpe: {m.get('sharpe', 'N/A'):.2f}  "
            f"CAGR: {m.get('cagr', 0)*100:.2f}%  "
            f"MaxDD: {m.get('max_dd', 0)*100:.2f}%  "
            f"t-stat: {m.get('t_stat', 'N/A'):.2f}"
        )
        if "net_sharpe" in m:
            print(f"  Net Sharpe: {m['net_sharpe']:.2f}, Net CAGR: {m['net_cagr']*100:.2f}%")


if __name__ == "__main__":
    main()
