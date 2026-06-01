"""PL983_trendforce_hbm_rollover_mu_short
TrendForce HBM/DRAM Contract Price Rollover -> Short Micron (MU) Margin Compression

Event-study design using historical DRAM contract price rollover dates from
TrendForce/DRAMeXchange. Seed events (known from queue):
  1) 2018-08-15 -- DRAM price peak/rollover (TrendForce Q3 2018)
  2) 2021-05-01 -- DRAM spot price rollover (TrendForce Q2 2021)
  3) 2022-07-15 -- DRAM contract price collapse (TrendForce Q3 2022)
  4) 2024-10-15 -- HBM-specific pricing plateau/rollover (TrendForce Q4 2024)

ENTRY: Short MU within 2 business days of TrendForce pricing update publication.
EXIT:  Cover at +40 trading days from entry (per backtest_approach spec).
ALPHA: Measured vs SOXX (sector beta hedge), also vs SPY.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# --------- TrendForce DRAM / HBM pricing rollover events ---------
# Known from strategy spec (backtest_approach + known_events).
# Date = date TrendForce pricing update was published (signal date).
ROLLOVER_EVENTS = [
    {
        "signal_date": "2018-08-15",
        "cycle": "DRAM_Q3_2018",
        "description": "DRAM contract price peak/rollover; first QoQ decline after 5 consecutive increases",
    },
    {
        "signal_date": "2021-05-01",
        "cycle": "DRAM_Q2_2021",
        "description": "DRAM spot price rollover; cycle peak after post-COVID demand surge",
    },
    {
        "signal_date": "2022-07-15",
        "cycle": "DRAM_Q3_2022",
        "description": "DRAM contract price collapse; sharp QoQ decline accelerating",
    },
    {
        "signal_date": "2024-10-15",
        "cycle": "HBM_Q4_2024",
        "description": "HBM-specific pricing plateau/rollover; first HBM QoQ price decline signal",
    },
]

HOLD_DAYS = 40  # per backtest_approach spec: "compute MU return over [0, +40 trading days]"


def run_event_study(events, ret, hold_days=40):
    """
    For each event, short MU for hold_days after signal_date (next trading session).
    Returns (pnl_series, positions_series, event_log).
    pnl = negative of MU return each day (short position).
    """
    idx = ret.index
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)
    mu_ret = ret["MU"].fillna(0)

    event_log = []
    for ev in events:
        sig_dt = pd.Timestamp(ev["signal_date"])
        # Entry at next trading session after signal date
        future_sessions = idx[idx > sig_dt]
        if len(future_sessions) == 0:
            ev_rec = dict(ev)
            ev_rec["status"] = "no_data_after_signal"
            event_log.append(ev_rec)
            continue

        entry_dt = future_sessions[0]
        entry_pos = idx.get_loc(entry_dt)
        exit_pos = min(entry_pos + hold_days, len(idx))

        # Short MU: PnL = -1 * MU daily return
        mu_slice = mu_ret.iloc[entry_pos:exit_pos]
        short_pnl = -mu_slice
        # Cumulative return from short
        ev_ret_mu = float((1 + short_pnl).prod() - 1) if len(short_pnl) else None
        # Raw MU cumulative (for reference)
        ev_ret_mu_raw = float((1 + mu_slice).prod() - 1) if len(mu_slice) else None

        # SOXX return over same window (sector beta reference)
        soxx_slice = ret["SOXX"].fillna(0).iloc[entry_pos:exit_pos]
        ev_ret_soxx = float((1 + soxx_slice).prod() - 1) if len(soxx_slice) else None

        # SPY return over same window
        spy_slice = ret["SPY"].fillna(0).iloc[entry_pos:exit_pos]
        ev_ret_spy = float((1 + spy_slice).prod() - 1) if len(spy_slice) else None

        # Alpha vs SOXX = short_MU_return - SOXX_return (short captures negative divergence)
        alpha_vs_soxx = None
        if ev_ret_mu is not None and ev_ret_soxx is not None:
            alpha_vs_soxx = round(ev_ret_mu - ev_ret_soxx, 4)

        ev_rec = dict(ev)
        ev_rec["entry_date"] = str(entry_dt.date())
        ev_rec["exit_date"] = str(idx[exit_pos - 1].date()) if exit_pos > entry_pos else None
        ev_rec["n_hold_days"] = int(exit_pos - entry_pos)
        ev_rec["short_mu_return"] = round(ev_ret_mu, 4) if ev_ret_mu is not None else None
        ev_rec["mu_raw_return"] = round(ev_ret_mu_raw, 4) if ev_ret_mu_raw is not None else None
        ev_rec["soxx_return"] = round(ev_ret_soxx, 4) if ev_ret_soxx is not None else None
        ev_rec["spy_return"] = round(ev_ret_spy, 4) if ev_ret_spy is not None else None
        ev_rec["alpha_vs_soxx"] = alpha_vs_soxx
        ev_rec["status"] = "ok"
        event_log.append(ev_rec)

        # Fill in pnl + positions (no overlapping events expected given multi-year gaps)
        for j in range(entry_pos, exit_pos):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = -1.0   # short position
                pnl.iloc[j] = short_pnl.iloc[j - entry_pos] if (j - entry_pos) < len(short_pnl) else 0.0

    return pnl, positions, event_log


def main():
    sid = "PL983_trendforce_hbm_rollover_mu_short"
    tickers = ["MU", "SOXX", "SPY"]

    try:
        px = load_prices(tickers, start="2017-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()

    # Run the event study
    pnl, positions, event_log = run_event_study(ROLLOVER_EVENTS, ret, hold_days=HOLD_DAYS)

    n_events = sum(1 for e in event_log if e.get("status") == "ok")
    if n_events == 0:
        return mark_failed(sid, "no valid events found in price data")

    # Event-level summaries
    def event_summary(log):
        rets = [e["short_mu_return"] for e in log if e.get("short_mu_return") is not None]
        alphas = [e["alpha_vs_soxx"] for e in log if e.get("alpha_vs_soxx") is not None]
        if not rets:
            return None
        return {
            "n_events": len(rets),
            "avg_short_mu_return": round(float(np.mean(rets)), 4),
            "median_short_mu_return": round(float(np.median(rets)), 4),
            "win_rate": round(float(np.mean([r > 0 for r in rets])), 4),
            "best": round(float(np.max(rets)), 4),
            "worst": round(float(np.min(rets)), 4),
            "avg_alpha_vs_soxx": round(float(np.mean(alphas)), 4) if alphas else None,
            "median_alpha_vs_soxx": round(float(np.median(alphas)), 4) if alphas else None,
        }

    summary = event_summary(event_log)

    # Use held-days only for compute_metrics to avoid diluting Sharpe with flat stretches
    held_pnl = pnl[positions != 0]
    held_spy = spy_r.reindex(held_pnl.index).dropna()
    held_positions = positions.reindex(held_pnl.index).fillna(0)

    if len(held_pnl) < 20:
        # Too few held days for meaningful Sharpe — record anyway
        extra = {
            "status": "insufficient_data",
            "n_events": n_events,
            "n_held_days": len(held_pnl),
            "events": event_log,
            "summary": summary,
            "caveats": (
                f"Only {n_events} events in sample; insufficient held days ({len(held_pnl)}) "
                "for robust Sharpe estimation. N=4 events is the full universe."
            ),
        }
        return mark_failed(
            sid,
            f"insufficient held days ({len(held_pnl)}) across {n_events} events",
            extra=extra,
        )

    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name="TrendForce DRAM/HBM Rollover Short MU (held-days only)",
        positions=held_positions,
        cost_bps=10,
    )

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "TRIGGER: TrendForce publishes quarterly DRAM/HBM contract pricing "
                "report showing first QoQ price decline after 3+ consecutive quarterly "
                "increases (rollover signal). "
                "ENTRY: Short MU within 2 business days of TrendForce publication. "
                "EXIT: Cover at +40 trading days from entry."
            ),
            "mechanism": (
                "MU derives ~30-40% of revenue from DRAM and HBM. Contract price "
                "rollover signals margin compression cycle beginning: ASPs fall while "
                "wafer costs are sticky, compressing gross margins over the next 1-2 "
                "quarters. Market often remains anchored to peak-cycle P/E at the "
                "rollover date (consensus lags TrendForce data by 1-2 qtrs), creating "
                "a window to short before EPS downgrades and multiple compression."
            ),
            "source": (
                "TrendForce quarterly DRAM/HBM contract pricing reports "
                "(trendforce.com/news); event dates embedded from strategy spec. "
                "Prices via yfinance (auto_adjust=True)."
            ),
            "tickers": tickers,
            "events": event_log,
            "summary": summary,
            "hold_days": HOLD_DAYS,
            "n_events": n_events,
            "caveats": (
                "Event study with N=4 events; statistical significance is extremely "
                "limited. Event 4 (2024-10-15 HBM rollover) is the first HBM-specific "
                "cycle and may differ from prior DRAM cycles. Short MU carries "
                "significant event risk from positive AI demand surprises. "
                "Sharpe/CAGR computed on held-day returns only (160 trading days total). "
                "BH-significance unlikely given small N."
            ),
        },
        pnl=held_pnl,
    )

    print(f"Done: {sid}")
    print(f"  n_events={n_events}, hold_days={HOLD_DAYS}, held_days={len(held_pnl)}")
    print(f"  Event summary: {summary}")
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
