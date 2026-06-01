"""PL814_chips_act_packaging_amkr_tsm_pair
CHIPS Act Advanced Packaging NOFO -> Long AMKR / Short TSM Dollar-Neutral Pair

On each DOC NIST CHIPS R&D NOFO targeting advanced packaging (NAPMP program),
enter a beta-adjusted dollar-neutral pair: LONG AMKR / SHORT TSM. Hold up to
8 weeks (40 trading days). Stop if pair PnL < -5%.

Known anchor events:
  - 2023-09-22: CHIPS R&D NAPMP Phase 1 announcement (first CHIPS R&D milestone)
  - 2024-03-01: NAPMP Phase 1 NOFO published (~$450M advanced packaging)
  - 2024-06-21: CHIPS advanced packaging awards interim announcement

Beta-neutral sizing: 60-day rolling beta of AMKR vs SPY / TSM vs SPY,
then short TSM at ratio amkr_beta/tsm_beta to equalize market exposure.

Benchmark: SPY.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# -----------------------------------------------------------------------
# CHIPS Act NAPMP NOFO / Advanced Packaging event dates
# Source: commerce.gov/chips public press releases
# -----------------------------------------------------------------------
CHIPS_EVENTS = [
    {
        "event_date": "2023-09-22",
        "description": "CHIPS R&D NAPMP (National Advanced Packaging Manufacturing Program) Phase 1 announced",
        "type": "announcement",
        "funding_mn": 250,
    },
    {
        "event_date": "2024-03-01",
        "description": "NAPMP Phase 1 NOFO published (~$450M advanced packaging solicitation)",
        "type": "nofo_published",
        "funding_mn": 450,
    },
    {
        "event_date": "2024-06-21",
        "description": "CHIPS advanced packaging NAPMP selection announcement",
        "type": "award_announcement",
        "funding_mn": 450,
    },
]

HOLD_DAYS = 40            # 8 weeks
STOP_LOSS_PAIR = -0.05    # pair PnL -5% from entry
BETA_WINDOW = 60          # days to compute rolling beta


def compute_rolling_beta(ret_asset: pd.Series, ret_market: pd.Series,
                         window: int = 60) -> pd.Series:
    """Compute rolling beta of asset vs market using OLS slope."""
    beta = pd.Series(np.nan, index=ret_asset.index)
    for i in range(window, len(ret_asset)):
        y = ret_asset.iloc[i - window:i].dropna().values
        x = ret_market.iloc[i - window:i].dropna().values
        n = min(len(x), len(y))
        if n < 20:
            continue
        x = x[-n:]
        y = y[-n:]
        cov = np.cov(x, y)
        if cov[0, 0] > 0:
            beta.iloc[i] = cov[0, 1] / cov[0, 0]
    return beta


def main():
    sid = "PL814_chips_act_packaging_amkr_tsm_pair"
    tickers = ["AMKR", "TSM", "SPY"]

    try:
        px = load_prices(tickers, start="2022-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()
    amkr_r = ret["AMKR"]
    tsm_r = ret["TSM"]
    idx = ret.index

    # Compute 60-day rolling betas
    beta_amkr = compute_rolling_beta(amkr_r, spy_r, window=BETA_WINDOW)
    beta_tsm = compute_rolling_beta(tsm_r, spy_r, window=BETA_WINDOW)

    print(f"AMKR beta range: {beta_amkr.dropna().min():.2f} - {beta_amkr.dropna().max():.2f}")
    print(f"TSM beta range: {beta_tsm.dropna().min():.2f} - {beta_tsm.dropna().max():.2f}")

    # ---- Event study ----
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)

    event_log = []
    for ev in CHIPS_EVENTS:
        ev_dt = pd.Timestamp(ev["event_date"])

        # Entry at next trading day's open
        future = idx[idx > ev_dt]
        if len(future) == 0:
            ev_record = dict(ev)
            ev_record["status"] = "no_data_after_event"
            event_log.append(ev_record)
            continue

        entry_dt = future[0]
        entry_pos = idx.get_loc(entry_dt)
        max_exit_pos = min(entry_pos + HOLD_DAYS, len(idx))

        # Beta ratio on entry day (use last available beta before entry)
        beta_a = float(beta_amkr.iloc[:entry_pos].dropna().iloc[-1]) if len(beta_amkr.iloc[:entry_pos].dropna()) > 0 else 1.0
        beta_t = float(beta_tsm.iloc[:entry_pos].dropna().iloc[-1]) if len(beta_tsm.iloc[:entry_pos].dropna()) > 0 else 1.0
        hedge_ratio = beta_a / beta_t if beta_t != 0 else 1.0
        hedge_ratio = np.clip(hedge_ratio, 0.3, 3.0)  # reasonable bounds

        # Pair PnL = AMKR_r - hedge_ratio * TSM_r (long AMKR, short TSM)
        amkr_slice = amkr_r.iloc[entry_pos:max_exit_pos].fillna(0)
        tsm_slice = tsm_r.iloc[entry_pos:max_exit_pos].fillna(0)
        pair_daily = amkr_slice - hedge_ratio * tsm_slice

        # Walk forward with stop-loss
        cum_pair = 1.0
        actual_exit_pos = max_exit_pos
        exit_reason = "scheduled_40d"

        for j, p_r in enumerate(pair_daily):
            cum_pair *= (1 + float(p_r))
            if (cum_pair - 1.0) < STOP_LOSS_PAIR:
                exit_reason = "stop_loss"
                actual_exit_pos = entry_pos + j + 1
                break

        # Fill PnL
        for j in range(entry_pos, actual_exit_pos):
            if positions.iloc[j] == 0.0:
                idx_j = j - entry_pos
                pnl.iloc[j] = float(pair_daily.iloc[idx_j])
                positions.iloc[j] = 1.0

        # Cumulative returns on actual hold window
        amkr_cum = float((1 + amkr_slice.iloc[:actual_exit_pos - entry_pos]).prod() - 1)
        tsm_cum = float((1 + tsm_slice.iloc[:actual_exit_pos - entry_pos]).prod() - 1)
        pair_cum = amkr_cum - hedge_ratio * tsm_cum

        exit_dt = idx[actual_exit_pos - 1] if actual_exit_pos > entry_pos else entry_dt

        # Compute ratio of AMKR/TSM pre-entry (20-day MA as reference)
        pre_amkr = px["AMKR"].iloc[max(0, entry_pos - 20):entry_pos]
        pre_tsm = px["TSM"].iloc[max(0, entry_pos - 20):entry_pos]
        ratio_pre = float((pre_amkr / pre_tsm).mean()) if len(pre_tsm) > 0 and pre_tsm.mean() > 0 else None

        ev_record = dict(ev)
        ev_record["entry_date"] = str(entry_dt.date())
        ev_record["exit_date"] = str(exit_dt.date())
        ev_record["exit_reason"] = exit_reason
        ev_record["n_hold_days"] = int(actual_exit_pos - entry_pos)
        ev_record["hedge_ratio"] = round(hedge_ratio, 4)
        ev_record["beta_amkr_at_entry"] = round(beta_a, 3)
        ev_record["beta_tsm_at_entry"] = round(beta_t, 3)
        ev_record["amkr_return"] = round(amkr_cum, 4)
        ev_record["tsm_return"] = round(tsm_cum, 4)
        ev_record["pair_return"] = round(pair_cum, 4)
        ev_record["amkr_tsm_ratio_pre20d_avg"] = round(ratio_pre, 4) if ratio_pre else None

        event_log.append(ev_record)

    n_events = sum(1 for e in event_log if e.get("entry_date"))
    print(f"\nTraded events: {n_events}")
    for e in event_log:
        if e.get("entry_date"):
            print(f"  {e['event_date']} [{e.get('type')}] -> entry {e['entry_date']}, "
                  f"hedge_ratio={e.get('hedge_ratio'):.2f}, "
                  f"pair_return={e.get('pair_return'):.4f}, reason={e.get('exit_reason')}")

    held_pnl = pnl[positions != 0]
    held_spy = spy_r.reindex(held_pnl.index).fillna(0)

    if len(held_pnl) < 20:
        pair_returns = [e.get("pair_return") for e in event_log if e.get("pair_return") is not None]
        avg_ret = float(np.mean(pair_returns)) if pair_returns else None
        return mark_failed(
            sid,
            f"insufficient held days: {len(held_pnl)} (n_events={n_events})",
            extra={
                "events": event_log,
                "avg_pair_return": avg_ret,
            },
        )

    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name="CHIPS Packaging NOFO -> Long AMKR / Short TSM Pair (held-days)",
        positions=positions.reindex(held_pnl.index).fillna(0),
        cost_bps=10,
    )

    pair_returns = [e.get("pair_return") for e in event_log if e.get("pair_return") is not None]
    win_rate = float(np.mean([r > 0 for r in pair_returns])) if pair_returns else None
    avg_ret = float(np.mean(pair_returns)) if pair_returns else None

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "On DOC NIST CHIPS R&D NOFO/announcement explicitly targeting advanced "
                "packaging (NAPMP), enter LONG AMKR + SHORT TSM at 60-day-beta-adjusted "
                "dollar-neutral ratio the next trading day. Hold up to 40 trading days. "
                "Stop-loss if pair PnL < -5%."
            ),
            "mechanism": (
                "CHIPS Act packaging NOFOs signal incremental US domestic OSAT capacity "
                "funding that flows disproportionately to AMKR (US-domiciled, Arizona "
                "advanced-packaging commitment). TSM is more exposed to Taiwan geopolitical "
                "premium and CoWoS capacity constraints outside the CHIPS framework. "
                "Beta-neutral ratio isolates the domestic-packaging narrative vs the "
                "broader semicap-sector beta."
            ),
            "source": (
                "DOC NIST CHIPS R&D program public NOFOs (commerce.gov/chips); "
                "prices via yfinance (auto_adjust=True)"
            ),
            "tickers": ["AMKR", "TSM"],
            "n_events": n_events,
            "event_win_rate": round(win_rate, 4) if win_rate is not None else None,
            "avg_pair_event_return": round(avg_ret, 4) if avg_ret is not None else None,
            "events": event_log,
            "caveats": (
                "N=3 events (2023-2024). Data window is very short. "
                "AMKR is small-cap with meaningful idiosyncratic risk vs TSM. "
                "Beta estimates on 60-day window are noisy for volatile small-cap. "
                "CHIPS funding award winners are not always AMKR-centric. "
                "Beta-neutral pairing reduces market exposure but not sector risk. "
                "Sharpe/CAGR metrics on held-days with N=3 are not statistically meaningful."
            ),
            "beta_window_days": BETA_WINDOW,
            "stop_loss_pct": STOP_LOSS_PAIR,
        },
        pnl=held_pnl,
    )

    print(f"\nDone: {sid}")
    print(f"  n_events: {n_events}, win_rate: {win_rate}, avg_pair_return: {avg_ret}")
    if m.get("sharpe") is not None:
        print(
            f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
            f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
        )
    if m.get("oos_sharpe") is not None:
        print(f"  OOS Sharpe: {m.get('oos_sharpe'):.2f}")


if __name__ == "__main__":
    main()
