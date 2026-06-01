"""PL911 — HPAI Receding + Egg Price Mean-Reversion → Short CALM / Long POST Hedge"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns

# Manually annotated trigger dates: HPAI wave receding + egg wholesale >25% off 90-day high.
# Sources: USDA APHIS HPAI commercial-flock detection dashboard; USDA AMS LM_EG201 egg price CSV;
#          BLS CPI eggs (FRED APU0000708111) as secondary confirmation.
#   2022-08-01: First HPAI wave flock detections near zero, egg price pulling back from Mar-2022 peak
#   2023-03-01: Egg price mean-reversion after Feb-2023 secondary peak; HPAI quiet
#   2025-02-01: Second major HPAI wave (late 2024) receding; Jan-2025 price highs passed

TRIGGER_DATES = ["2022-08-01", "2023-03-01", "2025-02-01"]
HOLD_DAYS = 30   # ~6 calendar weeks in trading days
HEDGE_RATIO = 0.5  # long POST at 0.5x dollar weight vs short CALM

def main():
    sid = "PL911_calm_hpai_recede_egg_meanrev_short"
    try:
        px = load_prices(["CALM", "POST", "SPY"], start="2020-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    if "CALM" not in ret.columns or "POST" not in ret.columns:
        return mark_failed(sid, "missing CALM or POST returns")

    calm_r = ret["CALM"]
    post_r = ret["POST"]
    spy_r  = ret["SPY"]

    # Strategy PnL: short CALM + 0.5x long POST (net short, partial hedge)
    # Net weight: -1 CALM + 0.5 POST
    strat_r = -calm_r + HEDGE_RATIO * post_r

    # Use egg CPI as secondary signal check (informational)
    try:
        egg_cpi = load_fred("APU0000708111", start="2020-01-01").squeeze()
    except Exception:
        egg_cpi = pd.Series(dtype=float)

    pnl = pd.Series(0.0, index=strat_r.index)
    evts = []

    for entry_str in TRIGGER_DATES:
        entry_dt = pd.Timestamp(entry_str)
        mask = strat_r.index >= entry_dt
        if mask.sum() < HOLD_DAYS:
            continue
        start_idx = strat_r.index[mask][0]
        p = strat_r.index.get_loc(start_idx)
        ep = min(p + HOLD_DAYS, len(strat_r))
        seg = strat_r.iloc[p:ep]

        # Accumulate pnl (avoid overlapping windows)
        for idx, v in seg.items():
            if pnl[idx] == 0.0:
                pnl[idx] = v

        cum_strat = float((1 + seg).prod() - 1)
        calm_seg  = calm_r.iloc[p:ep]
        post_seg  = post_r.iloc[p:ep]
        spy_seg   = spy_r.iloc[p:ep]
        cum_calm  = float((1 + calm_seg).prod() - 1)
        cum_post  = float((1 + post_seg).prod() - 1)
        cum_spy   = float((1 + spy_seg).prod() - 1) if len(spy_seg) > 0 else None

        # RSI check at entry (informational)
        calm_window = calm_r.iloc[max(0, p-14):p]
        if len(calm_window) >= 7:
            gains = calm_window.clip(lower=0).mean()
            losses = (-calm_window.clip(upper=0)).mean()
            rsi = 100 - 100 / (1 + gains / losses) if losses > 0 else 100.0
        else:
            rsi = None

        evts.append({
            "entry_date": str(start_idx.date()),
            "strat_return": round(cum_strat, 4),
            "calm_return": round(cum_calm, 4),
            "post_return": round(cum_post, 4),
            "spy_return": round(cum_spy, 4) if cum_spy is not None else None,
            "calm_rsi14_at_entry": round(rsi, 1) if rsi is not None else None,
        })

    if not evts:
        return mark_failed(sid, "no valid trigger events")

    ip = pnl[pnl != 0]
    if len(ip) < 30:
        return mark_failed(sid, f"insufficient active days ({len(ip)})")

    m = compute_metrics(ip, benchmark=spy_r, name="HPAI Recede: Short CALM / Long POST")
    sr = [e["strat_return"] for e in evts]
    save_result(sid, m, extra={
        "rule": "Short CALM + 0.5x long POST when HPAI detections zero 4wks and egg price >25% off 90-day high; hold ~6wk",
        "mechanism": "Egg price mean-reversion into CALM earnings compresses gross margins; POST hedge dampens broad-staples beta",
        "source": "USDA APHIS HPAI dashboard; USDA AMS LM_EG201; FRED APU0000708111; yfinance",
        "n_events": len(evts),
        "avg_strat_return": round(float(np.mean(sr)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in sr])), 4),
        "hold_days": HOLD_DAYS,
        "hedge_ratio": HEDGE_RATIO,
        "events": evts,
        "caveats": "Only 3 events identified. Trigger dates hand-annotated from HPAI/egg-price archives; live signal needs real-time USDA data feed.",
    })
    print(f"Done: {len(evts)} events, avg strat return {np.mean(sr):.2%}")

if __name__ == "__main__":
    main()
