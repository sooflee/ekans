"""PL909 — Plant-Based Meat YoY Decline >15% → Short BYND / Long TSN+PPC Pair"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns

# Event dates from GFI State-of-the-Industry quarterly reports and Circana press releases.
# Trigger: plant-based retail dollar sales YoY < -15% for 3+ consecutive months.
# Sources: GFI SPINS/Circana retail scan data 2022-2024.
#   2022-06-01: Q2 2022 first major -15%+ decline confirmed by GFI mid-year data
#   2022-09-01: Q3 2022 sustained -20%+ decline
#   2023-03-01: Q1 2023 continued -15 to -25% YoY
#   2023-09-01: Q3 2023 continued decline (-20% range per GFI annual report)

TRIGGER_DATES = ["2022-06-01", "2022-09-01", "2023-03-01", "2023-09-01"]
HOLD_DAYS = 40   # ~55 calendar days in trading days

def main():
    sid = "PL909_circana_plant_based_decline_bynd_short"
    try:
        px = load_prices(["BYND", "TSN", "PPC", "SPY"], start="2019-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    missing = [t for t in ["BYND", "TSN", "PPC"] if t not in ret.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    bynd_r = ret["BYND"]
    tsn_r  = ret["TSN"]
    ppc_r  = ret["PPC"]
    spy_r  = ret["SPY"]

    # Pair return: short BYND + 0.5*long TSN + 0.5*long PPC (dollar neutral basket)
    long_basket = 0.5 * tsn_r + 0.5 * ppc_r
    strat_r = -bynd_r + long_basket

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

        # Accumulate pnl without overlap
        for idx, v in seg.items():
            if pnl[idx] == 0.0:
                pnl[idx] = v

        cum_strat = float((1 + seg).prod() - 1)
        cum_bynd  = float((1 + bynd_r.iloc[p:ep]).prod() - 1)
        cum_long  = float((1 + long_basket.iloc[p:ep]).prod() - 1)
        cum_spy   = float((1 + spy_r.iloc[p:ep]).prod() - 1) if p < len(spy_r) else None

        # Check BYND not already in capitulation (>40% off 52w high)
        bynd_52w_window = px["BYND"].iloc[max(0, p - 252):p]
        bynd_entry_px = px["BYND"].iloc[p] if p < len(px) else np.nan
        if len(bynd_52w_window) > 10:
            high_52w = bynd_52w_window.max()
            pct_off_high = (bynd_entry_px - high_52w) / high_52w if high_52w > 0 else None
        else:
            pct_off_high = None

        evts.append({
            "entry_date": str(start_idx.date()),
            "strat_return": round(cum_strat, 4),
            "bynd_return": round(cum_bynd, 4),
            "long_basket_return": round(cum_long, 4),
            "spy_return": round(cum_spy, 4) if cum_spy is not None else None,
            "bynd_pct_off_52w_high_at_entry": round(float(pct_off_high), 3) if pct_off_high is not None else None,
        })

    if not evts:
        return mark_failed(sid, "no valid trigger events")

    ip = pnl[pnl != 0]
    if len(ip) < 30:
        return mark_failed(sid, f"insufficient active days ({len(ip)})")

    m = compute_metrics(ip, benchmark=spy_r, name="Plant-Based Decline: Short BYND / Long TSN+PPC")
    sr = [e["strat_return"] for e in evts]
    save_result(sid, m, extra={
        "rule": "Short BYND / long equal-weighted TSN+PPC when GFI/Circana plant-based retail YoY < -15% for 3+ months; hold ~55 calendar days",
        "mechanism": "Declining plant-based demand hurts BYND revenue + guides lower; conventional protein incumbents benefit as consumers revert",
        "source": "GFI State of the Industry quarterly; Circana scan data; yfinance",
        "n_events": len(evts),
        "avg_strat_return": round(float(np.mean(sr)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in sr])), 4),
        "hold_days": HOLD_DAYS,
        "events": evts,
        "caveats": "Circana data not freely accessible; trigger dates proxied from GFI reports. BYND had severe structural decline 2022-2023 — short bias may partly reflect distressed equity rather than signal. Small sample (4 events).",
    })
    print(f"Done: {len(evts)} events, avg strat return {np.mean(sr):.2%}")

if __name__ == "__main__":
    main()
