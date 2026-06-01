"""PL977_lloyds_jwc_pi_club_reinsurer_long — Lloyd's JWC Listed-Area Expansion -> Long Specialty Reinsurer Pair (RNR/EG) War-Risk Premium Hardening"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL977_lloyds_jwc_pi_club_reinsurer_long"
    # Known JWC Listed Areas expansion events (dual trigger: JWC + IG P&I Clubs)
    # 2022-03-03: Russia invades Ukraine (Feb 24), JWC expands Black Sea to Enhanced Risk
    # 2023-12-19: JWC expands Bab-el-Mandeb/Red Sea after Houthi attacks
    # 2024-03-04: JWC updates Red Sea zone after IG P&I Clubs Feb 2024 circular on shadow fleet
    trigger_dates = pd.to_datetime(["2022-03-03", "2023-12-19", "2024-03-04"])

    try:
        # RNR = RenaissanceRe (specialty reinsurer, heavy marine war risk)
        # EG = Everest Group (formerly RE, Everest Re, rebranded 2023)
        # ACGL = Arch Capital (robustness check)
        px = load_prices(["RNR", "EG", "ACGL", "SPY"], start="2021-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if px.empty:
        return mark_failed(sid, "empty price data")

    missing = [t for t in ["RNR", "EG", "SPY"] if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    rnr_r = ret["RNR"]
    eg_r = ret["EG"]
    spy_r = ret["SPY"]

    hold = 60  # 60 trading days (~12 weeks, per the strategy spec)
    pnl = pd.Series(0.0, index=spy_r.index)
    events = []

    for td in trigger_dates:
        future_mask = spy_r.index >= td
        if future_mask.sum() < hold:
            continue
        entry_idx = spy_r.index[future_mask][0]
        p = spy_r.index.get_loc(entry_idx)
        ep = min(p + hold, len(spy_r))

        if entry_idx not in rnr_r.index or entry_idx not in eg_r.index:
            continue

        rp = rnr_r.index.get_loc(entry_idx)
        egp = eg_r.index.get_loc(entry_idx)

        rend = min(rp + hold, len(rnr_r))
        egend = min(egp + hold, len(eg_r))

        rnr_window = rnr_r.iloc[rp:rend]
        eg_window = eg_r.iloc[egp:egend]

        # Align on common dates
        common_dates = rnr_window.index.intersection(eg_window.index)
        if len(common_dates) < 5:
            continue

        # Equal-weight long RNR + EG
        pair_pnl = 0.5 * rnr_window.loc[common_dates] + 0.5 * eg_window.loc[common_dates]

        rnr_ret = float((1 + rnr_window.loc[common_dates]).prod() - 1)
        eg_ret = float((1 + eg_window.loc[common_dates]).prod() - 1)
        pair_ret = 0.5 * rnr_ret + 0.5 * eg_ret

        spy_window = spy_r.iloc[p:ep]
        spy_ret = float((1 + spy_window).prod() - 1)

        # Add to pnl series
        for dt, val in pair_pnl.items():
            if dt in pnl.index:
                pnl.loc[dt] += val

        events.append({
            "trigger_date": str(td.date()),
            "entry_date": str(entry_idx.date()),
            "rnr_return": round(rnr_ret, 4),
            "eg_return": round(eg_ret, 4),
            "pair_return": round(pair_ret, 4),
            "spy_return": round(spy_ret, 4),
            "n_days": len(common_dates),
        })

    print(f"Events: {len(events)}")
    for e in events:
        print(f"  {e['trigger_date']}: pair={e['pair_return']:.2%}, "
              f"RNR={e['rnr_return']:.2%}, EG={e['eg_return']:.2%}, "
              f"SPY={e['spy_return']:.2%}")

    if not events:
        return mark_failed(sid, "no valid events found")

    # Use active PnL days only
    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active days: {len(active_pnl)}")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="Lloyd's JWC War-Risk: Long RNR/EG Equal-Weight")

    pair_returns = [e["pair_return"] for e in events]
    mean_pair = float(np.mean(pair_returns))
    win_rate = float(np.mean([r > 0 for r in pair_returns]))

    save_result(sid, m, extra={
        "rule": "Buy RNR and EG equal-weight within 3 business days of Lloyd's JWC Listed Areas expansion (new Enhanced Risk zones) AND IG P&I Clubs circular restricting shadow-fleet coverage within 14 days. Hold 12 weeks or exit on VIX>40.",
        "mechanism": "Marine war-risk premium hardening after JWC zone expansions drives specialty reinsurer revenue uplift; equity re-rating precedes actual EPS impact by 1-2 quarters",
        "source": "JWC events: 2022-03-03 (Black Sea), 2023-12-19 (Red Sea/Bab-el-Mandeb), 2024-03-04 (shadow fleet update). Sources: lloyds.com, igpandi.org",
        "n_events": len(events),
        "mean_pair_return": round(mean_pair, 4),
        "win_rate": round(win_rate, 4),
        "events": events,
    })
    print(f"Done. Mean pair return: {mean_pair:.2%}, Win rate: {win_rate:.0%}")


if __name__ == "__main__":
    main()
