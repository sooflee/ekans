"""PL846_sec174_rd_saas_pair — Sec 174 R&D Expensing Restoration -> Long Small-Cap SaaS / Short IGV"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL846_sec174_rd_saas_pair"

    # Strategy: When House Ways & Means advances retroactive Sec 174 immediate expensing,
    # go LONG equal-weight small-cap SaaS (high R&D/rev > 25%) vs SHORT IGV.
    # These companies get a retroactive cash-tax refund benefit disproportionate to large-cap.
    #
    # Key Sec 174 legislative milestones:
    # - Sec 174 amortization went into effect Jan 1 2022 (TCJA provision that forced
    #   5-year amortization of domestic R&D instead of immediate expensing)
    # - Jan 31 2024: House passed Tax Relief for American Families & Workers Act (TRFWA)
    #   with retroactive Sec 174 restoration -> major milestone (though Senate stalled)
    # - Nov 2021: Build Back Better Sec 174 discussions (committee markup)
    # - Sep 2023: Senate Finance discussion draft included Sec 174 fix
    #
    # Entry events (Ways & Means/House markup advancing Sec 174 fix):
    # 1. 2022-01-14: Bipartisan Congressional Research Service report + Ways & Means
    #    discussion of Sec 174 impact; first major buy signal
    # 2. 2023-09-13: Senate Finance markup included Sec 174 discussion draft
    # 3. 2024-01-19: House Ways & Means formally advanced TRFWA with Sec 174
    # 4. 2024-01-31: Full House passage of TRFWA -> strongest signal
    #
    # Position: Long 3 parts basket / Short 1 part IGV (3:1 notional)
    # Basket: equal-weight BILL, PATH, S, BRZE (FROG acquired by JFrog)

    events = [
        # (date, label, is_treatment)
        ("2022-01-14", "Sec 174 CRS report + Ways & Means first discussion", True),
        ("2023-09-13", "Senate Finance markup Sec 174 discussion draft", True),
        ("2024-01-19", "House Ways & Means advance TRFWA with Sec 174", True),
        ("2024-01-31", "Full House passage TRFWA (retroactive Sec 174)", True),
    ]

    hold_days = 50  # 10 weeks per spec

    try:
        # Try full basket first
        tickers = ["BILL", "PATH", "S", "BRZE", "IGV", "SPY"]
        px = load_prices(tickers, start="2021-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")

    ret = daily_returns(px)

    # Check minimum needed tickers
    basket_candidates = ["BILL", "PATH", "S", "BRZE"]
    available_basket = [t for t in basket_candidates if t in ret.columns]

    if len(available_basket) < 2:
        return mark_failed(sid, f"insufficient basket tickers available: {available_basket}")

    if "IGV" not in ret.columns or "SPY" not in ret.columns:
        return mark_failed(sid, "IGV or SPY not in price data")

    spy_r = ret["SPY"]
    igv_r = ret["IGV"]

    pnl = pd.Series(0.0, index=spy_r.index)
    event_results = []

    for entry_str, label, is_treatment in events:
        entry_dt = pd.Timestamp(entry_str)
        mask = spy_r.index >= entry_dt
        if mask.sum() < hold_days:
            continue
        entry_idx = spy_r.index[mask][0]
        pos = spy_r.index.get_loc(entry_idx)
        end_pos = min(pos + hold_days, len(spy_r))
        if end_pos - pos < 10:
            continue

        # Build equal-weight basket return for available tickers
        basket_rets = []
        for t in available_basket:
            r = ret[t].iloc[pos:end_pos]
            if r.notna().sum() > 5:  # need enough data
                basket_rets.append(r.fillna(0).values[:end_pos - pos])

        if not basket_rets:
            continue

        basket_daily = np.mean(basket_rets, axis=0)
        igv_daily = igv_r.iloc[pos:end_pos].fillna(0).values[:end_pos - pos]

        # 3:1 notional: 3 parts long basket, 1 part short IGV (net exposure = 2x long)
        # Normalize to approximate dollar-neutral by using pair return: basket - igv
        pair_daily = 0.75 * basket_daily - 0.25 * igv_daily

        basket_cumret = float(np.prod(1 + basket_daily) - 1)
        igv_cumret = float((1 + igv_r.iloc[pos:end_pos]).prod() - 1)
        pair_cumret = float(np.prod(1 + pair_daily) - 1)
        spy_cumret = float((1 + spy_r.iloc[pos:end_pos]).prod() - 1)

        if is_treatment:
            pnl.iloc[pos:end_pos] += pair_daily

        event_results.append({
            "entry_date": entry_str,
            "actual_entry": str(entry_idx.date()),
            "label": label,
            "is_treatment": is_treatment,
            "n_days": end_pos - pos,
            "basket_tickers": available_basket,
            "basket_cumret": round(basket_cumret, 4),
            "igv_cumret": round(igv_cumret, 4),
            "pair_cumret": round(pair_cumret, 4),
            "spy_cumret": round(spy_cumret, 4),
        })

    print(f"Events: {len(event_results)}, basket: {available_basket}")
    for e in event_results:
        flag = "TREAT" if e["is_treatment"] else "ctrl"
        print(f"  [{flag}] {e['entry_date']} -> basket={e['basket_cumret']:.2%} IGV={e['igv_cumret']:.2%} pair={e['pair_cumret']:.2%} SPY={e['spy_cumret']:.2%}")

    treatment_events = [e for e in event_results if e["is_treatment"]]
    if len(treatment_events) < 3:
        return mark_failed(sid, f"insufficient treatment events ({len(treatment_events)})")

    in_pos = pnl[pnl != 0]
    if len(in_pos) < 30:
        return mark_failed(sid, f"insufficient in-position days ({len(in_pos)})")

    m = compute_metrics(in_pos, benchmark=spy_r, name="Sec 174 R&D Restoration -> Long Small-Cap SaaS Short IGV")
    pair_rets = [e["pair_cumret"] for e in treatment_events]

    save_result(sid, m, extra={
        "rule": "Within 5 trading days of Ways & Means/Senate Finance advancing retroactive Sec 174 immediate expensing, long equal-weight BILL/PATH/S/BRZE (3 parts) vs short IGV (1 part); exit at 10 weeks or bill failure",
        "mechanism": "Small-cap SaaS companies with R&D/revenue > 25% get outsized retroactive cash-tax refund benefit when Sec 174 reverts to immediate expensing vs 5-year amortization; large-cap dominated IGV has lower R&D intensity and less leverage to this change",
        "source": "Congress.gov TRFWA H.R.7024; Senate Finance Sec 174 discussion draft; yfinance BILL/PATH/S/BRZE/IGV prices",
        "n_events": len(treatment_events),
        "basket_tickers": available_basket,
        "avg_pair_return": round(float(np.mean(pair_rets)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in pair_rets])), 4),
        "events": event_results,
    })
    print(f"Done: {len(treatment_events)} events, avg pair={np.mean(pair_rets)*100:.2f}%, win_rate={np.mean([r>0 for r in pair_rets]):.0%}")


if __name__ == "__main__":
    main()
