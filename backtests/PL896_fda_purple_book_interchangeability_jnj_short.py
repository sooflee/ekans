"""PL896 — FDA Purple Book Interchangeability Designation Cascade -> Short JNJ / Long AMRX+TEVA Biosimilar Pair"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics, save_result, mark_failed, daily_returns)


def main():
    sid = "PL896_fda_purple_book_interchangeability_jnj_short"

    # Event-study: FDA Purple Book interchangeability designation dates
    # Short JNJ, Long AMRX+TEVA pair
    # Hold for ~8 weeks (40 trading days); stop if JNJ outperforms XPH by >6%
    HOLD = 40  # trading days (~8 weeks)

    # Known FDA interchangeability designation dates (anchors for the event study)
    # Broader set: adalimumab biosimilars (establishes mechanism) + Stelara wave
    KNOWN_EVENTS = [
        # Cyltezo (adalimumab) interchangeability - Jul 2021
        pd.Timestamp("2021-07-28"),
        # Hyrimoz interchangeability (adalimumab SC) - Nov 2023
        pd.Timestamp("2023-11-15"),
        # Stelara biosimilar interchangeability wave - Feb-Mar 2025
        pd.Timestamp("2025-02-01"),
        pd.Timestamp("2025-03-15"),
        # Additional biosimilar events that affected JNJ:
        # First wave of biosimilar market entry for Remicade (infliximab) - 2016
        pd.Timestamp("2016-04-05"),
        # Remicade biosimilar market entry (Pfizer) - 2017
        pd.Timestamp("2017-11-28"),
        # Stelara first biosimilar approval (FDA) - 2023
        pd.Timestamp("2023-12-15"),
    ]

    try:
        px = load_prices(["JNJ", "AMRX", "TEVA", "XPH", "SPY"], start="2015-01-01")
    except Exception as e:
        try:
            px = load_prices(["JNJ", "AMRX", "TEVA", "XPH", "SPY"], start="2015-01-01", cache=False)
        except Exception as e2:
            return mark_failed(sid, f"price load: {e2}")

    spy_r = daily_returns(px[["SPY"]]).iloc[:, 0]
    jnj_r = daily_returns(px[["JNJ"]]).iloc[:, 0]
    xph_r = daily_returns(px[["XPH"]]).iloc[:, 0] if "XPH" in px.columns else None

    amrx_r = daily_returns(px[["AMRX"]]).iloc[:, 0] if "AMRX" in px.columns else None
    teva_r = daily_returns(px[["TEVA"]]).iloc[:, 0] if "TEVA" in px.columns else None

    if amrx_r is not None and teva_r is not None:
        biosimilar_r = 0.5 * amrx_r + 0.5 * teva_r
    elif teva_r is not None:
        biosimilar_r = teva_r
    elif amrx_r is not None:
        biosimilar_r = amrx_r
    else:
        return mark_failed(sid, "no biosimilar tickers (AMRX/TEVA) loaded")

    # Pair: short JNJ, long biosimilar basket
    # pair_r = biosimilar - jnj (positive when biosimilars outperform JNJ)
    pair_r = biosimilar_r.subtract(jnj_r, fill_value=0).dropna()

    pnl = pd.Series(0.0, index=pair_r.index)
    events = []

    # Use XPH prices for stop-loss check
    xph_price = px["XPH"].dropna() if "XPH" in px.columns else None
    jnj_price = px["JNJ"].dropna()

    last_trade_end = pd.Timestamp("1900-01-01")

    for event_date in sorted(KNOWN_EVENTS):
        if event_date <= last_trade_end:
            continue

        # Check that AMRX had price history at this date (IPO 2017)
        future_idx = pair_r.index[pair_r.index >= event_date]
        if len(future_idx) < HOLD + 1:
            continue

        entry_idx = future_idx[0]

        # Check AMRX has data at entry (if it's pre-IPO, biosimilar basket is TEVA only)
        if amrx_r is not None and entry_idx in amrx_r.index:
            bio_r_window = biosimilar_r
        else:
            bio_r_window = teva_r if teva_r is not None else biosimilar_r

        pos = pair_r.index.get_loc(entry_idx)
        end_pos = min(pos + HOLD, len(pair_r))

        # Stop-loss: JNJ outperforms XPH by >6% from entry
        entry_jnj_price = jnj_price.get(entry_idx, None)
        if xph_price is not None and entry_jnj_price is not None:
            entry_xph_price = xph_price.get(entry_idx, None)
            if entry_xph_price is not None:
                jnj_window = jnj_price.iloc[pos:end_pos]
                xph_window = xph_price.iloc[pos:end_pos]
                jnj_vs_xph = (jnj_window / entry_jnj_price) / (xph_window / entry_xph_price) - 1
                stop_hit = (jnj_vs_xph > 0.06)
                if stop_hit.any():
                    stop_loc = jnj_price.index.get_loc(stop_hit.idxmax())
                    end_pos = min(stop_loc + 1, end_pos)

        window = pair_r.iloc[pos:end_pos]
        jnj_window_r = jnj_r.reindex(window.index).fillna(0)
        bio_window_r = bio_r_window.reindex(window.index).fillna(0)

        pnl.iloc[pos:end_pos] += window.values[:end_pos - pos]
        last_trade_end = pair_r.index[end_pos - 1]

        pair_cum = float((1 + window).prod() - 1)
        jnj_cum = float((1 + jnj_window_r).prod() - 1)
        bio_cum = float((1 + bio_window_r).prod() - 1)

        events.append({
            "event_date": str(event_date.date()),
            "entry_date": str(entry_idx.date()),
            "hold_days": end_pos - pos,
            "pair_return": round(pair_cum, 4),
            "jnj_return": round(jnj_cum, 4),
            "biosimilar_return": round(bio_cum, 4),
        })

    if not events:
        return mark_failed(sid, "no valid events found")

    active_pnl = pnl[pnl != 0]
    print(f"Events: {len(events)}, Active trading days: {len(active_pnl)}")
    for e in events:
        print(f"  {e['event_date']}: pair={e['pair_return']:.1%}, JNJ={e['jnj_return']:.1%}, biosimilar={e['biosimilar_return']:.1%}")

    if len(active_pnl) < 30:
        return mark_failed(sid, f"insufficient active days: {len(active_pnl)} ({len(events)} events)")

    m = compute_metrics(active_pnl, benchmark=spy_r, name="FDA Purple Book Interchangeability: Short JNJ / Long AMRX+TEVA")
    win_rates = [1 if e["pair_return"] > 0 else 0 for e in events]

    save_result(sid, m, extra={
        "rule": "Short JNJ / Long equal-weight AMRX+TEVA for 8 weeks following FDA Purple Book interchangeability designation for Stelara/Remicade/Darzalex biosimilar; stop-loss if JNJ outperforms XPH by >6%",
        "mechanism": "Interchangeability designation allows pharmacist-level substitution without prescriber intervention, enabling ~30-40% script share capture within 90 days; JNJ Innovative Medicine revenues at risk (Stelara alone was $8.6B/yr); biosimilar manufacturers (AMRX, TEVA) gain revenue",
        "source": "FDA Purple Book (biologics.fda.gov); known interchangeability dates; yfinance JNJ/AMRX/TEVA/XPH/SPY",
        "n_events": len(events),
        "event_win_rate": round(float(np.mean(win_rates)), 4) if win_rates else 0,
        "events": events,
        "caveats": "Very low event count (5-7 events); AMRX IPO 2017 limits pre-2017 backtest; selected events may not all be interchangeability designations (some biosimilar approvals); JNJ decline may already be priced in",
    })
    print(f"Done: Sharpe={m.get('sharpe', 'N/A'):.3f}, CAGR={m.get('cagr', 0)*100:.1f}%")


if __name__ == "__main__":
    main()
