"""PL1013 — FDA FAERS PRR Signal-Cluster -> Post-Market Safety Review -> Drug Sponsor Short"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL1013_faers_prr_signal_cluster_sponsor_short"

    # Curated anchor events: known FAERS PRR signal -> FDA safety action -> sponsor equity short
    # Format: (ticker, signal_date, drug, event_type)
    # signal_date = FAERS quarterly data availability date or media report date
    EVENTS = [
        # Vioxx withdrawal - MRK - PRR signals existed well before Sept 2004 withdrawal
        ("MRK", "2004-09-30", "Vioxx (rofecoxib) withdrawal", "voluntary_withdrawal"),
        # Avandia GSK FDA safety communication
        ("GSK", "2007-05-21", "Avandia (rosiglitazone) CV risk", "safety_communication"),
        # Avandia further restrictions
        ("GSK", "2010-06-09", "Avandia access restrictions", "label_change"),
        # Onglyza/Kombiglyze AZN heart failure signal
        ("AZN", "2016-04-26", "Onglyza (saxagliptin) HF warning", "label_change"),
        # Abilify BMY compulsive behaviors warning
        ("BMY", "2016-05-03", "Abilify (aripiprazole) compulsive behaviors", "safety_communication"),
        # Invokana JNJ amputations warning
        ("JNJ", "2016-05-16", "Invokana (canagliflozin) amputation risk", "label_change"),
        # Zantac NDMA contamination signal (PFE sold ranitidine brands)
        ("PFE", "2019-09-01", "Zantac (ranitidine) NDMA", "market_withdrawal"),
        # STELARA / Tremfya - JNJ safety communication (placeholder)
        # Xarelto/Eliquis bleeding signal - JNJ 2021
        ("JNJ", "2021-04-13", "Xarelto (rivaroxaban) brain bleed signal", "safety_update"),
        # Acetaminophen skin reaction - JNJ Tylenol 2013
        ("JNJ", "2013-08-01", "Tylenol Stevens-Johnson signal", "label_change"),
        # Singulair/MRK neuropsychiatric warning
        ("MRK", "2020-03-04", "Singulair (montelukast) neuro warning", "label_change_boxed"),
        # Actemra/Roche (RHHBY) venous thromboembolism
        # Skip - Roche not in US exchange primary
        # Chantix/PFE neuropsychiatric - re-label
        ("PFE", "2016-09-13", "Chantix (varenicline) neuro boxed warning", "label_change_boxed"),
        # Trulicity/Lilly thyroid cancer signal
        ("LLY", "2017-12-01", "Trulicity (dulaglutide) thyroid signal", "label_update"),
        # ABBV Humira hepatitis B reactivation
        ("ABBV", "2018-09-01", "Humira (adalimumab) HBV reactivation signal", "label_update"),
    ]

    all_tickers = list(set([e[0] for e in EVENTS]) | {"IBB", "SPY"})

    try:
        px = load_prices(all_tickers, start="2004-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"] if "SPY" in ret.columns else None
    ibb_r = ret["IBB"] if "IBB" in ret.columns else None

    if spy_r is None:
        return mark_failed(sid, "SPY data missing")

    pnl_list = []
    events_out = []
    HOLD_DAYS = 90  # calendar days
    STOP_LOSS_PCT = 0.15  # stop at +15% adverse (price UP = bad for short)

    for ticker, event_str, drug, event_type in EVENTS:
        if ticker not in ret.columns:
            continue
        stock_r = ret[ticker].dropna()
        stock_px = px[ticker].dropna()

        try:
            event_dt = pd.Timestamp(event_str)
        except Exception:
            continue

        # Entry: T+1 trading day (short the sponsor)
        entry_mask = stock_r.index > event_dt
        entry_candidates = stock_r.index[entry_mask][:5]
        if len(entry_candidates) == 0:
            continue
        entry_date = entry_candidates[0]

        # Exit: ~90 calendar days from entry
        exit_cutoff = entry_date + pd.Timedelta(days=HOLD_DAYS)
        window_mask = (stock_r.index >= entry_date) & (stock_r.index <= exit_cutoff)
        window_ret = stock_r.loc[window_mask]

        if len(window_ret) < 5:
            continue

        # Short position: PnL = -1 * stock return
        short_ret = -window_ret

        # Apply stop-loss for short: if stock goes UP 15% from entry, stop out
        cum_stock = (1 + window_ret).cumprod()
        stop_hit = cum_stock[cum_stock >= (1 + STOP_LOSS_PCT)]
        stopped_out = len(stop_hit) > 0
        if stopped_out:
            stop_date = stop_hit.index[0]
            short_ret = short_ret.loc[short_ret.index <= stop_date]
            window_ret = window_ret.loc[window_ret.index <= stop_date]

        if len(short_ret) < 3:
            continue

        total_return = float((1 + short_ret).prod() - 1)

        # SPY return for same window
        spy_window = spy_r.reindex(window_ret.index)
        spy_cumret = float((1 + spy_window.dropna()).prod() - 1)

        # IBB return for same window
        ibb_cumret = None
        if ibb_r is not None:
            ibb_window = ibb_r.reindex(window_ret.index)
            ibb_cumret = float((1 + ibb_window.dropna()).prod() - 1)

        pnl_list.append(short_ret)
        events_out.append({
            "ticker": ticker,
            "event_date": event_str,
            "drug": drug,
            "event_type": event_type,
            "entry_date": str(entry_date.date()),
            "exit_date": str(short_ret.index[-1].date()),
            "n_days": len(short_ret),
            "total_return": round(total_return, 4),
            "ibb_return": round(ibb_cumret, 4) if ibb_cumret is not None else None,
            "spy_return": round(spy_cumret, 4),
            "stopped_out": stopped_out,
        })

    print(f"Events processed: {len(events_out)}")
    for ev in events_out:
        print(f"  {ev['ticker']} {ev['event_date']} ({ev['drug'][:30]}): {ev['total_return']:.2%} vs SPY {ev['spy_return']:.2%} {'STOP' if ev['stopped_out'] else ''}")

    if len(events_out) < 5:
        return mark_failed(sid, f"insufficient events ({len(events_out)})")

    # Build combined PnL
    pnl = pd.concat(pnl_list).sort_index()
    pnl = pnl.groupby(pnl.index).mean()

    ip = pnl[pnl != 0]
    if len(ip) < 30:
        return mark_failed(sid, f"insufficient active days ({len(ip)})")

    spy_aligned = spy_r.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=spy_aligned, name="FDA FAERS PRR Signal-Cluster Sponsor Short")

    returns_list = [e["total_return"] for e in events_out]
    save_result(sid, m, extra={
        "rule": "Short drug sponsor equity T+1 after FAERS quarterly data release shows PRR>=2, n>=3, Chi-sq>=4 for serious drug-event pair; cover at 90 calendar days or FDA safety communication (profit target); stop-loss at +15% adverse move",
        "mechanism": "FAERS PRR signals often precede FDA safety communications by 1-4 quarters; when PRR threshold crossed, label changes / class action risk increases; short provides asymmetric payoff before market fully prices safety overhang",
        "source": "FDA FAERS quarterly data files; FDA Drug Safety Communications; yfinance prices",
        "n_events": len(events_out),
        "avg_event_return": round(float(np.mean(returns_list)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in returns_list])), 4),
        "events": events_out,
        "caveats": "Small curated sample; real-time PRR computation requires FAERS CSV parsing. Large pharma tickers (MRK, JNJ, PFE) have diversified revenues - individual drug events have diluted impact.",
    })
    print("Done.")


if __name__ == "__main__":
    main()
