"""PL802 — Argentine RIGI Reform Vote -> YPF Vaca Muerta Re-Rating Event Study"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL802_argentina_rigi_ypf_vote_event"
    # Known Argentine Congress RIGI/reform vote dates
    # Entry: T-5 trading days before vote, hold through vote + T+5 (total ~10 trading days)
    vote_events = [
        {"vote_date": "2024-06-12", "label": "Ley Bases House final passage"},
        {"vote_date": "2024-06-27", "label": "Senate RIGI ratification"},
        {"vote_date": "2023-11-19", "label": "Milei election night reform signal"},
    ]
    DAYS_BEFORE = 5   # enter T-5 before vote
    DAYS_AFTER = 5    # hold T+5 after vote
    YPF_WEIGHT = 1.0
    PAM_WEIGHT = -0.6  # short PAM

    try:
        px = load_prices(["YPF", "PAM", "SPY"], start="2020-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if px.empty or "YPF" not in px.columns:
        return mark_failed(sid, "missing YPF price data")

    available = [t for t in ["YPF", "PAM", "SPY"] if t in px.columns and not px[t].dropna().empty]
    print(f"Available tickers: {available}")

    ret = daily_returns(px)
    ypf_r = ret["YPF"].dropna()
    spy_r = ret["SPY"].dropna()
    if "PAM" in available:
        pam_r = ret["PAM"].dropna()
        use_pair = True
    else:
        pam_r = pd.Series(dtype=float)
        use_pair = False
        print("PAM not available, using YPF long only")

    all_idx = spy_r.index
    pnl = pd.Series(0.0, index=all_idx)
    event_details = []

    for ev in vote_events:
        vote_date = pd.Timestamp(ev["vote_date"])

        # Build common index
        if use_pair:
            common = ypf_r.index.intersection(pam_r.index).intersection(spy_r.index)
        else:
            common = ypf_r.index.intersection(spy_r.index)

        # Find vote date in common index (or nearest future date)
        future_of_vote = common[common >= vote_date]
        if len(future_of_vote) == 0:
            print(f"Event {vote_date.date()}: no data after vote date, skipping")
            continue

        vote_idx = common.get_loc(future_of_vote[0])

        # Entry: T-5 before vote
        entry_pos = max(0, vote_idx - DAYS_BEFORE)
        # Exit: T+5 after vote
        exit_pos = min(vote_idx + DAYS_AFTER, len(common) - 1)
        window = common[entry_pos:exit_pos + 1]

        if len(window) < 3:
            print(f"Event {vote_date.date()}: window too small ({len(window)}), skipping")
            continue

        # PnL: long YPF + short PAM (if available)
        if use_pair:
            pair_ret = (YPF_WEIGHT * ypf_r.reindex(window) +
                        PAM_WEIGHT * pam_r.reindex(window))
        else:
            pair_ret = YPF_WEIGHT * ypf_r.reindex(window)

        pnl.loc[window] = pair_ret.values

        ypf_cum = float((1 + ypf_r.reindex(window)).prod() - 1)
        spy_cum = float((1 + spy_r.reindex(window)).prod() - 1)
        pair_cum = float((1 + pair_ret).prod() - 1)
        pam_cum = float((1 + pam_r.reindex(window)).prod() - 1) if use_pair else None

        det = {
            "vote_date": str(vote_date.date()),
            "entry_date": str(common[entry_pos].date()),
            "label": ev["label"],
            "ypf_return": round(ypf_cum, 4),
            "spy_return": round(spy_cum, 4),
            "pair_return": round(pair_cum, 4),
            "days_held": len(window),
        }
        if pam_cum is not None:
            det["pam_return"] = round(pam_cum, 4)
        event_details.append(det)
        print(f"Event {vote_date.date()} [{ev['label'][:30]}]: YPF {ypf_cum:.2%}, pair {pair_cum:.2%}, SPY {spy_cum:.2%}")

    active_pnl = pnl[pnl != 0]
    print(f"Active trading days: {len(active_pnl)}")

    if len(active_pnl) < 5:
        return mark_failed(sid, f"insufficient active days ({len(active_pnl)})")

    spy_bench = spy_r.reindex(active_pnl.index).dropna()
    m = compute_metrics(active_pnl, benchmark=spy_bench,
                        name="RIGI Vote → Long YPF / Short PAM")
    m["n_events"] = len(event_details)

    save_result(sid, m, extra={
        "rule": "Long YPF (1x) + Short PAM (0.6x) from T-5 before Argentine Congress RIGI/reform vote through T+5 after vote (~10 trading days total). Known votes: Ley Bases House (Jun 12 2024), Senate RIGI (Jun 27 2024), Milei election (Nov 19 2023).",
        "mechanism": "RIGI grants YPF 30-year FX repatriation rights + royalty reductions for Vaca Muerta concessions. Pre-vote drift captures market repricing of RIGI probability. PAM short isolates Vaca Muerta beta from broad Argentine EM risk (PAM is more regulated utility, less Vaca Muerta exposed).",
        "source": "Argentine Congress vote calendar (Direccion de Informacion Parlamentaria); yfinance YPF, PAM, SPY",
        "known_events": [ev["vote_date"] for ev in vote_events],
        "events": event_details,
        "caveats": "Only 3 events total, overlapping dates (Jun 12 and Jun 27 are close). YPF ADR is illiquid and subject to Argentine FX risk. Milei election event is a political analog, not a formal RIGI vote.",
    })
    print(f"Saved result: Sharpe={m.get('sharpe', 'N/A'):.2f}, CAGR={m.get('cagr', 'N/A'):.2%}")


if __name__ == "__main__":
    main()
