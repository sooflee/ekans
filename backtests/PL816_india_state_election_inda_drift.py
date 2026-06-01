"""PL816_india_state_election_inda_drift — India Hindi-Belt State Election Surprise -> INDA vs EEM Post-Result Drift"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL816_india_state_election_inda_drift"

    # Known India Hindi-belt state election counting days with BJP surprise direction
    # direction: +1 = BJP positive surprise (long INDA / short EEM)
    #            -1 = BJP negative surprise (short INDA / long EEM)
    # Source: Wikipedia election results + pre-election poll-of-polls comparisons (manual)
    events = [
        # 2013 results (pre-2018 period, before INDA was well-established)
        # 2018: MP, Rajasthan, Chhattisgarh - BJP expected to win all 3; won only Chhattisgarh small margin
        #       -> BJP negative surprise overall (Congress took MP, Raj)
        ("2018-12-11", -1, "2018 MP/Raj/CG - BJP surprise loss (Congress sweep)"),
        # 2019: Maharashtra/Haryana BJP underperformed polls (expected 220+ in MH, got 105 in alliance)
        ("2019-10-24", -1, "2019 Maharashtra/Haryana BJP reduced majority vs polls"),
        # 2021: West Bengal - BJP expected close contest vs TMC; TMC won decisively (BJP negative surprise)
        ("2021-05-02", -1, "2021 West Bengal - TMC landslide, BJP surprise loss"),
        # 2022: UP state elections - BJP won UP convincingly, surprising most exit polls on margin
        ("2022-03-10", +1, "2022 UP - BJP strong retention (positive surprise vs pessimistic polls)"),
        # 2022: Himachal Pradesh - BJP expected to retain; Congress won (BJP negative surprise)
        ("2022-12-08", -1, "2022 Himachal Pradesh - BJP surprise defeat"),
        # 2022: Gujarat - BJP won with record majority (large positive surprise vs polls)
        ("2022-12-08", +1, "2022 Gujarat - BJP record majority (positive surprise)"),
        # 2023: Karnataka - BJP expected competitive; Congress won comfortable majority (negative surprise)
        ("2023-05-13", -1, "2023 Karnataka - Congress win, BJP surprise loss"),
        # 2023: MP, Raj, CG: BJP swept all 3 vs polls showing Congress ahead in Raj, MP close
        ("2023-12-03", +1, "2023 MP/Raj/CG - BJP sweep (positive surprise, polls showed close race)"),
        # 2024: Haryana - BJP surprise win, most polls showing Congress lead
        ("2024-10-08", +1, "2024 Haryana - BJP surprise win vs Congress-leading polls"),
        # 2024: Maharashtra - BJP/MahaYuti alliance won decisively vs close polls
        ("2024-11-23", +1, "2024 Maharashtra - BJP alliance surprise win"),
        # 2024: Jharkhand - JMM (anti-BJP) won, BJP expected to challenge (negative surprise for BJP)
        ("2024-11-23", -1, "2024 Jharkhand - JMM retention, BJP surprise loss"),
    ]

    # Note: Same date events (2022-12-08, 2024-11-23) — we handle by averaging or taking most significant
    # For simplicity, use a deduplicated version prioritizing the surprise-direction signal
    # For same-day events: use net direction (if mixed, skip or use +1 only if stronger)
    # We'll use a simplified dedup: keep first occurrence per date (Gujarat positive for 2022-12-08,
    # Maharashtra positive for 2024-11-23)
    deduped = {}
    for (date, direction, desc) in events:
        if date not in deduped:
            deduped[date] = (direction, desc)
        else:
            # Average direction if both present
            prev_dir, prev_desc = deduped[date]
            avg_dir = prev_dir + direction  # 0 if mixed, +2 if both positive, -2 if both negative
            if avg_dir == 0:
                # Mixed signal — skip
                deduped[date] = (0, f"Mixed: {prev_desc} + {desc}")
            else:
                net = 1 if avg_dir > 0 else -1
                deduped[date] = (net, f"Net: {prev_desc} + {desc}")

    events_clean = [(d, v[0], v[1]) for d, v in sorted(deduped.items()) if v[0] != 0]

    tickers = ["INDA", "EEM", "SPY"]
    try:
        px = load_prices(tickers, start="2012-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    if "INDA" not in px.columns or "EEM" not in px.columns:
        return mark_failed(sid, "INDA or EEM not available")

    ret = daily_returns(px)
    inda_r = ret["INDA"]
    eem_r = ret["EEM"]
    spy_r = ret["SPY"]

    hold = 15  # 3 weeks = 15 trading days
    event_results = []
    pnl_parts = []

    for event_date_str, direction, desc in events_clean:
        event_date = pd.Timestamp(event_date_str)

        # Entry on next trading day after event
        future_mask = inda_r.index > event_date
        if future_mask.sum() < hold // 2:
            continue

        entry_idx = inda_r.index[future_mask][0]
        pos = inda_r.index.get_loc(entry_idx)
        end_pos = min(pos + hold, len(inda_r))

        inda_window = inda_r.iloc[pos:end_pos]
        eem_window = eem_r.reindex(inda_window.index).fillna(0)
        spy_window = spy_r.reindex(inda_window.index).fillna(0)

        # Pair: long/short INDA vs EEM (beta-1 dollar-neutral)
        # direction=+1: long INDA, short EEM
        # direction=-1: short INDA, long EEM
        trade_pnl = direction * (inda_window - eem_window)

        pnl_parts.append(trade_pnl)

        inda_car = float((1 + inda_window).prod() - 1)
        eem_car = float((1 + eem_window).prod() - 1)
        pair_car = float((1 + trade_pnl).prod() - 1)
        spy_car = float((1 + spy_window).prod() - 1)

        event_results.append({
            "event_date": event_date_str,
            "entry_date": str(entry_idx.date()),
            "description": desc[:60],
            "direction": direction,
            "hold_days": len(inda_window),
            "inda_return": round(inda_car, 4),
            "eem_return": round(eem_car, 4),
            "pair_return": round(pair_car, 4),
            "spy_return": round(spy_car, 4),
        })

    if not event_results:
        return mark_failed(sid, "no valid election events found")

    if len(pnl_parts) < 3:
        return mark_failed(sid, f"insufficient events: only {len(pnl_parts)} valid trades")

    # Combine all trade PnL
    all_pnl = pd.concat(pnl_parts).sort_index()
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep='first')]

    if len(all_pnl) < 30:
        return mark_failed(sid, f"insufficient trading days: {len(all_pnl)}")

    m = compute_metrics(all_pnl, benchmark=spy_r, name="India State Election Surprise -> INDA/EEM Drift")
    m["n_events"] = len(event_results)

    save_result(sid, m, extra={
        "rule": "On India Hindi-belt state election counting day: compute BJP seat-share surprise vs poll-of-polls. If surprise >= +5pp: long INDA / short EEM. If <= -5pp: short INDA / long EEM. Hold 15 trading days.",
        "mechanism": "BJP election surprises signal India reform-narrative persistence or setback, directly impacting foreign institutional investment flows into India equities vs broader EM. Market underreacts to poll-of-polls errors.",
        "source": "Election Commission of India (eci.gov.in) results; Wikipedia election pages for poll-of-polls; yfinance INDA/EEM prices",
        "n_events": len(event_results),
        "avg_pair_return": round(float(np.mean([e["pair_return"] for e in event_results])), 4),
        "win_rate": round(float(np.mean([e["pair_return"] > 0 for e in event_results])), 4),
        "events": event_results,
        "caveat": "Small sample (~8-10 events). BJP surprise classification requires manual poll-of-polls lookup; backtester uses simplified hand-coded directions. EEM beta to INDA varies; correlation-adjusted hedge not applied here.",
    })

    print(f"Done: {len(event_results)} events")
    for e in event_results:
        print(f"  {e['event_date']} dir={e['direction']:+d} ({e['description'][:50]}): pair={e['pair_return']:.3f}")


if __name__ == "__main__":
    main()
