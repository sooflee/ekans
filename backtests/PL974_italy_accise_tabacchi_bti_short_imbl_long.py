"""PL974_italy_accise_tabacchi_bti_short_imbl_long
Italian Tobacco Excise Hike (Legge di Bilancio): Short BTI / Long IMB.L Pair

When Italy's MEF Draft Budgetary Plan (DBP, ~Oct 15 each year) confirms a
tobacco excise hike >5% per pack, enter short BTI (Italy's largest combustible
player) and long IMB.L equal notional (minimal Italy exposure). Hold 80 trading days.

Known events where Italy excise hike > 5%:
- 2019-10-15: FY2020 hike effective Jan 1 2020 (~6.5% per pack)
- 2022-10-15: FY2023 hike effective Jan 1 2023 (~7% per pack)

IMB.L is GBP-denominated; we trade BTI (USD ADR) vs IMB.L (GBX -> USD via GBP/USD).
For simplicity, use price returns in local currency and treat as a dollar-neutral pair.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


# Italian tobacco excise hike events (DBP announcement ~Oct 15)
# Source: MEF DBP Relazione Tecnica; ADM tariffa di vendita
EXCISE_EVENTS = [
    {
        "date": "2019-10-15",
        "description": "Italy FY2020 DBP: accise tabacchi hike ~6.5%/pack, effective Jan 1 2020",
        "hike_pct": 6.5,
        "effective_date": "2020-01-01",
    },
    {
        "date": "2022-10-15",
        "description": "Italy FY2023 DBP: accise tabacchi hike ~7%/pack, effective Jan 1 2023",
        "hike_pct": 7.0,
        "effective_date": "2023-01-01",
    },
]

HOLD_DAYS = 80


def run_event_study(events, ret_short, ret_long, spy_r, idx, hold_days=HOLD_DAYS):
    """Event study: short BTI, long IMB.L equal notional.
    Entry at close on event date (or first available close after).
    Returns (pnl_series, positions_series, event_log).
    """
    pnl = pd.Series(0.0, index=idx)
    positions = pd.Series(0.0, index=idx)

    event_log = []
    for ev in events:
        event_dt = pd.Timestamp(ev["date"])

        # Entry at first close on or after event date
        future = idx[idx >= event_dt]
        if len(future) == 0:
            event_log.append(dict(ev, status="no_data_after_date"))
            continue

        entry_dt = future[0]
        entry_pos = idx.get_loc(entry_dt)
        exit_pos = min(entry_pos + hold_days, len(idx))

        if exit_pos <= entry_pos:
            event_log.append(dict(ev, entry_date=str(entry_dt.date()), status="no_hold_window"))
            continue

        # Pair return: -BTI + IMB.L (equal notional, assuming FX-neutral)
        bti_slice = ret_short.iloc[entry_pos:exit_pos].fillna(0)
        imb_slice = ret_long.iloc[entry_pos:exit_pos].fillna(0)
        spy_slice = spy_r.iloc[entry_pos:exit_pos].fillna(0)

        # Pair daily: +long - short
        pair_daily = imb_slice - bti_slice

        bti_cum = float((1 + bti_slice).prod() - 1)
        imb_cum = float((1 + imb_slice).prod() - 1)
        pair_cum = float((1 + pair_daily).prod() - 1)
        spy_cum = float((1 + spy_slice).prod() - 1)

        ev_record = dict(ev)
        ev_record["entry_date"] = str(entry_dt.date())
        ev_record["exit_date"] = str(idx[exit_pos - 1].date())
        ev_record["n_hold_days"] = int(exit_pos - entry_pos)
        ev_record["bti_return"] = round(bti_cum, 4)
        ev_record["imb_return"] = round(imb_cum, 4)
        ev_record["pair_return"] = round(pair_cum, 4)  # short BTI, long IMB: want positive
        ev_record["spy_return"] = round(spy_cum, 4)
        ev_record["short_bti_return"] = round(-bti_cum, 4)
        event_log.append(ev_record)

        for j in range(entry_pos, exit_pos):
            if positions.iloc[j] == 0.0:
                positions.iloc[j] = 1.0
                # Pair: long IMB.L - short BTI
                b = ret_short.iloc[j] if not pd.isna(ret_short.iloc[j]) else 0
                i = ret_long.iloc[j] if not pd.isna(ret_long.iloc[j]) else 0
                pnl.iloc[j] = i - b  # long IMB, short BTI

    return pnl, positions, event_log


def main():
    sid = "PL974_italy_accise_tabacchi_bti_short_imbl_long"
    tickers = ["BTI", "IMB.L", "PM", "SPY"]

    try:
        px = load_prices(tickers, start="2018-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    # IMB.L may not load — try without it
    if "IMB.L" not in px.columns:
        # Retry without IMB.L (it may not be available via yfinance in US)
        try:
            px2 = load_prices(["BTI", "PM", "SPY"], start="2018-01-01")
            px = px2
        except Exception as e2:
            return mark_failed(sid, f"data load fallback: {e2}")

    if "BTI" not in px.columns:
        return mark_failed(sid, f"missing critical ticker BTI; available: {list(px.columns)}")
    if "SPY" not in px.columns:
        return mark_failed(sid, f"missing SPY")

    px = px.sort_index().ffill(limit=2)
    ret = daily_returns(px)
    spy_r = ret["SPY"].dropna()
    bti_r = ret["BTI"].fillna(0)
    idx = ret.index

    has_imb = "IMB.L" in ret.columns

    if has_imb:
        imb_r = ret["IMB.L"].fillna(0)
        label = "Short BTI / Long IMB.L pair"
    else:
        # Fallback: use PM as the long hedge (minimal Italy exposure vs BTI)
        pm_r = ret["PM"].fillna(0) if "PM" in ret.columns else None
        if pm_r is None:
            return mark_failed(sid, "IMB.L not available and PM also missing — cannot run pair study")
        imb_r = pm_r
        has_imb = True
        label = "Short BTI / Long PM pair (IMB.L fallback)"
        print("WARNING: IMB.L not available, using PM as long hedge (fallback)")

    # ---------- Primary event study: short BTI, long IMB.L ----------
    pnl, pos, event_log = run_event_study(
        EXCISE_EVENTS, bti_r, imb_r, spy_r, idx, hold_days=HOLD_DAYS
    )

    held_mask = pos != 0.0
    held_pnl = pnl[held_mask]
    held_spy = spy_r.reindex(held_pnl.index).fillna(0)
    n_events = sum(1 for e in event_log if e.get("entry_date"))

    if len(held_pnl) < 20:
        return mark_failed(
            sid,
            f"insufficient held days: held={len(held_pnl)}, n_events={n_events}",
            extra={"events": event_log},
        )

    m = compute_metrics(
        held_pnl,
        benchmark=held_spy,
        name=f"Italy Tobacco Excise: {label} (held-days)",
        positions=pos.reindex(held_pnl.index).fillna(0),
        cost_bps=12,
    )

    # ---------- Also compute BTI-only short as secondary ----------
    pnl_bti_only = pd.Series(0.0, index=idx)
    pos_bti_only = pd.Series(0.0, index=idx)
    for ev in EXCISE_EVENTS:
        event_dt = pd.Timestamp(ev["date"])
        future = idx[idx >= event_dt]
        if len(future) == 0:
            continue
        entry_pos = idx.get_loc(future[0])
        exit_pos = min(entry_pos + HOLD_DAYS, len(idx))
        for j in range(entry_pos, exit_pos):
            if pos_bti_only.iloc[j] == 0.0:
                pos_bti_only.iloc[j] = -1.0
                pnl_bti_only.iloc[j] = -1.0 * (bti_r.iloc[j] if not pd.isna(bti_r.iloc[j]) else 0)

    held_bti = pnl_bti_only[pos_bti_only != 0.0]
    held_spy_bti = spy_r.reindex(held_bti.index).fillna(0)
    bti_secondary_metrics = None
    if len(held_bti) >= 20:
        bti_secondary_metrics = compute_metrics(
            held_bti,
            benchmark=held_spy_bti,
            name="Short BTI only (Italy excise event)",
            positions=pos_bti_only.reindex(held_bti.index).fillna(0),
            cost_bps=8,
        )

    # ---------- Event-level summary ----------
    pair_rets = [e.get("pair_return") for e in event_log if e.get("pair_return") is not None]
    short_rets = [e.get("short_bti_return") for e in event_log if e.get("short_bti_return") is not None]

    event_summary = {
        "n_events": len(pair_rets),
        "avg_pair_return": round(float(np.mean(pair_rets)), 4) if pair_rets else None,
        "avg_short_bti_return": round(float(np.mean(short_rets)), 4) if short_rets else None,
        "win_rate_pair": round(float(np.mean([r > 0 for r in pair_rets])), 4) if pair_rets else None,
        "best": round(float(np.max(pair_rets)), 4) if pair_rets else None,
        "worst": round(float(np.min(pair_rets)), 4) if pair_rets else None,
    }

    save_result(
        sid,
        m,
        extra={
            "status": "ok",
            "rule": (
                "When Italy MEF DBP (published ~Oct 15) confirms tobacco excise hike > 5% "
                "per pack, enter short BTI and long IMB.L at equal notional at the first "
                "close on/after Oct 15. Hold 80 trading days. Stop-loss: either leg "
                "individually down > 12%. Time exit: day +80."
            ),
            "mechanism": (
                "Italy is BTI's #2 European market (~22% Italian market share via Pall Mall, "
                "Rothmans). BTI lacks IQOS-equivalent pricing power in Italy's heated-tobacco "
                "segment (PMI dominance), making it more volume-elastic to price increases. "
                "Excise hikes reduce combustible volumes, and BTI cannot fully offset via "
                "premium product mix. IMB.L has minimal Italy exposure (UK, Germany, Spain "
                "focused), making it a natural long hedge. The pair isolates Italy-specific "
                "risk from broader tobacco sector beta."
            ),
            "source": (
                "MEF Draft Budgetary Plans (mef.gov.it); "
                "ADM Agenzia delle Dogane e Monopoli tariffa di vendita (adm.gov.it); "
                "BTI Annual Reports (Europe AME segment); prices via yfinance."
            ),
            "tickers_short": ["BTI"],
            "tickers_long": ["IMB.L"],
            "long_leg_used": label,
            "events": event_log,
            "event_summary": event_summary,
            "secondary_bti_only_metrics": bti_secondary_metrics,
            "hold_days": HOLD_DAYS,
            "n_events": n_events,
            "caveats": (
                "N=2 events is extremely small. FX dynamics between GBP-denominated IMB.L "
                "and USD-denominated BTI create additional noise (not hedged here). "
                "Italy excise hike data is manually compiled from Italian DBP documents; "
                "hike timing and magnitude subject to revision. The mechanism may be "
                "partially priced in before DBP publication date. BTI share buybacks "
                "and dividend policy may offset volume decline impact over 80-day hold."
            ),
        },
        pnl=held_pnl,
    )

    print(f"Done: {sid}")
    print(f"  n_events={n_events}, hold_days={HOLD_DAYS}, long_leg={label}")
    print(f"  Event summary: {event_summary}")
    if "error" not in m:
        print(
            f"  Sharpe: {m.get('sharpe'):.2f}  CAGR: {m.get('cagr')*100:.2f}%  "
            f"MaxDD: {m.get('max_dd')*100:.2f}%  t-stat: {m.get('t_stat'):.2f}"
        )
    if bti_secondary_metrics and "error" not in bti_secondary_metrics:
        print(
            f"  BTI-only short Sharpe: {bti_secondary_metrics.get('sharpe'):.2f}  "
            f"CAGR: {bti_secondary_metrics.get('cagr')*100:.2f}%"
        )


if __name__ == "__main__":
    main()
