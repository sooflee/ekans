"""PL586_fda_eop2_typeb_8k_phase3_alignment_drift - FDA EOP2/Type B Meeting 8-K -> Sponsor Drift
30-day long after EOP2 alignment, hedged via XBI.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL586_fda_eop2_typeb_8k_phase3_alignment_drift"
    # (ticker, 8-K date) - curated EOP2 / Type B alignment 8-Ks
    events_raw = [
        ("VRTX", "2018-11-08"),
        ("BIIB", "2019-03-12"),
        ("RGNX", "2020-05-15"),
        ("MGTA", "2021-02-10"),
        ("CRSP", "2022-01-20"),
        ("EDIT", "2022-06-15"),
        ("BLUE", "2018-04-23"),
        ("SRPT", "2019-12-04"),
        ("ALNY", "2020-08-12"),
        ("RXDX", "2021-11-09"),
        ("VKTX", "2023-07-25"),
        ("RYTM", "2022-09-15"),
        ("REPL", "2023-05-22"),
        ("PRTA", "2023-10-30"),
    ]
    hold = 30
    tickers = sorted(set([t for t, _ in events_raw] + ["XBI", "SPY"]))
    try:
        px = load_prices(tickers, start="2017-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    ret = daily_returns(px)
    spy_r = ret["SPY"]
    xbi_r = ret["XBI"] if "XBI" in ret.columns else None

    legs = []
    event_results = []
    for tkr, d in events_raw:
        if tkr not in ret.columns:
            continue
        dt = pd.Timestamp(d)
        # Anti-chase filter: skip if 30-day prior return > 30%
        prior_mask = (ret.index < dt) & (ret.index >= dt - pd.Timedelta(days=45))
        prior_ret = ret[tkr][prior_mask]
        if len(prior_ret) < 5:
            continue
        prior_cum = float((1 + prior_ret).prod() - 1)
        if prior_cum > 0.30:
            continue
        mask = ret.index >= dt
        if mask.sum() < hold:
            continue
        entry = ret.index[mask][0]
        loc = ret.index.get_loc(entry)
        end = min(loc + hold, len(ret))
        if end - loc < 10:
            continue
        long_leg = ret[tkr].iloc[loc:end]
        hedge = -0.5 * xbi_r.iloc[loc:end] if xbi_r is not None else 0
        net = long_leg + hedge
        legs.append(net)
        event_results.append({"ticker": tkr, "trigger_date": d, "ret": round(float((1+net).prod()-1), 4)})

    if not legs:
        return mark_failed(sid, "no valid events")
    df = pd.concat(legs, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")
    m = compute_metrics(pnl, benchmark=spy_r, name="FDA EOP2 Alignment Drift")
    save_result(sid, m, extra={
        "rule": "Long sponsor T+0 close after EOP2/TypeB 8-K (anti-chase), hold 30d, -50% XBI hedge.",
        "mechanism": "EOP2 alignment de-risks Phase 3 -> analyst rerating",
        "source": "EDGAR EOP2/TypeB 8-Ks + yfinance",
        "n_events": len(event_results),
        "events": event_results,
    })
    print(f"Done {sid}: events={len(event_results)} Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
