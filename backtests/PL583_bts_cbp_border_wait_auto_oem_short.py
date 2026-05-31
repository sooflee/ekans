"""PL583_bts_cbp_border_wait_auto_oem_short - Border Wait Surge -> Short Auto OEMs, Long UNP/EWW
Curated cluster events. 21-day hold.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL583_bts_cbp_border_wait_auto_oem_short"
    events_raw = ["2021-04-15", "2022-04-15", "2023-09-19", "2023-12-19"]
    hold = 21
    tickers = ["F", "GM", "STLA", "UNP", "EWW", "SPY"]
    try:
        px = load_prices(tickers, start="2020-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    ret = daily_returns(px)
    spy_r = ret["SPY"]

    legs = []
    event_results = []
    for d in events_raw:
        dt = pd.Timestamp(d)
        mask = ret.index >= dt
        if mask.sum() < hold:
            continue
        entry = ret.index[mask][0]
        loc = ret.index.get_loc(entry)
        end = min(loc + hold, len(ret))
        if end - loc < 10:
            continue
        avail_short = [t for t in ["F", "GM", "STLA"] if t in ret.columns]
        avail_long = [t for t in ["UNP", "EWW"] if t in ret.columns]
        if not avail_short or not avail_long:
            continue
        short_basket = -1.0 * ret[avail_short].mean(axis=1).iloc[loc:end]
        long_basket = ret[avail_long].mean(axis=1).iloc[loc:end]
        net = (short_basket + long_basket) / 2.0
        legs.append(net)
        event_results.append({"trigger_date": d, "ret": round(float((1+net).prod()-1), 4)})

    if not legs:
        return mark_failed(sid, "no valid events")
    df = pd.concat(legs, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")
    m = compute_metrics(pnl, benchmark=spy_r, name="Border Wait -> Auto OEM Pair")
    save_result(sid, m, extra={
        "rule": "Short F/GM/STLA, long UNP/EWW when Laredo/Otay truck wait exceeds 95th pct.",
        "mechanism": "Border friction -> JIT auto-supply disruption + rail substitution / Mexico export tailwind",
        "source": "CBP BWT + yfinance",
        "n_events": len(event_results),
        "events": event_results,
    })
    print(f"Done {sid}: events={len(event_results)} Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
