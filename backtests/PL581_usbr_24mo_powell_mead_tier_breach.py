"""PL581_usbr_24mo_powell_mead_tier_breach - USBR 24-Mo Study Tier Breach -> Long IPP / Short Pacific NW Utility
2-6 month hold on USBR August projection release events.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL581_usbr_24mo_powell_mead_tier_breach"
    events_raw = ["2021-08-16", "2022-08-15", "2024-08-15"]
    hold = 126
    long_basket = ["VST", "NRG"]
    short_basket = ["AVA", "IDA"]
    tickers = sorted(set(long_basket + short_basket + ["SPY"]))
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
        if end - loc < 30:
            continue
        avail_l = [t for t in long_basket if t in ret.columns]
        avail_s = [t for t in short_basket if t in ret.columns]
        if not avail_l or not avail_s:
            continue
        long_leg = ret[avail_l].mean(axis=1).iloc[loc:end]
        short_leg = -1.0 * ret[avail_s].mean(axis=1).iloc[loc:end]
        net = 0.5 * long_leg + 0.5 * short_leg
        legs.append(net)
        event_results.append({"trigger_date": d, "ret": round(float((1+net).prod()-1), 4)})

    if not legs:
        return mark_failed(sid, "no valid events")
    df = pd.concat(legs, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")
    m = compute_metrics(pnl, benchmark=spy_r, name="USBR Tier Breach IPP/Utility Pair")
    save_result(sid, m, extra={
        "rule": "Long Permian/Desert IPP (VST, NRG), short Pacific NW utility (AVA, IDA) when USBR projects Powell <3525 or Mead <1025. Hold 6mo.",
        "mechanism": "Hydro curtailment -> gas/coal CCGT demand pulse; PNW peer relative weakness",
        "source": "USBR 24-Month Study + yfinance",
        "n_events": len(event_results),
        "events": event_results,
    })
    print(f"Done {sid}: events={len(event_results)} Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
