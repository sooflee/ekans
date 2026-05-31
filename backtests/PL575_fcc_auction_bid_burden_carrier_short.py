"""PL575_fcc_auction_bid_burden_carrier_short - FCC Spectrum Auction Bid Burden -> Carrier Short/Long Pair
Short top bidders, long sit-outs, T+10 after auction close, hold 6 months.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL575_fcc_auction_bid_burden_carrier_short"
    # Each event: (auction_close_date, [high_burden_shorts], [low_burden_longs])
    # Derived from public FCC Public Notices + carrier 10-K capex
    events_raw = [
        # Auction 97 AWS-3 (Jan 2015): T+VZ heavy
        ("2015-01-29", ["T", "VZ"], ["TMUS"]),
        # Auction 107 C-band (Feb 2021): VZ + T heavy
        ("2021-02-24", ["VZ", "T"], ["TMUS"]),
        # Auction 108 2.5GHz (Aug 2022): TMUS heavy
        ("2022-08-29", ["TMUS"], ["T", "VZ"]),
        # Auction 110 3.45GHz (Jan 2022): T heavy
        ("2022-01-04", ["T"], ["TMUS", "VZ"]),
        # Auction 103 (May 2019)
        ("2019-05-28", ["T", "VZ"], ["TMUS"]),
    ]
    hold = 126
    wait = 10  # T+10 trading days
    tickers = ["T", "VZ", "TMUS", "SPY"]
    try:
        px = load_prices(tickers, start="2014-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    ret = daily_returns(px)
    spy_r = ret["SPY"] if "SPY" in ret.columns else None

    legs = []
    event_results = []
    for d, shorts, longs in events_raw:
        dt = pd.Timestamp(d)
        mask = ret.index >= dt
        if mask.sum() < hold + wait + 1:
            continue
        idxs = ret.index[mask]
        if len(idxs) < wait + 1:
            continue
        entry = idxs[wait]
        loc = ret.index.get_loc(entry)
        end = min(loc + hold, len(ret))
        if end - loc < 30:
            continue
        avail_shorts = [t for t in shorts if t in ret.columns]
        avail_longs = [t for t in longs if t in ret.columns]
        if not avail_shorts or not avail_longs:
            continue
        s_basket = -1.0 * ret[avail_shorts].mean(axis=1).iloc[loc:end]
        l_basket = ret[avail_longs].mean(axis=1).iloc[loc:end]
        net = (s_basket + l_basket) / 2.0
        legs.append(net)
        cum = float((1 + net).prod() - 1)
        event_results.append({"trigger_date": d, "shorts": avail_shorts, "longs": avail_longs,
                              "net_return": round(cum, 4)})

    if not legs:
        return mark_failed(sid, "no valid events")
    df = pd.concat(legs, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")
    m = compute_metrics(pnl, benchmark=spy_r, name="FCC Auction Bid-Burden Carrier Pair")
    save_result(sid, m, extra={
        "rule": "Short heavy bidders, long sit-outs T+10 after auction close, hold 6mo",
        "mechanism": "Bid burden compresses heavy-bidder FCF -> equity rerating; sit-outs benefit relatively",
        "source": "FCC Public Notices + yfinance",
        "n_events": len(event_results),
        "events": event_results,
    })
    print(f"Done {sid}: events={len(event_results)} Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
