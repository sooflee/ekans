"""PL826_scotus_cert_sec_alj_kre_kbe_pair — SCOTUS Cert Grant on SEC/Agency ALJ Authority -> Short KRE / Long KBE Spread"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL826_scotus_cert_sec_alj_kre_kbe_pair"
    # Known cert-grant event dates
    # 2022-10-03: SCOTUS cert granted SEC v. Jarkesy
    # 2017-09-28: SCOTUS cert granted Lucia v. SEC
    event_dates = [
        pd.Timestamp("2017-09-28"),
        pd.Timestamp("2022-10-03"),
    ]
    HOLD_DAYS = 30  # Trading days to hold

    try:
        px = load_prices(["KRE", "KBE", "SPY"], start="2016-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    kre_r = ret["KRE"]
    kbe_r = ret["KBE"]
    spy_r = ret["SPY"]

    # Pair PnL: Short KRE + Long KBE (dollar-neutral)
    # Daily pair return = KBE_return - KRE_return
    pnl = pd.Series(0.0, index=kre_r.index)
    evts = []

    for event_date in event_dates:
        # Find entry: next trading day on or after event date
        mask_entry = kre_r.index >= event_date
        if mask_entry.sum() < HOLD_DAYS:
            continue
        ei = kre_r.index[mask_entry][0]
        p = kre_r.index.get_loc(ei)
        ep = min(p + HOLD_DAYS, len(kre_r))

        if ep <= p:
            continue

        # Short KRE / Long KBE pair return
        kre_seg = kre_r.iloc[p:ep]
        kbe_seg = kbe_r.iloc[p:ep]
        pair_r = kbe_seg.values - kre_seg.values  # Long KBE, Short KRE

        pnl.iloc[p:ep] = pair_r

        # Cumulative returns
        kre_cum = float((1 + kre_seg).prod() - 1)
        kbe_cum = float((1 + kbe_seg).prod() - 1)
        pair_cum = float(np.sum(pair_r))  # approximate pair PnL

        spy_cum = None
        if ei in spy_r.index:
            sp = spy_r.index.get_loc(ei)
            spy_cum = float((1 + spy_r.iloc[sp:min(sp + HOLD_DAYS, len(spy_r))]).prod() - 1)

        evts.append({
            "event_date": str(event_date.date()),
            "entry_date": str(ei.date()),
            "n_days": ep - p,
            "kre_return": round(kre_cum, 4),
            "kbe_return": round(kbe_cum, 4),
            "pair_excess": round(float(kbe_cum - kre_cum), 4),
            "spy_return": round(spy_cum, 4) if spy_cum is not None else None,
        })

    print(f"Events processed: {len(evts)}")
    if not evts:
        return mark_failed(sid, "no valid events found")

    ip = pnl[pnl != 0]
    print(f"Active PnL days: {len(ip)}")

    if len(ip) < 5:
        return mark_failed(sid, f"insufficient active days ({len(ip)})")

    m = compute_metrics(ip, benchmark=spy_r, name="SCOTUS ALJ Cert → Short KRE / Long KBE")
    m["n_events"] = len(evts)

    save_result(sid, m, extra={
        "rule": "Short KRE + Long KBE (dollar-neutral) for 30 trading days after SCOTUS cert grant on case challenging SEC/FDIC/OCC in-house ALJ adjudication authority or enforcement-remedy powers.",
        "mechanism": "SCOTUS cert on agency enforcement authority expands regulatory liability uncertainty specifically for sub-B asset regional banks with active OCC/CFPB consent orders (overweight in KRE). Larger-cap banks in KBE have cleaner enforcement profiles. Market re-prices enforcement tail risk differentially across the two ETFs.",
        "source": "SCOTUS cert events: Lucia v. SEC 2017-09-28, SEC v. Jarkesy 2022-10-03; KRE, KBE, SPY via yfinance",
        "n_events": len(evts),
        "events": evts,
    })

    print(f"Done: {len(evts)} events")
    for e in evts:
        spy_str = f", spy={e['spy_return']:.1%}" if e['spy_return'] is not None else ""
        print(f"  {e['event_date']}: pair_excess={e['pair_excess']:.1%} (kbe={e['kbe_return']:.1%} - kre={e['kre_return']:.1%}){spy_str}")


if __name__ == "__main__":
    main()
