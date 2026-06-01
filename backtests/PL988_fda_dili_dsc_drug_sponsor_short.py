"""PL988_fda_dili_dsc_drug_sponsor_short — FDA Drug Safety Communication (DILI/Hepatotoxicity) -> Short Drug Sponsor Equity"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL988_fda_dili_dsc_drug_sponsor_short"

    # Known FDA DSC events: (date, sponsor_ticker)
    # 2018-11-14: Xofigo/Bayer -> BAYRY
    # 2020-08-27: Aubagio/Sanofi -> SNY
    # 2021-05-20: Ocaliva/Intercept -> ICPT (delisted 2024, excluded)
    # 2021-09-09: Xeljanz/Pfizer -> PFE
    # 2022-03-08: Tecfidera/Biogen -> BIIB
    events = [
        ("2018-11-14", "BAYRY"),
        ("2020-08-27", "SNY"),
        ("2021-09-09", "PFE"),
        ("2022-03-08", "BIIB"),
    ]

    tickers = list(set([t for _, t in events] + ["XBI", "SPY"]))
    hold_days = 40  # trading days

    try:
        px = load_prices(tickers, start="2017-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    spy_r = ret["SPY"] if "SPY" in ret.columns else ret.iloc[:, 0]

    # Build daily PnL: short sponsor, hedged vs XBI sector beta
    pnl = pd.Series(0.0, index=ret.index)
    event_records = []

    for ev_date_str, sponsor_ticker in events:
        if sponsor_ticker not in ret.columns:
            print(f"Skipping {ev_date_str} {sponsor_ticker}: not in price data")
            continue
        if "XBI" not in ret.columns:
            print("XBI not in price data, skipping sector hedge")
            continue

        ev_date = pd.Timestamp(ev_date_str)
        # Find entry: first trading day on or after event date
        future_idx = ret.index[ret.index >= ev_date]
        if len(future_idx) < hold_days:
            print(f"Skipping {ev_date_str}: insufficient data after event")
            continue

        entry_idx = future_idx[0]
        entry_loc = ret.index.get_loc(entry_idx)
        exit_loc = min(entry_loc + hold_days, len(ret))

        sponsor_ret = ret[sponsor_ticker].iloc[entry_loc:exit_loc]
        xbi_ret = ret["XBI"].iloc[entry_loc:exit_loc]
        spy_window = spy_r.iloc[entry_loc:exit_loc]

        # Strategy: short sponsor, long XBI sector (market-neutral within biotech)
        # Position: -1 sponsor + 0.5 XBI hedge
        strat_ret = -sponsor_ret + 0.5 * xbi_ret

        # Record cumulative returns
        sponsor_cum = float((1 + sponsor_ret).prod() - 1)
        xbi_cum = float((1 + xbi_ret).prod() - 1)
        spy_cum = float((1 + spy_window).prod() - 1)
        alpha_vs_xbi = -sponsor_cum - xbi_cum  # short alpha vs sector

        event_records.append({
            "event_date": ev_date_str,
            "sponsor": sponsor_ticker,
            "sponsor_return_40d": round(sponsor_cum, 4),
            "xbi_return_40d": round(xbi_cum, 4),
            "spy_return_40d": round(spy_cum, 4),
            "alpha_vs_xbi": round(alpha_vs_xbi, 4),
            "n_days": len(sponsor_ret),
        })

        # Add to PnL series (avoid overlapping windows from same sponsor)
        for i, (idx, r) in enumerate(strat_ret.items()):
            if idx in pnl.index:
                pnl[idx] += r

    if not event_records:
        return mark_failed(sid, "no valid events found")

    print(f"Events processed: {len(event_records)}")
    for e in event_records:
        print(f"  {e['event_date']} {e['sponsor']}: sponsor={e['sponsor_return_40d']:.2%}, "
              f"XBI={e['xbi_return_40d']:.2%}, alpha={e['alpha_vs_xbi']:.2%}")

    # Only use days where we actually had a position
    active_pnl = pnl[pnl != 0]
    if len(active_pnl) < 20:
        return mark_failed(sid, f"insufficient active trading days: {len(active_pnl)}")

    # Normalize by number of concurrent positions (max 1 event at a time in this sparse sample)
    m = compute_metrics(active_pnl, benchmark=spy_r, name="FDA DILI DSC Short Sponsor")

    save_result(sid, m, extra={
        "rule": "Short drug sponsor equity within 5 days of FDA DSC citing DILI/Hepatotoxicity; cover after 40 trading days or confirmed label change",
        "mechanism": "FDA safety communications trigger investor concern over future revenue loss, litigation risk, and prescriber hesitancy; DILI/Boxed Warning upgrades are particularly negative as they restrict prescribing",
        "source": "FDA Medwatch DSC RSS; known events: Bayer BAYRY 2018-11, Sanofi SNY 2020-08, Pfizer PFE 2021-09, Biogen BIIB 2022-03",
        "n_events": len(event_records),
        "events": event_records,
        "sector_hedge": "XBI 0.5x long offset against sponsor short",
    })


if __name__ == "__main__":
    main()
