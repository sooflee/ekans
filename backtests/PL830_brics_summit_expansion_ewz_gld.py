"""PL830_brics_summit_expansion_ewz_gld — BRICS Annual Summit New-Member Admission -> Long EWZ + Long GLD"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL830_brics_summit_expansion_ewz_gld"
    # BRICS summit event dates (summit opening date)
    # Entry: T-5 trading days before summit; Exit: T+10 trading days after summit close
    # Summit dates: Johannesburg Aug 22 2023, Kazan Oct 22 2024
    summit_opens = [
        pd.Timestamp("2023-08-22"),  # Johannesburg
        pd.Timestamp("2024-10-22"),  # Kazan
    ]
    summit_closes = [
        pd.Timestamp("2023-08-24"),  # Johannesburg 3-day summit
        pd.Timestamp("2024-10-24"),  # Kazan 3-day summit
    ]

    # Portfolio weights
    EWZ_W = 0.70
    GLD_W = 0.30

    try:
        px = load_prices(["EWZ", "GLD", "UUP", "SPY"], start="2022-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    ret = daily_returns(px)
    ewz_r = ret["EWZ"]
    gld_r = ret["GLD"]
    spy_r = ret["SPY"]

    pnl = pd.Series(0.0, index=ewz_r.index)
    evts = []

    for s_open, s_close in zip(summit_opens, summit_closes):
        # Entry: T-5 trading days before summit open
        mask_before = ewz_r.index < s_open
        if mask_before.sum() < 5:
            continue
        entry_i = ewz_r.index[mask_before][-5]  # 5 trading days before summit

        # Exit: T+10 trading days after summit close
        mask_after = ewz_r.index > s_close
        if mask_after.sum() < 10:
            # Might still be in-sample near present; use end of data
            exit_i = ewz_r.index[-1]
        else:
            exit_i = ewz_r.index[mask_after][9]  # T+10

        p = ewz_r.index.get_loc(entry_i)
        ep = ewz_r.index.get_loc(exit_i) + 1

        if ep <= p:
            continue

        # Blended portfolio return
        portfolio_r = EWZ_W * ewz_r.iloc[p:ep] + GLD_W * gld_r.iloc[p:ep]
        pnl.iloc[p:ep] = portfolio_r.values

        # Cumulative returns for diagnostics
        ewz_cum = float((1 + ewz_r.iloc[p:ep]).prod() - 1)
        gld_cum = float((1 + gld_r.iloc[p:ep]).prod() - 1)
        port_cum = float((1 + portfolio_r).prod() - 1)

        spy_cum = None
        if entry_i in spy_r.index:
            sp = spy_r.index.get_loc(entry_i)
            spy_cum = float((1 + spy_r.iloc[sp:min(sp + (ep - p), len(spy_r))]).prod() - 1)

        evts.append({
            "summit_date": str(s_open.date()),
            "entry_date": str(entry_i.date()),
            "exit_date": str(exit_i.date()),
            "n_days": ep - p,
            "ewz_return": round(ewz_cum, 4),
            "gld_return": round(gld_cum, 4),
            "portfolio_return": round(port_cum, 4),
            "spy_return": round(spy_cum, 4) if spy_cum is not None else None,
        })

    print(f"Events processed: {len(evts)}")
    if not evts:
        return mark_failed(sid, "no valid events found")

    ip = pnl[pnl != 0]
    print(f"Active PnL days: {len(ip)}")

    if len(ip) < 5:
        return mark_failed(sid, f"insufficient active days ({len(ip)})")

    m = compute_metrics(ip, benchmark=spy_r, name="BRICS Summit Expansion → Long EWZ+GLD")
    m["n_events"] = len(evts)

    save_result(sid, m, extra={
        "rule": "Go long 70% EWZ + 30% GLD from T-5 to T+10 around BRICS annual summit when new member admission and local-currency settlement language are confirmed on the agenda. Stop-loss: -4% on EWZ from entry.",
        "mechanism": "BRICS expansion signals de-dollarization momentum and capital rotation into EM equities and gold as alternative reserve assets. Market anticipates increased bilateral trade in local currencies reducing USD demand.",
        "source": "Known summit dates: Johannesburg 2023-08-22, Kazan 2024-10-22; EWZ, GLD, SPY via yfinance",
        "n_events": len(evts),
        "events": evts,
        "portfolio_weights": {"EWZ": EWZ_W, "GLD": GLD_W},
    })

    print(f"Done: {len(evts)} events")
    for e in evts:
        print(f"  Summit {e['summit_date']} (entry {e['entry_date']} -> exit {e['exit_date']}): port={e['portfolio_return']:.1%}, ewz={e['ewz_return']:.1%}, gld={e['gld_return']:.1%}, spy={e['spy_return']:.1%}" if e['spy_return'] else f"  Summit {e['summit_date']}: port={e['portfolio_return']:.1%}")


if __name__ == "__main__":
    main()
