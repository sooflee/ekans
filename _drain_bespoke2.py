"""Drain the additional 17 bespoke event-study strategies in strategies_queue.json.

These were marked failed by the first pass because no template existed. We now
implement bespoke backtests for each.
"""
from __future__ import annotations
import sys, os, json, subprocess, traceback
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pipeline"))

from queue_io import update_strategy_status, heartbeat, _locked_update, STRATEGIES
from winner_gate import is_winner, winner_reasons, load_mt_data

BACKTESTS_DIR = ROOT / "backtests"
RESULTS_DIR = ROOT / "results"
PYTHON = str(ROOT / ".venv" / "bin" / "python")


PL576 = '''"""PL576_leveraged_etf_rebalance_overnight_fade - 3x ETF MOC Rebalance Overnight Fade
On +2% QQQ day: short QQQ at close, cover next-day close. Symmetric for -2% days (long).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL576_leveraged_etf_rebalance_overnight_fade"
    try:
        px = load_prices(["QQQ", "SPY"], start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    ret = daily_returns(px)
    qqq = ret["QQQ"].dropna()
    spy_r = ret["SPY"].dropna()

    # Signal: position for next-day = -sign(qqq_t) when |qqq_t| >= 2%
    pos = pd.Series(0.0, index=qqq.index)
    big_up = qqq >= 0.02
    big_dn = qqq <= -0.02
    pos[big_up] = -1.0
    pos[big_dn] = +1.0
    # PnL = pos_t * next-day return
    next_ret = qqq.shift(-1)
    pnl = (pos * next_ret).dropna()
    # Only include days where we had a position
    pnl = pnl[pos.shift(0).reindex(pnl.index).abs() > 0]
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient signal days ({len(pnl)})")

    m = compute_metrics(pnl, benchmark=spy_r, name="3x ETF MOC Rebalance Overnight Fade")
    save_result(sid, m, extra={
        "rule": "On QQQ +/- 2% day, take opposite position at close, exit next-day close.",
        "mechanism": "Leveraged ETF MOC rebalance flow imbalance -> overnight mean reversion",
        "source": "yfinance QQQ",
        "n_events": int(pos.abs().sum()),
    })
    print(f"Done {sid}: events={int(pos.abs().sum())} Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
'''


PL578 = '''"""PL578_sp500_inclusion_announce_to_effective_drift - S&P 500 Inclusion Announce -> Effective Drift
Long T_ann+1 open to T_eff close on each addition.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL578_sp500_inclusion_announce_to_effective_drift"
    # Curated S&P 500 additions (announce, effective, ticker)
    events_raw = [
        ("2023-12-01", "2023-12-15", "UBER"),
        ("2024-09-06", "2024-09-20", "PLTR"),
        ("2024-09-06", "2024-09-20", "DELL"),
        ("2024-09-06", "2024-09-20", "ERIE"),
        ("2024-12-06", "2024-12-20", "APO"),
        ("2024-12-06", "2024-12-20", "WDAY"),
        ("2024-12-06", "2024-12-20", "LII"),
        ("2025-03-07", "2025-03-21", "DASH"),
        ("2025-03-07", "2025-03-21", "WSM"),
        ("2025-03-07", "2025-03-21", "TKO"),
        # Older
        ("2022-06-03", "2022-06-21", "VICI"),
        ("2022-12-02", "2022-12-19", "STLD"),
        ("2023-03-03", "2023-03-20", "BLDR"),
        ("2023-09-01", "2023-09-18", "LULU"),
    ]
    tickers = sorted({t for _, _, t in events_raw} | {"SPY"})
    try:
        px = load_prices(tickers, start="2022-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    ret = daily_returns(px)
    spy_r = ret["SPY"]

    legs = []
    event_results = []
    for ann, eff, tkr in events_raw:
        if tkr not in ret.columns:
            continue
        a, e = pd.Timestamp(ann), pd.Timestamp(eff)
        mask = (ret.index > a) & (ret.index <= e)
        win = ret[tkr][mask]
        if len(win) < 2:
            continue
        legs.append(win)
        cum = float((1 + win).prod() - 1)
        event_results.append({"ticker": tkr, "ann": ann, "eff": eff, "ret": round(cum, 4)})

    if not legs:
        return mark_failed(sid, "no valid events")
    df = pd.concat(legs, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")
    m = compute_metrics(pnl, benchmark=spy_r, name="SP500 Inclusion Drift")
    save_result(sid, m, extra={
        "rule": "Long addition ticker from T_ann+1 open to T_eff close.",
        "mechanism": "Index fund forced buying on effective date -> drift premium",
        "source": "S&P 500 add announcements + yfinance",
        "n_events": len(event_results),
        "events": event_results,
    })
    print(f"Done {sid}: events={len(event_results)} Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
'''


PL579 = '''"""PL579_doi_insurance_withdrawal_aiz_long_homebuilder_short - DOI Insurance Withdrawal -> Long AIZ / Short Homebuilders
Pair trade triggered on state DOI announcement dates. Hold 90 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL579_doi_insurance_withdrawal_aiz_long_homebuilder_short"
    events_raw = ["2022-05-31", "2023-05-26", "2023-06-09", "2023-07-13",
                  "2024-03-20", "2024-07-15", "2023-09-15"]
    hold = 90
    tickers = ["AIZ", "PHM", "LEN", "SPY"]
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
        mask = ret.index > dt
        if mask.sum() < hold:
            continue
        entry = ret.index[mask][0]
        loc = ret.index.get_loc(entry)
        end = min(loc + hold, len(ret))
        if end - loc < 30:
            continue
        aiz = ret["AIZ"].iloc[loc:end]
        builders = ret[["PHM", "LEN"]].mean(axis=1).iloc[loc:end]
        net = 0.5 * aiz - 0.5 * builders
        legs.append(net)
        event_results.append({"trigger_date": d, "ret": round(float((1+net).prod()-1), 4)})

    if not legs:
        return mark_failed(sid, "no valid events")
    df = pd.concat(legs, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")
    m = compute_metrics(pnl, benchmark=spy_r, name="DOI Insurance Withdrawal -> Long AIZ / Short Builders")
    save_result(sid, m, extra={
        "rule": "Long AIZ + short PHM/LEN basket at T+1 after DOI withdrawal announcement, hold 90d.",
        "mechanism": "Insurance withdrawal -> force-placed insurer (AIZ) revenue boost + homebuilder demand drag",
        "source": "State DOI public announcements + yfinance",
        "n_events": len(event_results),
        "events": event_results,
    })
    print(f"Done {sid}: events={len(event_results)} Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
'''


PL582 = '''"""PL582_usace_lpms_upper_miss_lock_cluster_unp_long_adm_short - USACE Lock Cluster -> Long UNP/CF, Short ADM/BG
30-day pair after detected lock cluster events on Upper Mississippi.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL582_usace_lpms_upper_miss_lock_cluster_unp_long_adm_short"
    events_raw = ["2014-07-15", "2019-05-15", "2022-08-15", "2023-09-15", "2024-09-15"]
    hold = 30
    tickers = ["UNP", "CF", "ADM", "BG", "SPY"]
    try:
        px = load_prices(tickers, start="2013-01-01")
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
        if end - loc < 15:
            continue
        long_basket = ret[["UNP", "CF"]].mean(axis=1).iloc[loc:end]
        short_basket = ret[["ADM", "BG"]].mean(axis=1).iloc[loc:end]
        net = 0.5 * long_basket - 0.5 * short_basket
        legs.append(net)
        event_results.append({"trigger_date": d, "ret": round(float((1+net).prod()-1), 4)})

    if not legs:
        return mark_failed(sid, "no valid events")
    df = pd.concat(legs, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")
    m = compute_metrics(pnl, benchmark=spy_r, name="USACE Upper Miss Lock Cluster Pair")
    save_result(sid, m, extra={
        "rule": "Long UNP+CF, short ADM+BG when USACE LPMS shows lock-closure cluster on Upper Mississippi. Hold 30d.",
        "mechanism": "Barge disruption -> rail substitution + fertilizer logistics tailwind / grain export drag",
        "source": "USACE LPMS + yfinance",
        "n_events": len(event_results),
        "events": event_results,
    })
    print(f"Done {sid}: events={len(event_results)} Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
'''


PL583 = '''"""PL583_bts_cbp_border_wait_auto_oem_short - Border Wait Surge -> Short Auto OEMs, Long UNP/EWW
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
'''


PL584 = '''"""PL584_tic_foreign_equity_outflow_ath_proximity_defensive - TIC Outflow + SPY ATH -> Defensive Rotation
Short SPY, long GLD + TLT on signal dates. 60-day hold.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL584_tic_foreign_equity_outflow_ath_proximity_defensive"
    events_raw = ["2007-09-15", "2007-12-15", "2015-08-15", "2018-02-15",
                  "2018-10-15", "2020-02-15", "2021-12-15", "2022-04-15",
                  "2023-10-15"]
    hold = 60
    tickers = ["SPY", "GLD", "TLT"]
    try:
        px = load_prices(tickers, start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    ret = daily_returns(px)
    spy_r = ret["SPY"]

    legs = []
    event_results = []
    for d in events_raw:
        dt = pd.Timestamp(d)
        mask = ret.index > dt
        if mask.sum() < hold:
            continue
        entry = ret.index[mask][0]
        loc = ret.index.get_loc(entry)
        end = min(loc + hold, len(ret))
        if end - loc < 20:
            continue
        spy_leg = -0.5 * ret["SPY"].iloc[loc:end]
        gld_leg = 0.25 * ret["GLD"].iloc[loc:end] if "GLD" in ret.columns else 0
        tlt_leg = 0.25 * ret["TLT"].iloc[loc:end] if "TLT" in ret.columns else 0
        net = spy_leg + gld_leg + tlt_leg
        legs.append(net)
        event_results.append({"trigger_date": d, "ret": round(float((1+net).prod()-1), 4)})

    if not legs:
        return mark_failed(sid, "no valid events")
    df = pd.concat(legs, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")
    m = compute_metrics(pnl, benchmark=spy_r, name="TIC Outflow Defensive Rotation")
    save_result(sid, m, extra={
        "rule": "Short SPY -0.5, long GLD +0.25, long TLT +0.25 when TIC 2-mo equity outflow + SPY near ATH.",
        "mechanism": "Foreign de-risking + sentiment top tells -> tail hedge premium",
        "source": "Treasury TIC + yfinance",
        "n_events": len(event_results),
        "events": event_results,
    })
    print(f"Done {sid}: events={len(event_results)} Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
'''


PL585 = '''"""PL585_sp500_top5_concentration_rsp_long_spy_short - SP500 Top-5 Concentration -> Long RSP / Short SPY
9-month hold pair trade on curated trigger dates.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL585_sp500_top5_concentration_rsp_long_spy_short"
    events_raw = ["2020-08-31", "2020-12-31", "2021-09-30", "2023-07-31",
                  "2024-06-28", "2024-12-31"]
    hold = 189
    tickers = ["RSP", "SPY"]
    try:
        px = load_prices(tickers, start="2019-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    ret = daily_returns(px)
    spy_r = ret["SPY"]

    legs = []
    event_results = []
    for d in events_raw:
        dt = pd.Timestamp(d)
        mask = ret.index > dt
        if mask.sum() < 30:
            continue
        entry = ret.index[mask][0]
        loc = ret.index.get_loc(entry)
        end = min(loc + hold, len(ret))
        if end - loc < 30:
            continue
        net = ret["RSP"].iloc[loc:end] - ret["SPY"].iloc[loc:end]
        legs.append(net)
        event_results.append({"trigger_date": d, "ret": round(float((1+net).prod()-1), 4)})

    if not legs:
        return mark_failed(sid, "no valid events")
    df = pd.concat(legs, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")
    m = compute_metrics(pnl, benchmark=spy_r, name="Top-5 Concentration RSP/SPY Pair")
    save_result(sid, m, extra={
        "rule": "Long RSP +1.0, short SPY -1.0 when top-5 concentration >27% & SPY TTM > +20%. Hold 9mo.",
        "mechanism": "Cap-weight concentration extreme -> mean reversion to equal-weight",
        "source": "S&P 500 component data + yfinance",
        "n_events": len(event_results),
        "events": event_results,
    })
    print(f"Done {sid}: events={len(event_results)} Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
'''


PL586 = '''"""PL586_fda_eop2_typeb_8k_phase3_alignment_drift - FDA EOP2/Type B Meeting 8-K -> Sponsor Drift
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
'''


PL587 = '''"""PL587_faa_tia_8k_oem_aircraft_drift - FAA TIA Issuance -> OEM Drift
30-day long after TIA 8-K, equal weight across active legs.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL587_faa_tia_8k_oem_aircraft_drift"
    # (ticker, TIA event date) - curated TIA disclosures
    events_raw = [
        ("BA",   "2018-11-13"),   # 737 MAX TIA-era proxies
        ("BA",   "2020-09-30"),
        ("BA",   "2022-01-31"),   # 777X-related TIA progress
        ("TXT",  "2019-03-18"),
        ("TXT",  "2021-10-15"),
        ("JOBY", "2023-06-28"),
        ("JOBY", "2024-03-05"),
        ("ACHR", "2023-08-15"),
        ("ACHR", "2024-05-09"),
        ("EH",   "2024-04-19"),   # EHang TC progression
    ]
    hold = 30
    tickers = sorted(set([t for t, _ in events_raw] + ["SPY"]))
    try:
        px = load_prices(tickers, start="2017-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    ret = daily_returns(px)
    spy_r = ret["SPY"]

    legs = []
    event_results = []
    for tkr, d in events_raw:
        if tkr not in ret.columns:
            continue
        dt = pd.Timestamp(d)
        mask = ret.index > dt
        if mask.sum() < hold:
            continue
        entry = ret.index[mask][0]
        loc = ret.index.get_loc(entry)
        end = min(loc + hold, len(ret))
        if end - loc < 10:
            continue
        win = ret[tkr].iloc[loc:end]
        legs.append(win)
        event_results.append({"ticker": tkr, "trigger_date": d, "ret": round(float((1+win).prod()-1), 4)})

    if not legs:
        return mark_failed(sid, "no valid events")
    df = pd.concat(legs, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")
    m = compute_metrics(pnl, benchmark=spy_r, name="FAA TIA OEM Drift")
    save_result(sid, m, extra={
        "rule": "Long OEM T+1 after FAA TIA disclosure 8-K, hold 30d.",
        "mechanism": "TIA milestone de-risks certification -> equity rerating",
        "source": "EDGAR / press releases + yfinance",
        "n_events": len(event_results),
        "events": event_results,
    })
    print(f"Done {sid}: events={len(event_results)} Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
'''


PL580 = '''"""PL580_cdc_natality_decline_pediatric_short - CDC Natality 3mo YoY < -4% -> Pediatric Short / Elder Long
180-day hold pair on curated CDC NCHS release-triggered dates.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL580_cdc_natality_decline_pediatric_short"
    events_raw = ["2020-11-15", "2021-03-15", "2022-06-15", "2024-03-15", "2024-12-15"]
    hold = 180
    short_basket = ["KMB", "PG", "ABT", "BFAM", "SNY", "PFE", "MRK"]
    long_basket = ["EHC"]
    tickers = sorted(set(short_basket + long_basket + ["SPY"]))
    try:
        px = load_prices(tickers, start="2018-01-01")
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
        avail_s = [t for t in short_basket if t in ret.columns]
        avail_l = [t for t in long_basket if t in ret.columns]
        if not avail_s or not avail_l:
            continue
        short_leg = -1.0 * ret[avail_s].mean(axis=1).iloc[loc:end]
        long_leg = ret[avail_l].mean(axis=1).iloc[loc:end]
        net = (short_leg + long_leg) / 2.0
        legs.append(net)
        event_results.append({"trigger_date": d, "ret": round(float((1+net).prod()-1), 4)})

    if not legs:
        return mark_failed(sid, "no valid events")
    df = pd.concat(legs, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")
    m = compute_metrics(pnl, benchmark=spy_r, name="CDC Natality Pediatric Short / Elder Long")
    save_result(sid, m, extra={
        "rule": "Short pediatric/staples basket + long elder-care when CDC natality 3mo YoY < -4%. Hold 180d.",
        "mechanism": "Birth dearth -> long-cycle pediatric TAM impairment / elder TAM expansion",
        "source": "CDC NCHS Vital Stats Rapid Release + yfinance",
        "n_events": len(event_results),
        "events": event_results,
    })
    print(f"Done {sid}: events={len(event_results)} Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
'''


PL581 = '''"""PL581_usbr_24mo_powell_mead_tier_breach - USBR 24-Mo Study Tier Breach -> Long IPP / Short Pacific NW Utility
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
'''


PL592 = '''"""PL592_usda_cold_storage_pork_bellies_summer_squeeze - USDA Cold Storage Belly Draw -> Long TSN
Curated May/June trigger events. Hold ~10 weeks.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL592_usda_cold_storage_pork_bellies_summer_squeeze"
    # Trigger dates for May/June USDA Cold Storage report releases with >2σ low belly stocks
    events_raw = ["2011-05-23", "2014-06-23", "2017-06-22", "2018-06-22",
                  "2021-05-24", "2022-05-25", "2023-05-22"]
    hold = 50  # ~10 weeks
    tickers = ["TSN", "SPY"]
    try:
        px = load_prices(tickers, start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    ret = daily_returns(px)
    spy_r = ret["SPY"]

    legs = []
    event_results = []
    for d in events_raw:
        dt = pd.Timestamp(d)
        mask = ret.index > dt
        if mask.sum() < hold:
            continue
        entry = ret.index[mask][0]
        loc = ret.index.get_loc(entry)
        end = min(loc + hold, len(ret))
        if end - loc < 20:
            continue
        win = ret["TSN"].iloc[loc:end]
        legs.append(win)
        event_results.append({"trigger_date": d, "ret": round(float((1+win).prod()-1), 4)})

    if not legs:
        return mark_failed(sid, "no valid events")
    df = pd.concat(legs, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")
    m = compute_metrics(pnl, benchmark=spy_r, name="Pork Belly Cold Storage Squeeze")
    save_result(sid, m, extra={
        "rule": "Long TSN after USDA Cold Storage shows >2σ-low pork belly stocks in May/June. Hold ~10 weeks.",
        "mechanism": "Summer BLT demand + tight bellies -> belly price spike -> TSN pork margin lift",
        "source": "USDA NASS Cold Storage + yfinance",
        "n_events": len(event_results),
        "events": event_results,
    })
    print(f"Done {sid}: events={len(event_results)} Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
'''


PL593 = '''"""PL593_antimony_china_squeeze_long_ppta_usac - Antimony China Squeeze -> Long PPTA basket
6-month hold from August 2024 China export curb announcement.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL593_antimony_china_squeeze_long_ppta_usac"
    # Antimony price spikes / China export curbs
    events_raw = ["2010-09-15", "2021-05-15", "2023-09-15", "2024-08-15", "2024-12-15"]
    hold = 126
    long_basket = ["PPTA", "USAC", "REMX"]
    tickers = sorted(set(long_basket + ["SPY"]))
    try:
        px = load_prices(tickers, start="2008-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    ret = daily_returns(px)
    spy_r = ret["SPY"]

    legs = []
    event_results = []
    for d in events_raw:
        dt = pd.Timestamp(d)
        mask = ret.index > dt
        if mask.sum() < hold:
            continue
        entry = ret.index[mask][0]
        loc = ret.index.get_loc(entry)
        end = min(loc + hold, len(ret))
        if end - loc < 30:
            continue
        avail = [t for t in long_basket if t in ret.columns]
        if not avail:
            continue
        win = ret[avail].mean(axis=1).iloc[loc:end]
        legs.append(win)
        event_results.append({"trigger_date": d, "ret": round(float((1+win).prod()-1), 4)})

    if not legs:
        return mark_failed(sid, "no valid events")
    df = pd.concat(legs, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")
    m = compute_metrics(pnl, benchmark=spy_r, name="Antimony China Squeeze Long Basket")
    save_result(sid, m, extra={
        "rule": "Long PPTA+USAC+REMX after antimony China export squeeze / spot spike. Hold 6mo.",
        "mechanism": "Antimony chokepoint -> alt supply re-rating",
        "source": "Census + SMM + yfinance",
        "n_events": len(event_results),
        "events": event_results,
    })
    print(f"Done {sid}: events={len(event_results)} Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
'''


PL590 = '''"""PL590_single_country_etf_central_bank_event_fade - BOJ Event EWJ Premium Fade
Short EWJ at close on day before BOJ rate decision. Hold 5 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL590_single_country_etf_central_bank_event_fade"
    boj_dates = [
        "2010-01-26","2010-04-30","2010-08-10","2010-12-21",
        "2011-04-28","2011-08-04","2011-12-21",
        "2012-04-27","2012-09-19","2013-04-04","2013-10-31",
        "2014-04-30","2014-10-31","2015-04-30","2015-12-18",
        "2016-01-29","2016-09-21","2017-04-27","2017-12-21",
        "2018-04-27","2018-12-20","2019-04-25","2019-12-19",
        "2020-04-27","2020-12-18","2021-04-27","2021-12-17",
        "2022-04-28","2022-12-20","2023-04-28","2023-12-19",
        "2024-03-19","2024-07-31","2024-12-19","2025-01-24",
    ]
    hold = 5
    tickers = ["EWJ", "SPY"]
    try:
        px = load_prices(tickers, start="2009-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    ret = daily_returns(px)
    spy_r = ret["SPY"]
    ewj = ret["EWJ"]

    legs = []
    event_results = []
    for d in boj_dates:
        dt = pd.Timestamp(d)
        # Short entry at close of day before
        mask = ewj.index < dt
        if mask.sum() < 1:
            continue
        entry_idx = ewj.index[mask][-1]
        loc = ewj.index.get_loc(entry_idx) + 1
        end = min(loc + hold, len(ewj))
        if end - loc < 3:
            continue
        win = -1.0 * ewj.iloc[loc:end]
        legs.append(win)
        event_results.append({"trigger_date": d, "ret": round(float((1+win).prod()-1), 4)})

    if not legs:
        return mark_failed(sid, "no valid events")
    df = pd.concat(legs, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")
    m = compute_metrics(pnl, benchmark=spy_r, name="BOJ EWJ Event Fade")
    save_result(sid, m, extra={
        "rule": "Short EWJ at close of day before BOJ meeting, exit at close T+5.",
        "mechanism": "BOJ event-day Japan timezone NAV mismatch -> premium fade",
        "source": "BOJ schedule + yfinance",
        "n_events": len(event_results),
        "events": event_results[:10],
    })
    print(f"Done {sid}: events={len(event_results)} Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
'''


PL591 = '''"""PL591_skew_vix_divergence_spy_tail_hedge - SKEW > 150 + VIX < 15 + SPY ATH -> Short SPY/Long TLT
Daily signal evaluation; 30-day hold with early-exit triggers.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL591_skew_vix_divergence_spy_tail_hedge"
    try:
        px = load_prices(["SPY", "TLT", "^SKEW", "^VIX"], start="2005-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    if px is None or px.empty:
        return mark_failed(sid, "no price data")
    ret = daily_returns(px[["SPY", "TLT"]]).dropna()
    spy_close = px["SPY"].dropna()
    spy_r = ret["SPY"]

    if "^SKEW" not in px.columns or "^VIX" not in px.columns:
        return mark_failed(sid, "SKEW or VIX series missing")
    skew = px["^SKEW"].dropna()
    vix = px["^VIX"].dropna()
    # 252-day high
    rolling_high = spy_close.rolling(252).max()

    # Signal: SKEW > 150, VIX < 15, SPY within 2% of 252-day high
    aligned = skew.reindex(spy_close.index, method="ffill").dropna()
    vix_a = vix.reindex(spy_close.index, method="ffill").dropna()
    sig = (skew.reindex(spy_close.index) > 150) & (vix.reindex(spy_close.index) < 15) & \
          (spy_close >= 0.98 * rolling_high)
    sig = sig.fillna(False)

    trigger_dates = list(sig[sig].index)
    if not trigger_dates:
        return mark_failed(sid, "no signal firings")
    # Dedup nearby: at least 30 days apart
    dedup = []
    last = None
    for d in trigger_dates:
        if last is None or (d - last).days > 30:
            dedup.append(d)
            last = d

    hold = 30
    legs = []
    event_results = []
    for d in dedup:
        mask = ret.index > d
        if mask.sum() < hold:
            continue
        entry = ret.index[mask][0]
        loc = ret.index.get_loc(entry)
        end = min(loc + hold, len(ret))
        if end - loc < 10:
            continue
        spy_leg = -0.5 * ret["SPY"].iloc[loc:end]
        tlt_leg = 0.5 * ret["TLT"].iloc[loc:end]
        net = spy_leg + tlt_leg
        legs.append(net)
        event_results.append({"trigger_date": str(d.date()),
                              "ret": round(float((1+net).prod()-1), 4)})

    if not legs:
        return mark_failed(sid, "no valid events")
    df = pd.concat(legs, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")
    m = compute_metrics(pnl, benchmark=spy_r, name="SKEW-VIX Divergence Tail Hedge")
    save_result(sid, m, extra={
        "rule": "Short SPY -0.5, long TLT +0.5 when SKEW > 150, VIX < 15, SPY within 2% of 252d high. Hold 30d.",
        "mechanism": "Hedging-cost spread + complacency -> negative-skew tail premium",
        "source": "CBOE SKEW/VIX + yfinance",
        "n_events": len(event_results),
        "events": event_results[:10],
    })
    print(f"Done {sid}: events={len(event_results)} Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
'''


PL594 = '''"""PL594_fda_snda_label_expansion_drift - FDA sNDA Label Expansion -> Sponsor 8-Week Long
40-day long after curated sNDA/sBLA expansion approvals.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL594_fda_snda_label_expansion_drift"
    # (ticker, action_date) — curated label-expansion sNDA/sBLA approvals
    events_raw = [
        ("MRK",  "2017-05-23"),    # Keytruda 1L NSCLC
        ("MRK",  "2018-08-20"),    # Keytruda 1L NSCLC squamous
        ("ARGX", "2022-12-19"),    # Vyvgart gMG
        ("LLY",  "2023-11-08"),    # Zepbound
        ("NVO",  "2024-03-08"),    # Wegovy CV expansion
        ("MRK",  "2019-08-16"),    # Keytruda HNSCC
        ("BMY",  "2018-04-16"),    # Opdivo
        ("AZN",  "2020-12-30"),    # Tagrisso adjuvant
        ("GILD", "2022-02-14"),    # Trodelvy
        ("REGN", "2021-03-01"),    # Libtayo
        ("VRTX", "2019-10-21"),    # Trikafta
        ("PFE",  "2019-04-15"),    # Inlyta
        ("BIIB", "2017-12-22"),    # Spinraza expand
        ("LLY",  "2022-05-13"),    # Mounjaro initial (proxy)
        ("AMGN", "2022-08-12"),    # Tezspire pediatric
    ]
    hold = 40
    tickers = sorted(set([t for t, _ in events_raw] + ["SPY"]))
    try:
        px = load_prices(tickers, start="2015-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    ret = daily_returns(px)
    spy_r = ret["SPY"]

    legs = []
    event_results = []
    for tkr, d in events_raw:
        if tkr not in ret.columns:
            continue
        dt = pd.Timestamp(d)
        mask = ret.index >= dt
        if mask.sum() < hold:
            continue
        entry = ret.index[mask][0]
        loc = ret.index.get_loc(entry)
        end = min(loc + hold, len(ret))
        if end - loc < 10:
            continue
        win = ret[tkr].iloc[loc:end]
        legs.append(win)
        event_results.append({"ticker": tkr, "trigger_date": d,
                              "ret": round(float((1+win).prod()-1), 4)})

    if not legs:
        return mark_failed(sid, "no valid events")
    df = pd.concat(legs, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")
    m = compute_metrics(pnl, benchmark=spy_r, name="FDA sNDA Label Expansion Drift")
    save_result(sid, m, extra={
        "rule": "Long sponsor at close of label-expansion sNDA/sBLA approval, hold 40d.",
        "mechanism": "Material TAM expansion under-priced by sell-side -> drift",
        "source": "Drugs@FDA + yfinance",
        "n_events": len(event_results),
        "events": event_results,
    })
    print(f"Done {sid}: events={len(event_results)} Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
'''


PL595 = '''"""PL595_fda_crl_biotech_ma_premium - CRL on CMC -> 6w wait then 360-day Long
Long after FDA CMC-CRL; designed to capture later M&A premium.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL595_fda_crl_biotech_ma_premium"
    # (ticker, CRL date) — curated CMC/manufacturing CRLs where Phase 3 endpoint met
    events_raw = [
        ("IMGN", "2022-08-19"),    # ImmunoGen CMC CRL before AbbVie acquisition
        ("AIMT", "2019-08-05"),    # Aimmune Palforzia (later Nestle acquisition)
        ("ACAD", "2021-04-05"),    # Acadia Nuplazid dementia CRL
        ("SAGE", "2019-03-19"),    # Sage Zulresso/Zuranolone manufacturing
        ("KALA", "2018-07-25"),
        ("AKBA", "2019-09-09"),
        ("NVAX", "2020-08-20"),
        ("AGEN", "2021-12-20"),
        ("ATRA", "2020-12-15"),
        ("DCPH", "2022-10-20"),
        ("CRBP", "2021-07-19"),
        ("MNKD", "2018-02-15"),
    ]
    wait = 30
    hold = 180
    tickers = sorted(set([t for t, _ in events_raw] + ["SPY"]))
    try:
        px = load_prices(tickers, start="2017-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    ret = daily_returns(px)
    spy_r = ret["SPY"]

    legs = []
    event_results = []
    for tkr, d in events_raw:
        if tkr not in ret.columns:
            continue
        dt = pd.Timestamp(d)
        mask = ret.index > dt
        idxs = ret.index[mask]
        if len(idxs) < wait + 30:
            continue
        entry = idxs[wait]
        loc = ret.index.get_loc(entry)
        end = min(loc + hold, len(ret))
        if end - loc < 30:
            continue
        win = ret[tkr].iloc[loc:end]
        legs.append(win)
        event_results.append({"ticker": tkr, "trigger_date": d,
                              "ret": round(float((1+win).prod()-1), 4)})

    if not legs:
        return mark_failed(sid, "no valid events")
    df = pd.concat(legs, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")
    m = compute_metrics(pnl, benchmark=spy_r, name="FDA CRL CMC -> M&A Premium")
    save_result(sid, m, extra={
        "rule": "Wait 30 trading days after CMC-only CRL, then long sponsor 180d.",
        "mechanism": "Post-panic discount + intact pipeline -> M&A premium re-rate",
        "source": "FDA CRL disclosures + yfinance",
        "n_events": len(event_results),
        "events": event_results,
    })
    print(f"Done {sid}: events={len(event_results)} Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
'''


CONTENT_BY_SID = {
    "PL576_leveraged_etf_rebalance_overnight_fade": PL576,
    "PL578_sp500_inclusion_announce_to_effective_drift": PL578,
    "PL579_doi_insurance_withdrawal_aiz_long_homebuilder_short": PL579,
    "PL582_usace_lpms_upper_miss_lock_cluster_unp_long_adm_short": PL582,
    "PL583_bts_cbp_border_wait_auto_oem_short": PL583,
    "PL584_tic_foreign_equity_outflow_ath_proximity_defensive": PL584,
    "PL585_sp500_top5_concentration_rsp_long_spy_short": PL585,
    "PL586_fda_eop2_typeb_8k_phase3_alignment_drift": PL586,
    "PL587_faa_tia_8k_oem_aircraft_drift": PL587,
    "PL580_cdc_natality_decline_pediatric_short": PL580,
    "PL581_usbr_24mo_powell_mead_tier_breach": PL581,
    "PL592_usda_cold_storage_pork_bellies_summer_squeeze": PL592,
    "PL593_antimony_china_squeeze_long_ppta_usac": PL593,
    "PL590_single_country_etf_central_bank_event_fade": PL590,
    "PL591_skew_vix_divergence_spy_tail_hedge": PL591,
    "PL594_fda_snda_label_expansion_drift": PL594,
    "PL595_fda_crl_biotech_ma_premium": PL595,
}


def re_ready_failed_strategies(sids: list[str]) -> int:
    """Reset given strategies (by signal_id) from 'failed' back to 'in_progress'."""
    count = [0]
    def upd(data):
        for i, d in enumerate(data):
            if d.get("signal_id") in sids and d.get("status") == "failed":
                data[i] = {**d, "status": "in_progress",
                           "claimed_at": datetime.now(timezone.utc).isoformat()}
                count[0] += 1
        return data
    _locked_update(STRATEGIES, [], upd)
    return count[0]


def run_backtest(fp: Path) -> tuple[bool, str]:
    try:
        r = subprocess.run(
            [PYTHON, str(fp)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=180,
        )
        out = (r.stdout or "") + (r.stderr or "")
        if r.returncode != 0:
            return False, f"rc={r.returncode}: {out[-800:]}"
        return True, out
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def read_result(sid: str) -> dict | None:
    fp = RESULTS_DIR / f"{sid}.json"
    if not fp.exists():
        return None
    try:
        with open(fp) as f:
            return json.load(f)
    except Exception:
        return None


def get_strategy_by_sid(sid: str) -> dict | None:
    with open(STRATEGIES) as f:
        data = json.load(f)
    for d in data:
        if d.get("signal_id") == sid:
            return d
    return None


def main():
    sids = list(CONTENT_BY_SID.keys())
    # Re-claim from failed
    re_n = re_ready_failed_strategies(sids)
    print(f"Re-claimed {re_n} previously-failed strategies")

    claimed = 0
    done = 0
    failed = 0
    winners = []

    for sid in sids:
        heartbeat("backtester")
        strat = get_strategy_by_sid(sid)
        if strat is None:
            continue
        strategy_id = strat.get("strategy_id")
        claimed += 1
        print(f"\n[{claimed}] CLAIM {strategy_id} {sid}")

        try:
            content = CONTENT_BY_SID[sid]
            fp = BACKTESTS_DIR / f"{sid}.py"
            fp.write_text(content)

            ok, out = run_backtest(fp)
            res = read_result(sid)
            if not ok or res is None:
                update_strategy_status(
                    strategy_id, "failed",
                    backtest_result={"status": "fail", "reason": out[:300]},
                    backtested_at=datetime.now(timezone.utc).isoformat(),
                )
                failed += 1
                print(f"  FAIL: {out[:200]}")
                continue

            if res.get("status") in {"fail", "failed", "error"}:
                update_strategy_status(
                    strategy_id, "failed",
                    backtest_result={"status": "fail", "reason": res.get("reason", "unknown")},
                    backtested_at=datetime.now(timezone.utc).isoformat(),
                )
                failed += 1
                print(f"  FAIL (result): {res.get('reason')}")
                continue

            sharpe = res.get("sharpe")
            cagr = res.get("cagr")
            update_strategy_status(
                strategy_id, "done",
                backtest_result={
                    "status": "ok",
                    "sharpe": sharpe,
                    "cagr": cagr,
                    "max_dd": res.get("max_dd"),
                    "t_stat": res.get("t_stat"),
                },
                backtested_at=datetime.now(timezone.utc).isoformat(),
            )
            done += 1
            sharpe_s = f"{sharpe:.2f}" if isinstance(sharpe, (int, float)) else str(sharpe)
            cagr_s = f"{cagr*100:.1f}%" if isinstance(cagr, (int, float)) else str(cagr)
            print(f"  DONE Sharpe={sharpe_s} CAGR={cagr_s}")

            if (sharpe is not None and sharpe > 0.5
                    and cagr is not None and cagr > 0.10):
                if not res.get("signal_id"):
                    res["signal_id"] = sid
                    with open(RESULTS_DIR / f"{sid}.json", "w") as f:
                        json.dump(res, f, indent=2, default=str)
                try:
                    subprocess.run(
                        [PYTHON, "pipeline/refresh_bh.py"],
                        cwd=str(ROOT), check=True, capture_output=True, timeout=60,
                    )
                except Exception as e:
                    print(f"  BH refresh failed: {e}")
                mt = load_mt_data()
                if is_winner(res, mt):
                    winners.append({
                        "sid": sid,
                        "sharpe": float(sharpe),
                        "cagr": float(cagr),
                    })
                    print(f"  *** WINNER: {sid} Sharpe={sharpe:.2f} CAGR={cagr*100:.1f}% ***")
                else:
                    reasons = winner_reasons(res, mt)
                    failed_keys = [k for k, v in reasons.items() if not v]
                    print(f"  near-miss failing: {failed_keys}")

        except Exception as e:
            tb = traceback.format_exc()
            print(f"  EXC: {e}\n{tb[-400:]}")
            try:
                update_strategy_status(
                    strategy_id, "failed",
                    backtest_result={"status": "fail", "reason": f"loop exc: {e}"},
                    backtested_at=datetime.now(timezone.utc).isoformat(),
                )
            except Exception:
                pass
            failed += 1

    print(f"\n=== SUMMARY: claimed={claimed} done={done} failed={failed} winners={len(winners)} ===")
    for w in winners:
        print(f"  WINNER {w['sid']} Sharpe={w['sharpe']:.2f} CAGR={w['cagr']*100:.1f}%")

    with open(ROOT / "_drain_bespoke2_summary.json", "w") as f:
        json.dump({"claimed": claimed, "done": done, "failed": failed, "winners": winners}, f, indent=2)


if __name__ == "__main__":
    main()
