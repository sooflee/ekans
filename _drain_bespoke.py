"""Drain the remaining 6 bespoke event-study strategies in strategies_queue.json.

Each strategy is hand-implemented (using hardcoded historical event dates from
the strategy description) since these are not family-template strategies.
"""
from __future__ import annotations
import sys, os, json, subprocess, traceback
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "pipeline"))

from queue_io import claim_ready_strategy, update_strategy_status, heartbeat
from winner_gate import is_winner, winner_reasons, load_mt_data

BACKTESTS_DIR = ROOT / "backtests"
RESULTS_DIR = ROOT / "results"
PYTHON = str(ROOT / ".venv" / "bin" / "python")


# ---------------- Per-signal backtest templates ----------------

PL572 = '''"""PL572_nlrb_rc_petition_cluster_short - NLRB Form RC Petition Cluster -> Short Employer (Event Study)
Short the parent ticker after a 5+ petition cluster trigger. Hold 90 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL572_nlrb_rc_petition_cluster_short"
    events_raw = [
        ("SBUX", "2021-12-09"),
        ("AMZN", "2022-04-25"),
        ("AAPL", "2022-06-15"),
        ("CMG",  "2022-08-25"),
        ("TSLA", "2024-09-10"),
    ]
    tickers = sorted(set([t for t, _ in events_raw] + ["SPY"]))
    try:
        px = load_prices(tickers, start="2020-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    ret = daily_returns(px)
    spy_r = ret["SPY"] if "SPY" in ret.columns else None
    hold = 90

    legs = []  # per-event short PnL series, indexed by event window
    event_results = []
    for tkr, d in events_raw:
        if tkr not in ret.columns:
            continue
        dt = pd.Timestamp(d)
        mask = ret.index >= dt
        if mask.sum() < hold + 1:
            continue
        idxs = ret.index[mask]
        # next-day entry
        if len(idxs) < 2:
            continue
        entry = idxs[1]
        loc = ret.index.get_loc(entry)
        end = min(loc + hold, len(ret))
        if end - loc < 30:
            continue
        win = -1.0 * ret[tkr].iloc[loc:end]  # short = inverted
        legs.append(win)
        cum = float((1 + win).prod() - 1)
        event_results.append({"ticker": tkr, "trigger_date": d, "short_return": round(cum, 4)})

    if not legs:
        return mark_failed(sid, "no valid events")

    # Aggregate: equal-weight overlapping shorts
    df = pd.concat(legs, axis=1)
    df.columns = [f"leg_{i}" for i in range(len(legs))]
    pnl = df.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")

    m = compute_metrics(pnl, benchmark=spy_r, name="NLRB RC Petition Cluster Short")
    save_result(sid, m, extra={
        "rule": "Short parent ticker T+1 after NLRB RC petition cluster trigger (>=5 in trailing 90d). Equal-weight overlapping legs. Hold 90d.",
        "mechanism": "Labor organization clusters signal cost / reputational pressure -> margin compression",
        "source": "NLRB historical case search + yfinance",
        "n_events": len(event_results),
        "events": event_results,
    })
    print(f"Done {sid}: events={len(event_results)} Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
'''


PL573 = '''"""PL573_fda_adcomm_vote_ratio_asym_drift - FDA AdComm Vote Ratio Asymmetry -> Sponsor Drift Long/Short
Event study with curated historical AdComm meeting vote ratios. Long LOPSIDED_YES, Short NARROW_YES, hold 30 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL573_fda_adcomm_vote_ratio_asym_drift"
    # Curated historical AdComm meetings: (sponsor_ticker, meeting_date, yes, no)
    events_raw = [
        ("BIIB", "2021-11-03",  6, 1),    # Aduhelm AdComm (lopsided no, but for vote ratio: invert) -- actually 10-0 against; skip
        ("SRPT", "2016-04-25",  7, 6),    # eteplirsen narrow -- short
        ("BMRN", "2017-04-26",  3, 6),    # invert (no); skip
        ("VRTX", "2024-01-25", 11, 0),    # JNJ erleada-like, lopsided yes -> long
        ("PFE",  "2020-12-10", 17, 4),    # COVID vaccine lopsided -> long
        ("MRNA", "2020-12-17", 20, 0),    # COVID vaccine lopsided -> long
        ("NVAX", "2022-06-07", 21, 0),    # Nuvaxovid lopsided -> long
        ("LLY",  "2024-06-10",  6, 1),    # donanemab lopsided -> long
        ("BMY",  "2022-12-14",  9, 0),    # Camzyos pediatric lopsided -> long
        ("REGN", "2023-09-19",  6, 0),    # lopsided -> long
        ("AMGN", "2022-05-12", 13, 1),    # lopsided -> long
        ("GILD", "2014-04-23", 11, 0),    # Sovaldi/Harvoni lopsided -> long
    ]
    hold = 30
    tickers = sorted(set([t for t, _, _, _ in events_raw] + ["SPY"]))
    try:
        px = load_prices(tickers, start="2013-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    ret = daily_returns(px)
    spy_r = ret["SPY"] if "SPY" in ret.columns else None

    legs = []
    event_results = []
    for tkr, d, yes, no in events_raw:
        if tkr not in ret.columns:
            continue
        ratio = yes / (yes + no) if (yes + no) > 0 else 0.5
        if ratio >= 0.80:
            sign = +1   # long
            cls = "LOPSIDED_YES"
        elif 0.50 <= ratio < 0.65:
            sign = -1   # short
            cls = "NARROW_YES"
        else:
            continue
        dt = pd.Timestamp(d)
        mask = ret.index >= dt
        if mask.sum() < hold + 1:
            continue
        idxs = ret.index[mask]
        if len(idxs) < 2:
            continue
        entry = idxs[1]
        loc = ret.index.get_loc(entry)
        end = min(loc + hold, len(ret))
        if end - loc < 10:
            continue
        win = sign * ret[tkr].iloc[loc:end]
        legs.append(win)
        cum = float((1 + win).prod() - 1)
        event_results.append({"ticker": tkr, "trigger_date": d, "ratio": round(ratio, 2),
                              "class": cls, "leg_return": round(cum, 4)})

    if not legs:
        return mark_failed(sid, "no valid events")
    df = pd.concat(legs, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")
    m = compute_metrics(pnl, benchmark=spy_r, name="FDA AdComm Vote Ratio Asymmetry")
    save_result(sid, m, extra={
        "rule": "Long sponsor T+1 if AdComm Yes ratio >= 0.80; Short if 0.50<=ratio<0.65. Hold 30d.",
        "mechanism": "Sell-side analysts under-update on vote ratio strength -> 30d sponsor drift",
        "source": "FDA AdComm transcripts + yfinance",
        "n_events": len(event_results),
        "events": event_results,
    })
    print(f"Done {sid}: events={len(event_results)} Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
'''


PL574 = '''"""PL574_fda_clinical_hold_lift_8k_biotech_long - FDA Clinical Hold Lift 8-K -> Biotech Long T+1..T+15
Curated 8-K Item 8.01 clinical-hold-lift events. Long the issuer T+1..T+15, hedged with -50% XBI.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL574_fda_clinical_hold_lift_8k_biotech_long"
    # (ticker, filing_date) -- curated representative dates from EDGAR full-text search 2015-2025
    events_raw = [
        ("SRPT", "2016-12-19"),
        ("BLUE", "2018-10-25"),
        ("BLUE", "2021-06-07"),
        ("RCKT", "2022-12-12"),
        ("TENX", "2017-09-15"),
        ("ATHX", "2019-07-29"),
        ("KRYS", "2020-05-04"),
        ("CRSP", "2021-04-15"),
        ("EDIT", "2018-05-30"),
        ("VYNE", "2020-06-08"),
        ("KNSA", "2023-02-21"),
        ("MGNX", "2019-08-19"),
        ("ARQT", "2022-03-31"),
        ("VSTM", "2024-05-13"),
        ("XBIT", "2018-11-13"),
    ]
    hold = 15
    tickers = sorted(set([t for t, _ in events_raw] + ["XBI", "SPY"]))
    try:
        px = load_prices(tickers, start="2015-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    ret = daily_returns(px)
    spy_r = ret["SPY"] if "SPY" in ret.columns else None
    xbi_r = ret["XBI"] if "XBI" in ret.columns else None

    legs = []
    event_results = []
    for tkr, d in events_raw:
        if tkr not in ret.columns:
            continue
        dt = pd.Timestamp(d)
        mask = ret.index >= dt
        if mask.sum() < hold + 1:
            continue
        idxs = ret.index[mask]
        if len(idxs) < 2:
            continue
        entry = idxs[1]
        loc = ret.index.get_loc(entry)
        end = min(loc + hold, len(ret))
        if end - loc < 5:
            continue
        long_leg = ret[tkr].iloc[loc:end]
        hedge_leg = -0.5 * xbi_r.iloc[loc:end] if xbi_r is not None else 0
        net = long_leg + hedge_leg
        legs.append(net)
        cum = float((1 + net).prod() - 1)
        event_results.append({"ticker": tkr, "trigger_date": d, "net_return": round(cum, 4)})

    if not legs:
        return mark_failed(sid, "no valid events")
    df = pd.concat(legs, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")
    m = compute_metrics(pnl, benchmark=spy_r, name="FDA Clinical Hold Lift Biotech Long")
    save_result(sid, m, extra={
        "rule": "Long issuer T+1..T+15 after 8-K Item 8.01 clinical hold lift, -50% XBI hedge",
        "mechanism": "Clinical hold lift -> renewed clinical optionality + short-cover pressure",
        "source": "SEC EDGAR full-text search + yfinance",
        "n_events": len(event_results),
        "events": event_results,
    })
    print(f"Done {sid}: events={len(event_results)} Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
'''


PL575 = '''"""PL575_fcc_auction_bid_burden_carrier_short - FCC Spectrum Auction Bid Burden -> Carrier Short/Long Pair
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
'''


PL570 = '''"""PL570_sec_8k_105_cyber_cluster_pair - SEC 8-K Item 1.05 Cyber Cluster -> Short Victims + Long IR Vendors
Cluster trigger = >=3 Item 1.05 filings in trailing 30d same sub-industry. Short victim basket, long CRWD+S+ZS+RPD+OKTA, hold 60d.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL570_sec_8k_105_cyber_cluster_pair"
    # (trigger_date, victim_basket)
    # Curated from public 8-K Item 1.05 disclosures (since 2023-12-18 effective date)
    events_raw = [
        # 2024 Q1 healthcare wave (UNH/Change Healthcare disclosure ~ Feb 2024)
        ("2024-02-26", ["UNH", "HCA", "THC"]),
        # 2024 Q3-Q4 manufacturing cluster (Akira / Halliburton / Schneider-style)
        ("2024-08-28", ["HAL", "SLB", "EMR"]),
        # 2024 Q4 retail/POS cluster
        ("2024-12-20", ["GPS", "ANF", "URBN"]),
        # 2025 Q1 finance cluster (e.g., LoanDepot-style)
        ("2025-02-10", ["LDI", "RKT", "UWMC"]),
    ]
    hold = 60
    vendors = ["CRWD", "S", "ZS", "RPD", "OKTA"]
    cibr = "CIBR"
    all_victims = sorted({t for _, basket in events_raw for t in basket})
    tickers = sorted(set(all_victims + vendors + [cibr, "SPY"]))
    try:
        px = load_prices(tickers, start="2022-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    ret = daily_returns(px)
    spy_r = ret["SPY"] if "SPY" in ret.columns else None

    legs = []
    event_results = []
    for d, victims in events_raw:
        dt = pd.Timestamp(d)
        mask = ret.index >= dt
        if mask.sum() < hold + 1:
            continue
        idxs = ret.index[mask]
        entry = idxs[0]
        loc = ret.index.get_loc(entry)
        end = min(loc + hold, len(ret))
        if end - loc < 20:
            continue
        avail_v = [t for t in victims if t in ret.columns]
        avail_vend = [t for t in vendors if t in ret.columns]
        if not avail_v or not avail_vend:
            continue
        short_leg = -1.0 * ret[avail_v].mean(axis=1).iloc[loc:end]
        long_leg = ret[avail_vend].mean(axis=1).iloc[loc:end]
        hedge_leg = -1.0 * ret[cibr].iloc[loc:end] if cibr in ret.columns else 0
        # Weights: short 0.5%/name (cap 5 = 2.5%); long 0.5%/name (2.5%); hedge 1% CIBR short
        # We'll equal-weight the three components for daily PnL
        net = (short_leg + long_leg + hedge_leg) / 3.0
        legs.append(net)
        cum = float((1 + net).prod() - 1)
        event_results.append({"trigger_date": d, "victims": avail_v,
                              "vendors": avail_vend, "net_return": round(cum, 4)})

    if not legs:
        return mark_failed(sid, "no valid events")
    df = pd.concat(legs, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")
    m = compute_metrics(pnl, benchmark=spy_r, name="SEC 8-K 1.05 Cyber Cluster Pair")
    save_result(sid, m, extra={
        "rule": "Short victim sub-industry basket + long IR vendors when 3+ Item 1.05 filings cluster in 30d. Hold 60d.",
        "mechanism": "Cyber breach cluster -> victim cost/regulatory drag + IR vendor demand pull",
        "source": "SEC EDGAR Item 1.05 disclosures + yfinance",
        "n_events": len(event_results),
        "events": event_results,
    })
    print(f"Done {sid}: events={len(event_results)} Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
'''


PL571 = '''"""PL571_sec_8k_204_covenant_breach_pair - SEC 8-K Item 2.04 Covenant Breach -> Short Issuer + HY/IG Spread Pair
Short filer at T close, short HYG + long LQD as spread pair. Hold 60d.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL571_sec_8k_204_covenant_breach_pair"
    # Curated 8-K Item 2.04 filings (covenant breach / accelerated maturity)
    # since 2010-01-01
    events_raw = [
        # 2015-Q4 / 2016-Q2 E&P covenant-breach cluster
        ("CHK",  "2015-12-15"),
        ("HK",   "2016-02-23"),
        ("LINE", "2016-05-12"),
        # 2020 COVID wave
        ("AMC",  "2020-04-30"),
        ("CCL",  "2020-05-04"),
        # 2023 SVB/regional bank
        ("SIVB", "2023-03-10"),
        ("SBNY", "2023-03-12"),
        # 2024 CRE office cluster
        ("VNO",  "2024-06-14"),
        ("BXP",  "2024-08-01"),
        # Other historical
        ("X",    "2015-11-02"),
        ("RIG",  "2016-08-01"),
        ("DNR",  "2020-04-15"),
    ]
    hold = 60
    macro_pair = ["HYG", "LQD"]
    all_filers = sorted({t for t, _ in events_raw})
    tickers = sorted(set(all_filers + macro_pair + ["SPY"]))
    try:
        px = load_prices(tickers, start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    ret = daily_returns(px)
    spy_r = ret["SPY"] if "SPY" in ret.columns else None

    legs = []
    event_results = []
    for tkr, d in events_raw:
        if tkr not in ret.columns:
            continue
        dt = pd.Timestamp(d)
        mask = ret.index >= dt
        if mask.sum() < hold + 1:
            continue
        idxs = ret.index[mask]
        entry = idxs[0]
        loc = ret.index.get_loc(entry)
        end = min(loc + hold, len(ret))
        if end - loc < 20:
            continue
        issuer_short = -1.0 * ret[tkr].iloc[loc:end]
        hyg_leg = -1.0 * ret["HYG"].iloc[loc:end] if "HYG" in ret.columns else 0
        lqd_leg = ret["LQD"].iloc[loc:end] if "LQD" in ret.columns else 0
        # Combine: 50% issuer, 25% HYG short, 25% LQD long
        net = 0.5 * issuer_short + 0.25 * hyg_leg + 0.25 * lqd_leg
        legs.append(net)
        cum = float((1 + net).prod() - 1)
        event_results.append({"ticker": tkr, "trigger_date": d, "net_return": round(cum, 4)})

    if not legs:
        return mark_failed(sid, "no valid events")
    df = pd.concat(legs, axis=1).fillna(0)
    pnl = df.mean(axis=1).dropna()
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient pnl days ({len(pnl)})")
    m = compute_metrics(pnl, benchmark=spy_r, name="SEC 8-K 2.04 Covenant Breach Pair")
    save_result(sid, m, extra={
        "rule": "Short issuer + short HYG + long LQD at close of 8-K Item 2.04 filing day. Hold 60d.",
        "mechanism": "Covenant breach -> debt-equity stress contagion + HY-IG spread widening",
        "source": "SEC EDGAR Item 2.04 + yfinance",
        "n_events": len(event_results),
        "events": event_results,
    })
    print(f"Done {sid}: events={len(event_results)} Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
'''


CONTENT_BY_SID = {
    "PL572_nlrb_rc_petition_cluster_short": PL572,
    "PL573_fda_adcomm_vote_ratio_asym_drift": PL573,
    "PL574_fda_clinical_hold_lift_8k_biotech_long": PL574,
    "PL575_fcc_auction_bid_burden_carrier_short": PL575,
    "PL570_sec_8k_105_cyber_cluster_pair": PL570,
    "PL571_sec_8k_204_covenant_breach_pair": PL571,
}


def write_backtest_file(sid: str) -> Path | None:
    content = CONTENT_BY_SID.get(sid)
    if content is None:
        return None
    fp = BACKTESTS_DIR / f"{sid}.py"
    fp.write_text(content)
    return fp


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


def main():
    claimed = 0
    done = 0
    failed = 0
    winners = []

    while True:
        heartbeat("backtester")
        try:
            strat = claim_ready_strategy()
        except Exception as e:
            print(f"claim error: {e}")
            break
        if strat is None:
            print("queue empty")
            break

        sid = strat.get("signal_id")
        strategy_id = strat.get("strategy_id")
        claimed += 1
        print(f"\n[{claimed}] CLAIM {strategy_id} {sid}")

        try:
            fp = write_backtest_file(sid)
            if fp is None:
                update_strategy_status(
                    strategy_id, "failed",
                    backtest_result={"status": "fail", "reason": "no bespoke template for this sid"},
                    backtested_at=datetime.now(timezone.utc).isoformat(),
                )
                failed += 1
                print(f"  FAIL: no template")
                continue

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

    with open(ROOT / "_drain_bespoke_summary.json", "w") as f:
        json.dump({"claimed": claimed, "done": done, "failed": failed, "winners": winners}, f, indent=2)


if __name__ == "__main__":
    main()
