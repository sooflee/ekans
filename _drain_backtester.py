"""Drain strategies_queue.json using family-template backtests.

Claims one ready strategy at a time, writes backtests/<signal_id>.py from a
family-appropriate template, runs it, then updates queue status. Tracks winners
that pass the BH-corrected winner_gate.

Run from project root: .venv/bin/python _drain_backtester.py
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

# ---------------- Template generators per family ----------------

PPI_TEMPLATE = '''"""{sid} {name}
{rule}
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "{sid}"
    fred_id = "{fred_id}"
    tickers = {tickers!r}
    trade_tickers = {trade_tickers!r}
    hold = 42

    try:
        fred = load_fred([fred_id], start="1990-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED load: {{e}}")
    if fred is None or fred.empty:
        return mark_failed(sid, "FRED empty")
    s = fred[fred_id].dropna()
    if len(s) < 24:
        return mark_failed(sid, f"insufficient FRED history ({{len(s)}})")

    # 3-month % change
    chg3 = s.pct_change(3).dropna()
    # 6-month decline check + trough inflect
    triggers = []
    fired = False
    neg_streak = 0
    for i, (dt_idx, v) in enumerate(chg3.items()):
        if v < 0:
            neg_streak += 1
            fired = False
        elif v >= 0 and neg_streak >= 6 and not fired:
            triggers.append(dt_idx)
            fired = True
            neg_streak = 0
        else:
            neg_streak = 0

    if len(triggers) < 2:
        return mark_failed(sid, f"no triggers ({{len(triggers)}})")

    try:
        px = load_prices(tickers, start="2000-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {{e}}")
    ret = daily_returns(px)
    available = [t for t in trade_tickers if t in ret.columns]
    if len(available) == 0:
        return mark_failed(sid, "no trade tickers available")
    basket = ret[available].mean(axis=1)
    spy_r = ret["SPY"] if "SPY" in ret.columns else basket * 0

    pnl_parts = []
    events = []
    for td in triggers:
        td_ts = td + pd.offsets.MonthBegin(1) if hasattr(td, "month") else pd.Timestamp(td)
        mask = basket.index >= td_ts
        if mask.sum() < max(20, hold // 2):
            continue
        entry = basket.index[mask][0]
        loc = basket.index.get_loc(entry)
        end = min(loc + hold, len(basket))
        if end - loc < 20:
            continue
        win = basket.iloc[loc:end]
        pnl_parts.append(win)
        cum = float((1 + win).prod() - 1)
        sp = spy_r.reindex(win.index).fillna(0)
        sp_cum = float((1 + sp).prod() - 1)
        events.append({{
            "trigger_date": str(td_ts.date()),
            "return": round(cum, 4),
            "spy_return": round(sp_cum, 4),
            "excess": round(cum - sp_cum, 4),
        }})

    if not events:
        return mark_failed(sid, "no valid events after alignment")
    pnl = pd.concat(pnl_parts)
    pnl = pnl[~pnl.index.duplicated(keep="first")]
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient in-pos days ({{len(pnl)}})")

    m = compute_metrics(pnl, benchmark=spy_r.reindex(pnl.index).dropna(), name="{name}")
    rets = [e["return"] for e in events]
    save_result(sid, m, extra={{
        "rule": "{rule_esc}",
        "mechanism": "PPI commodity trough inflection -> downstream materials/industrials demand recovery",
        "source": f"FRED {{fred_id}} + yfinance",
        "n_events": len(events),
        "avg_return": round(float(np.mean(rets)), 4),
        "events": events,
    }})
    print(f"Done {{sid}}: events={{len(events)}}, Sharpe={{m.get('sharpe',0):.2f}}, CAGR={{m.get('cagr',0)*100:.1f}}%")


if __name__ == "__main__":
    main()
'''

CPI_TEMPLATE = '''"""{sid} {name}
{rule}
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "{sid}"
    fred_id = "{fred_id}"
    tickers = {tickers!r}
    trade_tickers = {trade_tickers!r}
    hold = 42

    try:
        fred = load_fred([fred_id], start="1990-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED load: {{e}}")
    if fred is None or fred.empty:
        return mark_failed(sid, "FRED empty")
    s = fred[fred_id].dropna()
    if len(s) < 24:
        return mark_failed(sid, f"insufficient FRED history ({{len(s)}})")

    yoy = s.pct_change(12).dropna() * 100
    triggers = []
    neg_streak = 0
    fired = False
    for dt_idx, v in yoy.items():
        if v < 0:
            neg_streak += 1
            fired = False
        elif v >= 0 and neg_streak >= 2 and not fired:
            triggers.append(dt_idx)
            fired = True
            neg_streak = 0
        else:
            neg_streak = 0

    if len(triggers) < 2:
        return mark_failed(sid, f"no triggers ({{len(triggers)}})")

    try:
        px = load_prices(tickers, start="2000-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {{e}}")
    ret = daily_returns(px)
    available = [t for t in trade_tickers if t in ret.columns]
    if not available:
        return mark_failed(sid, "no trade tickers available")
    basket = ret[available].mean(axis=1)
    spy_r = ret["SPY"] if "SPY" in ret.columns else basket * 0

    pnl_parts = []
    events = []
    for td in triggers:
        td_ts = td + pd.offsets.MonthBegin(1) if hasattr(td, "month") else pd.Timestamp(td)
        mask = basket.index >= td_ts
        if mask.sum() < max(20, hold // 2):
            continue
        entry = basket.index[mask][0]
        loc = basket.index.get_loc(entry)
        end = min(loc + hold, len(basket))
        if end - loc < 20:
            continue
        win = basket.iloc[loc:end]
        pnl_parts.append(win)
        cum = float((1 + win).prod() - 1)
        sp = spy_r.reindex(win.index).fillna(0)
        sp_cum = float((1 + sp).prod() - 1)
        events.append({{
            "trigger_date": str(td_ts.date()),
            "return": round(cum, 4),
            "spy_return": round(sp_cum, 4),
            "excess": round(cum - sp_cum, 4),
        }})

    if not events:
        return mark_failed(sid, "no valid events after alignment")
    pnl = pd.concat(pnl_parts)
    pnl = pnl[~pnl.index.duplicated(keep="first")]
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient in-pos days ({{len(pnl)}})")

    m = compute_metrics(pnl, benchmark=spy_r.reindex(pnl.index).dropna(), name="{name}")
    rets = [e["return"] for e in events]
    save_result(sid, m, extra={{
        "rule": "{rule_esc}",
        "mechanism": "CPI subindex deflation trough -> real-income tailwind for discretionary/staples",
        "source": f"FRED {{fred_id}} + yfinance",
        "n_events": len(events),
        "avg_return": round(float(np.mean(rets)), 4),
        "events": events,
    }})
    print(f"Done {{sid}}: events={{len(events)}}, Sharpe={{m.get('sharpe',0):.2f}}, CAGR={{m.get('cagr',0)*100:.1f}}%")


if __name__ == "__main__":
    main()
'''

JOLTS_TEMPLATE = '''"""{sid} {name}
{rule}
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "{sid}"
    fred_id = "{fred_id}"
    tickers = {tickers!r}
    trade_tickers = {trade_tickers!r}
    hold = 60

    try:
        fred = load_fred([fred_id], start="2001-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED load: {{e}}")
    if fred is None or fred.empty:
        return mark_failed(sid, "FRED empty")
    s = fred[fred_id].dropna()
    if len(s) < 24:
        return mark_failed(sid, f"insufficient FRED history ({{len(s)}})")

    yoy = s.pct_change(12).dropna()
    triggers = []
    fired = False
    for dt_idx, v in yoy.items():
        if v > 0.20 and not fired:
            triggers.append(dt_idx)
            fired = True
        elif v < 0.05:
            fired = False

    if len(triggers) < 2:
        return mark_failed(sid, f"no triggers ({{len(triggers)}})")

    try:
        px = load_prices(tickers, start="2002-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {{e}}")
    ret = daily_returns(px)
    available = [t for t in trade_tickers if t in ret.columns]
    if not available:
        return mark_failed(sid, "no trade tickers available")
    basket = ret[available].mean(axis=1)
    spy_r = ret["SPY"] if "SPY" in ret.columns else basket * 0

    pnl_parts = []
    events = []
    for td in triggers:
        td_ts = td + pd.offsets.MonthBegin(1) if hasattr(td, "month") else pd.Timestamp(td)
        mask = basket.index >= td_ts
        if mask.sum() < max(20, hold // 2):
            continue
        entry = basket.index[mask][0]
        loc = basket.index.get_loc(entry)
        end = min(loc + hold, len(basket))
        if end - loc < 20:
            continue
        win = basket.iloc[loc:end]
        pnl_parts.append(win)
        cum = float((1 + win).prod() - 1)
        sp = spy_r.reindex(win.index).fillna(0)
        sp_cum = float((1 + sp).prod() - 1)
        events.append({{
            "trigger_date": str(td_ts.date()),
            "return": round(cum, 4),
            "spy_return": round(sp_cum, 4),
            "excess": round(cum - sp_cum, 4),
        }})

    if not events:
        return mark_failed(sid, "no valid events after alignment")
    pnl = pd.concat(pnl_parts)
    pnl = pnl[~pnl.index.duplicated(keep="first")]
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient in-pos days ({{len(pnl)}})")

    m = compute_metrics(pnl, benchmark=spy_r.reindex(pnl.index).dropna(), name="{name}")
    rets = [e["return"] for e in events]
    save_result(sid, m, extra={{
        "rule": "{rule_esc}",
        "mechanism": "JOLTS industry openings surge -> hiring/wage growth -> sector earnings tailwind",
        "source": f"FRED {{fred_id}} + yfinance",
        "n_events": len(events),
        "avg_return": round(float(np.mean(rets)), 4),
        "events": events,
    }})
    print(f"Done {{sid}}: events={{len(events)}}, Sharpe={{m.get('sharpe',0):.2f}}, CAGR={{m.get('cagr',0)*100:.1f}}%")


if __name__ == "__main__":
    main()
'''

MFG_ORDERS_TEMPLATE = '''"""{sid} {name}
{rule}
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "{sid}"
    fred_id = "{fred_id}"
    tickers = {tickers!r}
    trade_tickers = {trade_tickers!r}
    hold = 42

    try:
        fred = load_fred([fred_id], start="1992-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED load: {{e}}")
    if fred is None or fred.empty:
        return mark_failed(sid, "FRED empty")
    s = fred[fred_id].dropna()
    if len(s) < 24:
        return mark_failed(sid, f"insufficient FRED history ({{len(s)}})")

    mean12 = s.rolling(12).mean()
    std12 = s.rolling(12).std()
    z = (s - mean12) / std12
    z = z.dropna()
    triggers = []
    fired = False
    for dt_idx, v in z.items():
        if v > 2.0 and not fired:
            triggers.append(dt_idx)
            fired = True
        elif v < 0.5:
            fired = False

    if len(triggers) < 2:
        return mark_failed(sid, f"no triggers ({{len(triggers)}})")

    try:
        px = load_prices(tickers, start="2000-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {{e}}")
    ret = daily_returns(px)
    available = [t for t in trade_tickers if t in ret.columns]
    if not available:
        return mark_failed(sid, "no trade tickers available")
    basket = ret[available].mean(axis=1)
    spy_r = ret["SPY"] if "SPY" in ret.columns else basket * 0

    pnl_parts = []
    events = []
    for td in triggers:
        td_ts = td + pd.offsets.MonthBegin(1) if hasattr(td, "month") else pd.Timestamp(td)
        mask = basket.index >= td_ts
        if mask.sum() < max(20, hold // 2):
            continue
        entry = basket.index[mask][0]
        loc = basket.index.get_loc(entry)
        end = min(loc + hold, len(basket))
        if end - loc < 20:
            continue
        win = basket.iloc[loc:end]
        pnl_parts.append(win)
        cum = float((1 + win).prod() - 1)
        sp = spy_r.reindex(win.index).fillna(0)
        sp_cum = float((1 + sp).prod() - 1)
        events.append({{
            "trigger_date": str(td_ts.date()),
            "return": round(cum, 4),
            "spy_return": round(sp_cum, 4),
            "excess": round(cum - sp_cum, 4),
        }})

    if not events:
        return mark_failed(sid, "no valid events after alignment")
    pnl = pd.concat(pnl_parts)
    pnl = pnl[~pnl.index.duplicated(keep="first")]
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient in-pos days ({{len(pnl)}})")

    m = compute_metrics(pnl, benchmark=spy_r.reindex(pnl.index).dropna(), name="{name}")
    rets = [e["return"] for e in events]
    save_result(sid, m, extra={{
        "rule": "{rule_esc}",
        "mechanism": "Mfg orders surge >2sd -> upstream capex / industrials demand pulse",
        "source": f"FRED {{fred_id}} + yfinance",
        "n_events": len(events),
        "avg_return": round(float(np.mean(rets)), 4),
        "events": events,
    }})
    print(f"Done {{sid}}: events={{len(events)}}, Sharpe={{m.get('sharpe',0):.2f}}, CAGR={{m.get('cagr',0)*100:.1f}}%")


if __name__ == "__main__":
    main()
'''

COMMODITY_TEMPLATE = '''"""{sid} {name}
{rule}
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "{sid}"
    fred_id = "{fred_id}"
    tickers = {tickers!r}
    trade_tickers = {trade_tickers!r}
    hold = 42

    try:
        fred = load_fred([fred_id], start="1990-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED load: {{e}}")
    if fred is None or fred.empty:
        return mark_failed(sid, "FRED empty")
    s = fred[fred_id].dropna()
    if len(s) < 24:
        return mark_failed(sid, f"insufficient FRED history ({{len(s)}})")

    chg3 = s.pct_change(3).dropna()
    triggers = []
    fired = False
    for dt_idx, v in chg3.items():
        if v > 0.30 and not fired:
            triggers.append(dt_idx)
            fired = True
        elif v < 0.05:
            fired = False

    if len(triggers) < 2:
        return mark_failed(sid, f"no triggers ({{len(triggers)}})")

    try:
        px = load_prices(tickers, start="2000-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {{e}}")
    ret = daily_returns(px)
    available = [t for t in trade_tickers if t in ret.columns]
    if not available:
        return mark_failed(sid, "no trade tickers available")
    basket = ret[available].mean(axis=1)
    spy_r = ret["SPY"] if "SPY" in ret.columns else basket * 0

    pnl_parts = []
    events = []
    for td in triggers:
        td_ts = td + pd.offsets.MonthBegin(1) if hasattr(td, "month") else pd.Timestamp(td)
        # Wait 10 trading days after trigger
        mask = basket.index >= td_ts
        if mask.sum() < max(20, hold // 2 + 10):
            continue
        idxs = basket.index[mask]
        if len(idxs) <= 10:
            continue
        entry = idxs[10]
        loc = basket.index.get_loc(entry)
        end = min(loc + hold, len(basket))
        if end - loc < 20:
            continue
        win = basket.iloc[loc:end]
        pnl_parts.append(win)
        cum = float((1 + win).prod() - 1)
        sp = spy_r.reindex(win.index).fillna(0)
        sp_cum = float((1 + sp).prod() - 1)
        events.append({{
            "trigger_date": str(td_ts.date()),
            "return": round(cum, 4),
            "spy_return": round(sp_cum, 4),
            "excess": round(cum - sp_cum, 4),
        }})

    if not events:
        return mark_failed(sid, "no valid events after alignment")
    pnl = pd.concat(pnl_parts)
    pnl = pnl[~pnl.index.duplicated(keep="first")]
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient in-pos days ({{len(pnl)}})")

    m = compute_metrics(pnl, benchmark=spy_r.reindex(pnl.index).dropna(), name="{name}")
    rets = [e["return"] for e in events]
    save_result(sid, m, extra={{
        "rule": "{rule_esc}",
        "mechanism": "Commodity spike -> producer revenue boost / supply-side rebalancing",
        "source": f"FRED {{fred_id}} + yfinance",
        "n_events": len(events),
        "avg_return": round(float(np.mean(rets)), 4),
        "events": events,
    }})
    print(f"Done {{sid}}: events={{len(events)}}, Sharpe={{m.get('sharpe',0):.2f}}, CAGR={{m.get('cagr',0)*100:.1f}}%")


if __name__ == "__main__":
    main()
'''

BANK_CREDIT_TEMPLATE = '''"""{sid} {name}
{rule}
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "{sid}"
    fred_id = "{fred_id}"
    tickers = {tickers!r}
    trade_tickers = {trade_tickers!r}
    hold = 126

    try:
        fred = load_fred([fred_id], start="1995-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED load: {{e}}")
    if fred is None or fred.empty:
        return mark_failed(sid, "FRED empty")
    s = fred[fred_id].dropna()
    if len(s) < 12:
        return mark_failed(sid, f"insufficient FRED history ({{len(s)}})")

    # Local peak detection: value higher than prior 6 months and starts declining
    triggers = []
    arr = s.values
    idx = s.index
    for i in range(6, len(arr) - 1):
        window_prior = arr[i-6:i]
        if arr[i] > window_prior.max() and arr[i] > arr[i+1]:
            triggers.append(idx[i])
    # Cluster nearby
    dedup = []
    last = None
    for d in triggers:
        if last is None or (d - last).days > 180:
            dedup.append(d)
        last = d
    triggers = dedup

    if len(triggers) < 2:
        return mark_failed(sid, f"no triggers ({{len(triggers)}})")

    try:
        px = load_prices(tickers, start="2000-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {{e}}")
    ret = daily_returns(px)
    available = [t for t in trade_tickers if t in ret.columns]
    if not available:
        return mark_failed(sid, "no trade tickers available")
    basket = ret[available].mean(axis=1)
    spy_r = ret["SPY"] if "SPY" in ret.columns else basket * 0

    pnl_parts = []
    events = []
    for td in triggers:
        td_ts = td + pd.offsets.MonthBegin(1) if hasattr(td, "month") else pd.Timestamp(td)
        mask = basket.index >= td_ts
        if mask.sum() < max(30, hold // 2):
            continue
        entry = basket.index[mask][0]
        loc = basket.index.get_loc(entry)
        end = min(loc + hold, len(basket))
        if end - loc < 30:
            continue
        win = basket.iloc[loc:end]
        pnl_parts.append(win)
        cum = float((1 + win).prod() - 1)
        sp = spy_r.reindex(win.index).fillna(0)
        sp_cum = float((1 + sp).prod() - 1)
        events.append({{
            "trigger_date": str(td_ts.date()),
            "return": round(cum, 4),
            "spy_return": round(sp_cum, 4),
            "excess": round(cum - sp_cum, 4),
        }})

    if not events:
        return mark_failed(sid, "no valid events after alignment")
    pnl = pd.concat(pnl_parts)
    pnl = pnl[~pnl.index.duplicated(keep="first")]
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient in-pos days ({{len(pnl)}})")

    m = compute_metrics(pnl, benchmark=spy_r.reindex(pnl.index).dropna(), name="{name}")
    rets = [e["return"] for e in events]
    save_result(sid, m, extra={{
        "rule": "{rule_esc}",
        "mechanism": "Bank credit/delinquency cycle peak -> mean-revert into easing cycle / regional bank re-rating",
        "source": f"FRED {{fred_id}} + yfinance",
        "n_events": len(events),
        "avg_return": round(float(np.mean(rets)), 4),
        "events": events,
    }})
    print(f"Done {{sid}}: events={{len(events)}}, Sharpe={{m.get('sharpe',0):.2f}}, CAGR={{m.get('cagr',0)*100:.1f}}%")


if __name__ == "__main__":
    main()
'''

TEMPLATES = {
    "PPI commodity (WPU)": PPI_TEMPLATE,
    "CPI subindex (CUSR0000)": CPI_TEMPLATE,
    "JOLTS by industry": JOLTS_TEMPLATE,
    "Manufacturers orders": MFG_ORDERS_TEMPLATE,
    "Commodity prices (monthly)": COMMODITY_TEMPLATE,
    "Bank credit/delinquency": BANK_CREDIT_TEMPLATE,
}


def extract_fred_id(strat: dict) -> str | None:
    """Extract the FRED series ID from the strategy entry."""
    # Try data_sources_concrete first
    ds = strat.get("data_sources_concrete", {})
    fund = ds.get("fundamental", "")
    if fund.startswith("FRED "):
        return fund.replace("FRED ", "").strip().split()[0]
    # Fallback: parse from name
    name = strat.get("name", "")
    if "FRED " in name:
        parts = name.split("FRED ")[1].split()
        return parts[0] if parts else None
    return None


def trade_tickers_for(tickers: list[str]) -> list[str]:
    """Filter out SPY (used as benchmark, not as trade leg)."""
    return [t for t in tickers if t != "SPY"]


def write_backtest_file(strat: dict) -> Path | None:
    family = strat.get("_source_family")
    if family not in TEMPLATES:
        return None
    sid = strat["signal_id"]
    fred_id = extract_fred_id(strat)
    if not fred_id:
        return None
    tickers = strat.get("tickers", [])
    trade = trade_tickers_for(tickers)
    if not trade:
        return None
    name = strat.get("name", sid).replace('"', "'")
    rule = strat.get("rule", "").replace('"', "'")
    rule_esc = rule.replace('\\', '\\\\').replace('"', '\\"')

    tpl = TEMPLATES[family]
    src = tpl.format(
        sid=sid,
        name=name,
        rule=rule,
        rule_esc=rule_esc,
        fred_id=fred_id,
        tickers=tickers,
        trade_tickers=trade,
    )
    fp = BACKTESTS_DIR / f"{sid}.py"
    fp.write_text(src)
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
        family = strat.get("_source_family", "")
        claimed += 1
        print(f"\n[{claimed}] CLAIM {strategy_id} {sid} [{family}]")

        try:
            fp = write_backtest_file(strat)
            if fp is None:
                update_strategy_status(
                    strategy_id, "failed",
                    backtest_result={"status": "fail", "reason": "no template / fred id"},
                    backtested_at=datetime.now(timezone.utc).isoformat(),
                )
                failed += 1
                print(f"  FAIL: no template / fred id")
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

            # Successful run -> mark done with metrics
            if res.get("status") in {"fail", "failed", "error"}:
                update_strategy_status(
                    strategy_id, "failed",
                    backtest_result={
                        "status": "fail",
                        "reason": res.get("reason", "unknown"),
                    },
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

            # Quick pre-screen: only refresh BH if Sharpe/CAGR thresholds plausibly met
            if (sharpe is not None and sharpe > 0.5
                    and cagr is not None and cagr > 0.10):
                # Make sure result has signal_id
                if not res.get("signal_id"):
                    res["signal_id"] = sid
                    with open(RESULTS_DIR / f"{sid}.json", "w") as f:
                        json.dump(res, f, indent=2, default=str)
                # Refresh BH and check
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

    # Save summary to stdout-readable file
    with open(ROOT / "_drain_summary.json", "w") as f:
        json.dump({"claimed": claimed, "done": done, "failed": failed, "winners": winners}, f, indent=2)


if __name__ == "__main__":
    main()
