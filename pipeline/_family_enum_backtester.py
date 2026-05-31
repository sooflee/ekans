"""Driver script: pull family-enum strategies from the queue, generate their
backtest .py files using a unified per-family template, run them, and update
the queue + winner gate.

This is the loop driver — it processes strategies until the queue is empty
(or until a non-family-enum strategy is at the head, which it skips by
claiming-and-re-marking-as-ready... actually no, it just processes whatever
it claims).
"""
import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path("/Users/benson/Projects/ekans")
sys.path.insert(0, str(ROOT / "pipeline"))
from queue_io import claim_ready_strategy, update_strategy_status, heartbeat
from winner_gate import is_winner, load_mt_data, winner_reasons


# ----------------------- Backtest file templates ------------------------

# Common header/footer for the generated backtest files.
HEADER = '''"""{sid} -- {name}
{rule}
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "{sid}"
'''


# PPI commodity (WPU): trough recovery -> long XLB+XLI+IYM for 42 days
PPI_BODY = '''
    try:
        fred = load_fred(["{fred_id}"], start="1990-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED load: {{e}}")

    if fred is None or fred.empty or "{fred_id}" not in fred.columns:
        return mark_failed(sid, "FRED data empty or missing column")

    series = fred["{fred_id}"].dropna()
    if len(series) < 24:
        return mark_failed(sid, "insufficient FRED data")

    # 3-month % change inflection: positive after 6+ months of decline
    mom3 = series.pct_change(3).dropna()
    trigger_dates = []
    decline_count = 0
    fired = False
    for i in range(len(mom3)):
        v = mom3.iloc[i]
        if v < 0:
            decline_count += 1
            fired = False
        elif decline_count >= 6 and v > 0 and not fired:
            trigger_dates.append(mom3.index[i])
            fired = True
            decline_count = 0
        else:
            if v >= 0:
                decline_count = 0

    if len(trigger_dates) == 0:
        return mark_failed(sid, "no trigger events found")

    try:
        px = load_prices(["XLB", "XLI", "IYM", "SPY"], start="1998-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {{e}}")

    ret = daily_returns(px)
    spy_r = ret["SPY"] if "SPY" in ret.columns else None
    basket_tickers = [t for t in ["XLB", "XLI", "IYM"] if t in ret.columns]
    if not basket_tickers:
        return mark_failed(sid, "no basket tickers available")
    basket_r = ret[basket_tickers].mean(axis=1)

    hold_days = 42
    events = []
    pnl_parts = []
    for td in trigger_dates:
        mask = basket_r.index >= td
        if mask.sum() < 30:
            continue
        entry = basket_r.index[mask][0]
        loc = basket_r.index.get_loc(entry)
        end = min(loc + hold_days, len(basket_r) - 1)
        if end - loc < 20:
            continue
        win = basket_r.iloc[loc:end]
        pnl_parts.append(win)
        cumret = float((1 + win).prod() - 1)
        spy_cum = None
        if spy_r is not None:
            spy_win = spy_r.iloc[loc:end]
            spy_cum = float((1 + spy_win).prod() - 1)
        events.append({{
            "trigger_date": str(td.date()),
            "basket_return": round(cumret, 4),
            "spy_return": round(spy_cum, 4) if spy_cum is not None else None,
        }})

    if not events:
        return mark_failed(sid, "no valid events after price alignment")

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep="first")]
    m = compute_metrics(all_pnl, benchmark=spy_r, name="{name}")
    save_result(sid, m, extra={{
        "rule": {rule_repr},
        "mechanism": "PPI {family} trough recovery -> upstream input firms benefit",
        "source": "FRED {fred_id} + yfinance",
        "n_events": len(events),
        "events": events,
    }})
    print(f"Done: {{len(events)}} events, Sharpe={{m.get('sharpe', 0):.2f}}, CAGR={{m.get('cagr', 0)*100:.1f}}%")


if __name__ == "__main__":
    main()
'''


# CPI subindex (CUSR0000): YoY trough after 2+ months deflation -> long XLY+XLP 42d
CPI_BODY = '''
    try:
        fred = load_fred(["{fred_id}"], start="1990-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED load: {{e}}")

    if fred is None or fred.empty or "{fred_id}" not in fred.columns:
        return mark_failed(sid, "FRED data empty or missing column")

    series = fred["{fred_id}"].dropna()
    if len(series) < 24:
        return mark_failed(sid, "insufficient FRED data")

    yoy = series.pct_change(12).dropna() * 100
    trigger_dates = []
    neg_count = 0
    fired = False
    for i in range(len(yoy)):
        v = yoy.iloc[i]
        if v < 0:
            neg_count += 1
            fired = False
        elif neg_count >= 2 and v >= 0 and not fired:
            trigger_dates.append(yoy.index[i])
            fired = True
            neg_count = 0
        else:
            if v >= 0:
                neg_count = 0

    if len(trigger_dates) == 0:
        return mark_failed(sid, "no trigger events found")

    try:
        px = load_prices(["XLY", "XLP", "SPY"], start="1998-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {{e}}")

    ret = daily_returns(px)
    spy_r = ret["SPY"] if "SPY" in ret.columns else None
    basket_tickers = [t for t in ["XLY", "XLP"] if t in ret.columns]
    if not basket_tickers:
        return mark_failed(sid, "no basket tickers available")
    basket_r = ret[basket_tickers].mean(axis=1)

    hold_days = 42
    events = []
    pnl_parts = []
    for td in trigger_dates:
        mask = basket_r.index >= td
        if mask.sum() < 30:
            continue
        entry = basket_r.index[mask][0]
        loc = basket_r.index.get_loc(entry)
        end = min(loc + hold_days, len(basket_r) - 1)
        if end - loc < 20:
            continue
        win = basket_r.iloc[loc:end]
        pnl_parts.append(win)
        cumret = float((1 + win).prod() - 1)
        spy_cum = None
        if spy_r is not None:
            spy_win = spy_r.iloc[loc:end]
            spy_cum = float((1 + spy_win).prod() - 1)
        events.append({{
            "trigger_date": str(td.date()),
            "basket_return": round(cumret, 4),
            "spy_return": round(spy_cum, 4) if spy_cum is not None else None,
        }})

    if not events:
        return mark_failed(sid, "no valid events after price alignment")

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep="first")]
    m = compute_metrics(all_pnl, benchmark=spy_r, name="{name}")
    save_result(sid, m, extra={{
        "rule": {rule_repr},
        "mechanism": "CPI {family} YoY trough -> consumer relief -> staples/discretionary benefit",
        "source": "FRED {fred_id} + yfinance",
        "n_events": len(events),
        "events": events,
    }})
    print(f"Done: {{len(events)}} events, Sharpe={{m.get('sharpe', 0):.2f}}, CAGR={{m.get('cagr', 0)*100:.1f}}%")


if __name__ == "__main__":
    main()
'''


# JOLTS by industry: YoY > +20% (single trigger after streak) -> long basket 60d
JOLTS_BODY = '''
    try:
        fred = load_fred(["{fred_id}"], start="2001-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED load: {{e}}")

    if fred is None or fred.empty or "{fred_id}" not in fred.columns:
        return mark_failed(sid, "FRED data empty or missing column")

    series = fred["{fred_id}"].dropna()
    if len(series) < 24:
        return mark_failed(sid, "insufficient FRED data")

    yoy = series.pct_change(12).dropna() * 100
    trigger_dates = []
    streak = 0
    fired = False
    for i in range(len(yoy)):
        if yoy.iloc[i] > 20:
            streak += 1
            if streak >= 2 and not fired:
                trigger_dates.append(yoy.index[i])
                fired = True
        else:
            streak = 0
            fired = False

    if len(trigger_dates) == 0:
        return mark_failed(sid, "no trigger events found")

    try:
        px = load_prices(["XLI", "XLY", "XLV", "XLF", "SPY"], start="2001-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {{e}}")

    ret = daily_returns(px)
    spy_r = ret["SPY"] if "SPY" in ret.columns else None
    basket_tickers = [t for t in ["XLI", "XLY", "XLV", "XLF"] if t in ret.columns]
    if not basket_tickers:
        return mark_failed(sid, "no basket tickers available")
    basket_r = ret[basket_tickers].mean(axis=1)

    hold_days = 60
    events = []
    pnl_parts = []
    for td in trigger_dates:
        mask = basket_r.index >= td
        if mask.sum() < 30:
            continue
        entry = basket_r.index[mask][0]
        loc = basket_r.index.get_loc(entry)
        end = min(loc + hold_days, len(basket_r) - 1)
        if end - loc < 30:
            continue
        win = basket_r.iloc[loc:end]
        pnl_parts.append(win)
        cumret = float((1 + win).prod() - 1)
        spy_cum = None
        if spy_r is not None:
            spy_win = spy_r.iloc[loc:end]
            spy_cum = float((1 + spy_win).prod() - 1)
        events.append({{
            "trigger_date": str(td.date()),
            "basket_return": round(cumret, 4),
            "spy_return": round(spy_cum, 4) if spy_cum is not None else None,
        }})

    if not events:
        return mark_failed(sid, "no valid events after price alignment")

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep="first")]
    m = compute_metrics(all_pnl, benchmark=spy_r, name="{name}")
    save_result(sid, m, extra={{
        "rule": {rule_repr},
        "mechanism": "JOLTS {family} hiring surge -> sector demand strength",
        "source": "FRED {fred_id} + yfinance",
        "n_events": len(events),
        "events": events,
    }})
    print(f"Done: {{len(events)}} events, Sharpe={{m.get('sharpe', 0):.2f}}, CAGR={{m.get('cagr', 0)*100:.1f}}%")


if __name__ == "__main__":
    main()
'''


# Manufacturers orders: >2 std dev surge -> long XLI+ITA+XLB 42d
MFG_BODY = '''
    try:
        fred = load_fred(["{fred_id}"], start="1990-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED load: {{e}}")

    if fred is None or fred.empty or "{fred_id}" not in fred.columns:
        return mark_failed(sid, "FRED data empty or missing column")

    series = fred["{fred_id}"].dropna()
    if len(series) < 24:
        return mark_failed(sid, "insufficient FRED data")

    roll_mean = series.rolling(12).mean()
    roll_std = series.rolling(12).std()
    z = (series - roll_mean) / roll_std
    z = z.dropna()

    trigger_dates = []
    fired = False
    for i in range(len(z)):
        if z.iloc[i] > 2 and not fired:
            trigger_dates.append(z.index[i])
            fired = True
        elif z.iloc[i] < 0.5:
            fired = False

    if len(trigger_dates) == 0:
        return mark_failed(sid, "no trigger events found")

    try:
        px = load_prices(["XLI", "ITA", "XLB", "SPY"], start="1998-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {{e}}")

    ret = daily_returns(px)
    spy_r = ret["SPY"] if "SPY" in ret.columns else None
    basket_tickers = [t for t in ["XLI", "ITA", "XLB"] if t in ret.columns]
    if not basket_tickers:
        return mark_failed(sid, "no basket tickers available")
    basket_r = ret[basket_tickers].mean(axis=1)

    hold_days = 42
    events = []
    pnl_parts = []
    for td in trigger_dates:
        mask = basket_r.index >= td
        if mask.sum() < 30:
            continue
        entry = basket_r.index[mask][0]
        loc = basket_r.index.get_loc(entry)
        end = min(loc + hold_days, len(basket_r) - 1)
        if end - loc < 20:
            continue
        win = basket_r.iloc[loc:end]
        pnl_parts.append(win)
        cumret = float((1 + win).prod() - 1)
        spy_cum = None
        if spy_r is not None:
            spy_win = spy_r.iloc[loc:end]
            spy_cum = float((1 + spy_win).prod() - 1)
        events.append({{
            "trigger_date": str(td.date()),
            "basket_return": round(cumret, 4),
            "spy_return": round(spy_cum, 4) if spy_cum is not None else None,
        }})

    if not events:
        return mark_failed(sid, "no valid events after price alignment")

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep="first")]
    m = compute_metrics(all_pnl, benchmark=spy_r, name="{name}")
    save_result(sid, m, extra={{
        "rule": {rule_repr},
        "mechanism": "Mfg orders {family} surge -> industrial cycle acceleration",
        "source": "FRED {fred_id} + yfinance",
        "n_events": len(events),
        "events": events,
    }})
    print(f"Done: {{len(events)}} events, Sharpe={{m.get('sharpe', 0):.2f}}, CAGR={{m.get('cagr', 0)*100:.1f}}%")


if __name__ == "__main__":
    main()
'''


# Commodity prices (monthly): 3-month return > +30% (spike), wait 10d, then long XLE+XLB+DBA 42d
COMM_BODY = '''
    try:
        fred = load_fred(["{fred_id}"], start="1990-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED load: {{e}}")

    if fred is None or fred.empty or "{fred_id}" not in fred.columns:
        return mark_failed(sid, "FRED data empty or missing column")

    series = fred["{fred_id}"].dropna()
    if len(series) < 12:
        return mark_failed(sid, "insufficient FRED data")

    m3 = series.pct_change(3).dropna() * 100
    trigger_dates = []
    fired = False
    for i in range(len(m3)):
        if m3.iloc[i] > 30 and not fired:
            trigger_dates.append(m3.index[i])
            fired = True
        elif m3.iloc[i] < 15:
            fired = False

    if len(trigger_dates) == 0:
        return mark_failed(sid, "no trigger events found")

    try:
        px = load_prices(["XLE", "XLB", "DBA", "SPY"], start="2007-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {{e}}")

    ret = daily_returns(px)
    spy_r = ret["SPY"] if "SPY" in ret.columns else None
    basket_tickers = [t for t in ["XLE", "XLB", "DBA"] if t in ret.columns]
    if not basket_tickers:
        return mark_failed(sid, "no basket tickers available")
    basket_r = ret[basket_tickers].mean(axis=1)

    hold_days = 42
    delay = 10
    events = []
    pnl_parts = []
    for td in trigger_dates:
        mask = basket_r.index >= td
        if mask.sum() < (delay + 30):
            continue
        entry = basket_r.index[mask][0]
        loc = basket_r.index.get_loc(entry) + delay
        if loc >= len(basket_r):
            continue
        end = min(loc + hold_days, len(basket_r) - 1)
        if end - loc < 20:
            continue
        win = basket_r.iloc[loc:end]
        pnl_parts.append(win)
        cumret = float((1 + win).prod() - 1)
        spy_cum = None
        if spy_r is not None:
            spy_win = spy_r.iloc[loc:end]
            spy_cum = float((1 + spy_win).prod() - 1)
        events.append({{
            "trigger_date": str(td.date()),
            "basket_return": round(cumret, 4),
            "spy_return": round(spy_cum, 4) if spy_cum is not None else None,
        }})

    if not events:
        return mark_failed(sid, "no valid events after price alignment")

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep="first")]
    m = compute_metrics(all_pnl, benchmark=spy_r, name="{name}")
    save_result(sid, m, extra={{
        "rule": {rule_repr},
        "mechanism": "{family} spike -> upstream producer/farm benefit (with lag)",
        "source": "FRED {fred_id} + yfinance",
        "n_events": len(events),
        "events": events,
    }})
    print(f"Done: {{len(events)}} events, Sharpe={{m.get('sharpe', 0):.2f}}, CAGR={{m.get('cagr', 0)*100:.1f}}%")


if __name__ == "__main__":
    main()
'''


# Bank credit/delinquency: local max after 2+ quarters rising -> long KRE+XLF+KBE 126d
BANK_BODY = '''
    try:
        fred = load_fred(["{fred_id}"], start="1990-01-01")
    except Exception as e:
        return mark_failed(sid, f"FRED load: {{e}}")

    if fred is None or fred.empty or "{fred_id}" not in fred.columns:
        return mark_failed(sid, "FRED data empty or missing column")

    series = fred["{fred_id}"].dropna()
    if len(series) < 24:
        return mark_failed(sid, "insufficient FRED data")

    trigger_dates = []
    # Local maxima with 6+ months of prior rising
    rising = 0
    fired = False
    for i in range(1, len(series) - 1):
        v_prev = series.iloc[i - 1]
        v = series.iloc[i]
        v_next = series.iloc[i + 1]
        if v > v_prev:
            rising += 1
        else:
            rising = 0
        # local peak: v > prev and v > next, AND prior rising streak >=6
        if v > v_prev and v >= v_next and rising >= 6 and not fired:
            trigger_dates.append(series.index[i])
            fired = True
        if v < v_prev * 0.95:  # reset after substantial decline
            fired = False

    if len(trigger_dates) == 0:
        return mark_failed(sid, "no trigger events found")

    try:
        px = load_prices(["KRE", "XLF", "KBE", "SPY"], start="2006-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {{e}}")

    ret = daily_returns(px)
    spy_r = ret["SPY"] if "SPY" in ret.columns else None
    basket_tickers = [t for t in ["KRE", "XLF", "KBE"] if t in ret.columns]
    if not basket_tickers:
        return mark_failed(sid, "no basket tickers available")
    basket_r = ret[basket_tickers].mean(axis=1)

    hold_days = 126
    events = []
    pnl_parts = []
    for td in trigger_dates:
        mask = basket_r.index >= td
        if mask.sum() < 60:
            continue
        entry = basket_r.index[mask][0]
        loc = basket_r.index.get_loc(entry)
        end = min(loc + hold_days, len(basket_r) - 1)
        if end - loc < 60:
            continue
        win = basket_r.iloc[loc:end]
        pnl_parts.append(win)
        cumret = float((1 + win).prod() - 1)
        spy_cum = None
        if spy_r is not None:
            spy_win = spy_r.iloc[loc:end]
            spy_cum = float((1 + spy_win).prod() - 1)
        events.append({{
            "trigger_date": str(td.date()),
            "basket_return": round(cumret, 4),
            "spy_return": round(spy_cum, 4) if spy_cum is not None else None,
        }})

    if not events:
        return mark_failed(sid, "no valid events after price alignment")

    all_pnl = pd.concat(pnl_parts)
    all_pnl = all_pnl[~all_pnl.index.duplicated(keep="first")]
    m = compute_metrics(all_pnl, benchmark=spy_r, name="{name}")
    save_result(sid, m, extra={{
        "rule": {rule_repr},
        "mechanism": "Credit cycle peak in {family} -> bank stress eases on inflection",
        "source": "FRED {fred_id} + yfinance",
        "n_events": len(events),
        "events": events,
    }})
    print(f"Done: {{len(events)}} events, Sharpe={{m.get('sharpe', 0):.2f}}, CAGR={{m.get('cagr', 0)*100:.1f}}%")


if __name__ == "__main__":
    main()
'''


FAMILY_BODY = {
    "PPI commodity (WPU)": PPI_BODY,
    "CPI subindex (CUSR0000)": CPI_BODY,
    "JOLTS by industry": JOLTS_BODY,
    "Manufacturers orders": MFG_BODY,
    "Commodity prices (monthly)": COMM_BODY,
    "Bank credit/delinquency": BANK_BODY,
}


def extract_fred_id(strat):
    """Pull FRED series id out of data_sources_concrete.fundamental."""
    fundamental = strat.get("data_sources_concrete", {}).get("fundamental", "")
    # Format: "FRED WPS011"
    parts = fundamental.split()
    if len(parts) >= 2 and parts[0] == "FRED":
        return parts[1]
    # fallback: parse from name "FRED WPS011 PPI: Farm Products -> Family Variant"
    name = strat.get("name", "")
    nparts = name.split()
    if len(nparts) >= 2 and nparts[0] == "FRED":
        return nparts[1]
    return None


def write_backtest_file(strat):
    sid = strat["signal_id"]
    family = strat.get("_source_family")
    body_tpl = FAMILY_BODY.get(family)
    if body_tpl is None:
        raise RuntimeError(f"Unknown family: {family}")
    fred_id = extract_fred_id(strat)
    if not fred_id:
        raise RuntimeError(f"Could not extract FRED id from {sid}")

    name = strat.get("name", sid)
    rule = strat.get("rule", "")
    header = HEADER.format(sid=sid, name=name, rule=rule)
    body = body_tpl.format(
        sid=sid,
        name=name,
        family=family,
        fred_id=fred_id,
        rule_repr=repr(rule),
    )
    code = header + body
    fp = ROOT / "backtests" / f"{sid}.py"
    fp.write_text(code)
    return fp


def process_one(strat):
    """Returns (status, winner_info_or_None, err_msg)."""
    sid = strat["signal_id"]
    strategy_id = strat["strategy_id"]

    # 1. Determine backtest file path. If it already exists (e.g. another
    # agent or a prior run wrote a custom backtest for a bespoke strategy),
    # don't overwrite — just run it. Otherwise build from family template.
    fp = ROOT / "backtests" / f"{sid}.py"
    if not fp.exists():
        try:
            fp = write_backtest_file(strat)
        except Exception as e:
            try:
                from harness import mark_failed
                mark_failed(sid, f"template error: {e}")
            except Exception:
                pass
            update_strategy_status(
                strategy_id,
                "failed",
                backtest_result={"status": "fail", "reason": f"template error: {e}"},
                backtested_at=datetime.now(timezone.utc).isoformat(),
            )
            return "failed", None, str(e)

    # 2. Run it
    try:
        proc = subprocess.run(
            [str(ROOT / ".venv" / "bin" / "python"), str(fp)],
            capture_output=True, text=True, timeout=180,
            cwd=str(ROOT),
        )
    except subprocess.TimeoutExpired:
        update_strategy_status(
            strategy_id,
            "failed",
            backtest_result={"status": "fail", "reason": "timeout"},
            backtested_at=datetime.now(timezone.utc).isoformat(),
        )
        return "failed", None, "timeout"
    except Exception as e:
        update_strategy_status(
            strategy_id,
            "failed",
            backtest_result={"status": "fail", "reason": f"exec error: {e}"},
            backtested_at=datetime.now(timezone.utc).isoformat(),
        )
        return "failed", None, str(e)

    # 3. Read result
    result_fp = ROOT / "results" / f"{sid}.json"
    if not result_fp.exists():
        update_strategy_status(
            strategy_id,
            "failed",
            backtest_result={"status": "fail", "reason": "no result file", "stderr": proc.stderr[-500:]},
            backtested_at=datetime.now(timezone.utc).isoformat(),
        )
        return "failed", None, "no result file"
    try:
        result = json.load(open(result_fp))
    except Exception as e:
        update_strategy_status(
            strategy_id,
            "failed",
            backtest_result={"status": "fail", "reason": f"result parse: {e}"},
            backtested_at=datetime.now(timezone.utc).isoformat(),
        )
        return "failed", None, str(e)

    if not result.get("signal_id"):
        result["signal_id"] = sid

    status = (result.get("status") or "").lower()
    if status in {"fail", "failed", "error"}:
        update_strategy_status(
            strategy_id,
            "failed",
            backtest_result={"status": "fail", "reason": result.get("reason", "backtest fail")},
            backtested_at=datetime.now(timezone.utc).isoformat(),
        )
        return "failed", None, result.get("reason", "backtest fail")

    # 4. Mark done with metrics
    update_strategy_status(
        strategy_id,
        "done",
        backtest_result={
            "status": "ok",
            "sharpe": result.get("sharpe"),
            "cagr": result.get("cagr"),
            "max_dd": result.get("max_dd"),
            "t_stat": result.get("t_stat"),
        },
        backtested_at=datetime.now(timezone.utc).isoformat(),
    )

    return "done", result, None


def main():
    claimed = 0
    done = 0
    failed = 0
    winners = []

    while True:
        strat = claim_ready_strategy()
        if strat is None:
            break
        claimed += 1
        sid = strat["signal_id"]
        family = strat.get("_source_family", "?")
        print(f"\n[{claimed}] Processing {sid} (family={family})")

        status, result, err = process_one(strat)
        if status == "failed":
            failed += 1
            print(f"  FAILED: {err}")
        else:
            done += 1
            sharpe = result.get("sharpe", 0) or 0
            cagr = result.get("cagr", 0) or 0
            print(f"  ok: Sharpe={sharpe:.2f}, CAGR={cagr*100:.1f}%")

            # Quick pre-check: don't refresh BH for obvious non-winners
            if (result.get("sharpe", 0) or 0) > 0.5 and (result.get("cagr", 0) or 0) > 0.10:
                # Run BH refresh
                subprocess.run(
                    [str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "pipeline" / "refresh_bh.py")],
                    check=True, capture_output=True, cwd=str(ROOT),
                )
                mt = load_mt_data()
                if is_winner(result, mt):
                    winners.append({
                        "sid": sid,
                        "sharpe": round(result.get("sharpe", 0), 3),
                        "cagr": round(result.get("cagr", 0), 4),
                        "oos_sharpe": round(result.get("oos_sharpe", 0) or 0, 3),
                        "n_events": result.get("n_events"),
                        "name": result.get("name", sid),
                    })
                    print(f"  *** WINNER FOUND: {sid} — Sharpe {result.get('sharpe', 0):.2f}, "
                          f"CAGR {result.get('cagr', 0)*100:.1f}%, OOS {result.get('oos_sharpe', 0):.2f}, BH-sig ***")
                else:
                    reasons = winner_reasons(result, mt)
                    failed_reasons = [k for k, v in reasons.items() if not v]
                    print(f"  not a winner: failed criteria = {failed_reasons}")

        heartbeat("backtester")

    print(f"\n=== Drain complete ===")
    print(f"Claimed: {claimed}")
    print(f"Done: {done}")
    print(f"Failed: {failed}")
    print(f"Winners: {len(winners)}")
    for w in winners:
        print(f"  {w}")

    # Write summary for caller
    summary = {
        "claimed": claimed,
        "done": done,
        "failed": failed,
        "winners": winners,
    }
    (ROOT / "pipeline" / "_last_backtester_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
