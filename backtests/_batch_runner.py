"""Batch runner for all 29 ready strategies PL73-PL101.
Creates backtest scripts on the fly, runs each, updates strategies_queue.json.
"""
import sys, json, datetime, subprocess, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUEUE_FILE = ROOT / "pipeline" / "strategies_queue.json"
RESULTS_DIR = ROOT / "results"
BACKTESTS_DIR = ROOT / "backtests"
PYTHON = str(ROOT / ".venv" / "bin" / "python")


def load_queue():
    with open(QUEUE_FILE) as f:
        return json.load(f)


def save_queue(q):
    with open(QUEUE_FILE, "w") as f:
        json.dump(q, f, indent=2, default=str, ensure_ascii=False)


def set_status(q, signal_id, status, result=None):
    for s in q:
        if s.get("signal_id") == signal_id:
            s["status"] = status
            s["backtested_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            if result:
                s["backtest_result"] = result
            break
    save_queue(q)


def read_result(signal_id):
    fp = RESULTS_DIR / f"{signal_id}.json"
    if fp.exists():
        with open(fp) as f:
            r = json.load(f)
        if r.get("status") == "fail":
            return {"status": "fail", "reason": r.get("reason", "unknown"),
                    "sharpe": None, "cagr": None, "max_dd": None, "t_stat": None}
        return {
            "status": "ok",
            "sharpe": r.get("sharpe"),
            "cagr": r.get("cagr"),
            "max_dd": r.get("max_dd"),
            "t_stat": r.get("t_stat"),
        }
    return {"status": "fail", "reason": "no result file", "sharpe": None, "cagr": None, "max_dd": None, "t_stat": None}


def run_backtest(signal_id):
    script = BACKTESTS_DIR / f"{signal_id}.py"
    if not script.exists():
        return False, f"script not found: {script}"
    try:
        result = subprocess.run(
            [PYTHON, str(script)],
            capture_output=True, text=True, timeout=120, cwd=str(ROOT)
        )
        print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout, end="")
        if result.stderr:
            # Filter out common warnings
            err_lines = [l for l in result.stderr.split("\n")
                         if l and "FutureWarning" not in l and "UserWarning" not in l
                         and "DeprecationWarning" not in l]
            if err_lines:
                print("  STDERR:", "\n  ".join(err_lines[-5:]))
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "timeout (120s)"
    except Exception as e:
        return False, str(e)


# ===== STRATEGY DEFINITIONS =====
# Each entry: (signal_id, script_content)

STRATEGIES = {}

# PL73 - already written
# PL74 - already written
# PL75 - already written

# PL76 - yield curve uninversion cyclicals
STRATEGIES["PL76_yield_curve_uninversion_cyclicals"] = '''"""PL76_yield_curve_uninversion_cyclicals — 10Y-2Y Un-Inversion After 12mo → Long XLI+XLF"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns

def main():
    sid = "PL76_yield_curve_uninversion_cyclicals"
    try:
        fred = load_fred("T10Y2Y", start="1976-01-01")
        spread = fred.squeeze()
    except Exception as e:
        return mark_failed(sid, f"FRED data load: {e}")
    if spread.empty:
        return mark_failed(sid, "T10Y2Y data empty")
    # Find dates where spread crosses above 0 after 12+ months below 0
    trigger_dates = []
    below_zero_days = 0
    for i in range(1, len(spread)):
        val = float(spread.iloc[i])
        prev = float(spread.iloc[i-1])
        if np.isnan(val) or np.isnan(prev):
            continue
        if val < 0:
            below_zero_days += 1
        elif val >= 0 and prev < 0 and below_zero_days >= 252:
            if len(trigger_dates) == 0 or (spread.index[i] - trigger_dates[-1]).days > 365:
                trigger_dates.append(spread.index[i])
            below_zero_days = 0
        elif val >= 0:
            below_zero_days = 0
    print(f"Yield curve un-inversion events: {len(trigger_dates)}")
    for d in trigger_dates:
        print(f"  {d.date()}")
    if not trigger_dates:
        return mark_failed(sid, "no un-inversion events found")
    try:
        px = load_prices(["XLI", "XLF", "SPY"], start="1998-01-01")
    except Exception as e:
        return mark_failed(sid, f"equity data load: {e}")
    ret = daily_returns(px)
    available = [t for t in ["XLI", "XLF"] if t in ret.columns]
    if not available:
        return mark_failed(sid, "no tickers")
    basket_ret = ret[available].mean(axis=1)
    spy_ret = ret["SPY"]
    hold_days = 252
    pnl_series = pd.Series(0.0, index=basket_ret.index)
    event_results = []
    for td in trigger_dates:
        entry_mask = basket_ret.index >= td
        if entry_mask.sum() < hold_days:
            continue
        entry_idx = basket_ret.index[entry_mask][0]
        pos = basket_ret.index.get_loc(entry_idx)
        end_pos = min(pos + hold_days, len(basket_ret))
        event_rets = basket_ret.iloc[pos:end_pos]
        cumret = float((1 + event_rets).prod() - 1)
        pnl_series.iloc[pos:end_pos] = event_rets.values[:end_pos - pos]
        spy_cumret = None
        if entry_idx in spy_ret.index:
            sp = spy_ret.index.get_loc(entry_idx)
            se = min(sp + hold_days, len(spy_ret))
            spy_cumret = float((1 + spy_ret.iloc[sp:se]).prod() - 1)
        event_results.append({"trigger_date": str(td.date()), "basket_12m_return": round(cumret, 4),
                              "spy_12m_return": round(spy_cumret, 4) if spy_cumret is not None else None})
    if not event_results:
        return mark_failed(sid, "no valid events after alignment")
    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) < 30:
        return mark_failed(sid, f"insufficient days ({len(in_pos)})")
    m = compute_metrics(in_pos, benchmark=spy_ret, name="Yield Curve Un-Inversion → Long XLI+XLF")
    rets_arr = [e["basket_12m_return"] for e in event_results]
    save_result(sid, m, extra={"rule": "Long XLI+XLF 252d when T10Y2Y crosses above 0 after 12+ months inverted",
        "mechanism": "Un-inversion signals recession troughing → cyclical recovery",
        "source": "FRED T10Y2Y; yfinance", "n_events": len(event_results),
        "avg_event_return": round(float(np.mean(rets_arr)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in rets_arr])), 4), "events": event_results})
    print(f"Done: {len(event_results)} events, avg return={np.mean(rets_arr)*100:.2f}%")

if __name__ == "__main__":
    main()
'''

# PL77 - ISM Services expansion XLY
STRATEGIES["PL77_ism_services_expansion_xly"] = '''"""PL77_ism_services_expansion_xly — ISM Services PMI Crosses 55 from Below 50 → Long XLY"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns

def main():
    sid = "PL77_ism_services_expansion_xly"
    try:
        fred = load_fred("NMFBAI", start="2008-01-01")
        ism = fred.squeeze()
    except Exception as e:
        return mark_failed(sid, f"FRED data load: {e}")
    if ism.empty or len(ism) < 10:
        return mark_failed(sid, "NMFBAI data insufficient")
    trigger_dates = []
    below50_count = 0
    fired = False
    for i in range(len(ism)):
        val = float(ism.iloc[i])
        if np.isnan(val): continue
        if val < 50:
            below50_count += 1
            fired = False
        elif val >= 55 and below50_count >= 3 and not fired:
            trigger_dates.append(ism.index[i])
            fired = True
            below50_count = 0
        elif val >= 50:
            pass  # between 50 and 55, keep counting
    print(f"ISM Services expansion events: {len(trigger_dates)}")
    for d in trigger_dates:
        print(f"  {d.date()}: ISM = {ism.loc[d]:.1f}")
    if not trigger_dates:
        return mark_failed(sid, "no ISM Services expansion events found")
    try:
        px = load_prices(["XLY", "SPY"], start="1998-01-01")
    except Exception as e:
        return mark_failed(sid, f"equity data load: {e}")
    ret = daily_returns(px)
    xly_ret = ret["XLY"]; spy_ret = ret["SPY"]
    hold_days = 126
    pnl_series = pd.Series(0.0, index=xly_ret.index)
    event_results = []
    for td in trigger_dates:
        entry_date = td + pd.offsets.MonthBegin(1)
        entry_mask = xly_ret.index >= entry_date
        if entry_mask.sum() < hold_days: continue
        entry_idx = xly_ret.index[entry_mask][0]
        pos = xly_ret.index.get_loc(entry_idx)
        end_pos = min(pos + hold_days, len(xly_ret))
        event_rets = xly_ret.iloc[pos:end_pos]
        cumret = float((1 + event_rets).prod() - 1)
        pnl_series.iloc[pos:end_pos] = event_rets.values[:end_pos - pos]
        spy_cumret = None
        if entry_idx in spy_ret.index:
            sp = spy_ret.index.get_loc(entry_idx); se = min(sp + hold_days, len(spy_ret))
            spy_cumret = float((1 + spy_ret.iloc[sp:se]).prod() - 1)
        event_results.append({"trigger_date": str(td.date()), "ism_value": round(float(ism.loc[td]), 1),
            "xly_6m_return": round(cumret, 4), "spy_6m_return": round(spy_cumret, 4) if spy_cumret is not None else None})
    if not event_results:
        return mark_failed(sid, "no valid events")
    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) < 30:
        return mark_failed(sid, f"insufficient days ({len(in_pos)})")
    m = compute_metrics(in_pos, benchmark=spy_ret, name="ISM Services Expansion → Long XLY")
    rets_arr = [e["xly_6m_return"] for e in event_results]
    save_result(sid, m, extra={"rule": "Long XLY 126d when NMFBAI crosses 55 after 3+ months below 50",
        "source": "FRED NMFBAI; yfinance", "n_events": len(event_results),
        "avg_event_return": round(float(np.mean(rets_arr)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in rets_arr])), 4), "events": event_results})
    print(f"Done: {len(event_results)} events, avg return={np.mean(rets_arr)*100:.2f}%")

if __name__ == "__main__":
    main()
'''

# PL78 - GDPNow consensus gap SPY
STRATEGIES["PL78_gdpnow_consensus_gap_spy"] = '''"""PL78_gdpnow_consensus_gap_spy — GDPNow > 3.0% → Long SPY 42d"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns

def main():
    sid = "PL78_gdpnow_consensus_gap_spy"
    try:
        fred = load_fred("GDPNOW", start="2011-01-01")
        gdp = fred.squeeze()
    except Exception as e:
        return mark_failed(sid, f"FRED data load: {e}")
    if gdp.empty or len(gdp) < 10:
        return mark_failed(sid, "GDPNOW data insufficient")
    trigger_dates = []
    prev_above = False
    for i in range(len(gdp)):
        val = float(gdp.iloc[i])
        if np.isnan(val): continue
        if val > 3.0 and not prev_above:
            if len(trigger_dates) == 0 or (gdp.index[i] - trigger_dates[-1]).days > 60:
                trigger_dates.append(gdp.index[i])
            prev_above = True
        elif val <= 3.0:
            prev_above = False
    print(f"GDPNow > 3.0% events: {len(trigger_dates)}")
    if not trigger_dates:
        return mark_failed(sid, "no GDPNow events found")
    try:
        px = load_prices("SPY", start="2011-01-01")
    except Exception as e:
        return mark_failed(sid, f"equity data load: {e}")
    ret = daily_returns(px)
    if isinstance(ret, pd.DataFrame): spy_ret = ret.iloc[:, 0]
    else: spy_ret = ret
    hold_days = 42
    pnl_series = pd.Series(0.0, index=spy_ret.index)
    event_results = []
    for td in trigger_dates:
        entry_mask = spy_ret.index >= td
        if entry_mask.sum() < hold_days: continue
        entry_idx = spy_ret.index[entry_mask][0]
        pos = spy_ret.index.get_loc(entry_idx)
        end_pos = min(pos + hold_days, len(spy_ret))
        event_rets = spy_ret.iloc[pos:end_pos]
        cumret = float((1 + event_rets).prod() - 1)
        pnl_series.iloc[pos:end_pos] = event_rets.values[:end_pos - pos]
        event_results.append({"trigger_date": str(td.date()), "gdpnow": round(float(gdp.loc[td]), 2),
            "spy_return": round(cumret, 4)})
    if not event_results:
        return mark_failed(sid, "no valid events")
    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) < 30:
        return mark_failed(sid, f"insufficient days ({len(in_pos)})")
    m = compute_metrics(in_pos, benchmark=spy_ret, name="GDPNow > 3% → Long SPY")
    rets_arr = [e["spy_return"] for e in event_results]
    save_result(sid, m, extra={"rule": "Long SPY 42d when GDPNOW > 3.0%",
        "source": "FRED GDPNOW; yfinance", "n_events": len(event_results),
        "avg_event_return": round(float(np.mean(rets_arr)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in rets_arr])), 4), "events": event_results})
    print(f"Done: {len(event_results)} events, avg return={np.mean(rets_arr)*100:.2f}%")

if __name__ == "__main__":
    main()
'''

# Helper to generate FRED-threshold-with-streak strategies
def make_fred_yoy_streak_script(sid, fred_series, fred_start, threshold, consecutive, hold_days,
                                tickers, basket_name, rule_desc, mechanism, above=True,
                                after_below=None, after_below_months=6):
    """Generate a standard FRED YoY streak → long basket script."""
    tickers_str = str(tickers)
    return f'''"""
{sid} backtest
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns

def main():
    sid = "{sid}"
    try:
        fred = load_fred("{fred_series}", start="{fred_start}")
        data = fred.squeeze()
    except Exception as e:
        return mark_failed(sid, f"FRED data load: {{e}}")
    if data.empty or len(data) < 24:
        return mark_failed(sid, "{fred_series} data insufficient")
    yoy = data.pct_change(12)
    yoy = yoy.dropna()
    trigger_dates = []
    streak = 0
    for i in range(len(yoy)):
        val = float(yoy.iloc[i])
        if np.isnan(val): continue
        if {"val > " + str(threshold) if above else "val < " + str(threshold)}:
            streak += 1
            if streak == {consecutive}:
                if len(trigger_dates) == 0 or (yoy.index[i] - trigger_dates[-1]).days > 180:
                    trigger_dates.append(yoy.index[i])
        else:
            streak = 0
    print(f"Events: {{len(trigger_dates)}}")
    if not trigger_dates:
        return mark_failed(sid, "no events found")
    try:
        px = load_prices({tickers_str} + ["SPY"], start="1998-01-01")
    except Exception as e:
        return mark_failed(sid, f"equity data load: {{e}}")
    ret = daily_returns(px)
    available = [t for t in {tickers_str} if t in ret.columns]
    if not available:
        return mark_failed(sid, "no tickers available")
    basket_ret = ret[available].mean(axis=1)
    spy_ret = ret["SPY"]
    pnl_series = pd.Series(0.0, index=basket_ret.index)
    event_results = []
    for td in trigger_dates:
        entry_date = td + pd.offsets.MonthBegin(1)
        entry_mask = basket_ret.index >= entry_date
        if entry_mask.sum() < {hold_days}: continue
        entry_idx = basket_ret.index[entry_mask][0]
        pos = basket_ret.index.get_loc(entry_idx)
        end_pos = min(pos + {hold_days}, len(basket_ret))
        event_rets = basket_ret.iloc[pos:end_pos]
        cumret = float((1 + event_rets).prod() - 1)
        pnl_series.iloc[pos:end_pos] = event_rets.values[:end_pos - pos]
        spy_cumret = None
        if entry_idx in spy_ret.index:
            sp = spy_ret.index.get_loc(entry_idx); se = min(sp + {hold_days}, len(spy_ret))
            spy_cumret = float((1 + spy_ret.iloc[sp:se]).prod() - 1)
        event_results.append({{"trigger_date": str(td.date()), "basket_return": round(cumret, 4),
            "spy_return": round(spy_cumret, 4) if spy_cumret is not None else None}})
    if not event_results:
        return mark_failed(sid, "no valid events after alignment")
    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) < 30:
        return mark_failed(sid, f"insufficient days ({{len(in_pos)}})")
    m = compute_metrics(in_pos, benchmark=spy_ret, name="{basket_name}")
    rets_arr = [e["basket_return"] for e in event_results]
    save_result(sid, m, extra={{"rule": "{rule_desc}", "mechanism": "{mechanism}",
        "source": "FRED {fred_series}; yfinance", "n_events": len(event_results),
        "avg_event_return": round(float(np.mean(rets_arr)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in rets_arr])), 4), "events": event_results}})
    print(f"Done: {{len(event_results)}} events, avg return={{np.mean(rets_arr)*100:.2f}}%")

if __name__ == "__main__":
    main()
'''

# PL79 - resi construction cement
STRATEGIES["PL79_resi_construction_cement"] = '''"""PL79_resi_construction_cement — Private Resi Construction 3mo Ann. > +15% → Long EXP+SUM"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns

def main():
    sid = "PL79_resi_construction_cement"
    try:
        fred = load_fred("PRRESCONS", start="2000-01-01")
        data = fred.squeeze()
    except Exception as e:
        return mark_failed(sid, f"FRED data load: {e}")
    if data.empty or len(data) < 6:
        return mark_failed(sid, "PRRESCONS data insufficient")
    # 3-month annualized growth = (P_t / P_{t-3})^4 - 1
    ann3m = (data / data.shift(3)) ** 4 - 1
    ann3m = ann3m.dropna()
    trigger_dates = []
    prev_above = False
    for i in range(len(ann3m)):
        val = float(ann3m.iloc[i])
        if np.isnan(val): continue
        if val > 0.15 and not prev_above:
            if len(trigger_dates) == 0 or (ann3m.index[i] - trigger_dates[-1]).days > 180:
                trigger_dates.append(ann3m.index[i])
            prev_above = True
        elif val <= 0.15:
            prev_above = False
    print(f"Resi construction surge events: {len(trigger_dates)}")
    if not trigger_dates:
        return mark_failed(sid, "no events found")
    try:
        px = load_prices(["EXP", "SUM", "SPY"], start="2000-01-01")
    except Exception as e:
        return mark_failed(sid, f"equity data load: {e}")
    ret = daily_returns(px)
    available = [t for t in ["EXP", "SUM"] if t in ret.columns]
    if not available:
        return mark_failed(sid, "no tickers")
    basket_ret = ret[available].mean(axis=1)
    spy_ret = ret["SPY"]
    hold_days = 126
    pnl_series = pd.Series(0.0, index=basket_ret.index)
    event_results = []
    for td in trigger_dates:
        entry_date = td + pd.offsets.MonthBegin(1)
        entry_mask = basket_ret.index >= entry_date
        if entry_mask.sum() < hold_days: continue
        entry_idx = basket_ret.index[entry_mask][0]
        pos = basket_ret.index.get_loc(entry_idx)
        end_pos = min(pos + hold_days, len(basket_ret))
        event_rets = basket_ret.iloc[pos:end_pos]
        cumret = float((1 + event_rets).prod() - 1)
        pnl_series.iloc[pos:end_pos] = event_rets.values[:end_pos - pos]
        spy_cumret = None
        if entry_idx in spy_ret.index:
            sp = spy_ret.index.get_loc(entry_idx); se = min(sp + hold_days, len(spy_ret))
            spy_cumret = float((1 + spy_ret.iloc[sp:se]).prod() - 1)
        event_results.append({"trigger_date": str(td.date()), "ann3m": round(float(ann3m.loc[td])*100, 1),
            "basket_return": round(cumret, 4), "spy_return": round(spy_cumret, 4) if spy_cumret is not None else None})
    if not event_results:
        return mark_failed(sid, "no valid events")
    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) < 30:
        return mark_failed(sid, f"insufficient days ({len(in_pos)})")
    m = compute_metrics(in_pos, benchmark=spy_ret, name="Resi Construction → Long EXP+SUM")
    rets_arr = [e["basket_return"] for e in event_results]
    save_result(sid, m, extra={"rule": "Long EXP+SUM 126d when PRRESCONS 3mo ann. > +15%",
        "source": "FRED PRRESCONS; yfinance", "n_events": len(event_results),
        "avg_event_return": round(float(np.mean(rets_arr)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in rets_arr])), 4), "events": event_results})
    print(f"Done: {len(event_results)} events, avg return={np.mean(rets_arr)*100:.2f}%")

if __name__ == "__main__":
    main()
'''

# PL80 - northeast permits building products
STRATEGIES["PL80_northeast_permits_building_products"] = '''"""PL80_northeast_permits_building_products — NE Permits >+20% YoY (National Flat) → Long JELD+AZEK+BLDR"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns

def main():
    sid = "PL80_northeast_permits_building_products"
    try:
        ne = load_fred("PERMITNE", start="2000-01-01").squeeze()
        nat = load_fred("PERMIT", start="2000-01-01").squeeze()
    except Exception as e:
        return mark_failed(sid, f"FRED data load: {e}")
    if ne.empty or nat.empty:
        return mark_failed(sid, "permit data insufficient")
    ne_yoy = ne.pct_change(12).dropna()
    nat_yoy = nat.pct_change(12).dropna()
    idx = ne_yoy.index.intersection(nat_yoy.index)
    trigger_dates = []
    for d in idx:
        ne_v = float(ne_yoy.loc[d]); nat_v = float(nat_yoy.loc[d])
        if np.isnan(ne_v) or np.isnan(nat_v): continue
        if ne_v > 0.20 and nat_v < 0.05:
            if len(trigger_dates) == 0 or (d - trigger_dates[-1]).days > 180:
                trigger_dates.append(d)
    print(f"NE permit outperformance events: {len(trigger_dates)}")
    if not trigger_dates:
        return mark_failed(sid, "no events found")
    try:
        px = load_prices(["JELD", "AZEK", "BLDR", "SPY"], start="2017-01-01")
    except Exception as e:
        return mark_failed(sid, f"equity data load: {e}")
    ret = daily_returns(px)
    available = [t for t in ["JELD", "AZEK", "BLDR"] if t in ret.columns]
    if not available:
        return mark_failed(sid, "no tickers")
    basket_ret = ret[available].mean(axis=1)
    spy_ret = ret["SPY"]
    hold_days = 126
    pnl_series = pd.Series(0.0, index=basket_ret.index)
    event_results = []
    for td in trigger_dates:
        entry_date = td + pd.offsets.MonthBegin(1)
        entry_mask = basket_ret.index >= entry_date
        if entry_mask.sum() < hold_days: continue
        entry_idx = basket_ret.index[entry_mask][0]
        pos = basket_ret.index.get_loc(entry_idx)
        end_pos = min(pos + hold_days, len(basket_ret))
        event_rets = basket_ret.iloc[pos:end_pos]
        cumret = float((1 + event_rets).prod() - 1)
        pnl_series.iloc[pos:end_pos] = event_rets.values[:end_pos - pos]
        spy_cumret = None
        if entry_idx in spy_ret.index:
            sp = spy_ret.index.get_loc(entry_idx); se = min(sp + hold_days, len(spy_ret))
            spy_cumret = float((1 + spy_ret.iloc[sp:se]).prod() - 1)
        event_results.append({"trigger_date": str(td.date()), "basket_return": round(cumret, 4),
            "spy_return": round(spy_cumret, 4) if spy_cumret is not None else None})
    if not event_results:
        return mark_failed(sid, "no valid events after alignment (tickers too recent)")
    in_pos = pnl_series[pnl_series != 0]
    if len(in_pos) < 30:
        return mark_failed(sid, f"insufficient days ({len(in_pos)})")
    m = compute_metrics(in_pos, benchmark=spy_ret, name="NE Permits → Long Building Products")
    rets_arr = [e["basket_return"] for e in event_results]
    save_result(sid, m, extra={"rule": "Long JELD+AZEK+BLDR 126d when PERMITNE YoY>20% and PERMIT YoY<5%",
        "source": "FRED PERMITNE, PERMIT; yfinance", "n_events": len(event_results),
        "avg_event_return": round(float(np.mean(rets_arr)), 4),
        "event_win_rate": round(float(np.mean([r > 0 for r in rets_arr])), 4), "events": event_results})
    print(f"Done: {len(event_results)} events, avg return={np.mean(rets_arr)*100:.2f}%")

if __name__ == "__main__":
    main()
'''

# Generate remaining strategies using inline scripts
# I'll define the remaining as inline strings and write them all at once

remaining_scripts = {
"PL81_jolts_quits_staffing": '''"""PL81 — JOLTS Quits Rate Rise >+0.3pp from Trough → Long RHI"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns
def main():
    sid = "PL81_jolts_quits_staffing"
    try:
        fred = load_fred("JTSQUR", start="2000-01-01"); data = fred.squeeze()
    except Exception as e: return mark_failed(sid, f"FRED: {e}")
    if data.empty: return mark_failed(sid, "no data")
    low6 = data.rolling(6).min(); diff = data - low6; diff = diff.dropna()
    triggers = []
    for i in range(len(diff)):
        if diff.iloc[i] > 0.3:
            if not triggers or (diff.index[i] - triggers[-1]).days > 180: triggers.append(diff.index[i])
    print(f"Events: {len(triggers)}")
    if not triggers: return mark_failed(sid, "no events")
    try: px = load_prices(["RHI", "SPY"], start="2000-01-01")
    except Exception as e: return mark_failed(sid, f"price: {e}")
    ret = daily_returns(px); rhi_r = ret["RHI"]; spy_r = ret["SPY"]; hold = 126
    pnl = pd.Series(0.0, index=rhi_r.index); evts = []
    for td in triggers:
        ed = td + pd.offsets.MonthBegin(1); mask = rhi_r.index >= ed
        if mask.sum() < hold: continue
        ei = rhi_r.index[mask][0]; p = rhi_r.index.get_loc(ei); ep = min(p+hold, len(rhi_r))
        er = rhi_r.iloc[p:ep]; cr = float((1+er).prod()-1); pnl.iloc[p:ep] = er.values[:ep-p]
        sc = None
        if ei in spy_r.index: sp=spy_r.index.get_loc(ei); sc=float((1+spy_r.iloc[sp:min(sp+hold,len(spy_r))]).prod()-1)
        evts.append({"trigger_date":str(td.date()),"rhi_return":round(cr,4),"spy_return":round(sc,4) if sc else None})
    if not evts: return mark_failed(sid, "no valid events")
    ip = pnl[pnl!=0]
    if len(ip)<30: return mark_failed(sid, f"insufficient days ({len(ip)})")
    m = compute_metrics(ip, benchmark=spy_r, name="JOLTS Quits → Long RHI")
    ra = [e["rhi_return"] for e in evts]
    save_result(sid, m, extra={"rule":"Long RHI 126d when JTSQUR > 6mo low + 0.3pp","source":"FRED JTSQUR; yfinance",
        "n_events":len(evts),"avg_event_return":round(float(np.mean(ra)),4),
        "event_win_rate":round(float(np.mean([r>0 for r in ra])),4),"events":evts})
    print(f"Done: {len(evts)} events")
if __name__=="__main__": main()
''',

"PL82_mfg_weekly_hours_cyclicals": '''"""PL82 — Avg Weekly Hours Mfg Crosses 41.0 from Below 40.5 → Long XLI"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns
def main():
    sid = "PL82_mfg_weekly_hours_cyclicals"
    try: fred = load_fred("AWHMAN", start="1990-01-01"); data = fred.squeeze()
    except Exception as e: return mark_failed(sid, f"FRED: {e}")
    if data.empty: return mark_failed(sid, "no data")
    triggers = []; below_count = 0; fired = False
    for i in range(len(data)):
        v = float(data.iloc[i])
        if np.isnan(v): continue
        if v < 40.5: below_count += 1; fired = False
        elif v >= 41.0 and below_count >= 4 and not fired:
            triggers.append(data.index[i]); fired = True; below_count = 0
        elif v >= 40.5: pass
    print(f"Events: {len(triggers)}")
    if not triggers: return mark_failed(sid, "no events")
    try: px = load_prices(["XLI","SPY"], start="1998-01-01")
    except Exception as e: return mark_failed(sid, f"price: {e}")
    ret = daily_returns(px); xli_r = ret["XLI"]; spy_r = ret["SPY"]; hold = 126
    pnl = pd.Series(0.0, index=xli_r.index); evts = []
    for td in triggers:
        ed = td + pd.offsets.MonthBegin(1); mask = xli_r.index >= ed
        if mask.sum() < hold: continue
        ei = xli_r.index[mask][0]; p = xli_r.index.get_loc(ei); ep = min(p+hold, len(xli_r))
        er = xli_r.iloc[p:ep]; cr = float((1+er).prod()-1); pnl.iloc[p:ep] = er.values[:ep-p]
        sc = None
        if ei in spy_r.index: sp=spy_r.index.get_loc(ei); sc=float((1+spy_r.iloc[sp:min(sp+hold,len(spy_r))]).prod()-1)
        evts.append({"trigger_date":str(td.date()),"hours":round(float(data.loc[td]),1),"xli_return":round(cr,4),"spy_return":round(sc,4) if sc else None})
    if not evts: return mark_failed(sid, "no valid events")
    ip = pnl[pnl!=0]
    if len(ip)<30: return mark_failed(sid, f"insufficient days ({len(ip)})")
    m = compute_metrics(ip, benchmark=spy_r, name="Mfg Hours → Long XLI")
    ra = [e["xli_return"] for e in evts]
    save_result(sid, m, extra={"rule":"Long XLI 126d when AWHMAN crosses 41.0 after 4+ months below 40.5",
        "source":"FRED AWHMAN; yfinance","n_events":len(evts),"avg_event_return":round(float(np.mean(ra)),4),
        "event_win_rate":round(float(np.mean([r>0 for r in ra])),4),"events":evts})
    print(f"Done: {len(evts)} events")
if __name__=="__main__": main()
''',

"PL83_continued_claims_decline_iwm": '''"""PL83 — Continued Claims Decline 8+ Weeks from Peak → Long IWM"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns
def main():
    sid = "PL83_continued_claims_decline_iwm"
    try: fred = load_fred("CCSA", start="1990-01-01"); data = fred.squeeze()
    except Exception as e: return mark_failed(sid, f"FRED: {e}")
    if data.empty: return mark_failed(sid, "no data")
    peak26 = data.rolling(26).max()
    triggers = []; consec_decline = 0; near_peak_start = False
    for i in range(1, len(data)):
        v = float(data.iloc[i]); prev = float(data.iloc[i-1]); pk = float(peak26.iloc[i])
        if np.isnan(v) or np.isnan(prev) or np.isnan(pk): continue
        if prev >= pk * 0.95: near_peak_start = True
        if near_peak_start and v < prev: consec_decline += 1
        else: consec_decline = 0; near_peak_start = False if v >= pk * 0.95 else near_peak_start
        if consec_decline >= 8:
            if not triggers or (data.index[i] - triggers[-1]).days > 365:
                triggers.append(data.index[i])
            consec_decline = 0; near_peak_start = False
    print(f"Events: {len(triggers)}")
    if not triggers: return mark_failed(sid, "no events")
    try: px = load_prices(["IWM","SPY"], start="2000-01-01")
    except Exception as e: return mark_failed(sid, f"price: {e}")
    ret = daily_returns(px); iwm_r = ret["IWM"]; spy_r = ret["SPY"]; hold = 126
    pnl = pd.Series(0.0, index=iwm_r.index); evts = []
    for td in triggers:
        mask = iwm_r.index >= td
        if mask.sum() < hold: continue
        ei = iwm_r.index[mask][0]; p = iwm_r.index.get_loc(ei); ep = min(p+hold, len(iwm_r))
        er = iwm_r.iloc[p:ep]; cr = float((1+er).prod()-1); pnl.iloc[p:ep] = er.values[:ep-p]
        sc = None
        if ei in spy_r.index: sp=spy_r.index.get_loc(ei); sc=float((1+spy_r.iloc[sp:min(sp+hold,len(spy_r))]).prod()-1)
        evts.append({"trigger_date":str(td.date()),"iwm_return":round(cr,4),"spy_return":round(sc,4) if sc else None})
    if not evts: return mark_failed(sid, "no valid events")
    ip = pnl[pnl!=0]
    if len(ip)<30: return mark_failed(sid, f"insufficient days ({len(ip)})")
    m = compute_metrics(ip, benchmark=spy_r, name="Continued Claims Decline → Long IWM")
    ra = [e["iwm_return"] for e in evts]
    save_result(sid, m, extra={"rule":"Long IWM 126d when CCSA declines 8+ weeks from near-peak",
        "source":"FRED CCSA; yfinance","n_events":len(evts),"avg_event_return":round(float(np.mean(ra)),4),
        "event_win_rate":round(float(np.mean([r>0 for r in ra])),4),"events":evts})
    print(f"Done: {len(evts)} events")
if __name__=="__main__": main()
''',

"PL84_ted_spread_normalization_kre": '''"""PL84 — TED Spread Contracts Below 30bps After 50bps+ → Long KRE"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns
def main():
    sid = "PL84_ted_spread_normalization_kre"
    try: fred = load_fred("TEDRATE", start="1986-01-01"); data = fred.squeeze()
    except Exception as e: return mark_failed(sid, f"FRED: {e}")
    if data.empty: return mark_failed(sid, "no data")
    triggers = []; above50_days = 0
    for i in range(1, len(data)):
        v = float(data.iloc[i]); prev = float(data.iloc[i-1])
        if np.isnan(v) or np.isnan(prev): continue
        if v > 0.50: above50_days += 1
        elif v <= 0.30 and prev > 0.30 and above50_days >= 63:
            if not triggers or (data.index[i] - triggers[-1]).days > 365:
                triggers.append(data.index[i])
            above50_days = 0
        elif v <= 0.50: above50_days = 0
    print(f"Events: {len(triggers)}")
    if not triggers: return mark_failed(sid, "no events")
    try: px = load_prices(["KRE","SPY"], start="2006-01-01")
    except Exception as e: return mark_failed(sid, f"price: {e}")
    ret = daily_returns(px); kre_r = ret["KRE"]; spy_r = ret["SPY"]; hold = 126
    pnl = pd.Series(0.0, index=kre_r.index); evts = []
    for td in triggers:
        mask = kre_r.index >= td
        if mask.sum() < hold: continue
        ei = kre_r.index[mask][0]; p = kre_r.index.get_loc(ei); ep = min(p+hold, len(kre_r))
        er = kre_r.iloc[p:ep]; cr = float((1+er).prod()-1); pnl.iloc[p:ep] = er.values[:ep-p]
        sc = None
        if ei in spy_r.index: sp=spy_r.index.get_loc(ei); sc=float((1+spy_r.iloc[sp:min(sp+hold,len(spy_r))]).prod()-1)
        evts.append({"trigger_date":str(td.date()),"kre_return":round(cr,4),"spy_return":round(sc,4) if sc else None})
    if not evts: return mark_failed(sid, "no valid events")
    ip = pnl[pnl!=0]
    if len(ip)<30: return mark_failed(sid, f"insufficient days ({len(ip)})")
    m = compute_metrics(ip, benchmark=spy_r, name="TED Spread Norm → Long KRE")
    ra = [e["kre_return"] for e in evts]
    save_result(sid, m, extra={"rule":"Long KRE 126d when TEDRATE<0.30 after 3+ months >0.50",
        "source":"FRED TEDRATE; yfinance","n_events":len(evts),"avg_event_return":round(float(np.mean(ra)),4),
        "event_win_rate":round(float(np.mean([r>0 for r in ra])),4),"events":evts})
    print(f"Done: {len(evts)} events")
if __name__=="__main__": main()
''',

"PL85_excess_reserves_surge_spy": '''"""PL85 — Reserve Balances Surge > +20% in 3mo → Long SPY"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns
def main():
    sid = "PL85_excess_reserves_surge_spy"
    try: fred = load_fred("WRESBAL", start="2002-01-01"); data = fred.squeeze()
    except Exception as e: return mark_failed(sid, f"FRED: {e}")
    if data.empty: return mark_failed(sid, "no data")
    chg13 = data.pct_change(13).dropna()
    triggers = []; prev_above = False
    for i in range(len(chg13)):
        v = float(chg13.iloc[i])
        if np.isnan(v): continue
        if v > 0.20 and not prev_above:
            if not triggers or (chg13.index[i] - triggers[-1]).days > 180: triggers.append(chg13.index[i])
            prev_above = True
        elif v <= 0.20: prev_above = False
    print(f"Events: {len(triggers)}")
    if not triggers: return mark_failed(sid, "no events")
    try: px = load_prices("SPY", start="2002-01-01")
    except Exception as e: return mark_failed(sid, f"price: {e}")
    ret = daily_returns(px)
    if isinstance(ret, pd.DataFrame): spy_r = ret.iloc[:,0]
    else: spy_r = ret
    hold = 126; pnl = pd.Series(0.0, index=spy_r.index); evts = []
    for td in triggers:
        mask = spy_r.index >= td
        if mask.sum() < hold: continue
        ei = spy_r.index[mask][0]; p = spy_r.index.get_loc(ei); ep = min(p+hold, len(spy_r))
        er = spy_r.iloc[p:ep]; cr = float((1+er).prod()-1); pnl.iloc[p:ep] = er.values[:ep-p]
        evts.append({"trigger_date":str(td.date()),"spy_return":round(cr,4)})
    if not evts: return mark_failed(sid, "no valid events")
    ip = pnl[pnl!=0]
    if len(ip)<30: return mark_failed(sid, f"insufficient days ({len(ip)})")
    m = compute_metrics(ip, benchmark=spy_r, name="Reserves Surge → Long SPY")
    ra = [e["spy_return"] for e in evts]
    save_result(sid, m, extra={"rule":"Long SPY 126d when WRESBAL 13-week change > +20%",
        "source":"FRED WRESBAL; yfinance","n_events":len(evts),"avg_event_return":round(float(np.mean(ra)),4),
        "event_win_rate":round(float(np.mean([r>0 for r in ra])),4),"events":evts})
    print(f"Done: {len(evts)} events")
if __name__=="__main__": main()
''',

"PL86_ci_loan_growth_xlf": '''"""PL86 — C&I Loan Growth YoY Positive After 6mo Decline → Long XLF"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns
def main():
    sid = "PL86_ci_loan_growth_xlf"
    try: fred = load_fred("BUSLOANS", start="1990-01-01"); data = fred.squeeze()
    except Exception as e: return mark_failed(sid, f"FRED: {e}")
    if data.empty: return mark_failed(sid, "no data")
    monthly = data.resample("M").last().dropna()
    yoy = monthly.pct_change(12).dropna()
    triggers = []; neg_count = 0; fired = False
    for i in range(len(yoy)):
        v = float(yoy.iloc[i])
        if np.isnan(v): continue
        if v < 0: neg_count += 1; fired = False
        elif v >= 0 and neg_count >= 6 and not fired:
            triggers.append(yoy.index[i]); fired = True; neg_count = 0
        else: neg_count = 0
    print(f"Events: {len(triggers)}")
    if not triggers: return mark_failed(sid, "no events")
    try: px = load_prices(["XLF","SPY"], start="1998-01-01")
    except Exception as e: return mark_failed(sid, f"price: {e}")
    ret = daily_returns(px); xlf_r = ret["XLF"]; spy_r = ret["SPY"]; hold = 252
    pnl = pd.Series(0.0, index=xlf_r.index); evts = []
    for td in triggers:
        ed = td + pd.offsets.MonthBegin(1); mask = xlf_r.index >= ed
        if mask.sum() < hold: continue
        ei = xlf_r.index[mask][0]; p = xlf_r.index.get_loc(ei); ep = min(p+hold, len(xlf_r))
        er = xlf_r.iloc[p:ep]; cr = float((1+er).prod()-1); pnl.iloc[p:ep] = er.values[:ep-p]
        sc = None
        if ei in spy_r.index: sp=spy_r.index.get_loc(ei); sc=float((1+spy_r.iloc[sp:min(sp+hold,len(spy_r))]).prod()-1)
        evts.append({"trigger_date":str(td.date()),"xlf_return":round(cr,4),"spy_return":round(sc,4) if sc else None})
    if not evts: return mark_failed(sid, "no valid events")
    ip = pnl[pnl!=0]
    if len(ip)<30: return mark_failed(sid, f"insufficient days ({len(ip)})")
    m = compute_metrics(ip, benchmark=spy_r, name="C&I Loan Growth → Long XLF")
    ra = [e["xlf_return"] for e in evts]
    save_result(sid, m, extra={"rule":"Long XLF 252d when BUSLOANS YoY turns positive after 6+ months negative",
        "source":"FRED BUSLOANS; yfinance","n_events":len(evts),"avg_event_return":round(float(np.mean(ra)),4),
        "event_win_rate":round(float(np.mean([r>0 for r in ra])),4),"events":evts})
    print(f"Done: {len(evts)} events")
if __name__=="__main__": main()
''',

"PL87_twd_decline_industrial_exporters": '''"""PL87 — Trade-Weighted Dollar Declines >5% from 6mo Peak → Long HON+CAT+DE"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns
def main():
    sid = "PL87_twd_decline_industrial_exporters"
    try: fred = load_fred("DTWEXBGS", start="2006-01-01"); data = fred.squeeze()
    except Exception as e: return mark_failed(sid, f"FRED: {e}")
    if data.empty: return mark_failed(sid, "no data")
    peak126 = data.rolling(126).max().dropna()
    drawdown = data / peak126 - 1
    triggers = []; prev = False
    for i in range(len(drawdown)):
        v = float(drawdown.iloc[i])
        if np.isnan(v): continue
        if v < -0.05 and not prev:
            if not triggers or (drawdown.index[i] - triggers[-1]).days > 180: triggers.append(drawdown.index[i])
            prev = True
        elif v >= -0.03: prev = False
    print(f"Events: {len(triggers)}")
    if not triggers: return mark_failed(sid, "no events")
    try: px = load_prices(["HON","CAT","DE","SPY"], start="2006-01-01")
    except Exception as e: return mark_failed(sid, f"price: {e}")
    ret = daily_returns(px)
    available = [t for t in ["HON","CAT","DE"] if t in ret.columns]
    if not available: return mark_failed(sid, "no tickers")
    basket_r = ret[available].mean(axis=1); spy_r = ret["SPY"]; hold = 126
    pnl = pd.Series(0.0, index=basket_r.index); evts = []
    for td in triggers:
        mask = basket_r.index >= td
        if mask.sum() < hold: continue
        ei = basket_r.index[mask][0]; p = basket_r.index.get_loc(ei); ep = min(p+hold, len(basket_r))
        er = basket_r.iloc[p:ep]; cr = float((1+er).prod()-1); pnl.iloc[p:ep] = er.values[:ep-p]
        sc = None
        if ei in spy_r.index: sp=spy_r.index.get_loc(ei); sc=float((1+spy_r.iloc[sp:min(sp+hold,len(spy_r))]).prod()-1)
        evts.append({"trigger_date":str(td.date()),"basket_return":round(cr,4),"spy_return":round(sc,4) if sc else None})
    if not evts: return mark_failed(sid, "no valid events")
    ip = pnl[pnl!=0]
    if len(ip)<30: return mark_failed(sid, f"insufficient days ({len(ip)})")
    m = compute_metrics(ip, benchmark=spy_r, name="TWD Decline → Long Exporters")
    ra = [e["basket_return"] for e in evts]
    save_result(sid, m, extra={"rule":"Long HON+CAT+DE 126d when DTWEXBGS drops >5% from 6mo peak",
        "source":"FRED DTWEXBGS; yfinance","n_events":len(evts),"avg_event_return":round(float(np.mean(ra)),4),
        "event_win_rate":round(float(np.mean([r>0 for r in ra])),4),"events":evts})
    print(f"Done: {len(evts)} events")
if __name__=="__main__": main()
''',

"PL88_trade_deficit_widening_logistics": '''"""PL88 — Goods Trade Deficit Widens Beyond -$80B/mo → Long EXPD+CHRW+MATX"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns
def main():
    sid = "PL88_trade_deficit_widening_logistics"
    try: fred = load_fred("BOPGSTB", start="1992-01-01"); data = fred.squeeze()
    except Exception as e: return mark_failed(sid, f"FRED: {e}")
    if data.empty: return mark_failed(sid, "no data")
    # BOPGSTB is in millions; threshold -80000 (=$80B)
    unit = data.min()
    thresh = -80000 if unit < -10000 else (-80 if unit < -100 else -80)
    triggers = []; prev = False
    for i in range(len(data)):
        v = float(data.iloc[i])
        if np.isnan(v): continue
        if v < thresh and not prev:
            if not triggers or (data.index[i] - triggers[-1]).days > 180: triggers.append(data.index[i])
            prev = True
        elif v >= thresh * 0.9: prev = False
    print(f"Events: {len(triggers)} (threshold={thresh})")
    if not triggers: return mark_failed(sid, "no events found (deficit never exceeded $80B)")
    try: px = load_prices(["EXPD","CHRW","MATX","SPY"], start="1993-01-01")
    except Exception as e: return mark_failed(sid, f"price: {e}")
    ret = daily_returns(px)
    available = [t for t in ["EXPD","CHRW","MATX"] if t in ret.columns]
    if not available: return mark_failed(sid, "no tickers")
    basket_r = ret[available].mean(axis=1); spy_r = ret["SPY"]; hold = 126
    pnl = pd.Series(0.0, index=basket_r.index); evts = []
    for td in triggers:
        ed = td + pd.offsets.MonthBegin(1); mask = basket_r.index >= ed
        if mask.sum() < hold: continue
        ei = basket_r.index[mask][0]; p = basket_r.index.get_loc(ei); ep = min(p+hold, len(basket_r))
        er = basket_r.iloc[p:ep]; cr = float((1+er).prod()-1); pnl.iloc[p:ep] = er.values[:ep-p]
        sc = None
        if ei in spy_r.index: sp=spy_r.index.get_loc(ei); sc=float((1+spy_r.iloc[sp:min(sp+hold,len(spy_r))]).prod()-1)
        evts.append({"trigger_date":str(td.date()),"basket_return":round(cr,4),"spy_return":round(sc,4) if sc else None})
    if not evts: return mark_failed(sid, "no valid events")
    ip = pnl[pnl!=0]
    if len(ip)<30: return mark_failed(sid, f"insufficient days ({len(ip)})")
    m = compute_metrics(ip, benchmark=spy_r, name="Trade Deficit → Long Logistics")
    ra = [e["basket_return"] for e in evts]
    save_result(sid, m, extra={"rule":"Long EXPD+CHRW+MATX 126d when BOPGSTB < -$80B",
        "source":"FRED BOPGSTB; yfinance","n_events":len(evts),"avg_event_return":round(float(np.mean(ra)),4),
        "event_win_rate":round(float(np.mean([r>0 for r in ra])),4),"events":evts})
    print(f"Done: {len(evts)} events")
if __name__=="__main__": main()
''',

"PL89_wti_backwardation_xop": '''"""PL89 — WTI Backwardation (spot > 126d MA + $3) → Long XOP 63d"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns
def main():
    sid = "PL89_wti_backwardation_xop"
    try: fred = load_fred("DCOILWTICO", start="2005-01-01"); wti = fred.squeeze()
    except Exception as e: return mark_failed(sid, f"FRED: {e}")
    if wti.empty: return mark_failed(sid, "no data")
    ma126 = wti.rolling(126).mean().dropna()
    spread = wti - ma126; spread = spread.dropna()
    triggers = []; prev = False
    for i in range(len(spread)):
        v = float(spread.iloc[i])
        if np.isnan(v): continue
        if v > 3.0 and not prev:
            if not triggers or (spread.index[i] - triggers[-1]).days > 90: triggers.append(spread.index[i])
            prev = True
        elif v <= 1.0: prev = False
    print(f"Events: {len(triggers)}")
    if not triggers: return mark_failed(sid, "no events")
    try: px = load_prices(["XOP","SPY"], start="2006-01-01")
    except Exception as e: return mark_failed(sid, f"price: {e}")
    ret = daily_returns(px); xop_r = ret["XOP"]; spy_r = ret["SPY"]; hold = 63
    pnl = pd.Series(0.0, index=xop_r.index); evts = []
    for td in triggers:
        mask = xop_r.index >= td
        if mask.sum() < hold: continue
        ei = xop_r.index[mask][0]; p = xop_r.index.get_loc(ei); ep = min(p+hold, len(xop_r))
        er = xop_r.iloc[p:ep]; cr = float((1+er).prod()-1); pnl.iloc[p:ep] = er.values[:ep-p]
        sc = None
        if ei in spy_r.index: sp=spy_r.index.get_loc(ei); sc=float((1+spy_r.iloc[sp:min(sp+hold,len(spy_r))]).prod()-1)
        evts.append({"trigger_date":str(td.date()),"wti_spread":round(float(spread.loc[td]),2),
            "xop_return":round(cr,4),"spy_return":round(sc,4) if sc else None})
    if not evts: return mark_failed(sid, "no valid events")
    ip = pnl[pnl!=0]
    if len(ip)<30: return mark_failed(sid, f"insufficient days ({len(ip)})")
    m = compute_metrics(ip, benchmark=spy_r, name="WTI Backwardation → Long XOP")
    ra = [e["xop_return"] for e in evts]
    save_result(sid, m, extra={"rule":"Long XOP 63d when WTI > 126d MA + $3",
        "source":"FRED DCOILWTICO; yfinance","n_events":len(evts),"avg_event_return":round(float(np.mean(ra)),4),
        "event_win_rate":round(float(np.mean([r>0 for r in ra])),4),"events":evts})
    print(f"Done: {len(evts)} events")
if __name__=="__main__": main()
''',

"PL90_henry_hub_below_cash_cost_producers": '''"""PL90 — Henry Hub Below $2.50 → Long EQT+AR+RRC 12mo"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns
def main():
    sid = "PL90_henry_hub_below_cash_cost_producers"
    try: fred = load_fred("MHHNGSP", start="1997-01-01"); data = fred.squeeze()
    except Exception as e: return mark_failed(sid, f"FRED: {e}")
    if data.empty: return mark_failed(sid, "no data")
    triggers = []; prev = False
    for i in range(len(data)):
        v = float(data.iloc[i])
        if np.isnan(v): continue
        if v < 2.50 and not prev:
            if not triggers or (data.index[i] - triggers[-1]).days > 365: triggers.append(data.index[i])
            prev = True
        elif v >= 3.0: prev = False
    print(f"Events: {len(triggers)}")
    if not triggers: return mark_failed(sid, "no events")
    try: px = load_prices(["EQT","AR","RRC","SPY"], start="2005-01-01")
    except Exception as e: return mark_failed(sid, f"price: {e}")
    ret = daily_returns(px)
    available = [t for t in ["EQT","AR","RRC"] if t in ret.columns]
    if not available: return mark_failed(sid, "no tickers")
    basket_r = ret[available].mean(axis=1); spy_r = ret["SPY"]; hold = 252
    pnl = pd.Series(0.0, index=basket_r.index); evts = []
    for td in triggers:
        ed = td + pd.offsets.MonthBegin(1); mask = basket_r.index >= ed
        if mask.sum() < hold: continue
        ei = basket_r.index[mask][0]; p = basket_r.index.get_loc(ei); ep = min(p+hold, len(basket_r))
        er = basket_r.iloc[p:ep]; cr = float((1+er).prod()-1); pnl.iloc[p:ep] = er.values[:ep-p]
        sc = None
        if ei in spy_r.index: sp=spy_r.index.get_loc(ei); sc=float((1+spy_r.iloc[sp:min(sp+hold,len(spy_r))]).prod()-1)
        evts.append({"trigger_date":str(td.date()),"hh_price":round(float(data.loc[td]),2),
            "basket_return":round(cr,4),"spy_return":round(sc,4) if sc else None})
    if not evts: return mark_failed(sid, "no valid events")
    ip = pnl[pnl!=0]
    if len(ip)<30: return mark_failed(sid, f"insufficient days ({len(ip)})")
    m = compute_metrics(ip, benchmark=spy_r, name="HH Below Cash Cost → Long NatGas Producers")
    ra = [e["basket_return"] for e in evts]
    save_result(sid, m, extra={"rule":"Long EQT+AR+RRC 252d when MHHNGSP < $2.50",
        "source":"FRED MHHNGSP; yfinance","n_events":len(evts),"avg_event_return":round(float(np.mean(ra)),4),
        "event_win_rate":round(float(np.mean([r>0 for r in ra])),4),"events":evts})
    print(f"Done: {len(evts)} events")
if __name__=="__main__": main()
''',

"PL91_us_crude_production_decline_long": '''"""PL91 — US Crude Production Declines MoM for 3mo → Long CL=F 6mo"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns
def main():
    sid = "PL91_us_crude_production_decline_long"
    try: fred = load_fred("MCRFPUS1", start="2000-01-01"); data = fred.squeeze()
    except Exception as e: return mark_failed(sid, f"FRED: {e}")
    if data.empty: return mark_failed(sid, "no data")
    mom = data.diff().dropna()
    triggers = []; streak = 0
    for i in range(len(mom)):
        v = float(mom.iloc[i])
        if np.isnan(v): continue
        if v < 0: streak += 1
        else: streak = 0
        if streak == 3:
            if not triggers or (mom.index[i] - triggers[-1]).days > 180: triggers.append(mom.index[i])
    print(f"Events: {len(triggers)}")
    if not triggers: return mark_failed(sid, "no events")
    try: px = load_prices(["CL=F","SPY"], start="2000-01-01")
    except Exception as e: return mark_failed(sid, f"price: {e}")
    ret = daily_returns(px)
    if "CL=F" not in ret.columns: return mark_failed(sid, "CL=F data missing")
    cl_r = ret["CL=F"]; spy_r = ret["SPY"]; hold = 126
    pnl = pd.Series(0.0, index=cl_r.index); evts = []
    for td in triggers:
        ed = td + pd.offsets.MonthBegin(1); mask = cl_r.index >= ed
        if mask.sum() < hold: continue
        ei = cl_r.index[mask][0]; p = cl_r.index.get_loc(ei); ep = min(p+hold, len(cl_r))
        er = cl_r.iloc[p:ep]; cr = float((1+er).prod()-1); pnl.iloc[p:ep] = er.values[:ep-p]
        sc = None
        if ei in spy_r.index: sp=spy_r.index.get_loc(ei); sc=float((1+spy_r.iloc[sp:min(sp+hold,len(spy_r))]).prod()-1)
        evts.append({"trigger_date":str(td.date()),"cl_return":round(cr,4),"spy_return":round(sc,4) if sc else None})
    if not evts: return mark_failed(sid, "no valid events")
    ip = pnl[pnl!=0]
    if len(ip)<30: return mark_failed(sid, f"insufficient days ({len(ip)})")
    m = compute_metrics(ip, benchmark=spy_r, name="US Crude Production Decline → Long CL=F")
    ra = [e["cl_return"] for e in evts]
    save_result(sid, m, extra={"rule":"Long CL=F 126d when MCRFPUS1 declines MoM for 3 consecutive months",
        "source":"FRED MCRFPUS1; yfinance","n_events":len(evts),"avg_event_return":round(float(np.mean(ra)),4),
        "event_win_rate":round(float(np.mean([r>0 for r in ra])),4),"events":evts})
    print(f"Done: {len(evts)} events")
if __name__=="__main__": main()
''',

"PL92_air_rpm_recovery_jets": '''"""PL92 — Air RPM YoY > +5% After Decline → Long JETS"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns
def main():
    sid = "PL92_air_rpm_recovery_jets"
    try: fred = load_fred("AIRRPMTSI", start="2000-01-01"); data = fred.squeeze()
    except Exception as e: return mark_failed(sid, f"FRED: {e}")
    if data.empty: return mark_failed(sid, "no data")
    yoy = data.pct_change(12).dropna()
    triggers = []; neg_count = 0; fired = False
    for i in range(len(yoy)):
        v = float(yoy.iloc[i])
        if np.isnan(v): continue
        if v < 0: neg_count += 1; fired = False
        elif v >= 0.05 and neg_count >= 6 and not fired:
            triggers.append(yoy.index[i]); fired = True; neg_count = 0
        else: neg_count = 0
    print(f"Events: {len(triggers)}")
    if not triggers: return mark_failed(sid, "no events")
    try: px = load_prices(["JETS","DAL","UAL","SPY"], start="2005-01-01")
    except Exception as e: return mark_failed(sid, f"price: {e}")
    ret = daily_returns(px)
    available = [t for t in ["JETS","DAL","UAL"] if t in ret.columns]
    if not available: return mark_failed(sid, "no airline tickers")
    basket_r = ret[available].mean(axis=1); spy_r = ret["SPY"]; hold = 126
    pnl = pd.Series(0.0, index=basket_r.index); evts = []
    for td in triggers:
        ed = td + pd.offsets.MonthBegin(1); mask = basket_r.index >= ed
        if mask.sum() < hold: continue
        ei = basket_r.index[mask][0]; p = basket_r.index.get_loc(ei); ep = min(p+hold, len(basket_r))
        er = basket_r.iloc[p:ep]; cr = float((1+er).prod()-1); pnl.iloc[p:ep] = er.values[:ep-p]
        sc = None
        if ei in spy_r.index: sp=spy_r.index.get_loc(ei); sc=float((1+spy_r.iloc[sp:min(sp+hold,len(spy_r))]).prod()-1)
        evts.append({"trigger_date":str(td.date()),"basket_return":round(cr,4),"spy_return":round(sc,4) if sc else None})
    if not evts: return mark_failed(sid, "no valid events")
    ip = pnl[pnl!=0]
    if len(ip)<30: return mark_failed(sid, f"insufficient days ({len(ip)})")
    m = compute_metrics(ip, benchmark=spy_r, name="Air RPM Recovery → Long Airlines")
    ra = [e["basket_return"] for e in evts]
    save_result(sid, m, extra={"rule":"Long JETS/DAL/UAL 126d when AIRRPMTSI YoY > +5% after 6+ months negative",
        "source":"FRED AIRRPMTSI; yfinance","n_events":len(evts),"avg_event_return":round(float(np.mean(ra)),4),
        "event_win_rate":round(float(np.mean([r>0 for r in ra])),4),"events":evts})
    print(f"Done: {len(evts)} events")
if __name__=="__main__": main()
''',

"PL93_intermodal_traffic_turn_rails": '''"""PL93 — Railroad Intermodal Traffic YoY Positive After 6mo Decline → Long UNP+CSX"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns
def main():
    sid = "PL93_intermodal_traffic_turn_rails"
    try: fred = load_fred("RAILFRTINTERAM", start="2000-01-01"); data = fred.squeeze()
    except Exception as e: return mark_failed(sid, f"FRED: {e}")
    if data.empty: return mark_failed(sid, "no data")
    yoy = data.pct_change(12).dropna()
    triggers = []; neg_count = 0; fired = False
    for i in range(len(yoy)):
        v = float(yoy.iloc[i])
        if np.isnan(v): continue
        if v < 0: neg_count += 1; fired = False
        elif v >= 0 and neg_count >= 6 and not fired:
            triggers.append(yoy.index[i]); fired = True; neg_count = 0
        else: neg_count = 0
    print(f"Events: {len(triggers)}")
    if not triggers: return mark_failed(sid, "no events")
    try: px = load_prices(["UNP","CSX","SPY"], start="2000-01-01")
    except Exception as e: return mark_failed(sid, f"price: {e}")
    ret = daily_returns(px)
    available = [t for t in ["UNP","CSX"] if t in ret.columns]
    if not available: return mark_failed(sid, "no tickers")
    basket_r = ret[available].mean(axis=1); spy_r = ret["SPY"]; hold = 126
    pnl = pd.Series(0.0, index=basket_r.index); evts = []
    for td in triggers:
        ed = td + pd.offsets.MonthBegin(1); mask = basket_r.index >= ed
        if mask.sum() < hold: continue
        ei = basket_r.index[mask][0]; p = basket_r.index.get_loc(ei); ep = min(p+hold, len(basket_r))
        er = basket_r.iloc[p:ep]; cr = float((1+er).prod()-1); pnl.iloc[p:ep] = er.values[:ep-p]
        sc = None
        if ei in spy_r.index: sp=spy_r.index.get_loc(ei); sc=float((1+spy_r.iloc[sp:min(sp+hold,len(spy_r))]).prod()-1)
        evts.append({"trigger_date":str(td.date()),"basket_return":round(cr,4),"spy_return":round(sc,4) if sc else None})
    if not evts: return mark_failed(sid, "no valid events")
    ip = pnl[pnl!=0]
    if len(ip)<30: return mark_failed(sid, f"insufficient days ({len(ip)})")
    m = compute_metrics(ip, benchmark=spy_r, name="Intermodal Traffic Turn → Long UNP+CSX")
    ra = [e["basket_return"] for e in evts]
    save_result(sid, m, extra={"rule":"Long UNP+CSX 126d when RAILFRTINTERAM YoY turns positive after 6+ months negative",
        "source":"FRED RAILFRTINTERAM; yfinance","n_events":len(evts),"avg_event_return":round(float(np.mean(ra)),4),
        "event_win_rate":round(float(np.mean([r>0 for r in ra])),4),"events":evts})
    print(f"Done: {len(evts)} events")
if __name__=="__main__": main()
''',

"PL94_umich_sentiment_low_xrt": '''"""PL94 — UMich Sentiment Below 55 Then Rises 2mo → Long XRT"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns
def main():
    sid = "PL94_umich_sentiment_low_xrt"
    try: fred = load_fred("UMCSENT", start="1978-01-01"); data = fred.squeeze()
    except Exception as e: return mark_failed(sid, f"FRED: {e}")
    if data.empty: return mark_failed(sid, "no data")
    triggers = []; was_below55 = False
    for i in range(2, len(data)):
        v0=float(data.iloc[i-2]); v1=float(data.iloc[i-1]); v2=float(data.iloc[i])
        if any(np.isnan(x) for x in [v0,v1,v2]): continue
        if v0 < 55: was_below55 = True
        if was_below55 and v1 > v0 and v2 > v1:
            if not triggers or (data.index[i] - triggers[-1]).days > 365: triggers.append(data.index[i])
            was_below55 = False
    print(f"Events: {len(triggers)}")
    if not triggers: return mark_failed(sid, "no events")
    try: px = load_prices(["XRT","SPY"], start="2006-01-01")
    except Exception as e: return mark_failed(sid, f"price: {e}")
    ret = daily_returns(px); xrt_r = ret["XRT"]; spy_r = ret["SPY"]; hold = 126
    pnl = pd.Series(0.0, index=xrt_r.index); evts = []
    for td in triggers:
        ed = td + pd.offsets.MonthBegin(1); mask = xrt_r.index >= ed
        if mask.sum() < hold: continue
        ei = xrt_r.index[mask][0]; p = xrt_r.index.get_loc(ei); ep = min(p+hold, len(xrt_r))
        er = xrt_r.iloc[p:ep]; cr = float((1+er).prod()-1); pnl.iloc[p:ep] = er.values[:ep-p]
        sc = None
        if ei in spy_r.index: sp=spy_r.index.get_loc(ei); sc=float((1+spy_r.iloc[sp:min(sp+hold,len(spy_r))]).prod()-1)
        evts.append({"trigger_date":str(td.date()),"sentiment":round(float(data.loc[td]),1),
            "xrt_return":round(cr,4),"spy_return":round(sc,4) if sc else None})
    if not evts: return mark_failed(sid, "no valid events")
    ip = pnl[pnl!=0]
    if len(ip)<30: return mark_failed(sid, f"insufficient days ({len(ip)})")
    m = compute_metrics(ip, benchmark=spy_r, name="UMich Sentiment Trough → Long XRT")
    ra = [e["xrt_return"] for e in evts]
    save_result(sid, m, extra={"rule":"Long XRT 126d when UMCSENT below 55 then rises 2 consecutive months",
        "source":"FRED UMCSENT; yfinance","n_events":len(evts),"avg_event_return":round(float(np.mean(ra)),4),
        "event_win_rate":round(float(np.mean([r>0 for r in ra])),4),"events":evts})
    print(f"Done: {len(evts)} events")
if __name__=="__main__": main()
''',

"PL95_savings_rate_decline_leisure": '''"""PL95 — Savings Rate Declines >4pp from Peak → Long BKNG+MAR+H"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns
def main():
    sid = "PL95_savings_rate_decline_leisure"
    try: fred = load_fred("PSAVERT", start="1990-01-01"); data = fred.squeeze()
    except Exception as e: return mark_failed(sid, f"FRED: {e}")
    if data.empty: return mark_failed(sid, "no data")
    peak12 = data.rolling(12).max().dropna()
    diff = data - peak12; diff = diff.dropna()
    triggers = []; prev = False
    for i in range(len(diff)):
        v = float(diff.iloc[i])
        if np.isnan(v): continue
        if v < -4.0 and not prev:
            if not triggers or (diff.index[i] - triggers[-1]).days > 365: triggers.append(diff.index[i])
            prev = True
        elif v >= -2.0: prev = False
    print(f"Events: {len(triggers)}")
    if not triggers: return mark_failed(sid, "no events")
    try: px = load_prices(["BKNG","MAR","H","SPY"], start="2006-01-01")
    except Exception as e: return mark_failed(sid, f"price: {e}")
    ret = daily_returns(px)
    available = [t for t in ["BKNG","MAR","H"] if t in ret.columns]
    if not available: return mark_failed(sid, "no tickers")
    basket_r = ret[available].mean(axis=1); spy_r = ret["SPY"]; hold = 126
    pnl = pd.Series(0.0, index=basket_r.index); evts = []
    for td in triggers:
        ed = td + pd.offsets.MonthBegin(1); mask = basket_r.index >= ed
        if mask.sum() < hold: continue
        ei = basket_r.index[mask][0]; p = basket_r.index.get_loc(ei); ep = min(p+hold, len(basket_r))
        er = basket_r.iloc[p:ep]; cr = float((1+er).prod()-1); pnl.iloc[p:ep] = er.values[:ep-p]
        sc = None
        if ei in spy_r.index: sp=spy_r.index.get_loc(ei); sc=float((1+spy_r.iloc[sp:min(sp+hold,len(spy_r))]).prod()-1)
        evts.append({"trigger_date":str(td.date()),"basket_return":round(cr,4),"spy_return":round(sc,4) if sc else None})
    if not evts: return mark_failed(sid, "no valid events")
    ip = pnl[pnl!=0]
    if len(ip)<30: return mark_failed(sid, f"insufficient days ({len(ip)})")
    m = compute_metrics(ip, benchmark=spy_r, name="Savings Rate Decline → Long Leisure")
    ra = [e["basket_return"] for e in evts]
    save_result(sid, m, extra={"rule":"Long BKNG+MAR+H 126d when PSAVERT drops >4pp from 12mo peak",
        "source":"FRED PSAVERT; yfinance","n_events":len(evts),"avg_event_return":round(float(np.mean(ra)),4),
        "event_win_rate":round(float(np.mean([r>0 for r in ra])),4),"events":evts})
    print(f"Done: {len(evts)} events")
if __name__=="__main__": main()
''',

"PL96_retail_sales_ex_auto_xly": '''"""PL96 — Retail Sales ex-Auto MoM > +0.5% for 2mo After Flat → Long XLY"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns
def main():
    sid = "PL96_retail_sales_ex_auto_xly"
    try: fred = load_fred("RSFSXMV", start="1992-01-01"); data = fred.squeeze()
    except Exception as e: return mark_failed(sid, f"FRED: {e}")
    if data.empty: return mark_failed(sid, "no data")
    mom = data.pct_change().dropna()
    triggers = []; flat_count = 0
    for i in range(1, len(mom)):
        v = float(mom.iloc[i]); prev = float(mom.iloc[i-1])
        if np.isnan(v) or np.isnan(prev): continue
        if v <= 0: flat_count += 1
        elif v > 0.005 and prev > 0.005 and flat_count >= 3:
            if not triggers or (mom.index[i] - triggers[-1]).days > 180: triggers.append(mom.index[i])
            flat_count = 0
        else: flat_count = 0 if v > 0 else flat_count
    print(f"Events: {len(triggers)}")
    if not triggers: return mark_failed(sid, "no events")
    try: px = load_prices(["XLY","SPY"], start="1998-01-01")
    except Exception as e: return mark_failed(sid, f"price: {e}")
    ret = daily_returns(px); xly_r = ret["XLY"]; spy_r = ret["SPY"]; hold = 63
    pnl = pd.Series(0.0, index=xly_r.index); evts = []
    for td in triggers:
        ed = td + pd.offsets.MonthBegin(1); mask = xly_r.index >= ed
        if mask.sum() < hold: continue
        ei = xly_r.index[mask][0]; p = xly_r.index.get_loc(ei); ep = min(p+hold, len(xly_r))
        er = xly_r.iloc[p:ep]; cr = float((1+er).prod()-1); pnl.iloc[p:ep] = er.values[:ep-p]
        sc = None
        if ei in spy_r.index: sp=spy_r.index.get_loc(ei); sc=float((1+spy_r.iloc[sp:min(sp+hold,len(spy_r))]).prod()-1)
        evts.append({"trigger_date":str(td.date()),"xly_return":round(cr,4),"spy_return":round(sc,4) if sc else None})
    if not evts: return mark_failed(sid, "no valid events")
    ip = pnl[pnl!=0]
    if len(ip)<30: return mark_failed(sid, f"insufficient days ({len(ip)})")
    m = compute_metrics(ip, benchmark=spy_r, name="Retail Sales ex-Auto Turn → Long XLY")
    ra = [e["xly_return"] for e in evts]
    save_result(sid, m, extra={"rule":"Long XLY 63d when RSFSXMV MoM>+0.5% for 2mo after 3+ flat months",
        "source":"FRED RSFSXMV; yfinance","n_events":len(evts),"avg_event_return":round(float(np.mean(ra)),4),
        "event_win_rate":round(float(np.mean([r>0 for r in ra])),4),"events":evts})
    print(f"Done: {len(evts)} events")
if __name__=="__main__": main()
''',

"PL97_core_capex_orders_capital_goods": '''"""PL97 — Core Capex Orders 3mo MA YoY Positive After Decline → Long ETN+ROK+AME"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns
def main():
    sid = "PL97_core_capex_orders_capital_goods"
    try: fred = load_fred("NEWORDER", start="1990-01-01"); data = fred.squeeze()
    except Exception as e: return mark_failed(sid, f"FRED: {e}")
    if data.empty: return mark_failed(sid, "no data")
    ma3 = data.rolling(3).mean().dropna()
    yoy = ma3.pct_change(12).dropna()
    triggers = []; neg_count = 0; fired = False
    for i in range(len(yoy)):
        v = float(yoy.iloc[i])
        if np.isnan(v): continue
        if v < 0: neg_count += 1; fired = False
        elif v >= 0 and neg_count >= 6 and not fired:
            triggers.append(yoy.index[i]); fired = True; neg_count = 0
        else: neg_count = 0
    print(f"Events: {len(triggers)}")
    if not triggers: return mark_failed(sid, "no events")
    try: px = load_prices(["ETN","ROK","AME","SPY"], start="1998-01-01")
    except Exception as e: return mark_failed(sid, f"price: {e}")
    ret = daily_returns(px)
    available = [t for t in ["ETN","ROK","AME"] if t in ret.columns]
    if not available: return mark_failed(sid, "no tickers")
    basket_r = ret[available].mean(axis=1); spy_r = ret["SPY"]; hold = 126
    pnl = pd.Series(0.0, index=basket_r.index); evts = []
    for td in triggers:
        ed = td + pd.offsets.MonthBegin(1); mask = basket_r.index >= ed
        if mask.sum() < hold: continue
        ei = basket_r.index[mask][0]; p = basket_r.index.get_loc(ei); ep = min(p+hold, len(basket_r))
        er = basket_r.iloc[p:ep]; cr = float((1+er).prod()-1); pnl.iloc[p:ep] = er.values[:ep-p]
        sc = None
        if ei in spy_r.index: sp=spy_r.index.get_loc(ei); sc=float((1+spy_r.iloc[sp:min(sp+hold,len(spy_r))]).prod()-1)
        evts.append({"trigger_date":str(td.date()),"basket_return":round(cr,4),"spy_return":round(sc,4) if sc else None})
    if not evts: return mark_failed(sid, "no valid events")
    ip = pnl[pnl!=0]
    if len(ip)<30: return mark_failed(sid, f"insufficient days ({len(ip)})")
    m = compute_metrics(ip, benchmark=spy_r, name="Core Capex Orders Turn → Long ETN+ROK+AME")
    ra = [e["basket_return"] for e in evts]
    save_result(sid, m, extra={"rule":"Long ETN+ROK+AME 126d when NEWORDER 3mo MA YoY turns positive after 6+ months negative",
        "source":"FRED NEWORDER; yfinance","n_events":len(evts),"avg_event_return":round(float(np.mean(ra)),4),
        "event_win_rate":round(float(np.mean([r>0 for r in ra])),4),"events":evts})
    print(f"Done: {len(evts)} events")
if __name__=="__main__": main()
''',

"PL98_ipman_recovery_automation": '''"""PL98 — IPMAN Crosses Above 12mo MA After 6mo Below → Long ROK+TER"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns
def main():
    sid = "PL98_ipman_recovery_automation"
    try: fred = load_fred("IPMAN", start="1990-01-01"); data = fred.squeeze()
    except Exception as e: return mark_failed(sid, f"FRED: {e}")
    if data.empty: return mark_failed(sid, "no data")
    ma12 = data.rolling(12).mean().dropna()
    triggers = []; below_count = 0; fired = False
    for i in range(len(data)):
        if data.index[i] not in ma12.index: continue
        v = float(data.iloc[i]); m = float(ma12.loc[data.index[i]])
        if np.isnan(v) or np.isnan(m): continue
        if v < m: below_count += 1; fired = False
        elif v >= m and below_count >= 6 and not fired:
            triggers.append(data.index[i]); fired = True; below_count = 0
        elif v >= m: below_count = 0
    print(f"Events: {len(triggers)}")
    if not triggers: return mark_failed(sid, "no events")
    try: px = load_prices(["ROK","TER","SPY"], start="1997-01-01")
    except Exception as e: return mark_failed(sid, f"price: {e}")
    ret = daily_returns(px)
    available = [t for t in ["ROK","TER"] if t in ret.columns]
    if not available: return mark_failed(sid, "no tickers")
    basket_r = ret[available].mean(axis=1); spy_r = ret["SPY"]; hold = 126
    pnl = pd.Series(0.0, index=basket_r.index); evts = []
    for td in triggers:
        ed = td + pd.offsets.MonthBegin(1); mask = basket_r.index >= ed
        if mask.sum() < hold: continue
        ei = basket_r.index[mask][0]; p = basket_r.index.get_loc(ei); ep = min(p+hold, len(basket_r))
        er = basket_r.iloc[p:ep]; cr = float((1+er).prod()-1); pnl.iloc[p:ep] = er.values[:ep-p]
        sc = None
        if ei in spy_r.index: sp=spy_r.index.get_loc(ei); sc=float((1+spy_r.iloc[sp:min(sp+hold,len(spy_r))]).prod()-1)
        evts.append({"trigger_date":str(td.date()),"basket_return":round(cr,4),"spy_return":round(sc,4) if sc else None})
    if not evts: return mark_failed(sid, "no valid events")
    ip = pnl[pnl!=0]
    if len(ip)<30: return mark_failed(sid, f"insufficient days ({len(ip)})")
    m = compute_metrics(ip, benchmark=spy_r, name="IPMAN Recovery → Long ROK+TER")
    ra = [e["basket_return"] for e in evts]
    save_result(sid, m, extra={"rule":"Long ROK+TER 126d when IPMAN crosses above 12mo MA after 6+ months below",
        "source":"FRED IPMAN; yfinance","n_events":len(evts),"avg_event_return":round(float(np.mean(ra)),4),
        "event_win_rate":round(float(np.mean([r>0 for r in ra])),4),"events":evts})
    print(f"Done: {len(evts)} events")
if __name__=="__main__": main()
''',

"PL99_breakeven_below_target_tips": '''"""PL99 — 5Y Breakeven Drops Below 2.0% After 6mo Above 2.3% → Long TIP"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns
def main():
    sid = "PL99_breakeven_below_target_tips"
    try: fred = load_fred("T5YIE", start="2003-01-01"); data = fred.squeeze()
    except Exception as e: return mark_failed(sid, f"FRED: {e}")
    if data.empty: return mark_failed(sid, "no data")
    triggers = []; above23_days = 0
    for i in range(1, len(data)):
        v = float(data.iloc[i]); prev = float(data.iloc[i-1])
        if np.isnan(v) or np.isnan(prev): continue
        if v > 2.3: above23_days += 1
        elif v <= 2.0 and prev > 2.0 and above23_days >= 126:
            if not triggers or (data.index[i] - triggers[-1]).days > 365: triggers.append(data.index[i])
            above23_days = 0
        elif v <= 2.3: above23_days = 0
    print(f"Events: {len(triggers)}")
    if not triggers: return mark_failed(sid, "no events")
    try: px = load_prices(["TIP","IEF","SPY"], start="2003-01-01")
    except Exception as e: return mark_failed(sid, f"price: {e}")
    ret = daily_returns(px); tip_r = ret["TIP"]; spy_r = ret["SPY"]; hold = 252
    pnl = pd.Series(0.0, index=tip_r.index); evts = []
    for td in triggers:
        mask = tip_r.index >= td
        if mask.sum() < hold: continue
        ei = tip_r.index[mask][0]; p = tip_r.index.get_loc(ei); ep = min(p+hold, len(tip_r))
        er = tip_r.iloc[p:ep]; cr = float((1+er).prod()-1); pnl.iloc[p:ep] = er.values[:ep-p]
        sc = None
        if ei in spy_r.index: sp=spy_r.index.get_loc(ei); sc=float((1+spy_r.iloc[sp:min(sp+hold,len(spy_r))]).prod()-1)
        evts.append({"trigger_date":str(td.date()),"breakeven":round(float(data.loc[td]),2),
            "tip_return":round(cr,4),"spy_return":round(sc,4) if sc else None})
    if not evts: return mark_failed(sid, "no valid events")
    ip = pnl[pnl!=0]
    if len(ip)<30: return mark_failed(sid, f"insufficient days ({len(ip)})")
    m = compute_metrics(ip, benchmark=spy_r, name="Breakeven Below Target → Long TIP")
    ra = [e["tip_return"] for e in evts]
    save_result(sid, m, extra={"rule":"Long TIP 252d when T5YIE drops below 2.0% after 6+ months above 2.3%",
        "source":"FRED T5YIE; yfinance","n_events":len(evts),"avg_event_return":round(float(np.mean(ra)),4),
        "event_win_rate":round(float(np.mean([r>0 for r in ra])),4),"events":evts})
    print(f"Done: {len(evts)} events")
if __name__=="__main__": main()
''',

"PL100_negative_real_rate_qqq": '''"""PL100 — Real Fed Funds Rate Turns Negative → Long QQQ 12mo"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns
def main():
    sid = "PL100_negative_real_rate_qqq"
    try:
        dff = load_fred("DFF", start="1998-01-01").squeeze()
        pce = load_fred("PCEPILFE", start="1998-01-01").squeeze()
    except Exception as e: return mark_failed(sid, f"FRED: {e}")
    if dff.empty or pce.empty: return mark_failed(sid, "no data")
    # Forward-fill PCE to daily
    pce_daily = pce.resample("D").ffill()
    idx = dff.index.intersection(pce_daily.index)
    real_rate = dff.loc[idx] - pce_daily.loc[idx]
    real_rate = real_rate.dropna()
    triggers = []; pos_days = 0
    for i in range(1, len(real_rate)):
        v = float(real_rate.iloc[i]); prev = float(real_rate.iloc[i-1])
        if np.isnan(v) or np.isnan(prev): continue
        if v > 0: pos_days += 1
        elif v <= 0 and prev > 0 and pos_days >= 126:
            if not triggers or (real_rate.index[i] - triggers[-1]).days > 365: triggers.append(real_rate.index[i])
            pos_days = 0
        elif v <= 0: pos_days = 0
    print(f"Events: {len(triggers)}")
    if not triggers: return mark_failed(sid, "no events")
    try: px = load_prices(["QQQ","SPY"], start="1999-01-01")
    except Exception as e: return mark_failed(sid, f"price: {e}")
    ret = daily_returns(px); qqq_r = ret["QQQ"]; spy_r = ret["SPY"]; hold = 252
    pnl = pd.Series(0.0, index=qqq_r.index); evts = []
    for td in triggers:
        mask = qqq_r.index >= td
        if mask.sum() < hold: continue
        ei = qqq_r.index[mask][0]; p = qqq_r.index.get_loc(ei); ep = min(p+hold, len(qqq_r))
        er = qqq_r.iloc[p:ep]; cr = float((1+er).prod()-1); pnl.iloc[p:ep] = er.values[:ep-p]
        sc = None
        if ei in spy_r.index: sp=spy_r.index.get_loc(ei); sc=float((1+spy_r.iloc[sp:min(sp+hold,len(spy_r))]).prod()-1)
        evts.append({"trigger_date":str(td.date()),"qqq_return":round(cr,4),"spy_return":round(sc,4) if sc else None})
    if not evts: return mark_failed(sid, "no valid events")
    ip = pnl[pnl!=0]
    if len(ip)<30: return mark_failed(sid, f"insufficient days ({len(ip)})")
    m = compute_metrics(ip, benchmark=spy_r, name="Negative Real Rate → Long QQQ")
    ra = [e["qqq_return"] for e in evts]
    save_result(sid, m, extra={"rule":"Long QQQ 252d when DFF-PCEPILFE turns negative after 6+ months positive",
        "source":"FRED DFF, PCEPILFE; yfinance","n_events":len(evts),"avg_event_return":round(float(np.mean(ra)),4),
        "event_win_rate":round(float(np.mean([r>0 for r in ra])),4),"events":evts})
    print(f"Done: {len(evts)} events")
if __name__=="__main__": main()
''',

"PL101_yield_curve_steepening_tlt": '''"""PL101 — 10Y-3M Un-Inverts After Deep Inversion → Long TLT 12mo"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns
def main():
    sid = "PL101_yield_curve_steepening_tlt"
    try: fred = load_fred("T10Y3M", start="1982-01-01"); data = fred.squeeze()
    except Exception as e: return mark_failed(sid, f"FRED: {e}")
    if data.empty: return mark_failed(sid, "no data")
    triggers = []; deep_inv_days = 0
    for i in range(1, len(data)):
        v = float(data.iloc[i]); prev = float(data.iloc[i-1])
        if np.isnan(v) or np.isnan(prev): continue
        if v < -1.0: deep_inv_days += 1
        elif v >= 0 and prev < 0 and deep_inv_days >= 126:
            if not triggers or (data.index[i] - triggers[-1]).days > 365: triggers.append(data.index[i])
            deep_inv_days = 0
        elif v >= -1.0 and v < 0: pass  # still inverted but not deep
        else: deep_inv_days = 0
    print(f"Events: {len(triggers)}")
    if not triggers: return mark_failed(sid, "no events")
    try: px = load_prices(["TLT","SPY"], start="2002-01-01")
    except Exception as e: return mark_failed(sid, f"price: {e}")
    ret = daily_returns(px); tlt_r = ret["TLT"]; spy_r = ret["SPY"]; hold = 252
    pnl = pd.Series(0.0, index=tlt_r.index); evts = []
    for td in triggers:
        mask = tlt_r.index >= td
        if mask.sum() < hold: continue
        ei = tlt_r.index[mask][0]; p = tlt_r.index.get_loc(ei); ep = min(p+hold, len(tlt_r))
        er = tlt_r.iloc[p:ep]; cr = float((1+er).prod()-1); pnl.iloc[p:ep] = er.values[:ep-p]
        sc = None
        if ei in spy_r.index: sp=spy_r.index.get_loc(ei); sc=float((1+spy_r.iloc[sp:min(sp+hold,len(spy_r))]).prod()-1)
        evts.append({"trigger_date":str(td.date()),"tlt_return":round(cr,4),"spy_return":round(sc,4) if sc else None})
    if not evts: return mark_failed(sid, "no valid events")
    ip = pnl[pnl!=0]
    if len(ip)<30: return mark_failed(sid, f"insufficient days ({len(ip)})")
    m = compute_metrics(ip, benchmark=spy_r, name="10Y-3M Un-Inversion → Long TLT")
    ra = [e["tlt_return"] for e in evts]
    save_result(sid, m, extra={"rule":"Long TLT 252d when T10Y3M crosses above 0 after 6+ months below -1.0",
        "source":"FRED T10Y3M; yfinance","n_events":len(evts),"avg_event_return":round(float(np.mean(ra)),4),
        "event_win_rate":round(float(np.mean([r>0 for r in ra])),4),"events":evts})
    print(f"Done: {len(evts)} events")
if __name__=="__main__": main()
''',
}


def main():
    q = load_queue()
    ready = [s for s in q if s.get("status") == "ready"]
    print(f"=== BATCH BACKTEST: {len(ready)} ready strategies ===\n")

    # Write all strategy scripts that are defined in STRATEGIES or remaining_scripts
    all_scripts = {}
    all_scripts.update(STRATEGIES)
    all_scripts.update(remaining_scripts)

    for sid, content in all_scripts.items():
        script_path = BACKTESTS_DIR / f"{sid}.py"
        if not script_path.exists():
            with open(script_path, "w") as f:
                f.write(content)
            print(f"  Created {script_path.name}")

    # Now run each ready strategy
    results_summary = []
    for strat in ready:
        sid = strat["signal_id"]
        print(f"\n--- [{ready.index(strat)+1}/{len(ready)}] {sid} ---")

        # Set status to in_progress
        set_status(q, sid, "in_progress")

        # Check script exists
        script_path = BACKTESTS_DIR / f"{sid}.py"
        if not script_path.exists():
            print(f"  SKIP: no script for {sid}")
            set_status(q, sid, "failed", {"status": "fail", "reason": "no backtest script",
                "sharpe": None, "cagr": None, "max_dd": None, "t_stat": None})
            results_summary.append({"signal_id": sid, "status": "failed", "reason": "no script"})
            continue

        # Run backtest
        ok, err = run_backtest(sid)
        if not ok:
            print(f"  FAILED: {err}")
            set_status(q, sid, "failed", {"status": "fail", "reason": err,
                "sharpe": None, "cagr": None, "max_dd": None, "t_stat": None})
            results_summary.append({"signal_id": sid, "status": "failed", "reason": err})
            continue

        # Read result
        result = read_result(sid)
        status = "done" if result["status"] == "ok" else "failed"
        set_status(q, sid, status, result)

        sharpe = result.get("sharpe")
        cagr = result.get("cagr")
        winner = ""
        if sharpe is not None and cagr is not None and sharpe > 0.5 and cagr > 0.10:
            winner = " *** WINNER ***"
        print(f"  Result: Sharpe={sharpe}, CAGR={cagr}{winner}")
        results_summary.append({"signal_id": sid, "status": status, "sharpe": sharpe, "cagr": cagr,
                                "max_dd": result.get("max_dd"), "t_stat": result.get("t_stat")})

    # Update heartbeat
    status_file = ROOT / "pipeline" / "status.json"
    with open(status_file) as f:
        status_data = json.load(f)
    status_data["loops"]["backtester"]["last_run"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    with open(status_file, "w") as f:
        json.dump(status_data, f, indent=2)

    # Print summary table
    print("\n\n" + "=" * 90)
    print(f"{'Signal ID':<50} {'Status':<8} {'Sharpe':>8} {'CAGR':>8} {'MaxDD':>8} {'t-stat':>8}")
    print("-" * 90)
    winners = []
    for r in results_summary:
        s = r.get("sharpe")
        c = r.get("cagr")
        d = r.get("max_dd")
        t = r.get("t_stat")
        flag = " *" if s is not None and c is not None and s > 0.5 and c > 0.10 else ""
        print(f"{r['signal_id']:<50} {r['status']:<8} {s if s is not None else 'N/A':>8} "
              f"{f'{c*100:.1f}%' if c is not None else 'N/A':>8} "
              f"{f'{d*100:.1f}%' if d is not None else 'N/A':>8} "
              f"{t if t is not None else 'N/A':>8}{flag}")
        if s is not None and c is not None and s > 0.5 and c > 0.10:
            winners.append(r["signal_id"])

    print(f"\nTotal: {len(results_summary)} backtested")
    print(f"Winners (Sharpe>0.5, CAGR>10%): {len(winners)}")
    for w in winners:
        print(f"  {w}")

    return winners


if __name__ == "__main__":
    winners = main()
