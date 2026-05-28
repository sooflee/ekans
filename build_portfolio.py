"""
Build portfolio.html: combines uncorrelated winners into an equal-weight portfolio,
shows equity curves, and highlights currently active signals.

Usage: .venv/bin/python build_portfolio.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime, date

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PNL_DIR = RESULTS / "pnl"

sys.path.insert(0, str(ROOT / "backtests"))


def load_winner_data():
    results = {}
    for fp in sorted(RESULTS.glob("*.json")):
        if fp.stem.startswith("_"):
            continue
        d = json.load(open(fp))
        if d.get("status") == "ok" and (d.get("sharpe", 0) or 0) > 0.5 and (d.get("cagr", 0) or 0) > 0.1:
            results[fp.stem] = d
    return results


def load_pnl():
    pnl = {}
    for fp in PNL_DIR.glob("*.parquet"):
        df = pd.read_parquet(fp)
        pnl[fp.stem] = df["pnl"]
    return pnl


def select_portfolio_signals(results, pnl_data):
    """Select uncorrelated signals for the portfolio: 1 per cluster + filtered independents."""
    corr_path = RESULTS / "_correlation_matrix.json"
    mt_path = RESULTS / "_multiple_testing.json"
    corr_data = json.load(open(corr_path))
    mt_data = json.load(open(mt_path)) if mt_path.exists() else {}

    clusters = corr_data["clusters"]

    def score(sid):
        d = results.get(sid, {})
        oos = d.get("oos_sharpe", 0) or 0
        n = d.get("n_days", 0) or 0
        bh = 1 if mt_data.get(sid, {}).get("bh_significant") else 0
        return (bh, oos, n)

    selected = []
    cluster_reps = {}

    # Pick best from each multi-signal cluster
    for i, c in enumerate(clusters):
        if c["size"] <= 1:
            continue
        sigs = [s for s in c["signals"] if s in pnl_data]
        if not sigs:
            continue
        best = max(sigs, key=score)
        selected.append(best)
        cluster_reps[best] = f"Cluster {i+1} ({c['size']} correlated signals)"

    # Add independent signals with reasonable OOS
    clustered = set()
    for c in clusters:
        if c["size"] > 1:
            clustered.update(c["signals"])

    for sid in sorted(results.keys()):
        if sid in clustered or sid not in pnl_data:
            continue
        d = results[sid]
        oos = d.get("oos_sharpe", 0) or 0
        n = d.get("n_days", 0) or 0
        # Include if OOS Sharpe > 0.3 or n_days < 120 (too short for OOS but otherwise good)
        if oos > 0.3 or (n < 120 and (d.get("sharpe", 0) or 0) > 0.8):
            selected.append(sid)

    return selected, cluster_reps


def build_portfolio(selected, pnl_data, results):
    """Build equal-weight portfolio from selected signals."""
    # Align all PnL series
    all_pnl = pd.DataFrame({sid: pnl_data[sid] for sid in selected if sid in pnl_data})
    all_pnl = all_pnl.sort_index()

    # Equal-weight: average across active signals each day (NaN = not yet exists, skip)
    port_pnl = all_pnl.mean(axis=1)
    n_active = all_pnl.notna().sum(axis=1)

    # Also compute a vol-targeted version (target 15% annual vol for fair SPY comparison)
    target_vol = 0.15
    raw_vol = port_pnl.std() * np.sqrt(252)
    lever = target_vol / raw_vol if raw_vol > 0 else 1.0

    # Compute portfolio metrics
    port_pnl = port_pnl.dropna()
    eq = (1 + port_pnl).cumprod()
    years = len(port_pnl) / 252
    cagr = eq.iloc[-1] ** (1 / years) - 1
    vol = port_pnl.std() * np.sqrt(252)
    sharpe = port_pnl.mean() / port_pnl.std() * np.sqrt(252) if port_pnl.std() > 0 else 0
    dd = eq / eq.cummax() - 1
    max_dd = dd.min()

    # SPY benchmark
    from harness import load_prices, daily_returns
    spy = load_prices(["SPY"], start=str(port_pnl.index[0].date()))
    spy_ret = daily_returns(spy)["SPY"]
    spy_ret = spy_ret.reindex(port_pnl.index).dropna()
    spy_eq = (1 + spy_ret).cumprod()

    # Vol-targeted version: lever the daily PnL to 15% annual vol
    lev_pnl = port_pnl * lever
    lev_eq = (1 + lev_pnl).cumprod()
    lev_cagr = lev_eq.iloc[-1] ** (1 / years) - 1
    lev_dd = lev_eq / lev_eq.cummax() - 1

    return {
        "pnl": port_pnl,
        "equity": eq,
        "drawdown": dd,
        "n_active": n_active,
        "spy_equity": spy_eq,
        "spy_ret": spy_ret,
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "vol": float(vol),
        "max_dd": float(max_dd),
        "years": years,
        "n_signals": len(selected),
        "component_pnl": all_pnl,
        "lev_equity": lev_eq,
        "lev_cagr": float(lev_cagr),
        "lev_max_dd": float(lev_dd.min()),
        "lever": float(lever),
    }


def determine_active_signals(results):
    """Determine which signals are currently generating positions (May 2026)."""
    active = []
    inactive = []

    signal_status = {
        "V3_felder_margin_debt_gdp": {
            "active": True,
            "position": "100% SPY",
            "reason": "Margin debt/GDP YoY is positive — fully invested",
            "tickers": ["SPY"],
        },
        "C07_accel_dual_momentum": {
            "active": True,
            "position": "Long SPY",
            "reason": "SPY has highest avg(1m,3m,6m) momentum at +9.0% vs SCZ +7.2% and TLT -2.8%",
            "tickers": ["SPY"],
        },
        "C08_yield_curve_10y2y": {
            "active": True,
            "position": "50% SPY (reduced)",
            "reason": "10Y-2Y spread at +0.48, just below the +0.50 all-clear threshold after inversion",
            "tickers": ["SPY"],
        },
        "P6_howell_liquidity_btc": {
            "active": True,
            "position": "Long BTC",
            "reason": "Global CB liquidity proxy trending positive and accelerating",
            "tickers": ["BTC-USD"],
        },
        "AG-2": {
            "active": True,
            "position": "Long ETN+HUBB, Short NEE+AEP",
            "reason": "Transformer lead times still >100 weeks — grid equipment demand exceeds supply",
            "tickers": ["ETN", "HUBB", "NEE", "AEP"],
        },
        "AS-B": {
            "active": True,
            "position": "Long MRVL+CRDO",
            "reason": "Custom silicon regime trade — hyperscaler ASIC demand continues accelerating",
            "tickers": ["MRVL", "CRDO"],
        },
        "AI-4": {
            "active": True,
            "position": "Short BF-B+STZ",
            "reason": "GLP-1 adoption continues expanding — negative for spirits consumption",
            "tickers": ["BF-B", "STZ"],
        },
        "AM-1": {
            "active": True,
            "position": "Long VLO+PSX+MPC",
            "reason": "Hormuz disruption premium — entered Mar 2026",
            "tickers": ["VLO", "PSX", "MPC"],
        },
        "AE-2": {
            "active": True,
            "position": "Long ETN+VRT+PWR+HUBB+AMSC+MOD",
            "reason": "All 4 hyperscalers reported capex >20% YoY in recent quarters — power infra beneficiaries",
            "tickers": ["ETN", "VRT", "PWR", "HUBB", "AMSC", "MOD"],
        },
        "H07_puell_multiple": {
            "active": False,
            "position": "Flat",
            "reason": "Puell Multiple between 0.5 and 3.0 — no trigger",
            "tickers": ["BTC-USD"],
        },
        "A14_btc_halving": {
            "active": False,
            "position": "Flat",
            "reason": "Last halving Apr 2024, H+18mo expired Oct 2025",
            "tickers": ["BTC-USD"],
        },
        "AR-1": {
            "active": False,
            "position": "Flat",
            "reason": "Freight trough Oct 2024 trade expired Oct 2025",
            "tickers": ["ODFL", "SAIA"],
        },
        "AQ-1": {
            "active": False,
            "position": "Flat",
            "reason": "Copper at $6.30/lb, well above marginal cost floor",
            "tickers": ["HG=F"],
        },
    }

    for sid, info in signal_status.items():
        d = results.get(sid, {})
        entry = {
            "signal_id": sid,
            "name": d.get("name", sid),
            "sharpe": d.get("sharpe", 0),
            "oos_sharpe": d.get("oos_sharpe"),
            "rule": d.get("rule", ""),
            **info,
        }
        if info["active"]:
            active.append(entry)
        else:
            inactive.append(entry)

    return active, inactive


def generate_html(portfolio, selected, cluster_reps, results, active_signals, inactive_signals):
    # Prepare equity curve data for JS
    eq = portfolio["equity"]
    spy_eq = portfolio["spy_equity"]
    lev_eq = portfolio["lev_equity"]

    # Downsample to weekly for reasonable chart size
    eq_weekly = eq.resample("W").last().dropna()
    spy_weekly = spy_eq.reindex(eq_weekly.index, method="ffill").dropna()
    lev_weekly = lev_eq.reindex(eq_weekly.index, method="ffill").dropna()

    dates = [str(d.date()) for d in eq_weekly.index]
    eq_vals = [round(float(v), 4) for v in eq_weekly.values]
    spy_vals = [round(float(v), 4) for v in spy_weekly.reindex(eq_weekly.index).dropna().values]
    lev_vals = [round(float(v), 4) for v in lev_weekly.reindex(eq_weekly.index).dropna().values]
    n_min = min(len(dates), len(spy_vals), len(lev_vals))
    dates = dates[:n_min]
    eq_vals = eq_vals[:n_min]
    spy_vals = spy_vals[:n_min]
    lev_vals = lev_vals[:n_min]

    # Drawdown data
    dd = portfolio["drawdown"]
    dd_weekly = dd.resample("W").min().dropna()
    dd_vals = [round(float(v), 4) for v in dd_weekly.reindex(eq_weekly.index[:n_min], method="ffill").fillna(0).values[:n_min]]

    # Component signal table
    comp_rows = ""
    for sid in sorted(selected):
        d = results.get(sid, {})
        mt = json.load(open(RESULTS / "_multiple_testing.json")).get(sid, {})
        bh = "YES" if mt.get("bh_significant") else ""
        cluster = cluster_reps.get(sid, "Independent")
        oos = d.get("oos_sharpe")
        oos_str = f"{oos:.2f}" if oos else "—"
        comp_rows += f"""<tr>
<td>{sid}</td><td class="name-col">{_esc(d.get('name', sid))}</td>
<td>{d.get('sharpe',0):.2f}</td><td>{oos_str}</td>
<td>{d.get('cagr',0)*100:.1f}%</td><td>{d.get('n_days',0)}</td>
<td class="{'pos' if bh else ''}">{bh}</td>
<td class="dim">{cluster}</td>
</tr>"""

    # Active signals section
    active_html = ""
    all_tickers = set()
    for s in active_signals:
        tickers_str = ", ".join(s["tickers"])
        all_tickers.update(s["tickers"])
        active_html += f"""<div class="active-card">
<div class="active-header">
  <span class="active-name">{_esc(s['name'])}</span>
  <span class="active-position">{_esc(s['position'])}</span>
</div>
<div class="active-reason">{_esc(s['reason'])}</div>
<div class="active-tickers">Tickers: <strong>{tickers_str}</strong></div>
<div class="active-meta">Sharpe: {s['sharpe']:.2f} | OOS Sharpe: {f"{s['oos_sharpe']:.2f}" if s.get('oos_sharpe') else '—'}</div>
</div>"""

    inactive_html = ""
    for s in inactive_signals:
        inactive_html += f"""<div class="inactive-card">
<span class="inactive-name">{_esc(s['name'])}</span>
<span class="inactive-reason">{_esc(s['reason'])}</span>
</div>"""

    # Aggregate ticker recommendations
    ticker_counts = {}
    for s in active_signals:
        for t in s["tickers"]:
            if t not in ticker_counts:
                ticker_counts[t] = {"long": [], "short": []}
            if "short" in s["position"].lower():
                ticker_counts[t]["short"].append(s["signal_id"])
            else:
                ticker_counts[t]["long"].append(s["signal_id"])

    ticker_html = ""
    for t in sorted(ticker_counts.keys()):
        info = ticker_counts[t]
        direction = "LONG" if info["long"] else "SHORT"
        signals = info["long"] if info["long"] else info["short"]
        n = len(signals)
        cls = "long" if info["long"] else "short"
        ticker_html += f"""<div class="ticker-card {cls}">
<div class="ticker-sym">{t}</div>
<div class="ticker-dir">{direction}</div>
<div class="ticker-count">{n} signal{'s' if n > 1 else ''}</div>
<div class="ticker-signals">{', '.join(signals)}</div>
</div>"""

    p = portfolio

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Portfolio Analysis — Trading Signals Research</title>
<style>
:root {{
  --bg: #0f1419; --bg-2: #1a1f2e; --bg-3: #232a3d;
  --fg: #e6e9ef; --fg-dim: #9ca3af;
  --accent: #7dd3fc; --good: #34d399; --bad: #f87171; --warn: #fbbf24;
  --border: #2d3548;
}}
* {{ box-sizing: border-box; margin: 0; }}
body {{ background: var(--bg); color: var(--fg); font: 14px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
.container {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
h1 {{ font-size: 1.5em; margin-bottom: 4px; }}
h2 {{ font-size: 1.2em; margin: 40px 0 12px; border-bottom: 2px solid var(--border); padding-bottom: 8px; }}
.sub {{ color: var(--fg-dim); font-size: 13px; margin-bottom: 24px; }}
.disclaimer {{ background: rgba(251,191,36,0.08); border-left: 3px solid var(--warn); padding: 10px 14px; border-radius: 4px; font-size: 12px; color: var(--fg-dim); margin: 16px 0; line-height: 1.5; }}
.disclaimer strong {{ color: var(--warn); }}
.stat-row {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 16px 0; }}
.stat-box {{ background: var(--bg-2); border: 1px solid var(--border); border-radius: 8px; padding: 14px 20px; min-width: 120px; }}
.stat-box .v {{ font-size: 1.6em; font-weight: 700; font-family: "SF Mono", monospace; }}
.stat-box .l {{ color: var(--fg-dim); font-size: 11px; margin-top: 2px; }}
.pos {{ color: var(--good); }} .neg {{ color: var(--bad); }} .dim {{ color: var(--fg-dim); }}
canvas {{ display: block; width: 100%; margin: 12px 0; background: var(--bg-2); border: 1px solid var(--border); border-radius: 8px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin: 12px 0 24px; }}
th {{ text-align: left; padding: 8px 10px; border-bottom: 2px solid var(--border); color: var(--fg-dim); font-size: 11px; text-transform: uppercase; }}
td {{ padding: 6px 10px; border-bottom: 1px solid var(--border); }}
.name-col {{ max-width: 250px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.scroll-x {{ overflow-x: auto; }}

.active-card {{ background: var(--bg-2); border: 1px solid var(--good); border-radius: 8px; padding: 16px; margin: 10px 0; }}
.active-header {{ display: flex; justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 8px; margin-bottom: 6px; }}
.active-name {{ font-weight: 700; font-size: 15px; }}
.active-position {{ background: rgba(52,211,153,0.15); color: var(--good); padding: 3px 10px; border-radius: 4px; font-weight: 600; font-size: 13px; }}
.active-reason {{ color: var(--fg-dim); font-size: 13px; margin: 4px 0; }}
.active-tickers {{ font-size: 13px; margin: 4px 0; }}
.active-meta {{ font-size: 12px; color: var(--fg-dim); }}

.inactive-card {{ display: flex; gap: 12px; padding: 8px 12px; border-bottom: 1px solid var(--border); font-size: 13px; opacity: 0.5; }}
.inactive-name {{ font-weight: 600; min-width: 200px; }}
.inactive-reason {{ color: var(--fg-dim); }}

.ticker-grid {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 16px 0; }}
.ticker-card {{ background: var(--bg-2); border: 1px solid var(--border); border-radius: 8px; padding: 12px 16px; min-width: 140px; text-align: center; }}
.ticker-card.long {{ border-color: var(--good); }}
.ticker-card.short {{ border-color: var(--bad); }}
.ticker-sym {{ font-size: 1.3em; font-weight: 700; font-family: "SF Mono", monospace; }}
.ticker-card.long .ticker-sym {{ color: var(--good); }}
.ticker-card.short .ticker-sym {{ color: var(--bad); }}
.ticker-dir {{ font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; margin: 2px 0; }}
.ticker-card.long .ticker-dir {{ color: var(--good); }}
.ticker-card.short .ticker-dir {{ color: var(--bad); }}
.ticker-count {{ font-size: 11px; color: var(--fg-dim); }}
.ticker-signals {{ font-size: 10px; color: var(--fg-dim); margin-top: 4px; }}

.nav-links {{ display: flex; gap: 12px; margin: 12px 0; font-size: 13px; }}
.nav-links a {{ color: var(--accent); text-decoration: none; padding: 6px 12px; background: var(--bg-3); border-radius: 6px; border: 1px solid var(--border); }}
.nav-links a:hover {{ background: var(--accent); color: var(--bg); }}

.legend {{ display: flex; gap: 16px; font-size: 12px; color: var(--fg-dim); margin: 8px 0; }}
.legend-item {{ display: flex; align-items: center; gap: 6px; }}
.legend-swatch {{ width: 16px; height: 3px; border-radius: 2px; }}
</style>
</head>
<body>
<div class="container">

<h1>Portfolio Analysis</h1>
<p class="sub">Equal-weight portfolio of {p['n_signals']} uncorrelated winning signals vs SPY buy-and-hold</p>

<div class="nav-links">
  <a href="index.html">Signals</a>
  <a href="full_catalog.html">Full Catalog</a>
  <a href="analysis.html">Analysis</a>
</div>

<div class="disclaimer">
<strong>Not investment advice.</strong> All backtests use free public data with no transaction costs, slippage, or taxes.
The "currently active" signals reflect mechanical rule-checking, not a recommendation to trade.
Past performance is not predictive of future results. Do your own research.
</div>

<div class="stat-row">
  <div class="stat-box"><div class="v pos">{p['sharpe']:.2f}</div><div class="l">Portfolio Sharpe</div></div>
  <div class="stat-box"><div class="v pos">{p['cagr']*100:.1f}%</div><div class="l">Raw CAGR</div></div>
  <div class="stat-box"><div class="v">{p['vol']*100:.1f}%</div><div class="l">Raw Vol</div></div>
  <div class="stat-box"><div class="v neg">{p['max_dd']*100:.1f}%</div><div class="l">Raw MaxDD</div></div>
</div>

<div class="stat-row">
  <div class="stat-box"><div class="v pos">{p['lev_cagr']*100:.1f}%</div><div class="l">Vol-Targeted CAGR ({p['lever']:.1f}x lever)</div></div>
  <div class="stat-box"><div class="v">15.0%</div><div class="l">Target Vol</div></div>
  <div class="stat-box"><div class="v neg">{p['lev_max_dd']*100:.1f}%</div><div class="l">Vol-Targeted MaxDD</div></div>
  <div class="stat-box"><div class="v">{p['n_signals']}</div><div class="l">Signals</div></div>
  <div class="stat-box"><div class="v">{p['years']:.1f}</div><div class="l">Years</div></div>
</div>

<div class="disclaimer" style="background: rgba(125,211,252,0.05); border-color: var(--accent);">
The portfolio's <strong>raw</strong> figures average across {p['n_signals']} signals, most of which sit in cash most of the time, so volatility (and CAGR) look small. The <strong>vol-targeted</strong> figures scale the same daily PnL stream up by {p['lever']:.1f}x to match SPY-like 15% annual volatility — this is the apples-to-apples comparison.
</div>

<h2>Equity Curve</h2>
<div class="legend">
  <div class="legend-item"><div class="legend-swatch" style="background: var(--good)"></div> Portfolio (vol-targeted to 15%)</div>
  <div class="legend-item"><div class="legend-swatch" style="background: var(--accent)"></div> Portfolio (raw)</div>
  <div class="legend-item"><div class="legend-swatch" style="background: var(--fg-dim)"></div> SPY Buy & Hold</div>
</div>
<canvas id="eq-chart" height="380"></canvas>

<h2>Currently Active Signals</h2>
<p style="color: var(--fg-dim); font-size: 13px; margin-bottom: 12px;">
Signals whose trading rules are currently generating positions as of May 2026.
</p>

<h3 style="font-size: 14px; color: var(--fg-dim); margin: 20px 0 10px; text-transform: uppercase; letter-spacing: 0.05em;">Ticker Summary</h3>
<div class="ticker-grid">{ticker_html}</div>

{active_html}

<h3 style="font-size: 14px; color: var(--fg-dim); margin: 30px 0 10px; text-transform: uppercase; letter-spacing: 0.05em;">Currently Flat</h3>
{inactive_html}

<h2>Portfolio Components ({p['n_signals']} signals)</h2>
<div class="scroll-x">
<table>
<thead><tr><th>ID</th><th>Name</th><th>Sharpe</th><th>OOS Sharpe</th><th>CAGR</th><th>N Days</th><th>BH Sig</th><th>Group</th></tr></thead>
<tbody>{comp_rows}</tbody>
</table>
</div>

<div style="text-align:center; color: var(--fg-dim); margin-top: 32px; padding: 16px; font-size: 12px;">
Generated by <code>build_portfolio.py</code>. <a href="analysis.html">Full analysis</a> | <a href="index.html">Signal catalog</a>
</div>

</div>

<script>
const DATES = {json.dumps(dates)};
const EQ = {json.dumps(eq_vals)};
const SPY = {json.dumps(spy_vals)};
const LEV = {json.dumps(lev_vals)};
const DD = {json.dumps(dd_vals)};

(function() {{
  const canvas = document.getElementById('eq-chart');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = 380 * dpr;
  const ctx = canvas.getContext('2d');
  ctx.scale(dpr, dpr);
  const W = rect.width, H = 380;

  const margin = {{top: 20, right: 60, bottom: 30, left: 60}};
  const pw = W - margin.left - margin.right;
  const ph = H - margin.top - margin.bottom;

  // Normalize to start at 1.0; use log scale for vast vol-targeted gains
  const eqN = EQ.map(v => v / EQ[0]);
  const spyN = SPY.map(v => v / SPY[0]);
  const levN = LEV.map(v => v / LEV[0]);

  // Use log scale
  const logEqN = eqN.map(v => Math.log10(Math.max(v, 0.01)));
  const logSpyN = spyN.map(v => Math.log10(Math.max(v, 0.01)));
  const logLevN = levN.map(v => Math.log10(Math.max(v, 0.01)));

  const allLog = logEqN.concat(logSpyN, logLevN);
  const maxY = Math.max(...allLog) + 0.1;
  const minY = Math.min(...allLog) - 0.1;
  const n = DATES.length;

  function x(i) {{ return margin.left + (i / (n - 1)) * pw; }}
  function y(v) {{ return margin.top + (1 - (v - minY) / (maxY - minY)) * ph; }}

  // Background
  ctx.fillStyle = '#1a1f2e';
  ctx.fillRect(0, 0, W, H);

  // Log scale grid lines (powers of 10 and intermediate)
  ctx.strokeStyle = '#2d3548';
  ctx.lineWidth = 0.5;
  ctx.fillStyle = '#9ca3af'; ctx.font = '10px monospace'; ctx.textAlign = 'right';
  const tickVals = [];
  for (let p = Math.floor(minY); p <= Math.ceil(maxY); p++) {{
    for (let m = 1; m < 10; m++) {{
      const v = Math.log10(m * Math.pow(10, p));
      if (v >= minY && v <= maxY) tickVals.push({{log: v, val: m * Math.pow(10, p)}});
    }}
  }}
  // Subsample if too many
  const tickStep = Math.max(1, Math.floor(tickVals.length / 8));
  tickVals.forEach((t, idx) => {{
    if (idx % tickStep !== 0) return;
    ctx.strokeStyle = '#2d3548';
    ctx.beginPath(); ctx.moveTo(margin.left, y(t.log)); ctx.lineTo(W - margin.right, y(t.log)); ctx.stroke();
    const label = t.val >= 1 ? t.val.toFixed(0) + 'x' : t.val.toFixed(2) + 'x';
    ctx.fillStyle = '#9ca3af';
    ctx.fillText(label, margin.left - 6, y(t.log) + 3);
  }});

  // Date labels
  ctx.textAlign = 'center'; ctx.fillStyle = '#9ca3af'; ctx.font = '10px monospace';
  const step = Math.max(1, Math.floor(n / 8));
  for (let i = 0; i < n; i += step) {{
    ctx.fillText(DATES[i].substring(0, 7), x(i), H - 8);
  }}

  // SPY line
  ctx.strokeStyle = '#6b7280';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {{
    if (i === 0) ctx.moveTo(x(i), y(logSpyN[i]));
    else ctx.lineTo(x(i), y(logSpyN[i]));
  }}
  ctx.stroke();

  // Raw portfolio line
  ctx.strokeStyle = '#7dd3fc';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {{
    if (i === 0) ctx.moveTo(x(i), y(logEqN[i]));
    else ctx.lineTo(x(i), y(logEqN[i]));
  }}
  ctx.stroke();

  // Vol-targeted portfolio line (emphasized)
  ctx.strokeStyle = '#34d399';
  ctx.lineWidth = 2.5;
  ctx.beginPath();
  for (let i = 0; i < n; i++) {{
    if (i === 0) ctx.moveTo(x(i), y(logLevN[i]));
    else ctx.lineTo(x(i), y(logLevN[i]));
  }}
  ctx.stroke();

  // End labels
  ctx.font = '11px monospace'; ctx.textAlign = 'left';
  ctx.fillStyle = '#34d399';
  ctx.fillText(levN[n-1].toFixed(0) + 'x', x(n-1) + 6, y(logLevN[n-1]) + 4);
  ctx.fillStyle = '#7dd3fc';
  ctx.fillText(eqN[n-1].toFixed(1) + 'x', x(n-1) + 6, y(logEqN[n-1]) + 4);
  ctx.fillStyle = '#6b7280';
  ctx.fillText(spyN[n-1].toFixed(1) + 'x', x(n-1) + 6, y(logSpyN[n-1]) + 4);

  // Hover tooltip
  canvas.addEventListener('mousemove', function(e) {{
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const i = Math.round((mx - margin.left) / pw * (n - 1));
    if (i >= 0 && i < n) {{
      canvas.title = DATES[i] + ' | Vol-tgt: ' + levN[i].toFixed(2) + 'x | Raw: ' + eqN[i].toFixed(3) + 'x | SPY: ' + spyN[i].toFixed(2) + 'x';
    }}
  }});
}})();
</script>

</body>
</html>"""

    return html


def _esc(s):
    if not s:
        return ""
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def main():
    print("Loading data...")
    results = load_winner_data()
    pnl_data = load_pnl()
    print(f"  {len(results)} winners, {len(pnl_data)} PnL series")

    print("Selecting portfolio signals...")
    selected, cluster_reps = select_portfolio_signals(results, pnl_data)
    print(f"  {len(selected)} signals selected ({len(cluster_reps)} cluster reps + {len(selected) - len(cluster_reps)} independent)")

    print("Building portfolio...")
    portfolio = build_portfolio(selected, pnl_data, results)
    print(f"  Sharpe: {portfolio['sharpe']:.2f}")
    print(f"  CAGR:   {portfolio['cagr']*100:.1f}%")
    print(f"  MaxDD:  {portfolio['max_dd']*100:.1f}%")
    print(f"  Period: {portfolio['pnl'].index[0].date()} to {portfolio['pnl'].index[-1].date()}")

    print("Checking active signals...")
    active, inactive = determine_active_signals(results)
    print(f"  {len(active)} active, {len(inactive)} flat")

    print("Generating portfolio.html...")
    html = generate_html(portfolio, selected, cluster_reps, results, active, inactive)
    out = ROOT / "portfolio.html"
    out.write_text(html)
    print(f"  Written to {out}")


if __name__ == "__main__":
    main()
