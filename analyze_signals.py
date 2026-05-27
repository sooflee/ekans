"""
Signal analysis: multiple-testing correction, transaction costs, and correlation matrix.

Usage:
    .venv/bin/python analyze_signals.py                # full analysis (MT + rerun winners + correlation)
    .venv/bin/python analyze_signals.py --mt-only       # just multiple-testing correction (fast, no reruns)
    .venv/bin/python analyze_signals.py --rerun-only    # rerun winners and compute cost/IS-OOS/correlation

Outputs:
    results/_multiple_testing.json   — BH-adjusted p-values for all signals
    results/_correlation_matrix.json — pairwise correlation of winner PnL streams
    analysis.html                    — self-contained HTML report
"""

import json
import sys
import importlib
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
BACKTESTS = ROOT / "backtests"
PNL_DIR = RESULTS / "pnl"

sys.path.insert(0, str(BACKTESTS))


# ───────────────────────────────────────────────────────────────────
# 1. Multiple-testing correction (Benjamini-Hochberg)
# ───────────────────────────────────────────────────────────────────

def load_all_results():
    out = {}
    for fp in sorted(RESULTS.glob("*.json")):
        if fp.stem.startswith("_"):
            continue
        try:
            with open(fp) as f:
                d = json.load(f)
            out[fp.stem] = d
        except Exception:
            continue
    return out


def multiple_testing_correction(results, fdr=0.05):
    signals = []
    for sid, d in results.items():
        if d.get("status") != "ok":
            continue
        t = d.get("t_stat")
        n = d.get("n_days")
        if t is None or n is None or n < 30:
            continue
        p = 2 * (1 - sp_stats.t.cdf(abs(t), df=n - 1))
        signals.append({"signal_id": sid, "t_stat": t, "n_days": n, "p_value": p})

    signals.sort(key=lambda x: x["p_value"])
    m = len(signals)

    for i, s in enumerate(signals):
        rank = i + 1
        bh_threshold = fdr * rank / m
        s["bh_rank"] = rank
        s["bh_threshold"] = bh_threshold

    # BH: find largest k where p(k) <= fdr * k / m, then reject all i <= k
    max_reject = 0
    for i, s in enumerate(signals):
        if s["p_value"] <= s["bh_threshold"]:
            max_reject = i + 1

    for i, s in enumerate(signals):
        s["bh_significant"] = (i + 1) <= max_reject

    result = {s["signal_id"]: s for s in signals}
    return result, m, max_reject


# ───────────────────────────────────────────────────────────────────
# 2. Batch-rerun winners: capture PnL, compute costs + IS/OOS
# ───────────────────────────────────────────────────────────────────

def find_winner_ids(results):
    winners = []
    for sid, d in results.items():
        if d.get("status") != "ok":
            continue
        sharpe = d.get("sharpe", 0) or 0
        cagr = d.get("cagr", 0) or 0
        if sharpe > 0.5 and cagr > 0.10:
            winners.append(sid)
    return sorted(winners)


def find_backtest_file(signal_id):
    """Map signal_id to its backtest .py file."""
    # Try exact match first
    exact = BACKTESTS / f"{signal_id}.py"
    if exact.exists():
        return exact
    # Try case variations and prefix matches
    sid_lower = signal_id.lower().replace("-", "").replace("_", "")
    for fp in BACKTESTS.glob("*.py"):
        if fp.stem.startswith("_") or fp.stem == "harness":
            continue
        stem_lower = fp.stem.lower().replace("-", "").replace("_", "")
        if stem_lower == sid_lower:
            return fp
        # Try matching the prefix+number pattern (AC-4 -> AC4_...)
        norm_sid = signal_id.replace("-", "")
        if fp.stem.startswith(norm_sid + "_") or fp.stem == norm_sid:
            return fp
    return None


def rerun_winners_collect_pnl(winner_ids):
    """Re-run each winner's backtest, capturing PnL via monkey-patching.

    The key challenge: backtests do `from harness import compute_metrics`, which copies
    the function reference at import time. Patching `harness.compute_metrics` after import
    doesn't affect the local name. So we force a fresh import each run with our patches
    already installed on the module before the backtest's import copies them.
    """
    import runpy
    import backtests.harness as harness_mod

    collected_pnl = {}
    collected_positions = {}
    updated_metrics = {}

    orig_compute = harness_mod.compute_metrics.__wrapped__ if hasattr(harness_mod.compute_metrics, '__wrapped__') else harness_mod.compute_metrics
    orig_save = harness_mod.save_result
    orig_print = harness_mod.print_metrics
    orig_lsp = harness_mod.long_short_pnl

    for sid in winner_ids:
        bt_file = find_backtest_file(sid)
        if bt_file is None:
            print(f"  SKIP {sid}: no backtest file found")
            continue

        _captured = {"pnl": None, "positions": None, "metrics": None}

        def _patched_compute(pnl, benchmark=None, name="Strategy", positions=None, cost_bps=10,
                             _cap=_captured):
            _cap["pnl"] = pnl.copy()
            if positions is not None:
                _cap["positions"] = positions.copy()
            result = orig_compute(pnl, benchmark=benchmark, name=name,
                                  positions=positions, cost_bps=cost_bps)
            _cap["metrics"] = result
            return result

        def _patched_lsp(positions, returns, _cap=_captured):
            _cap["positions"] = positions.copy()
            return orig_lsp(positions, returns)

        def _patched_save(signal_id, metrics, extra=None, pnl=None):
            return None

        def _patched_print(m):
            pass

        # Patch the module BEFORE running the backtest, so `from harness import X`
        # picks up our patched versions. We also force-remove 'harness' from
        # sys.modules so the backtest's import triggers a fresh lookup that finds
        # our pre-patched module.
        harness_mod.compute_metrics = _patched_compute
        harness_mod.save_result = _patched_save
        harness_mod.print_metrics = _patched_print
        harness_mod.long_short_pnl = _patched_lsp

        # Clear any previously cached 'harness' module so the backtest's
        # `from harness import ...` re-imports and gets our patched functions.
        saved_modules = {}
        for key in list(sys.modules.keys()):
            if key == "harness" or (key.startswith("backtests") and key != "backtests.harness"):
                saved_modules[key] = sys.modules.pop(key)
        sys.modules["harness"] = harness_mod

        try:
            runpy.run_path(str(bt_file), run_name="__main__")

            if _captured["pnl"] is not None:
                pnl_series = _captured["pnl"].dropna()
                collected_pnl[sid] = pnl_series

                PNL_DIR.mkdir(exist_ok=True)
                pnl_series.to_frame("pnl").to_parquet(PNL_DIR / f"{sid}.parquet")

                positions = _captured["positions"]
                if positions is None:
                    # Estimate positions from PnL: in-position when PnL != 0
                    positions = (pnl_series != 0).astype(float)
                collected_positions[sid] = positions
                m = orig_compute(pnl_series, name=sid,
                                 positions=positions, cost_bps=10)
                updated_metrics[sid] = m

                print(f"  OK   {sid}: {len(pnl_series)} days captured")
            else:
                print(f"  MISS {sid}: no PnL captured (backtest may use non-standard flow)")
        except Exception as e:
            print(f"  FAIL {sid}: {e}")
        finally:
            harness_mod.compute_metrics = orig_compute
            harness_mod.save_result = orig_save
            harness_mod.print_metrics = orig_print
            harness_mod.long_short_pnl = orig_lsp
            # Restore module cache
            sys.modules.pop("harness", None)
            for key, mod in saved_modules.items():
                sys.modules[key] = mod

    return collected_pnl, collected_positions, updated_metrics


# ───────────────────────────────────────────────────────────────────
# 3. Correlation matrix
# ───────────────────────────────────────────────────────────────────

def compute_correlation(pnl_dict, min_overlap=120):
    """Compute pairwise correlation matrix from daily PnL series."""
    if len(pnl_dict) < 2:
        return None

    sids = sorted(pnl_dict.keys())
    df = pd.DataFrame({sid: pnl_dict[sid] for sid in sids})
    df = df.sort_index()

    n = len(sids)
    corr_matrix = np.full((n, n), np.nan)

    for i in range(n):
        corr_matrix[i, i] = 1.0
        for j in range(i + 1, n):
            pair = df[[sids[i], sids[j]]].dropna()
            if len(pair) >= min_overlap:
                c = pair.corr().iloc[0, 1]
                corr_matrix[i, j] = c
                corr_matrix[j, i] = c

    # Cluster by average-linkage on (1 - corr) distance
    clusters = cluster_signals(sids, corr_matrix)

    return {
        "signal_ids": sids,
        "matrix": corr_matrix.tolist(),
        "clusters": clusters,
        "n_signals": n,
        "mean_pairwise_corr": float(np.nanmean(corr_matrix[np.triu_indices(n, k=1)])),
    }


def cluster_signals(sids, corr_matrix):
    """Simple agglomerative clustering: group signals with avg corr > 0.5."""
    n = len(sids)
    assigned = [False] * n
    clusters = []

    for i in range(n):
        if assigned[i]:
            continue
        group = [i]
        assigned[i] = True
        for j in range(i + 1, n):
            if assigned[j]:
                continue
            if not np.isnan(corr_matrix[i, j]) and corr_matrix[i, j] > 0.5:
                avg_corr_with_group = np.nanmean([corr_matrix[j, k] for k in group])
                if avg_corr_with_group > 0.4:
                    group.append(j)
                    assigned[j] = True
        clusters.append({
            "signals": [sids[k] for k in group],
            "size": len(group),
            "mean_intra_corr": float(np.nanmean([
                corr_matrix[a, b] for a in group for b in group if a != b
            ])) if len(group) > 1 else 1.0,
        })

    clusters.sort(key=lambda c: c["size"], reverse=True)
    return clusters


# ───────────────────────────────────────────────────────────────────
# 4. HTML report generation
# ───────────────────────────────────────────────────────────────────

def generate_html(mt_results, mt_total, mt_surviving, corr_results, updated_metrics, all_results):
    """Generate a self-contained analysis.html."""

    # --- Multiple testing table ---
    mt_rows = sorted(mt_results.values(), key=lambda x: x["p_value"])
    mt_sig = [r for r in mt_rows if r["bh_significant"]]
    mt_not = [r for r in mt_rows if not r["bh_significant"]]

    mt_html = ""
    for r in mt_rows:
        sig_cls = "sig" if r["bh_significant"] else "not-sig"
        # Look up CAGR and Sharpe from all_results
        d = all_results.get(r["signal_id"], {})
        cagr = d.get("cagr")
        sharpe = d.get("sharpe")
        name = d.get("name", r["signal_id"])
        mt_html += f"""<tr class="{sig_cls}">
<td>{r['signal_id']}</td><td class="name-col">{_esc(name)}</td>
<td>{r['t_stat']:.2f}</td><td>{r['n_days']}</td>
<td>{r['p_value']:.4f}</td><td>{r['bh_threshold']:.4f}</td>
<td class="verdict">{'YES' if r['bh_significant'] else 'no'}</td>
<td>{f'{sharpe:.2f}' if sharpe else ''}</td>
<td>{f'{cagr*100:.1f}%' if cagr else ''}</td>
</tr>"""

    # --- Cost impact table (for winners that were re-run) ---
    cost_html = ""
    for sid in sorted(updated_metrics.keys()):
        m = updated_metrics[sid]
        d = all_results.get(sid, {})
        if "net_sharpe" not in m:
            continue
        cost_html += f"""<tr>
<td>{sid}</td><td class="name-col">{_esc(m.get('name', sid))}</td>
<td>{m['sharpe']:.2f}</td><td>{m.get('net_sharpe',0):.2f}</td>
<td class="{'neg' if m.get('net_sharpe',0) < 0.5 else 'pos'}">{m['sharpe'] - m.get('net_sharpe',0):.2f}</td>
<td>{m['cagr']*100:.1f}%</td><td>{m.get('net_cagr',0)*100:.1f}%</td>
<td>{m.get('cost_drag',0)*100:.1f}%</td>
<td>{m.get('turnover_annual',0):.0f}</td>
</tr>"""

    # --- IS/OOS table ---
    isoos_html = ""
    for sid in sorted(updated_metrics.keys()):
        m = updated_metrics[sid]
        if "is_sharpe" not in m:
            continue
        decay = m.get("oos_sharpe", 0) - m.get("is_sharpe", 0)
        isoos_html += f"""<tr>
<td>{sid}</td><td class="name-col">{_esc(m.get('name', sid))}</td>
<td>{m.get('is_sharpe',0):.2f}</td><td>{m.get('is_cagr',0)*100:.1f}%</td>
<td>{m.get('is_start','')}&ndash;{m.get('is_end','')}</td>
<td>{m.get('oos_sharpe',0):.2f}</td><td>{m.get('oos_cagr',0)*100:.1f}%</td>
<td>{m.get('oos_start','')}&ndash;{m.get('oos_end','')}</td>
<td class="{'neg' if decay < -0.3 else 'pos' if decay > 0 else ''}">{decay:+.2f}</td>
</tr>"""

    # --- Correlation heatmap ---
    heatmap_html = ""
    corr_summary = ""
    if corr_results and corr_results["n_signals"] >= 2:
        sids = corr_results["signal_ids"]
        matrix = corr_results["matrix"]
        n = len(sids)

        # Short labels
        labels_json = json.dumps(sids)
        matrix_json = json.dumps(matrix)

        # Cluster summary
        clusters = corr_results["clusters"]
        multi_clusters = [c for c in clusters if c["size"] > 1]
        corr_summary = f"""
        <p><strong>Mean pairwise correlation:</strong> {corr_results['mean_pairwise_corr']:.3f}</p>
        <p><strong>Correlated groups (avg r &gt; 0.5):</strong> {len(multi_clusters)}</p>
        """
        for i, c in enumerate(multi_clusters):
            corr_summary += f"<p class='cluster'>Cluster {i+1} ({c['size']} signals, avg r={c['mean_intra_corr']:.2f}): {', '.join(c['signals'])}</p>"

        singleton_count = sum(1 for c in clusters if c["size"] == 1)
        if singleton_count:
            corr_summary += f"<p class='dim'>{singleton_count} independent signals (not correlated with any other winner)</p>"

        heatmap_html = f"""
        <div id="heatmap-container"></div>
        <script>
        (function() {{
            const labels = {labels_json};
            const matrix = {matrix_json};
            const n = labels.length;
            const cellSize = Math.min(Math.max(14, Math.floor(800 / n)), 40);
            const margin = 160;
            const size = cellSize * n + margin;

            const container = document.getElementById('heatmap-container');
            const canvas = document.createElement('canvas');
            canvas.width = size + margin;
            canvas.height = size + 20;
            canvas.style.maxWidth = '100%';
            container.appendChild(canvas);
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = '#0f1419';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            function corrColor(v) {{
                if (isNaN(v) || v === null) return '#1a1f2e';
                const abs = Math.abs(v);
                if (v > 0) {{
                    const r = Math.round(52 + abs * 100);
                    const g = Math.round(211 - abs * 80);
                    const b = Math.round(153 - abs * 50);
                    return `rgb(${{r}},${{g}},${{b}})`;
                }} else {{
                    const r = Math.round(248 - abs * 50);
                    const g = Math.round(113 + abs * 30);
                    const b = Math.round(113 + abs * 30);
                    return `rgb(${{r}},${{g}},${{b}})`;
                }}
            }}

            for (let i = 0; i < n; i++) {{
                for (let j = 0; j < n; j++) {{
                    ctx.fillStyle = corrColor(matrix[i][j]);
                    ctx.fillRect(margin + j * cellSize, i * cellSize, cellSize - 1, cellSize - 1);
                }}
            }}

            ctx.fillStyle = '#9ca3af';
            ctx.font = `${{Math.min(11, cellSize - 2)}}px monospace`;
            ctx.textAlign = 'right';
            for (let i = 0; i < n; i++) {{
                ctx.fillText(labels[i].substring(0, 20), margin - 4, i * cellSize + cellSize - 2);
            }}
            ctx.textAlign = 'left';
            ctx.save();
            for (let j = 0; j < n; j++) {{
                ctx.save();
                ctx.translate(margin + j * cellSize + cellSize - 2, n * cellSize + 4);
                ctx.rotate(Math.PI / 4);
                ctx.fillText(labels[j].substring(0, 20), 0, 0);
                ctx.restore();
            }}
            ctx.restore();

            // Legend
            const lx = margin + n * cellSize + 20;
            const ly = 10;
            const lw = 20, lh = 120;
            for (let i = 0; i < lh; i++) {{
                const v = 1 - 2 * i / lh;
                ctx.fillStyle = corrColor(v);
                ctx.fillRect(lx, ly + i, lw, 1);
            }}
            ctx.fillStyle = '#9ca3af';
            ctx.font = '10px monospace';
            ctx.textAlign = 'left';
            ctx.fillText('+1.0', lx + lw + 4, ly + 10);
            ctx.fillText(' 0.0', lx + lw + 4, ly + lh / 2 + 4);
            ctx.fillText('-1.0', lx + lw + 4, ly + lh);

            // Tooltip on hover
            canvas.addEventListener('mousemove', function(e) {{
                const rect = canvas.getBoundingClientRect();
                const scaleX = canvas.width / rect.width;
                const x = (e.clientX - rect.left) * scaleX - margin;
                const y = (e.clientY - rect.top) * scaleX;
                const j = Math.floor(x / cellSize);
                const i = Math.floor(y / cellSize);
                if (i >= 0 && i < n && j >= 0 && j < n) {{
                    const v = matrix[i][j];
                    const vStr = v !== null && !isNaN(v) ? v.toFixed(3) : 'N/A';
                    canvas.title = `${{labels[i]}} × ${{labels[j]}}: r = ${{vStr}}`;
                }} else {{
                    canvas.title = '';
                }}
            }});
        }})();
        </script>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Signal Analysis — Multiple Testing, Costs & Correlation</title>
<style>
:root {{
  --bg: #0f1419; --bg-2: #1a1f2e; --bg-3: #232a3d;
  --fg: #e6e9ef; --fg-dim: #9ca3af;
  --accent: #7dd3fc; --good: #34d399; --bad: #f87171; --warn: #fbbf24;
  --border: #2d3548;
}}
* {{ box-sizing: border-box; margin: 0; }}
body {{ background: var(--bg); color: var(--fg); font: 14px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
.container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
h1 {{ font-size: 1.5em; margin-bottom: 4px; }}
h2 {{ font-size: 1.2em; margin: 40px 0 12px; border-bottom: 2px solid var(--border); padding-bottom: 8px; }}
.sub {{ color: var(--fg-dim); font-size: 13px; margin-bottom: 24px; }}
.stat-row {{ display: flex; gap: 24px; flex-wrap: wrap; margin: 16px 0; }}
.stat-box {{ background: var(--bg-2); border: 1px solid var(--border); border-radius: 8px; padding: 14px 20px; min-width: 140px; }}
.stat-box .v {{ font-size: 1.8em; font-weight: 700; font-family: "SF Mono", monospace; }}
.stat-box .l {{ color: var(--fg-dim); font-size: 12px; margin-top: 2px; }}
.pos {{ color: var(--good); }} .neg {{ color: var(--bad); }} .dim {{ color: var(--fg-dim); }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; margin: 12px 0 24px; }}
th {{ text-align: left; padding: 8px 10px; border-bottom: 2px solid var(--border); color: var(--fg-dim); font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em; }}
td {{ padding: 6px 10px; border-bottom: 1px solid var(--border); }}
tr.sig {{ background: rgba(52,211,153,0.06); }}
tr.not-sig {{ opacity: 0.6; }}
.verdict {{ font-weight: 700; }}
tr.sig .verdict {{ color: var(--good); }}
tr.not-sig .verdict {{ color: var(--fg-dim); }}
.name-col {{ max-width: 200px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.cluster {{ margin: 4px 0; padding: 6px 10px; background: var(--bg-2); border-radius: 4px; font-size: 13px; }}
.scroll-x {{ overflow-x: auto; }}
canvas {{ display: block; margin: 12px 0; }}
.section-note {{ color: var(--fg-dim); font-size: 12px; margin: 8px 0 16px; line-height: 1.5; }}
</style>
</head>
<body>
<div class="container">

<h1>Signal Analysis Report</h1>
<p class="sub">Multiple-testing correction, transaction costs, in-sample/out-of-sample, and cross-signal correlation</p>

<!-- ===== Multiple Testing ===== -->
<h2>1. Multiple-Testing Correction (Benjamini-Hochberg)</h2>
<p class="section-note">
With {mt_total} signals tested, some will look significant by chance alone.
The BH procedure controls the False Discovery Rate at 5%: of the signals marked "YES",
at most 5% are expected to be false positives.
</p>

<div class="stat-row">
  <div class="stat-box"><div class="v">{mt_total}</div><div class="l">signals tested</div></div>
  <div class="stat-box"><div class="v pos">{mt_surviving}</div><div class="l">survive BH @ 5% FDR</div></div>
  <div class="stat-box"><div class="v neg">{mt_total - mt_surviving}</div><div class="l">rejected</div></div>
  <div class="stat-box"><div class="v">{mt_surviving/mt_total*100:.0f}%</div><div class="l">survival rate</div></div>
</div>

<div class="scroll-x">
<table>
<thead><tr><th>ID</th><th>Name</th><th>t-stat</th><th>N days</th><th>p-value</th><th>BH threshold</th><th>Survives?</th><th>Sharpe</th><th>CAGR</th></tr></thead>
<tbody>{mt_html}</tbody>
</table>
</div>

<!-- ===== Transaction Costs ===== -->
<h2>2. Transaction Cost Impact (10 bps round-trip)</h2>
<p class="section-note">
Gross vs net metrics after deducting 10 basis points per unit of turnover.
"Sharpe drop" = gross Sharpe minus net Sharpe. Signals with high turnover lose the most.
{f'<strong>{sum(1 for m in updated_metrics.values() if m.get("net_sharpe",999) < 0.5)} of {len([m for m in updated_metrics.values() if "net_sharpe" in m])} winners drop below Sharpe 0.5 after costs.</strong>' if cost_html else '(Re-run winners to populate this section.)'}
</p>

{"<div class='scroll-x'><table><thead><tr><th>ID</th><th>Name</th><th>Gross Sharpe</th><th>Net Sharpe</th><th>Sharpe Drop</th><th>Gross CAGR</th><th>Net CAGR</th><th>Cost Drag</th><th>Ann Turnover</th></tr></thead><tbody>" + cost_html + "</tbody></table></div>" if cost_html else "<p class='dim'>No cost data — run with --rerun to compute.</p>"}

<!-- ===== IS/OOS ===== -->
<h2>3. In-Sample vs Out-of-Sample</h2>
<p class="section-note">
Each backtest period is split at the midpoint. Signals that hold up in the second half
are more credible. "Sharpe decay" = OOS Sharpe minus IS Sharpe (negative = degradation).
</p>

{"<div class='scroll-x'><table><thead><tr><th>ID</th><th>Name</th><th>IS Sharpe</th><th>IS CAGR</th><th>IS Period</th><th>OOS Sharpe</th><th>OOS CAGR</th><th>OOS Period</th><th>Decay</th></tr></thead><tbody>" + isoos_html + "</tbody></table></div>" if isoos_html else "<p class='dim'>No IS/OOS data — run with --rerun to compute.</p>"}

<!-- ===== Correlation Matrix ===== -->
<h2>4. Winner Correlation Matrix</h2>
<p class="section-note">
Pairwise correlation of daily PnL across winning signals.
Highly correlated signals represent the same underlying bet — a portfolio treating them as
independent would overstate diversification.
</p>

{corr_summary if corr_summary else "<p class='dim'>No correlation data — run with --rerun to compute.</p>"}
{heatmap_html}

</div>
</body>
</html>"""

    return html


def _esc(s):
    if not s:
        return ""
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


# ───────────────────────────────────────────────────────────────────
# Main
# ───────────────────────────────────────────────────────────────────

def main():
    mt_only = "--mt-only" in sys.argv
    rerun_only = "--rerun-only" in sys.argv

    print("Loading results...")
    all_results = load_all_results()
    print(f"  {len(all_results)} result files loaded")

    # --- Multiple testing ---
    mt_results, mt_total, mt_surviving = {}, 0, 0
    if not rerun_only:
        print("\n=== Multiple-testing correction (Benjamini-Hochberg, FDR=5%) ===")
        mt_results, mt_total, mt_surviving = multiple_testing_correction(all_results)
        print(f"  {mt_total} signals with valid t-stats")
        print(f"  {mt_surviving} survive BH correction at 5% FDR ({mt_surviving/mt_total*100:.0f}%)")

        fp = RESULTS / "_multiple_testing.json"
        with open(fp, "w") as f:
            json.dump(mt_results, f, indent=2)
        print(f"  Saved to {fp}")

    # --- Rerun winners ---
    collected_pnl = {}
    updated_metrics = {}
    corr_results = None

    if not mt_only:
        winner_ids = find_winner_ids(all_results)
        print(f"\n=== Re-running {len(winner_ids)} winners for cost/IS-OOS/PnL ===")
        collected_pnl, collected_positions, updated_metrics = rerun_winners_collect_pnl(winner_ids)
        print(f"\n  Collected PnL for {len(collected_pnl)} signals")
        print(f"  Collected positions for {len(collected_positions)} signals (cost metrics available)")

        # Merge updated metrics back into result JSONs
        for sid, m in updated_metrics.items():
            fp = RESULTS / f"{sid}.json"
            if fp.exists():
                with open(fp) as f:
                    existing = json.load(f)
                for key in ["net_cagr", "net_sharpe", "net_max_dd", "turnover_annual",
                            "cost_drag", "cost_bps", "is_cagr", "is_sharpe", "is_start",
                            "is_end", "oos_cagr", "oos_sharpe", "oos_start", "oos_end"]:
                    if key in m:
                        existing[key] = m[key]
                with open(fp, "w") as f:
                    json.dump(existing, f, indent=2, default=str)

        # --- Correlation matrix ---
        if len(collected_pnl) >= 2:
            print(f"\n=== Computing correlation matrix ({len(collected_pnl)} signals) ===")
            corr_results = compute_correlation(collected_pnl)
            print(f"  Mean pairwise correlation: {corr_results['mean_pairwise_corr']:.3f}")

            multi = [c for c in corr_results["clusters"] if c["size"] > 1]
            print(f"  Correlated clusters (r > 0.5): {len(multi)}")
            for i, c in enumerate(multi):
                print(f"    Cluster {i+1}: {', '.join(c['signals'])} (avg r={c['mean_intra_corr']:.2f})")

            fp = RESULTS / "_correlation_matrix.json"
            with open(fp, "w") as f:
                json.dump(corr_results, f, indent=2)
            print(f"  Saved to {fp}")

    # --- Generate HTML report ---
    # Always load MT data for the report
    if not mt_results:
        mt_path = RESULTS / "_multiple_testing.json"
        if mt_path.exists():
            with open(mt_path) as f:
                mt_results = json.load(f)
            mt_total = len(mt_results)
            mt_surviving = sum(1 for v in mt_results.values() if v.get("bh_significant"))
        else:
            mt_results, mt_total, mt_surviving = multiple_testing_correction(all_results)

    print("\n=== Generating analysis.html ===")
    html = generate_html(mt_results, mt_total, mt_surviving, corr_results,
                         updated_metrics, all_results)
    out_path = ROOT / "analysis.html"
    out_path.write_text(html)
    print(f"  Written to {out_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
