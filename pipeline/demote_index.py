"""
Demote/restore index.html cards based on the canonical winner gate.

The High Return tab in index.html is append-only by the backtester loop. Over
time, signals that once cleared `sharpe > 0.5 AND cagr > 0.10` accumulate, even
after a tightened gate (BH-significance + OOS + sample size) would no longer
qualify them. This script keeps the front page honest:

  - For each `<article class="card">` block in the main body of index.html, find
    the signal_id via its `<code>backtests/<id>.py</code>` reference and look up
    `results/<id>.json` and `_multiple_testing.json`.

  - If the signal no longer passes `pipeline/winner_gate.is_winner`, rewrite the
    card's class to `card demoted` and inject a banner listing the failed gate
    criteria. The card's prose (mechanism / rule / caveats) is preserved — the
    sweep is reversible.

  - If a previously-demoted card now passes, restore it.

The sweep is idempotent. Re-running with the same inputs leaves index.html
byte-identical. Refresh `_multiple_testing.json` first with `refresh_bh.py` for
accurate results.

Usage:
    .venv/bin/python pipeline/refresh_bh.py
    .venv/bin/python pipeline/demote_index.py            # dry-run report
    .venv/bin/python pipeline/demote_index.py --apply    # rewrite index.html
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from winner_gate import is_winner, winner_reasons, load_mt_data

INDEX_PATH = ROOT / "index.html"
RESULTS_DIR = ROOT / "results"

MAIN_OPEN = '<main class="container">'
APPENDIX_MARK = "<!-- ================== APPENDIX ================== -->"

ARTICLE_OPEN_RE = re.compile(r'<article class="(card(?:\s+demoted)?)"([^>]*)>')
BACKTEST_REF_RE = re.compile(r"<code>backtests/([A-Za-z0-9_\-]+)\.py</code>")
BADGE_RE = re.compile(r'<span class="badge[^"]*">([^<]+)</span>')
DEMOTED_BANNER_RE = re.compile(
    r'\n  <div class="demoted-banner"[^>]*>.*?</div>', re.DOTALL
)

DEMOTED_CSS = """
/* ===== Demoted cards (no longer pass winner gate) ===== */
article.card.demoted {
  border-color: var(--warn);
  position: relative;
}
article.card.demoted .demoted-banner {
  background: rgba(251,191,36,0.08);
  border-left: 3px solid var(--warn);
  padding: 8px 12px;
  margin: 0 0 12px 0;
  border-radius: 4px;
  font-size: 12px;
  color: var(--fg-dim);
  line-height: 1.5;
}
article.card.demoted .demoted-banner strong { color: var(--warn); }
article.card.demoted .demoted-banner code {
  background: var(--bg-3); padding: 1px 5px; border-radius: 3px;
  font-size: 11px; color: var(--fg);
}
"""

DEMOTED_CSS_MARKER = "/* ===== Demoted cards (no longer pass winner gate) ===== */"


def split_into_cards(body: str) -> list[str]:
    """Return list of slices: [pre, card1, card2, ..., post] preserving exact bytes."""
    parts = re.split(r'(?=<article class="card)', body)
    return parts


def find_main_section(html: str) -> tuple[int, int]:
    main_start = html.find(MAIN_OPEN)
    if main_start < 0:
        raise SystemExit(f"could not find {MAIN_OPEN!r} in index.html")
    appendix = html.find(APPENDIX_MARK)
    if appendix < 0:
        raise SystemExit(f"could not find {APPENDIX_MARK!r} in index.html")
    return main_start, appendix


def reason_to_label(key: str, reasons: dict, result: dict) -> str:
    """Human-readable explanation for a failed gate criterion."""
    if key == "status_ok":
        return f"status=<code>{result.get('status') or 'none'}</code>"
    if key == "sharpe":
        s = result.get("sharpe")
        return f"Sharpe <code>{s:.2f}</code> &le; 0.50" if s is not None else "Sharpe missing"
    if key == "cagr":
        c = result.get("cagr")
        return f"CAGR <code>{c*100:.1f}%</code> &le; 10%" if c is not None else "CAGR missing"
    if key == "bh_significant":
        return "fails Benjamini-Hochberg multiple-testing correction at FDR=0.05"
    if key == "oos_sharpe":
        oos = result.get("oos_sharpe")
        if oos is None:
            return "no OOS Sharpe computed (rerun via shared harness)"
        return f"OOS Sharpe <code>{oos:.2f}</code> &le; 0"
    if key == "sample_size":
        n = result.get("n_days") or 0
        e = result.get("n_events") or 0
        return f"sample too small: n_days=<code>{n}</code>, n_events=<code>{e}</code>"
    return key


def build_banner(reasons: dict, result: dict) -> str:
    failed = [k for k, v in reasons.items() if not v]
    labels = [reason_to_label(k, reasons, result) for k in failed]
    bullets = "; ".join(labels)
    # Banner is inserted as `\n  <div…>…</div>` (no trailing whitespace) right
    # after the article open tag. The original card content already starts with
    # `\n  <…>` for its first element, so the result is well-formed without us
    # needing to add separator whitespace here.
    return (
        '\n  <div class="demoted-banner">'
        '<strong>Demoted:</strong> no longer passes the tightened winner gate. '
        f'<em>Failed:</em> {bullets}.'
        '</div>'
    )


def resolve_signal_id(card_text: str) -> str | None:
    """Find the signal_id by checking badge first, then the backtest code ref.

    The badge is more reliable because some older cards reference batch runners
    (`_run_batch_pl.py`) or use a stale filename convention (`AE2_…`) while the
    actual result is keyed by the badge value (`AE-2`).
    """
    badge_m = BADGE_RE.search(card_text)
    if badge_m:
        cand = badge_m.group(1).strip()
        if (RESULTS_DIR / f"{cand}.json").exists():
            return cand
    code_m = BACKTEST_REF_RE.search(card_text)
    if code_m:
        cand = code_m.group(1).strip()
        if (RESULTS_DIR / f"{cand}.json").exists():
            return cand
    return None


def rewrite_card(card_text: str, mt: dict) -> tuple[str, str, dict | None]:
    """Return (new_card, action, reasons) where action is 'demote'|'restore'|'noop'|'unknown'."""
    sid = resolve_signal_id(card_text)
    if sid is None:
        return card_text, "missing_result", None
    result_path = RESULTS_DIR / f"{sid}.json"
    result = json.loads(result_path.read_text())
    if not result.get("signal_id"):
        result["signal_id"] = sid

    open_m = ARTICLE_OPEN_RE.search(card_text)
    if not open_m:
        return card_text, "unknown", None
    current_class = open_m.group(1)
    currently_demoted = "demoted" in current_class

    reasons = winner_reasons(result, mt)
    passes = all(reasons.values())

    if passes and not currently_demoted:
        return card_text, "noop", None
    if passes and currently_demoted:
        # Restore: change class back, strip banner.
        new_open = f'<article class="card"{open_m.group(2)}>'
        new = ARTICLE_OPEN_RE.sub(new_open, card_text, count=1)
        new = DEMOTED_BANNER_RE.sub("", new, count=1)
        return new, "restore", reasons

    # Demote (new) or refresh (existing). Either way, ensure class is `card
    # demoted` and the banner reflects current reasons. Strip any existing
    # banner first so this is byte-idempotent when reasons haven't changed.
    stripped = DEMOTED_BANNER_RE.sub("", card_text, count=1)
    open_m2 = ARTICLE_OPEN_RE.search(stripped)
    new_open = f'<article class="card demoted"{open_m2.group(2)}>'
    new = stripped.replace(open_m2.group(0), new_open + build_banner(reasons, result), 1)
    action = "demote" if not currently_demoted else "refresh"
    return new, action, reasons


def ensure_demoted_css(html: str) -> str:
    if DEMOTED_CSS_MARKER in html:
        return html
    # Inject before the closing </style> of the first <style> block
    head_style_end = html.find("</style>")
    if head_style_end < 0:
        return html
    return html[:head_style_end] + DEMOTED_CSS + html[head_style_end:]


def sweep(apply_changes: bool) -> dict:
    mt = load_mt_data()
    html = INDEX_PATH.read_text()
    main_start, appendix = find_main_section(html)

    pre = html[:main_start + len(MAIN_OPEN)]
    body = html[main_start + len(MAIN_OPEN):appendix]
    post = html[appendix:]

    parts = split_into_cards(body)
    counts = {"demote": [], "restore": [], "noop": 0, "refresh": [], "missing_result": [], "unknown": 0}
    rewritten = []
    for p in parts:
        if not p.startswith('<article class="card'):
            rewritten.append(p)
            continue
        new_card, action, _reasons = rewrite_card(p, mt)
        rewritten.append(new_card)
        if action == "noop":
            counts["noop"] += 1
        elif action == "unknown":
            counts["unknown"] += 1
        elif action == "missing_result":
            badge = BADGE_RE.search(p)
            ref = BACKTEST_REF_RE.search(p)
            label = (badge.group(1) if badge else None) or (ref.group(1) if ref else "?")
            counts["missing_result"].append(label)
        else:
            sid = resolve_signal_id(p) or "?"
            counts[action].append(sid)

    new_body = "".join(rewritten)
    new_html = pre + new_body + post
    new_html = ensure_demoted_css(new_html)

    if apply_changes and new_html != html:
        INDEX_PATH.write_text(new_html)

    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="write changes to index.html")
    args = parser.parse_args()

    counts = sweep(args.apply)
    mode = "APPLIED" if args.apply else "DRY-RUN"
    print(f"=== Index demote sweep [{mode}] ===")
    print(f"  pass gate (kept):        {counts['noop']}")
    print(f"  to demote:               {len(counts['demote'])}")
    print(f"  to restore:              {len(counts['restore'])}")
    print(f"  already demoted, refresh banner: {len(counts['refresh'])}")
    print(f"  missing result file:     {len(counts['missing_result'])}")
    print(f"  unknown structure:       {counts['unknown']}")

    if counts["demote"]:
        print("\n  Demoting:")
        for sid in counts["demote"][:50]:
            print(f"    - {sid}")
        if len(counts["demote"]) > 50:
            print(f"    ... and {len(counts['demote']) - 50} more")
    if counts["missing_result"]:
        print("\n  Cards reference missing result file:")
        for sid in counts["missing_result"][:20]:
            print(f"    - {sid}")
    if not args.apply:
        print("\n(dry run; pass --apply to rewrite index.html)")


if __name__ == "__main__":
    main()
