#!/usr/bin/env python3
"""
pipeline/daily_check.py — automated daily signal re-evaluation.

Single source of truth: reads each signal's CURRENT status directly from the
META / PLMETA blocks in index.html, recomputes the price- and calendar-driven
signals from fresh market data, diffs computed-vs-stored, and (unless --dry-run)
writes any changes back to:

  - index.html                          META/PLMETA s:'...' status + trailing
                                        comment refresh; CUR_VER bumped (change
                                        days only, to preserve users' UI state)
  - pipeline/signals_history.json       today's change list (prepend / replace)
  - pipeline/signals_last_update.json   fresh summary counts + timestamp

Only the 15 programmatically-checkable signals are evaluated. On-chain / event /
monthly-macro signals have no live feed and are left untouched.

This replaces the old copy-a-script-per-day workflow: there is no hand-maintained
STORED dict — it is parsed from index.html, so META is the only place a status
lives. Calendar signals (A06/A07/A16) are computed year-round from the shared
FOMC calendar and SPY seasonality, not hardcoded to a single month.

Usage:
  python pipeline/daily_check.py                # run for today, write changes
  python pipeline/daily_check.py --dry-run      # compute + report, write nothing
  python pipeline/daily_check.py --date 2026-06-07
  python pipeline/daily_check.py --quiet        # only print the summary line

Exit code is 0 on success (with or without changes), 1 on a hard failure.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import tempfile
import warnings
from collections import Counter
from pathlib import Path

warnings.filterwarnings("ignore")

REPO = Path(__file__).resolve().parent.parent
INDEX_HTML = REPO / "index.html"
HISTORY = REPO / "pipeline" / "signals_history.json"
LAST_UPDATE = REPO / "pipeline" / "signals_last_update.json"

# Reuse the canonical FOMC calendar that the A06/A07 backtests use.
sys.path.insert(0, str(REPO / "backtests"))
try:
    from _fomc_dates import FOMC_DATES  # noqa: E402
except Exception:  # pragma: no cover - fallback if backtests/ is unavailable
    FOMC_DATES = [
        "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
        "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09",
    ]


# ----------------------------------------------------------------------------
# Market data
# ----------------------------------------------------------------------------
class Market:
    """Lazy, cached daily-close downloader. end is exclusive, so a run gets data
    through the last *completed* trading day before `asof` (no partial bars)."""

    def __init__(self, asof: dt.date):
        import pandas as pd  # local import keeps --help fast and dependency-light
        self.pd = pd
        self.asof = asof
        self._cache: dict[tuple[str, str], object] = {}

    def get(self, ticker: str, start: str = "2024-06-01"):
        key = (ticker, start)
        if key in self._cache:
            return self._cache[key]
        import yfinance as yf
        pd = self.pd
        series = None
        for _ in range(3):  # yfinance occasionally returns an empty frame
            df = yf.download(ticker, start=start, end=self.asof.isoformat(),
                             progress=False, auto_adjust=True)
            if df is not None and not df.empty:
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0].lower() for c in df.columns]
                else:
                    df.columns = [c.lower() for c in df.columns]
                series = df["close"].dropna()
                break
        self._cache[key] = series
        return series


# Indicator helpers -----------------------------------------------------------
def ret(s, days):
    if s is None or len(s) <= days:
        return None
    return float(s.iloc[-1] / s.iloc[-days - 1] - 1.0)


def sma(s, n):
    if s is None or len(s) < n:
        return None
    return float(s.iloc[-n:].mean())


def rsi(s, n=2):
    d = s.diff()
    up = d.clip(lower=0).rolling(n).mean()
    dn = (-d.clip(upper=0)).rolling(n).mean()
    rs = up / dn.replace(0, float("nan"))
    return float((100 - 100 / (1 + rs)).iloc[-1])


def corr60(pd, a, b):
    df = pd.concat([a.pct_change(), b.pct_change()], axis=1).dropna().iloc[-60:]
    return float(df.iloc[:, 0].corr(df.iloc[:, 1]))


def next_business_day(d: dt.date) -> dt.date:
    nd = d + dt.timedelta(days=1)
    while nd.weekday() >= 5:
        nd += dt.timedelta(days=1)
    return nd


# ----------------------------------------------------------------------------
# Signal checks.  Each returns (status, detail).  Raising / returning None for
# the status means "could not compute" and the stored status is left unchanged.
# `meta`/`pl` mark which block the signal lives in (for parsing + writing).
# ----------------------------------------------------------------------------
def chk_F09(m, today):
    s = m.get("SPY", "2024-06-01")
    s50, s200 = sma(s, 50), sma(s, 200)
    return ("active" if s50 > s200 else "inactive",
            f"50d {s50:.1f} {'>' if s50 > s200 else '<'} 200d {s200:.1f}")


def chk_C02(m, today):
    r_spy, r_xlu = ret(m.get("SPY"), 20), ret(m.get("XLU"), 20)
    return ("active" if r_spy > r_xlu else "inactive",
            f"SPY 4wk {r_spy*100:+.1f}% vs XLU {r_xlu*100:+.1f}%")


def chk_F08(m, today):
    r2 = rsi(m.get("SPY"), 2)
    return ("active" if r2 < 10 else "inactive", f"RSI(2)={r2:.0f}")


def chk_B01(m, today):
    v = float(m.get("^VIX").iloc[-1]); v3 = float(m.get("^VIX3M").iloc[-1])
    return ("active" if v / v3 < 0.92 else "inactive", f"VIX/VIX3M={v/v3:.2f}")


def chk_C11(m, today):
    v = float(m.get("^VIX").iloc[-1]); mv = float(m.get("^MOVE").iloc[-1])
    return ("active" if mv / v < 6 else "inactive", f"MOVE/VIX={mv/v:.2f}")


def chk_C14(m, today):
    r12 = {k: ret(m.get(k), 252) for k in ("SPY", "TLT", "GLD")}
    ok = all(x is not None and x > 0 for x in r12.values())
    return ("active" if ok else "inactive",
            " ".join(f"{k} {v*100:+.0f}%" for k, v in r12.items()))


def chk_C05(m, today):
    above = []
    for k in ("SPY", "EFA", "EEM", "GLD", "IEF"):
        s = m.get(k); mm = sma(s, 210)
        if mm is not None and float(s.iloc[-1]) > mm:
            above.append(k)
    return ("active" if len(above) >= 3 else "inactive",
            f"{len(above)}/5 above 10mo SMA: {','.join(above)}")


def chk_C06(m, today):
    r_spy, r_acwx = ret(m.get("SPY"), 252), ret(m.get("ACWX"), 252)
    return ("active" if (r_spy > 0 and r_acwx > 0) else "inactive",
            f"SPY 12m {r_spy*100:+.1f}% ACWX {r_acwx*100:+.1f}%")


def chk_C07(m, today):
    mom = {k: ret(m.get(k), 252) for k in ("SPY", "EFA", "EEM", "TLT", "GLD", "IEF")}
    top = max(mom, key=lambda k: mom[k])
    return ("active" if top == "SPY" else "inactive",
            f"top mom={top} ({mom[top]*100:+.0f}%); SPY {mom['SPY']*100:+.0f}%")


def chk_D02(m, today):
    s = m.get("MTUM"); ytd = s[s.index >= f"{today.year}-01-01"]
    v = float(ytd.iloc[-1] / ytd.iloc[0] - 1) if len(ytd) > 1 else None
    return ("active" if v > 0 else "inactive", f"MTUM YTD {v*100:+.1f}%")


def chk_PL87(m, today):
    s = m.get("UUP")
    peak = float(s.iloc[-252:].max()); off = float(s.iloc[-1]) / peak - 1
    return ("active" if off < -0.02 else "inactive", f"UUP {off*100:+.1f}% from 52wk peak")


def chk_H11(m, today):
    pd = m.pd
    cbn = corr60(pd, m.get("BTC-USD"), m.get("QQQ"))
    cbg = corr60(pd, m.get("BTC-USD"), m.get("GLD"))
    return ("active" if (cbn < 0.5 and cbg < 0.5) else "inactive",
            f"BTC-NDX {cbn:.2f}, BTC-GLD {cbg:.2f}")


def chk_A16(m, today):
    """Heston-Sadka same-calendar-month seasonality: if SPY's prior-10y same-month
    average daily return is > 0, hold SPY (active); else cash (defensive)."""
    s = m.get("SPY", f"{today.year - 12}-01-01")
    r = s.pct_change().dropna()
    same = r[(r.index.month == today.month) & (r.index.year >= today.year - 10)
             & (r.index.year <= today.year - 1)]
    if len(same) < 30:
        return (None, "insufficient seasonality history")
    avg = float(same.mean())
    mname = today.strftime("%b")
    return ("active" if avg > 0 else "defensive",
            f"{mname} prior-10y avg {avg*100:+.3f}%/day -> {'long' if avg>0 else 'cash'}")


def chk_A07(m, today):
    """Pre-FOMC drift (Lucca-Moench): active on the trading day before (T-1) or of
    (T) a scheduled FOMC decision."""
    fomc = [dt.date.fromisoformat(d) for d in FOMC_DATES]
    upcoming = sorted(d for d in fomc if d >= today)
    if not upcoming:
        return ("inactive", "no scheduled FOMC ahead")
    nxt = upcoming[0]
    days = (nxt - today).days
    active = nxt == today or nxt == next_business_day(today)
    return ("active" if active else "inactive", f"next FOMC in {days}d ({nxt})")


def chk_A06(m, today):
    """FOMC even-week effect (Cieslak et al): long during weeks 0,2,4,6 of the cycle."""
    fomc = sorted(dt.date.fromisoformat(d) for d in FOMC_DATES)
    past = [d for d in fomc if d <= today]
    if not past:
        return ("inactive", "before first FOMC in calendar")
    last = past[-1]
    wk = (today - last).days // 7
    return ("active" if wk in (0, 2, 4, 6) else "inactive",
            f"week {wk} of FOMC cycle since {last}")


# (id, block, fn).  block: 'META' or 'PLMETA' — where the status lives in index.html
CHECKS = [
    ("F09", "META", chk_F09), ("C02", "META", chk_C02), ("F08", "META", chk_F08),
    ("B01", "META", chk_B01), ("C11", "META", chk_C11), ("C14", "META", chk_C14),
    ("C05", "META", chk_C05), ("C06", "META", chk_C06), ("C07", "META", chk_C07),
    ("D02", "META", chk_D02), ("H11", "META", chk_H11), ("A16", "META", chk_A16),
    ("A07", "META", chk_A07), ("A06", "META", chk_A06),
    ("PL87", "PLMETA", chk_PL87),
]

# How wide a window each check needs (calendar-day lookback for the download).
DEFAULT_START = "2024-06-01"  # ~2y: covers 252d returns, 200d/210d SMAs, 60d corr


# ----------------------------------------------------------------------------
# index.html parsing / editing  (single source of truth = the META blocks)
# ----------------------------------------------------------------------------
def _block_span(html: str, name: str) -> tuple[int, int]:
    start = html.index(f"var {name} = {{")
    end = html.index("\n};", start)
    return start, end


def read_status(html: str, sid: str) -> str | None:
    """Return the stored status for a signal id, or None if it has no s:'...'."""
    pat = re.compile(r"^\s*'" + re.escape(sid) + r"'\s*:\s*\{.*?\bs:'(\w+)'", re.M)
    m = pat.search(html)
    return m.group(1) if m else None


def set_status(html: str, sid: str, new_status: str, detail: str, date: str) -> str:
    """Update a signal's s:'...' and refresh its trailing // comment, in place,
    on that signal's single definition line only."""
    lines = html.split("\n")
    key = re.compile(r"^\s*'" + re.escape(sid) + r"'\s*:\s*\{")
    for i, line in enumerate(lines):
        if key.match(line):
            line = re.sub(r"(\bs:')(\w+)(')",
                          lambda mm: mm.group(1) + new_status + mm.group(3),
                          line, count=1)
            if "//" in line and detail:
                line = re.sub(r"//.*$", f"// {detail} [auto {date}]", line)
            lines[i] = line
            return "\n".join(lines)
    raise KeyError(f"{sid} definition line not found in index.html")


def bump_cur_ver(html: str, date: str) -> str:
    return re.sub(r"(var CUR_VER = ')[^']*(';)", r"\g<1>" + date + r"\2", html, count=1)


def status_counts(html: str) -> Counter:
    c: Counter = Counter()
    for name in ("META", "PLMETA"):
        start, end = _block_span(html, name)
        for line in html[start:end].split("\n"):
            if re.match(r"\s*'[^']+'\s*:\s*\{", line):
                m = re.search(r"\bs:'(\w+)'", line)
                c[m.group(1) if m else "unknown"] += 1
    return c


# ----------------------------------------------------------------------------
# State-file writers
# ----------------------------------------------------------------------------
def write_history(date: str, changes: list[dict]):
    """Record today's transitions. Merges with any existing same-date entry by id
    (preserving the original `from`, taking the latest `to`) so the script is safe
    to run multiple times a day or after a manual edit — it accumulates the day's
    net moves rather than overwriting them. Net no-ops (from == to) are dropped."""
    hist = json.loads(HISTORY.read_text()) if HISTORY.exists() else []
    existing = next((e for e in hist if e.get("date") == date), None)
    merged: dict[str, dict] = {}
    for c in (existing["changes"] if existing else []):
        merged[c["id"]] = dict(c)
    for c in changes:
        if c["id"] in merged:
            merged[c["id"]]["to"] = c["to"]          # keep original from, latest to
        else:
            merged[c["id"]] = dict(c)
    entry_changes = [c for c in merged.values() if c["from"] != c["to"]]
    hist = [e for e in hist if e.get("date") != date]
    hist.insert(0, {"date": date, "changes": entry_changes})
    HISTORY.write_text(json.dumps(hist, indent=2) + "\n")
    return len(entry_changes)  # day total (may exceed this run's changes)


def write_last_update(date: str, checked: int, n_changes: int, counts: Counter):
    ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    LAST_UPDATE.write_text(json.dumps({
        "timestamp": ts,
        "signals_checked": checked,
        "changes_count": n_changes,
        "active_count": counts.get("active", 0),
        "defensive_count": counts.get("defensive", 0),
        "inactive_count": counts.get("inactive", 0),
        "unknown_count": counts.get("unknown", 0),
    }, indent=2) + "\n")


def atomic_write(path: Path, text: str):
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Automated daily signal check.")
    ap.add_argument("--date", help="YYYY-MM-DD run date (default: today)")
    ap.add_argument("--dry-run", action="store_true", help="compute + report, write nothing")
    ap.add_argument("--quiet", action="store_true", help="print only the summary line")
    args = ap.parse_args(argv)

    today = dt.date.fromisoformat(args.date) if args.date else dt.date.today()
    date = today.isoformat()
    log = (lambda *a: None) if args.quiet else print

    html = INDEX_HTML.read_text()
    market = Market(today)

    log(f"=== DAILY CHECK {date} ===")
    rows, changes, errors = [], [], []
    for sid, block, fn in CHECKS:
        stored = read_status(html, sid)
        try:
            computed, detail = fn(market, today)
        except Exception as e:  # never let one bad ticker abort the whole run
            computed, detail = None, f"error: {e}"
        if computed is None:
            errors.append((sid, detail))
            rows.append(("??", sid, stored, "skip", detail))
            continue
        if computed != stored:
            changes.append({"id": sid, "from": stored, "to": computed})
            rows.append(("**", sid, stored, computed, detail))
        else:
            rows.append(("  ", sid, stored, computed, detail))

    if not args.quiet:
        for flag, sid, old, comp, detail in rows:
            log(f"{flag} {sid:5s} stored={str(old):9s} computed={comp:9s} | {detail}")

    # Apply changes to index.html (status + comment), bump CUR_VER on change days.
    new_html = html
    for c in changes:
        detail = next(d for f, s, o, cm, d in rows if s == c["id"])
        new_html = set_status(new_html, c["id"], c["to"], detail, date)
    if changes:
        new_html = bump_cur_ver(new_html, date)

    counts = status_counts(new_html)
    checked = len([r for r in rows if r[0] != "??"])

    if args.dry_run:
        log("\n[dry-run] no files written.")
    else:
        if changes:
            atomic_write(INDEX_HTML, new_html)
        day_total = write_history(date, changes)
        write_last_update(date, checked, day_total, counts)

    # Summary (always printed, even under --quiet)
    print(f"{date}: checked {checked}/{len(CHECKS)}, {len(changes)} change(s)"
          + (f", {len(errors)} skipped" if errors else "")
          + f" | active {counts.get('active',0)} / inactive {counts.get('inactive',0)}"
          + f" / defensive {counts.get('defensive',0)} / unknown {counts.get('unknown',0)}"
          + (" [dry-run]" if args.dry_run else ""))
    for c in changes:
        print(f"  CHANGE {c['id']}: {c['from']} -> {c['to']}")
    for sid, detail in errors:
        print(f"  SKIP   {sid}: {detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
