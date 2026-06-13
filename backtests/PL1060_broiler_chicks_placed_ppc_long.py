"""PL1060_broiler_chicks_placed_ppc_long — USDA Broiler Hatchery Chicks-Placed
YoY Contraction -> 9-Week Supply Cliff -> Long PPC.

Primary data source (USDA NASS Quick Stats API) requires an API key we don't
have (keyless verified {'error':['unauthorized']}), so this uses the documented
fallback: the USDA ESMIS / NAL archive of the weekly *Broiler Hatchery* report
(identifier BroiHatc), parsing chicks-placed totals from each as-released txt.

Signal (per spec):
  4-week MA of weekly chicks placed (19-state / US weekly program total, head)
  YoY < 0 for >= 4 consecutive weekly reports -> LONG PPC at the first trading
  day's close after the 4th qualifying report (report released Wednesday
  ~3pm ET -> Thursday close). Exit when the 4-week-MA YoY turns positive
  (next trading day's close after that report) OR after 60 trading days.
  Re-entry requires 4 fresh consecutive negative weekly reports after exit.

Point-in-time construction:
  - Old format (~2017 and earlier): each release prints the last ~6 weeks of
    the 19-state total AND the matching "Previous year total" row, so the
    4-week YoY is computed entirely from numbers visible in that release.
  - New format: no previous-year row, so the year-ago 4-week window is taken
    from the as-first-released totals of reports ~52 weeks earlier (week
    ending - 364 days), same coverage ("United States" weekly program).
  - Backtest window 2010-01 onward (PPC Chapter 11 Dec-2008..Dec-2009).
"""
import sys
import json
import re
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics, save_result, mark_failed,
                     daily_returns, print_metrics, DATA)

SID = "PL1060_broiler_chicks_placed_ppc_long"
API = "https://esmis.nal.usda.gov/api/v1/release/findByIdentifier/BroiHatc"
CACHE = DATA / "PL1060_broiler_hatchery_releases.json"
FETCH_FROM = "2008-01-01"   # need ~2y lead-in before 2010 backtest start
HOLD_MAX = 60               # trading-day time stop
NEG_WEEKS = 4               # consecutive negative weekly reports to enter


# ---------------- data: ESMIS archive fetch + parse ----------------

def _parse_release(text):
    """Parse one Broiler Hatchery txt. Returns dict or None."""
    lines = text.splitlines()
    start = None
    for i, ln in enumerate(lines):
        # "Broiler-Type Chicks Placed - 19 Selected States: 2012" (2010-2012)
        # "Broiler-Type Chicks Placed - Selected States and United States" (new)
        # "Broiler Chicks Placed, 19 Selected States" (pre-mid-2010)
        if re.match(r"\s*Broiler(?:[- ]Type)? Chicks Placed\s*[-,]", ln):
            start = i
            break
    if start is None:
        return None
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if ("Statistical Methodology" in lines[j]
                or re.match(r"\s*Broiler(?:[- ]Type)? Eggs Set\s*[-,]", lines[j])):
            end = j
            break
    section = lines[start:end]

    label_map = {"19 state total": "s19",
                 "united states": "us",
                 "previous year total": "prev"}
    rows = {}
    for ln in section:
        m = re.match(
            r"\s*(19 State [Tt]otal|United States|Previous [Yy]ear [Tt]otal)"
            r"\s*\.*\s*:?(.*)$", ln)
        if m:
            lab = label_map[m.group(1).lower()]
            nums = [int(x.replace(",", "")) for x in re.findall(r"[\d,]+", m.group(2))]
            if nums and lab not in rows:
                rows[lab] = nums

    s19, us, prev = rows.get("s19"), rows.get("us"), rows.get("prev")
    if s19 is None and us is None:
        return None

    norm = re.sub(r"\s+", " ", text)
    m = re.search(
        r"placed [\d.,]+ (?:million|billion) chicks[^.]*?week ending "
        r"([A-Za-z]+ \d{1,2}, \d{4})", norm)
    if not m:
        m = re.search(r"week ending ([A-Za-z]+ \d{1,2}, \d{4})", norm)
    if not m:
        return None
    week_ending = str(pd.to_datetime(m.group(1)).date())

    return {"week_ending": week_ending, "s19": s19, "us": us,
            "prev_year": prev}


def fetch_releases():
    """Fetch + parse all Broiler Hatchery releases since FETCH_FROM. Cached."""
    if CACHE.exists():
        with open(CACHE) as f:
            return json.load(f)

    import requests
    from concurrent.futures import ThreadPoolExecutor

    sess = requests.Session()
    sess.headers["User-Agent"] = "ekans-research/1.0"

    meta = []
    page = 0
    while True:
        for attempt in range(3):
            try:
                r = sess.get(API, params={"latest": "false", "page": page},
                             timeout=60)
                r.raise_for_status()
                d = r.json()
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
        results = d.get("results", [])
        if not results:
            break
        oldest = None
        for item in results:
            rd = item["release_datetime"][:10]
            oldest = rd
            txts = [f for f in item["files"] if f.lower().endswith(".txt")]
            if rd >= FETCH_FROM and txts:
                meta.append({"release_date": rd, "url": txts[0]})
        print(f"  page {page}: {len(results)} releases (oldest {oldest})")
        if oldest is not None and oldest < FETCH_FROM:
            break
        page += 1
        time.sleep(0.2)

    print(f"  fetching {len(meta)} release txts...")

    def grab(m):
        for attempt in range(3):
            try:
                r = sess.get(m["url"], timeout=60)
                r.raise_for_status()
                p = _parse_release(r.text)
                if p:
                    p["release_date"] = m["release_date"]
                return p or {"release_date": m["release_date"],
                             "parse_failed": True, "url": m["url"]}
            except Exception as e:
                if attempt == 2:
                    return {"release_date": m["release_date"],
                            "parse_failed": True, "url": m["url"],
                            "err": str(e)}
                time.sleep(2)

    with ThreadPoolExecutor(max_workers=8) as ex:
        parsed = list(ex.map(grab, meta))

    ok = [p for p in parsed if p and not p.get("parse_failed")]
    bad = [p for p in parsed if p and p.get("parse_failed")]
    print(f"  parsed OK: {len(ok)}, failed: {len(bad)}")
    for b in bad[:10]:
        print("   FAILED:", b["release_date"], b.get("url"), b.get("err", ""))

    out = sorted(ok, key=lambda p: p["release_date"])
    with open(CACHE, "w") as f:
        json.dump({"releases": out, "n_failed": len(bad),
                   "failed": bad}, f, indent=1)
    return {"releases": out, "n_failed": len(bad), "failed": bad}


# ---------------- signal construction ----------------

def build_signal(releases):
    """Per release: 4-week-MA YoY of chicks placed, point-in-time.

    Returns DataFrame indexed by release_date with columns
    [week_ending, cur_week, yoy4, method].
    """
    # as-first-released current-week totals keyed by (coverage, week_ending)
    cur_by_we = {}
    for p in releases:
        for cov in ("s19", "us"):
            if p.get(cov):
                cur_by_we[(cov, p["week_ending"])] = p[cov][-1]

    def lag364(p, cov):
        """4wk YoY from as-released totals of the same coverage 364d earlier."""
        totals = p.get(cov)
        if not totals or len(totals) < NEG_WEEKS:
            return np.nan
        we = pd.Timestamp(p["week_ending"])
        s_cur = s_prev = 0
        for k in range(NEG_WEEKS):
            ago = str((we - pd.Timedelta(days=7 * k + 364)).date())
            v = cur_by_we.get((cov, ago))
            if v is None:
                return np.nan
            s_cur += totals[-1 - k]
            s_prev += v
        return s_cur / s_prev - 1.0 if s_prev > 0 else np.nan

    recs = []
    for p in releases:
        s19, prev = p.get("s19"), p.get("prev_year")
        yoy4, method = np.nan, None
        if (s19 and prev and len(s19) >= NEG_WEEKS and len(prev) >= NEG_WEEKS
                and sum(prev[-NEG_WEEKS:]) > 0):
            # in-report previous-year row: fully point-in-time, same coverage
            yoy4 = sum(s19[-NEG_WEEKS:]) / sum(prev[-NEG_WEEKS:]) - 1.0
            method = "in_report"
        if np.isnan(yoy4):
            yoy4 = lag364(p, "s19")
            method = "s19_lag364" if not np.isnan(yoy4) else None
        if np.isnan(yoy4):
            yoy4 = lag364(p, "us")
            method = "us_lag364" if not np.isnan(yoy4) else None
        recs.append({"release_date": p["release_date"],
                     "week_ending": p["week_ending"],
                     "label": "us" if p.get("us") else "s19",
                     "cur_week": (p.get("us") or p.get("s19"))[-1],
                     "yoy4": yoy4, "method": method})
    df = pd.DataFrame(recs).drop_duplicates("release_date")
    df["release_date"] = pd.to_datetime(df["release_date"])
    return df.sort_values("release_date").set_index("release_date")


# ---------------- backtest ----------------

def main():
    try:
        data = fetch_releases()
    except Exception as e:
        return mark_failed(SID, f"ESMIS archive fetch: {e}")
    releases = data["releases"]
    if len(releases) < 400:
        return mark_failed(SID, f"too few parsed releases ({len(releases)})")

    sig = build_signal(releases)
    n_nan = int(sig["yoy4"].isna().sum())
    lab_change = sig["label"].ne(sig["label"].shift()).sum() - 1
    print(f"releases parsed: {len(sig)}  yoy4 NaN: {n_nan}  "
          f"label changes: {lab_change}")
    print(sig.groupby("method", dropna=False)["yoy4"].agg(["count", "mean"]))

    try:
        px = load_prices(["PPC", "TSN", "SPY"], start="2009-06-01")
    except Exception as e:
        return mark_failed(SID, f"price load: {e}")
    ret = daily_returns(px)
    idx = ret.index[ret.index >= "2010-01-01"]
    ppc_r = ret["PPC"].reindex(idx).fillna(0.0)
    spy_r = ret["SPY"].reindex(idx).dropna()

    # --- state machine over weekly reports ---
    sig_bt = sig[(sig.index >= "2009-10-01")]
    trades = []
    pos = pd.Series(0.0, index=idx)
    state, streak = "flat", 0
    entry_loc = None
    entry_info = None

    def first_td_after(d):
        """index location of first trading day strictly after date d."""
        locs = np.searchsorted(idx, d + pd.Timedelta(days=1))
        return int(locs) if locs < len(idx) else None

    for rd, row in sig_bt.iterrows():
        v = row["yoy4"]
        if state == "long":
            stop_loc = min(entry_loc + HOLD_MAX, len(idx) - 1)
            sig_loc = first_td_after(rd)
            if sig_loc is None:
                break
            if sig_loc > stop_loc:
                # time stop hit before this report could be acted on
                _close_trade(trades, entry_info, entry_loc, stop_loc, idx,
                             ppc_r, spy_r, pos, "time_stop")
                state, streak = "flat", 0
                entry_loc, entry_info = None, None
                # this report now counts toward a fresh streak (falls through)
            elif not np.isnan(v) and v > 0:
                exit_loc = min(sig_loc, stop_loc)
                _close_trade(trades, entry_info, entry_loc, exit_loc, idx,
                             ppc_r, spy_r, pos, "yoy_positive")
                state, streak = "flat", 0
                entry_loc, entry_info = None, None
                continue
            else:
                continue
        # flat: count consecutive negative weekly reports
        if not np.isnan(v) and v < 0:
            streak += 1
        else:
            streak = 0
        if streak >= NEG_WEEKS and rd >= pd.Timestamp("2009-12-01"):
            e = first_td_after(rd)
            if e is None or idx[e] < pd.Timestamp("2010-01-04"):
                # don't enter before the 2010 window opens
                if e is None:
                    break
                continue
            state, entry_loc = "long", e
            entry_info = {"signal_date": str(rd.date()),
                          "week_ending": row["week_ending"],
                          "yoy4_at_entry": round(float(v), 4)}
            streak = 0

    if state == "long" and entry_loc is not None:
        stop_loc = min(entry_loc + HOLD_MAX, len(idx) - 1)
        _close_trade(trades, entry_info, entry_loc, stop_loc, idx,
                     ppc_r, spy_r, pos, "open_at_end" if stop_loc == len(idx) - 1
                     else "time_stop")

    if len(trades) < 3:
        return mark_failed(SID, f"too few trades ({len(trades)})")

    # entry at close of entry day -> earn returns from next day through exit close
    pnl = pos.shift(1).fillna(0.0) * ppc_r

    m = compute_metrics(pnl, benchmark=spy_r, positions=pos,
                        name="Broiler Chicks-Placed YoY Contraction -> Long PPC")

    # diagnostics: active-day-only stats and PPC buy-and-hold comparison
    active = pnl[pos.shift(1).fillna(0.0) != 0]
    m_active = compute_metrics(active, benchmark=spy_r, name="active_days_only")
    m_ppc_bh = compute_metrics(ppc_r, benchmark=spy_r, name="PPC buy-and-hold")

    tr = [t["trade_return"] for t in trades]
    in_pos_frac = float((pos != 0).mean())
    print(f"\ntrades: {len(trades)}  in-position fraction: {in_pos_frac:.2f}")
    print(f"avg trade return: {np.mean(tr)*100:+.2f}%  "
          f"hit rate: {np.mean([t > 0 for t in tr])*100:.0f}%")
    print_metrics(m)
    print_metrics(m_active)
    print_metrics(m_ppc_bh)

    save_result(SID, m, pnl=pnl, extra={
        "rule": ("Weekly USDA Broiler Hatchery report: 4-week sum of chicks "
                 "placed (19-state/US weekly program, head) vs same 4-week "
                 "window a year earlier. After >=4 consecutive weekly reports "
                 "with 4wk YoY < 0, go LONG PPC at next trading day's close "
                 "(report is Wed ~3pm ET -> Thu close). Exit at next close "
                 "after 4wk YoY turns positive, or after 60 trading days. "
                 "Re-entry needs 4 fresh negative weeks. Window 2010-present."),
        "mechanism": ("Chick placements are biologically locked ~9 weeks ahead "
                      "of slaughter: a sustained placement contraction is a "
                      "pre-announced supply cliff -> chicken prices and "
                      "spot-exposed processor margins (PPC) rise 6-12 weeks "
                      "later, before the cutback shows in slaughter data."),
        "source": ("USDA NASS Broiler Hatchery weekly (ESMIS/NAL archive "
                   "identifier BroiHatc, as-released txt vintages); "
                   "yfinance PPC/SPY"),
        "n_events": len(trades),
        "trades": trades,
        "event_hit_rate": round(float(np.mean([t > 0 for t in tr])), 3),
        "avg_trade_return": round(float(np.mean(tr)), 4),
        "in_position_fraction": round(in_pos_frac, 3),
        "active_days_sharpe": m_active.get("sharpe"),
        "active_days_cagr": m_active.get("cagr"),
        "ppc_bh_sharpe": m_ppc_bh.get("sharpe"),
        "ppc_bh_cagr": m_ppc_bh.get("cagr"),
        "signal_releases": int(len(sig_bt)),
        "yoy4_nan_releases": n_nan,
        "caveats": ("NASS Quick Stats key unavailable -> parsed as-released "
                    "ESMIS report txts (documented fallback). Pre-transition "
                    "YoY uses in-report previous-year rows (fully point-in-"
                    "time); after NASS dropped that row the year-ago window "
                    "uses as-first-released totals 364d earlier (ignores the "
                    "5-week revision window; first ~52 weeks after any "
                    "coverage change are NaN and treated as non-negative). "
                    "PPC float is ~80% JBS-owned and thin; closes only. 2022 "
                    "hatchability episode overlaps HPAI demand scares. PnL is "
                    "full-window (zeros when flat); active-day stats in "
                    "extras."),
    })


def _close_trade(trades, entry_info, entry_loc, exit_loc, idx, ppc_r, spy_r,
                 pos, reason):
    pos.iloc[entry_loc:exit_loc] = 1.0
    hold = idx[entry_loc + 1:exit_loc + 1]
    t_ret = float((1 + ppc_r.reindex(hold)).prod() - 1)
    s_ret = float((1 + spy_r.reindex(hold).fillna(0)).prod() - 1)
    trades.append(dict(entry_info,
                       entry_date=str(idx[entry_loc].date()),
                       exit_date=str(idx[exit_loc].date()),
                       exit_reason=reason,
                       n_days=int(exit_loc - entry_loc),
                       trade_return=round(t_ret, 4),
                       spy_return=round(s_ret, 4)))
    print(f"  {idx[entry_loc].date()} -> {idx[exit_loc].date()} "
          f"({exit_loc - entry_loc:3d}d, {reason:12s}) "
          f"PPC {t_ret*100:+6.1f}%  SPY {s_ret*100:+6.1f}%  "
          f"(entry yoy4 {entry_info['yoy4_at_entry']*100:+.1f}%)")


if __name__ == "__main__":
    main()
