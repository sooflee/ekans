"""PL1057_cftc_btc_cot_extreme_fade — CFTC TFF Bitcoin Asset-Manager Net-Long Extreme -> Short BTC"""
import io
import sys
import zipfile
import urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics, print_metrics,
                     save_result, mark_failed, daily_returns, DATA)

SID = "PL1057_cftc_btc_cot_extreme_fade"
BTC_CODE = "133741"  # BITCOIN - CHICAGO MERCANTILE EXCHANGE (full-size, NOT micro)
YEARS = range(2017, 2027)
CACHE = DATA / "cftc_tff_btc_133741.parquet"

COLS = [
    "Report_Date_as_YYYY-MM-DD",
    "CFTC_Contract_Market_Code",
    "Open_Interest_All",
    "Asset_Mgr_Positions_Long_All",
    "Asset_Mgr_Positions_Short_All",
    "Lev_Money_Positions_Long_All",
    "Lev_Money_Positions_Short_All",
]


def load_tff_btc():
    """Weekly TFF futures-only positioning for CME Bitcoin (code 133741)."""
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    frames = []
    for y in YEARS:
        url = f"https://www.cftc.gov/files/dea/history/fut_fin_txt_{y}.zip"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            raw = urllib.request.urlopen(req, timeout=60).read()
        except Exception as e:
            print(f"  {y}: download failed ({e}), skipping")
            continue
        with zipfile.ZipFile(io.BytesIO(raw)) as zf:
            name = zf.namelist()[0]
            df = pd.read_csv(io.BytesIO(zf.read(name)), low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        df = df[df["CFTC_Contract_Market_Code"].astype(str).str.strip() == BTC_CODE]
        if df.empty:
            print(f"  {y}: no BTC rows")
            continue
        frames.append(df[COLS].copy())
        print(f"  {y}: {len(df)} BTC weekly reports")
    if not frames:
        raise RuntimeError("no TFF data downloaded")
    out = pd.concat(frames, ignore_index=True)
    out["report_date"] = pd.to_datetime(out["Report_Date_as_YYYY-MM-DD"])
    for c in COLS[2:]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna().sort_values("report_date").drop_duplicates("report_date")
    out = out.set_index("report_date")
    out.to_parquet(CACHE)
    return out


def trailing_pct_rank(s, window=104, min_obs=52):
    """Percentile rank of each obs within trailing `window` obs (incl. itself),
    NaN until `min_obs` observations are available."""
    vals = s.values
    out = np.full(len(vals), np.nan)
    for i in range(len(vals)):
        if i + 1 < min_obs:
            continue
        w = vals[max(0, i - window + 1): i + 1]
        out[i] = (w <= vals[i]).mean()
    return pd.Series(out, index=s.index)


def run_state_machine(pct, btc_idx, hold_td=30, entry_p=0.90, exit_p=0.50):
    """Build daily short-flag series (1 = short active) on btc_idx (weekdays).

    Report Tuesday -> released Friday ~3:30pm ET -> position change effective the
    next trading day (Monday). Entry is implementable at the Friday 00:00-UTC BTC
    close, which occurs AFTER the 3:30pm ET release, so Monday's Fri->Mon weekday
    close-to-close return is fully capturable with no look-ahead.
    """
    # effective trading date for each report = first weekday after report Tue + 3d (Friday)
    events = []  # (effective_date, pct_value)
    for rd, p in pct.dropna().items():
        friday = rd + pd.Timedelta(days=3)
        later = btc_idx[btc_idx > friday]
        if len(later) == 0:
            continue
        events.append((later[0], p))
    ev = dict(events)  # effective_date -> latest pct (one report per week)

    pos = pd.Series(0, index=btc_idx, dtype=float)
    state, days_in, entry_d, entry_p_val = 0, 0, None, None
    episodes = []
    for d in btc_idx:
        if d in ev:
            p = ev[d]
            if state == 0 and p > entry_p:
                state, days_in, entry_d, entry_p_val = 1, 0, d, p
            elif state == 1 and p < exit_p:
                episodes.append({"entry": entry_d, "exit": d, "reason": "pct_normalized",
                                 "entry_pct": entry_p_val})
                state = 0
        if state == 1 and days_in >= hold_td:
            episodes.append({"entry": entry_d, "exit": d, "reason": "time_stop",
                             "entry_pct": entry_p_val})
            state = 0
        if state == 1:
            pos[d] = 1
            days_in += 1
    if state == 1:
        episodes.append({"entry": entry_d, "exit": btc_idx[-1], "reason": "open",
                         "entry_pct": entry_p_val})
    return pos, episodes


def main():
    sid = SID
    print("Loading CFTC TFF futures-only Bitcoin (133741)...")
    try:
        tff = load_tff_btc()
    except Exception as e:
        return mark_failed(sid, f"CFTC TFF data load: {e}")
    print(f"TFF weekly reports: {len(tff)} ({tff.index[0].date()} -> {tff.index[-1].date()})")

    tff = tff[tff["Open_Interest_All"] > 0]  # guard thin/zero OI
    nl = (tff["Asset_Mgr_Positions_Long_All"] - tff["Asset_Mgr_Positions_Short_All"]
          + tff["Lev_Money_Positions_Long_All"] - tff["Lev_Money_Positions_Short_All"]
          ) / tff["Open_Interest_All"]
    nl_am = (tff["Asset_Mgr_Positions_Long_All"] - tff["Asset_Mgr_Positions_Short_All"]
             ) / tff["Open_Interest_All"]

    pct = trailing_pct_rank(nl)
    pct_am = trailing_pct_rank(nl_am)
    print(f"Valid percentile obs: {pct.notna().sum()} (first: {pct.dropna().index[0].date()})")

    try:
        px = load_prices(["BTC-USD", "SPY"], start="2017-01-01")
    except Exception as e:
        return mark_failed(sid, f"price data load: {e}")
    if "BTC-USD" not in px.columns:
        return mark_failed(sid, "BTC-USD not in price data")

    # weekday-only BTC series: Monday return spans Fri close -> Mon close, so
    # weekend moves are captured while staying aligned with the 252-day harness
    # annualization and the SPY benchmark calendar.
    btc = px["BTC-USD"].dropna()
    btc_wd = btc[btc.index.dayofweek < 5]
    btc_r = btc_wd.pct_change().dropna()
    spy_r = daily_returns(px[["SPY"]]).iloc[:, 0].dropna()

    pos, episodes = run_state_machine(pct, btc_r.index)
    pos_am, _ = run_state_machine(pct_am, btc_r.index)

    first_valid = pct.dropna().index[0] + pd.Timedelta(days=3)
    idx = btc_r.index[btc_r.index > first_valid]
    pos, btc_w = pos.reindex(idx).fillna(0), btc_r.reindex(idx).fillna(0)

    pnl = -pos * btc_w  # short BTC while active, flat (0) otherwise

    ep_out = []
    for ep in episodes:
        sl = btc_w.loc[ep["entry"]:ep["exit"]]
        sl = sl.loc[pos.loc[ep["entry"]:ep["exit"]] == 1]
        btc_cum = float((1 + sl).prod() - 1)
        ep_out.append({
            "entry": str(ep["entry"].date()), "exit": str(ep["exit"].date()),
            "n_days": int(len(sl)), "exit_reason": ep["reason"],
            "entry_pct": round(float(ep["entry_pct"]), 3),
            "btc_return": round(btc_cum, 4), "trade_pnl": round(-btc_cum, 4),
        })
        print(f"  {ep_out[-1]['entry']} -> {ep_out[-1]['exit']} ({ep_out[-1]['n_days']}d, "
              f"{ep['reason']}, pct={ep_out[-1]['entry_pct']}): BTC {btc_cum:+.2%}, "
              f"pnl {-btc_cum:+.2%}")

    if not ep_out:
        return mark_failed(sid, "no signal episodes triggered")

    exposure = float((pos == 1).mean())
    print(f"\nEpisodes: {len(ep_out)} | exposure: {exposure:.1%} | days: {len(pnl)}")

    m = compute_metrics(pnl, benchmark=spy_r, positions=-pos,
                        name="CFTC TFF BTC Net-Long Extreme -> Short BTC")
    print_metrics(m)

    # --- variants ---
    # (a) counter overlay: long BTC baseline, to CASH while signal active
    overlay = (1 - pos.reindex(idx).fillna(0)) * btc_w
    bh = btc_w
    m_ov = compute_metrics(overlay, benchmark=spy_r, name="overlay")
    m_bh = compute_metrics(bh, benchmark=spy_r, name="btc_bh")
    # (b) AssetMgr-only NL robustness
    pos_am = pos_am.reindex(idx).fillna(0)
    m_am = compute_metrics(-pos_am * btc_w, benchmark=spy_r, name="am_only")

    print(f"\nOverlay (long BTC, cash on signal): CAGR {m_ov['cagr']:.1%} Sharpe {m_ov['sharpe']:.2f} "
          f"MaxDD {m_ov['max_dd']:.1%}")
    print(f"BTC buy-and-hold:                   CAGR {m_bh['cagr']:.1%} Sharpe {m_bh['sharpe']:.2f} "
          f"MaxDD {m_bh['max_dd']:.1%}")
    print(f"AssetMgr-only short variant:        CAGR {m_am['cagr']:.1%} Sharpe {m_am['sharpe']:.2f}")

    save_result(sid, m, extra={
        "rule": ("Weekly CFTC TFF futures-only report for CME Bitcoin (code 133741): "
                 "NL = (AssetMgr net long + LevMoney net long) / open interest. When NL's "
                 "percentile rank over the trailing 104 weekly reports (min 52 obs) exceeds 0.90, "
                 "short BTC-USD at the next Monday open (Friday-release lag respected); cover when "
                 "the percentile falls below 0.50 or after 30 trading days. Flat otherwise."),
        "mechanism": ("When institutional (asset manager + leveraged fund) net-long positioning in "
                      "CME Bitcoin futures reaches a 2-year extreme, the marginal institutional "
                      "buyer is exhausted and crowded longs unwind, producing negative forward "
                      "BTC returns over 2-6 weeks."),
        "source": ("CFTC Traders in Financial Futures, futures-only yearly archives "
                   "(cftc.gov/files/dea/history/fut_fin_txt_YYYY.zip), contract 133741; "
                   "BTC-USD via yfinance. IBIT is the live-trading vehicle (history too short to backtest)."),
        "n_events": len(ep_out),
        "exposure": round(exposure, 4),
        "events": ep_out,
        "variant_overlay_long_btc_cash_on_signal": {k: round(m_ov[k], 4) for k in
                                                    ("cagr", "sharpe", "max_dd") if k in m_ov},
        "variant_btc_buy_hold": {k: round(m_bh[k], 4) for k in ("cagr", "sharpe", "max_dd") if k in m_bh},
        "variant_assetmgr_only_short": {k: round(m_am[k], 4) for k in ("cagr", "sharpe", "max_dd") if k in m_am},
        "notes": ("Weekday-only BTC series (Monday return spans the weekend); entry implementable "
                  "at the Friday 00:00 UTC close, after the ~3:30pm ET release. Full-size contract "
                  "only; micros excluded. PnL series includes flat days (short/flat strategy)."),
    }, pnl=pnl)


if __name__ == "__main__":
    main()
