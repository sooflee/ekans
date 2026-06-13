"""PL1059_fed_ebp_rising_spy_high_short — Fed Excess Bond Premium positive & rising
with SPY near highs -> short SPY / long IEF (counter long_SPY)."""
import io
import sys
import urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics, print_metrics,
                     save_result, mark_failed, daily_returns, DATA)

SID = "PL1059_fed_ebp_rising_spy_high_short"
EBP_URL = "https://www.federalreserve.gov/econres/notes/feds-notes/ebp_csv.csv"
EBP_CACHE = DATA / "fed_ebp_csv.parquet"
EBP_TMP = Path("/tmp/ebp_csv.csv")

EBP_LEVEL_MIN = 0.20      # month-t EBP >= +0.20
SLOPE_LAG = 3             # EBP(t) > EBP(t-3)
NEAR_HIGH_PCT = 0.03      # SPY month-end close within 3% of trailing 252td high
EXEC_TD = 5               # execute at close of 5th trading day of month t+1
HARD_STOP = 0.10          # exit if SPY closes 10% below entry-date close


def load_ebp():
    """Fed FEDS Notes EBP monthly CSV (columns date, gz_spread, ebp, est_prob)."""
    if EBP_CACHE.exists():
        return pd.read_parquet(EBP_CACHE)
    raw = None
    try:
        req = urllib.request.Request(EBP_URL, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=60).read()
    except Exception as e:
        print(f"  live fetch failed ({e}); trying /tmp cache")
        if EBP_TMP.exists():
            raw = EBP_TMP.read_bytes()
    if raw is None:
        raise RuntimeError("EBP CSV unavailable (live + /tmp cache)")
    df = pd.read_csv(io.BytesIO(raw))
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df["ebp"] = pd.to_numeric(df["ebp"], errors="coerce")
    df = df.dropna(subset=["ebp"])
    df.to_parquet(EBP_CACHE)
    return df


def nth_trading_day_of_next_month(month_period, cal, n=EXEC_TD):
    """Close of the n-th trading day of the month after `month_period`.
    cal: DatetimeIndex of trading days. Returns None if not enough days yet."""
    nxt = (month_period + 1).to_timestamp()
    days = cal[(cal >= nxt) & (cal < (month_period + 2).to_timestamp())]
    if len(days) < n:
        return None
    return days[n - 1]


def main():
    sid = SID
    try:
        ebp = load_ebp()
    except Exception as e:
        return mark_failed(sid, f"EBP data load: {e}")
    print(f"EBP monthly: {len(ebp)} obs ({ebp.index[0].date()} -> {ebp.index[-1].date()})")

    try:
        px = load_prices(["SPY", "IEF"], start="2002-01-01")
    except Exception as e:
        return mark_failed(sid, f"price data load: {e}")
    px = px.dropna()
    ret = daily_returns(px)
    spy, ief = px["SPY"], px["IEF"]
    spy_r, ief_r = ret["SPY"], ret["IEF"]
    cal = px.index

    # --- monthly signal state (evaluated on month-t data, executed 5td into t+1) ---
    ebp_m = ebp["ebp"].copy()
    ebp_m.index = ebp_m.index.to_period("M")
    roll_hi = spy.rolling(252).max()
    spy_me = spy.groupby(spy.index.to_period("M")).last()           # month-end close
    hi_me = roll_hi.groupby(roll_hi.index.to_period("M")).last()    # 252td high at month-end
    near_high = spy_me >= (1 - NEAR_HIGH_PCT) * hi_me

    # events: effective_date -> (entry_ok, exit_trig) decided from month-t data
    events = {}
    months = [m for m in ebp_m.index if m - SLOPE_LAG in ebp_m.index]
    for m in months:
        if m < pd.Period("2002-12", "M"):   # window starts 2003-01 (IEF inception buffer)
            continue
        eff = nth_trading_day_of_next_month(m, cal)
        if eff is None:
            continue
        slope_pos = ebp_m[m] > ebp_m[m - SLOPE_LAG]
        entry_ok = (ebp_m[m] >= EBP_LEVEL_MIN and slope_pos
                    and bool(near_high.get(m, False)))
        events[eff] = {"month": str(m), "entry_ok": entry_ok,
                       "exit_trig": not slope_pos, "ebp": float(ebp_m[m]),
                       "ebp_lag3": float(ebp_m[m - SLOPE_LAG])}

    # --- daily state machine ---
    # pos[d] = 1 means position is held over the close of day d (entered at d's
    # close or earlier); PnL accrues via pos.shift(1), so the entry-day close
    # itself earns nothing and the exit-day return IS captured (held into that close).
    sim_idx = cal[cal >= "2003-01-01"]
    pos = pd.Series(0.0, index=sim_idx)
    state, entry_d, entry_px, stop_pending = 0, None, None, False
    episodes = []

    for d in sim_idx:
        # hard stop fires at NEXT close after trigger (per spec: exit next close)
        if state == 1 and stop_pending:
            episodes.append({"entry": entry_d, "exit": d, "reason": "hard_stop",
                             "month": ev_month})
            state, stop_pending = 0, False
            pos[d] = 0.0
            continue
        ev = events.get(d)
        if ev is not None:
            if state == 0 and ev["entry_ok"]:
                state, entry_d, entry_px = 1, d, spy.loc[d]
                ev_month = ev["month"]
            elif state == 1 and ev["exit_trig"]:
                episodes.append({"entry": entry_d, "exit": d, "reason": "slope_flip",
                                 "month": ev_month})
                state = 0
                pos[d] = 0.0   # exit at this close; today's return still captured via shift
                continue
        if state == 1:
            pos[d] = 1.0
            if spy.loc[d] <= (1 - HARD_STOP) * entry_px:
                stop_pending = True
    if state == 1:
        episodes.append({"entry": entry_d, "exit": sim_idx[-1], "reason": "open",
                         "month": ev_month})

    if not episodes:
        return mark_failed(sid, "no signal episodes triggered")

    # --- PnL: equal-notional short SPY / long IEF, daily rebalanced ---
    leg = (ief_r - spy_r).reindex(sim_idx).fillna(0)
    pnl = pos.shift(1).fillna(0) * leg
    spy_bench = spy_r.reindex(sim_idx).dropna()

    ep_out = []
    for ep in episodes:
        held = pos.loc[ep["entry"]:ep["exit"]]
        sl = pnl.loc[ep["entry"]:ep["exit"]]
        spy_cum = float((1 + spy_r.loc[ep["entry"]:ep["exit"]]).prod() - 1)
        ep_pnl = float((1 + sl).prod() - 1)
        ep_out.append({
            "entry": str(ep["entry"].date()), "exit": str(ep["exit"].date()),
            "signal_month": ep["month"], "n_days": int((held == 1).sum()),
            "exit_reason": ep["reason"], "spy_return": round(spy_cum, 4),
            "episode_pnl": round(ep_pnl, 4),
        })
        print(f"  {ep_out[-1]['entry']} -> {ep_out[-1]['exit']} "
              f"({ep_out[-1]['n_days']}d, {ep['reason']}, sig_month={ep['month']}): "
              f"SPY {spy_cum:+.2%}, pnl {ep_pnl:+.2%}")

    exposure = float((pos == 1).mean())
    print(f"\nEpisodes: {len(ep_out)} | exposure: {exposure:.1%} | days: {len(pnl)}")

    m = compute_metrics(pnl, benchmark=spy_bench, positions=pos * 2,  # 2 legs
                        name="Fed EBP Rising + SPY Near Highs -> Short SPY / Long IEF")
    print_metrics(m)

    save_result(sid, m, extra={
        "rule": ("Monthly, Fed FEDS Notes EBP CSV (column 'ebp'): if EBP(t) >= +0.20 AND "
                 "EBP(t) > EBP(t-3) AND SPY month-t-end close is within 3% of its trailing "
                 "252-trading-day high, enter equal-notional short SPY / long IEF at the close "
                 "of the 5th trading day of month t+1 (publication lag). Exit at the close of "
                 "the 5th trading day after the EBP 3-month slope turns non-positive, or next "
                 "close after SPY closes 10% below entry-date close. Flat otherwise."),
        "mechanism": ("The excess bond premium (Gilchrist-Zakrajsek) measures credit-market risk "
                      "appetite net of default fundamentals; a positive and rising EBP while "
                      "equities sit near highs flags credit stress not yet priced by stocks, "
                      "preceding equity drawdowns and duration rallies."),
        "source": ("Fed FEDS Notes EBP monthly CSV "
                   "(federalreserve.gov/econres/notes/feds-notes/ebp_csv.csv), 1973-01..2026-05, "
                   "fetched live 2026-06-12; SPY/IEF via yfinance."),
        "n_events": len(ep_out),
        "exposure": round(exposure, 4),
        "events": ep_out,
        "notes": ("COUNTER-SIGNAL to long_SPY. 5-trading-day execution lag into month t+1 "
                  "respects EBP publication timing (no lookahead vs release). CAVEAT: the Fed "
                  "re-estimates the full EBP history each monthly update, so the series is a "
                  "revised vintage — results are in-sample-revised relative to what was knowable "
                  "in real time. PnL includes flat days; costs charged on 2 legs (positions=2x)."),
    }, pnl=pnl)


if __name__ == "__main__":
    main()
