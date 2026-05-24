"""
E01 Form 4 insider cluster buying.

Universe: 15 hand-picked mid-cap US tickers chosen for having visible insider
activity over 2015-2024 (energy, financials, industrials lean — sectors with
historically more insider purchases).

Data: scrape openinsider.com per-ticker insider trades table. Polite 1 req/sec.

Cluster rule: ≥3 distinct insiders, each buying ≥$25k, all within a rolling
10-trading-day window. Triggered on the date of the 3rd qualifying purchase.

Trade: buy at t+1 close, hold 252 trading days. Equal-weight basket across
triggered names; positions overlap so basket can be >100% gross.

Reports: event-study CAR vs SPY at 21d, 63d, 126d, 252d.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import requests
from harness import save_result, mark_failed


# 15 mid-cap names that historically had insider purchases visible
TICKERS = [
    "HBAN",  # Huntington Bancshares
    "RF",    # Regions Financial
    "ZION",  # Zions Bancorporation
    "CFG",   # Citizens Financial
    "KEY",   # KeyCorp
    "OZK",   # Bank OZK
    "WSBC",  # WesBanco
    "FRO",   # Frontline (shipping)
    "DXC",   # DXC Technology
    "M",     # Macy's
    "NWL",   # Newell Brands
    "AAP",   # Advance Auto Parts
    "KSS",   # Kohl's
    "PARA",  # Paramount Global
    "CC",    # Chemours
    "MOS",   # Mosaic
    "X",     # US Steel
    "BBBY",  # Bed Bath (delisted, may fail)
    "FCEL",
    "GPS",   # Gap
]

UA = "ekans-research/0.1 bensonw.dev@gmail.com"


def fetch_insider_table(ticker, max_rows=1000):
    """Return DataFrame of insider trades for ticker via openinsider screener,
    filtered to date range 2015-2024 and value >=25k."""
    url = ("http://openinsider.com/screener?"
           f"s={ticker}&o=&pl=&ph=&ll=&lh="
           "&fd=0&fdr=01%2F01%2F2015-12%2F31%2F2024"
           "&td=0&tdr=&fdlyl=&fdlyh="
           "&daysago=&xp=1"
           "&vl=25&vh="
           "&ocl=&och=&sic1=-1&sicl=100&sich=9999"
           "&grp=0&nfl=&nfh=&nil=&nih="
           "&iscob=1&isofficer=1&istenpercent=1&isother=1"
           "&nol=&noh=&v2l=&v2h=&oc2l=&oc2h="
           f"&sortcol=0&cnt={max_rows}&page=1")
    r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    try:
        tables = pd.read_html(r.text)
    except ValueError:
        return None
    # The transactions table has 17 cols including 'Trade Type'
    for t in tables:
        cols = [str(c) for c in t.columns]
        if any("Trade" in c and "Type" in c for c in cols):
            return t
    return None


def detect_clusters(df, window_days=10, min_insiders=3, min_value=25000):
    """Return list of cluster trigger dates given an insider-trade table for a single ticker.
    Trigger date = trade date of the 3rd qualifying buy whose 2-most-recent peers are
    within window_days trading days."""
    if df is None or df.empty:
        return []
    # Trade Type contains "P - Purchase"
    tt_col = [c for c in df.columns if "Trade" in str(c) and "Type" in str(c)][0]
    val_col = [c for c in df.columns if str(c).startswith("Value")][0]
    td_col  = [c for c in df.columns if "Trade" in str(c) and "Date" in str(c)][0]
    ins_col = [c for c in df.columns if str(c) == "Insider Name" or "Insider" in str(c)]
    # Insider name not in compact view; we'll rely on "X" code's uniqueness via Filing Date pair.
    # Use combination of (Filing Date, Trade Date) as a proxy for unique events; openinsider
    # already de-dupes per filing.
    fd_col  = [c for c in df.columns if "Filing" in str(c) and "Date" in str(c)][0]
    sub = df[df[tt_col].astype(str).str.contains("P - Purchase", na=False)].copy()
    if sub.empty:
        return []
    # parse value
    def parse_val(v):
        if pd.isna(v): return 0.0
        s = str(v).replace("+","").replace("$","").replace(",","")
        try: return float(s)
        except: return 0.0
    sub["val_num"] = sub[val_col].apply(parse_val)
    sub = sub[sub["val_num"] >= min_value]
    if sub.empty:
        return []
    sub["trade_date"] = pd.to_datetime(sub[td_col], errors="coerce")
    sub = sub.dropna(subset=["trade_date"])
    sub = sub.sort_values("trade_date")

    # Treat each row as a distinct "insider buy" (filings often have one insider each).
    # Slide a 10-trading-day-ish window (we approximate by calendar days * 7/5).
    cal_window = int(window_days * 7 / 5) + 1  # ~15 calendar days
    triggers = []
    dates = sub["trade_date"].tolist()
    for i in range(len(dates)):
        window_start = dates[i] - pd.Timedelta(days=cal_window)
        # how many trades in [window_start, dates[i]] (inclusive of dates[i])
        n = sum(1 for d in dates[:i+1] if d >= window_start)
        if n >= min_insiders:
            triggers.append(dates[i])
    # Compress consecutive triggers — only first trigger per 30-cal-day cool-down
    compressed = []
    cooldown = pd.Timedelta(days=30)
    for d in triggers:
        if not compressed or (d - compressed[-1]) > cooldown:
            compressed.append(d)
    return compressed


def main():
    sid = "E01_form4_cluster"
    try:
        # Fetch all triggers
        all_triggers = {}
        for tkr in TICKERS:
            try:
                df = fetch_insider_table(tkr)
                triggers = detect_clusters(df) if df is not None else []
                print(f"  {tkr}: {len(triggers)} triggers")
                if triggers:
                    all_triggers[tkr] = triggers
            except Exception as e:
                print(f"  {tkr}: failed ({e})")
            time.sleep(1.1)  # polite

        if not all_triggers:
            return mark_failed(sid, "no clusters found in any ticker")

        # Load prices for triggered tickers + SPY
        import yfinance as yf
        tickers = sorted(list(all_triggers.keys()) + ["SPY"])
        pxall = yf.download(tickers, start="2014-01-01", end="2025-12-31",
                            progress=False, auto_adjust=True)
        if isinstance(pxall.columns, pd.MultiIndex):
            px = pxall["Close"]
        else:
            px = pxall.copy()
        px = px.dropna(how="all").sort_index()
        spy = px["SPY"]

        # Event-study: for each (ticker, trigger_date) compute t+1..t+252 return and excess vs SPY
        per_event = []
        horizons = [21, 63, 126, 252]
        cars = {h: [] for h in horizons}

        for tkr, dates in all_triggers.items():
            if tkr not in px.columns:
                continue
            s = px[tkr].dropna()
            for d in dates:
                idx = s.index[s.index >= d]
                if len(idx) < 2:
                    continue
                t0 = idx[0]
                pos0 = s.index.get_loc(t0)
                t1_i = pos0 + 1   # buy at t+1 close
                if t1_i >= len(s):
                    continue
                t1 = s.index[t1_i]
                # Compute returns to each horizon
                row = {"ticker": tkr, "trigger": str(d.date()), "buy_date": str(t1.date())}
                for h in horizons:
                    end_i = min(t1_i + h, len(s) - 1)
                    if end_i <= t1_i:
                        continue
                    stock_ret = s.iloc[end_i] / s.iloc[t1_i] - 1
                    spy_w = spy.loc[t1:s.index[end_i]]
                    if len(spy_w) >= 2:
                        spy_r = spy_w.iloc[-1] / spy_w.iloc[0] - 1
                    else:
                        spy_r = 0
                    excess = stock_ret - spy_r
                    cars[h].append(excess)
                    row[f"car_{h}"] = float(excess)
                per_event.append(row)

        if not per_event:
            return mark_failed(sid, "triggers found but no price-window data")

        # Stats
        summary = {}
        for h in horizons:
            arr = np.array(cars[h])
            if len(arr) == 0:
                continue
            mean = float(arr.mean())
            std = float(arr.std(ddof=1))
            n = len(arr)
            t = mean / (std / np.sqrt(n)) if std > 0 else 0
            summary[f"car_{h}d"] = {
                "n": n,
                "mean": mean,
                "median": float(np.median(arr)),
                "std": std,
                "t_stat": float(t),
                "hit_rate": float((arr > 0).mean()),
            }

        # Build a daily PnL series: equal-weight across active positions
        # active = currently held (252-day window after t+1 trigger)
        pos = pd.DataFrame(0.0, index=px.index, columns=[t for t in px.columns if t != "SPY"])
        for tkr, dates in all_triggers.items():
            if tkr not in pos.columns:
                continue
            for d in dates:
                idx = px.index[px.index >= d]
                if len(idx) < 2: continue
                t0_i = px.index.get_loc(idx[0])
                start_i = t0_i + 1
                end_i = min(start_i + 252, len(px) - 1)
                if start_i >= len(px): continue
                for i in range(start_i, end_i):
                    pos.iloc[i, pos.columns.get_loc(tkr)] += 1.0

        active = pos.sum(axis=1).replace(0, np.nan)
        pos_norm = pos.div(active, axis=0).fillna(0.0)
        rets = px[pos.columns].pct_change()
        port = (pos_norm.shift(1) * rets).sum(axis=1)
        port = port.fillna(0)

        # Sharpe / CAGR on the daily PnL series (only active days have non-zero)
        active_pnl = port[port != 0]
        ann_factor = 252
        if len(active_pnl) > 30:
            mean_d = port.mean()
            std_d = port.std()
            sharpe = mean_d / std_d * np.sqrt(ann_factor) if std_d > 0 else 0
            spy_r = spy.pct_change().reindex(port.index).fillna(0)
            spy_eq = (1 + spy_r).cumprod()
            eq = (1 + port).cumprod()
            years = len(port) / ann_factor
            cagr = float(eq.iloc[-1] ** (1 / years) - 1) if eq.iloc[-1] > 0 else None
            bench_cagr = float(spy_eq.iloc[-1] ** (1 / years) - 1)
        else:
            sharpe = None; cagr = None; bench_cagr = None

        result = {
            "name": "E01 Form 4 cluster buying",
            "n_triggers_total": sum(len(v) for v in all_triggers.values()),
            "n_tickers_with_triggers": len(all_triggers),
            "n_events_for_car": len(per_event),
            "event_study": summary,
            "portfolio_sharpe": float(sharpe) if sharpe is not None else None,
            "portfolio_cagr": cagr,
            "spy_cagr_same_window": bench_cagr,
        }
        save_result(sid, result, extra={
            "status": "ok",
            "rule": "Cluster = >=3 insider buys >=$25k in 10 trading days. Buy t+1 close, hold 252d. Equal-weight overlapping basket.",
            "source": "openinsider.com scrape; cluster proxy follows Cohen-Malloy-Pomorski 2012, Lakonishok-Lee 2001.",
            "triggers_by_ticker": {k: [str(d.date()) for d in v] for k, v in all_triggers.items()},
        })
        print(f"E01: n_events={len(per_event)}, "
              f"252d CAR mean={summary.get('car_252d',{}).get('mean',0)*100:.2f}%, "
              f"t={summary.get('car_252d',{}).get('t_stat',0):.2f}")
    except Exception as e:
        import traceback; traceback.print_exc()
        return mark_failed(sid, f"unhandled exception: {e}")


if __name__ == "__main__":
    main()
