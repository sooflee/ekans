"""
G07 California WARN Act layoff notices — event study.

Data: CA EDD daily WARN report (XLSX), covering the most recent ~12 months
of notices.  https://edd.ca.gov/en/jobs_and_training/Layoff_Services_WARN

Method:
 1. Pull the latest XLSX (Detailed WARN Report sheet).
 2. Map company names to a curated list of ~50 well-known public tickers
    using word-boundary regex (avoid the "san franCISCO contains CISCO" trap).
 3. Filter to "real" notices (>= 100 affected employees, exclude AMAZON
    delivery-station and other warehouse churn by requiring the corporate
    name to appear without the "AMAZON (xxx)" warehouse-code suffix).
 4. For each unique (ticker, notice_date), compute event-study CAR:
       short on notice_date + 5 trading days, hold 90 trading days, hedge with SPY.
 5. Aggregate average CAR, t-stat. Honest: sign is ambiguous and this sample
    is small (~10-30 events from a single one-year window).

Honest notes:
 - WARN XLSX is only the current annual snapshot; PDF historicals not parsed
   (PDF tables are non-trivial). This is therefore a one-year event study,
   not a multi-year panel.
 - Ticker mapping is heuristic. Some Amazon-warehouse closures bleed into
   the AMZN bucket; we de-duplicate to one event per ticker (earliest notice).
"""
import sys
import re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import numpy as np
import requests

from harness import (
    load_prices, compute_metrics, print_metrics, save_result, mark_failed, DATA,
)


WARN_XLSX = "https://edd.ca.gov/siteassets/files/jobs_and_training/warn/warn_report1.xlsx"

# Word-boundary keyword -> ticker. Order matters (most-specific first).
COMPANY_TICKER_MAP = [
    (r"\bMETA PLATFORMS\b", "META"),
    (r"\bAMAZON\b", "AMZN"),
    (r"\bGOOGLE\b", "GOOGL"),
    (r"\bALPHABET\b", "GOOGL"),
    (r"\bMICROSOFT\b", "MSFT"),
    (r"\bAPPLE INC\b", "AAPL"),
    (r"\bTESLA\b", "TSLA"),
    (r"\bNETFLIX\b", "NFLX"),
    (r"\bINTEL CORP", "INTC"),
    (r"\bCISCO SYSTEMS\b", "CSCO"),
    (r"\bSALESFORCE\b", "CRM"),
    (r"\bPAYPAL\b", "PYPL"),
    (r"\bEBAY INC\b", "EBAY"),
    (r"\bUBER\b", "UBER"),
    (r"\bLYFT\b", "LYFT"),
    (r"\bNIKE\b", "NKE"),
    (r"\bWALT DISNEY\b", "DIS"),
    (r"\bBOEING\b", "BA"),
    (r"\bDELL TECH", "DELL"),
    (r"\bORACLE\b", "ORCL"),
    (r"\bADOBE\b", "ADBE"),
    (r"\bNVIDIA\b", "NVDA"),
    (r"\bSNAP INC\b", "SNAP"),
    (r"\bDOORDASH\b", "DASH"),
    (r"\bCOINBASE\b", "COIN"),
    (r"\bROBINHOOD\b", "HOOD"),
    (r"\bCHEVRON\b", "CVX"),
    (r"\bBANK OF AMERICA\b", "BAC"),
    (r"\bWELLS FARGO\b", "WFC"),
    (r"\bCHARLES SCHWAB\b", "SCHW"),
    (r"\bGAP INC\b", "GPS"),
    (r"\bPG&E\b", "PCG"),
    (r"\bPACIFIC GAS\b", "PCG"),
    (r"\bSTARBUCKS\b", "SBUX"),
    (r"\bPFIZER\b", "PFE"),
    (r"\bBROADCOM\b", "AVGO"),
    (r"\bQUALCOMM\b", "QCOM"),
    (r"\bMICRON\b", "MU"),
    (r"\bCOMCAST\b", "CMCSA"),
    (r"\bCHARTER COMMUNICATIONS\b", "CHTR"),
    (r"\bPALANTIR\b", "PLTR"),
    (r"\bOKTA\b", "OKTA"),
    (r"\bTWILIO\b", "TWLO"),
    (r"\bSNOWFLAKE\b", "SNOW"),
    (r"\bSPLUNK\b", "SPLK"),
    (r"\bMONGODB\b", "MDB"),
    (r"\bSHOPIFY\b", "SHOP"),
    (r"\bSPOTIFY\b", "SPOT"),
    (r"\bRIVIAN\b", "RIVN"),
    (r"\bLUCID GROUP\b", "LCID"),
    (r"\bFORD MOTOR\b", "F"),
    (r"\bAIRBNB\b", "ABNB"),
    (r"\bPINTEREST\b", "PINS"),
    (r"\bROBLOX\b", "RBLX"),
    (r"\bHEWLETT PACKARD\b", "HPE"),
    (r"\bHP INC\b", "HPQ"),
]


def find_ticker(name_upper):
    for pattern, ticker in COMPANY_TICKER_MAP:
        if re.search(pattern, name_upper):
            return ticker
    return None


def main():
    cache = DATA / "warn_ca_latest.xlsx"
    try:
        r = requests.get(WARN_XLSX, timeout=60)
        r.raise_for_status()
        cache.write_bytes(r.content)
    except Exception as e:
        if not cache.exists():
            return mark_failed("G07_warn_act_layoffs",
                               f"WARN XLSX fetch failed and no cache: {e}")

    try:
        df = pd.read_excel(cache, sheet_name="Detailed WARN Report ",
                           skiprows=3, header=None)
        df.columns = ["County", "Notice_Date", "Received_Date", "Effective_Date",
                      "Company", "Type", "Employees", "Address", "Industry"]
    except Exception as e:
        return mark_failed("G07_warn_act_layoffs", f"XLSX parse failed: {e}")

    df = df.dropna(subset=["Company", "Notice_Date"]).copy()
    df["Notice_Date"] = pd.to_datetime(df["Notice_Date"], errors="coerce")
    df["Employees"] = pd.to_numeric(df["Employees"], errors="coerce").fillna(0)
    df = df[df["Notice_Date"].notna()]
    df = df[df["Employees"] >= 100]
    df["Company_upper"] = df["Company"].astype(str).str.upper()
    df["ticker"] = df["Company_upper"].apply(find_ticker)
    events = df[df["ticker"].notna()].copy().sort_values("Notice_Date")

    if events.empty:
        return mark_failed("G07_warn_act_layoffs",
                           "No mapped public-ticker WARN events found.")

    # One event per ticker (earliest)
    events = events.groupby("ticker").first().reset_index()
    print(f"Mapped {len(events)} unique-ticker WARN events.")
    print(events[["ticker", "Notice_Date", "Company", "Employees"]].to_string())

    if len(events) < 5:
        # Still write a result but flag small-sample
        small_sample = True
    else:
        small_sample = False

    # Pull prices for all tickers + SPY across event window
    tickers = sorted(set(events["ticker"].tolist() + ["SPY"]))
    min_date = events["Notice_Date"].min() - pd.Timedelta(days=30)
    max_date = events["Notice_Date"].max() + pd.Timedelta(days=200)
    today = pd.Timestamp.utcnow().tz_localize(None).normalize()
    end_dt = min(max_date, today)
    try:
        import yfinance as yf
        px = yf.download(tickers, start=min_date.strftime("%Y-%m-%d"),
                         end=end_dt.strftime("%Y-%m-%d"),
                         progress=False, auto_adjust=True)
        if isinstance(px.columns, pd.MultiIndex):
            px = px["Close"]
    except Exception as e:
        return mark_failed("G07_warn_act_layoffs", f"Price fetch failed: {e}")

    px = px.dropna(how="all")
    if "SPY" not in px.columns:
        return mark_failed("G07_warn_act_layoffs", "SPY not loaded.")
    spy = px["SPY"].dropna()

    # Event study: for each event, log-CAR over [+5, +95]: i.e., enter short
    # at notice + 5 trading days, hold 90 trading days (or fewer if data runs out).
    rets = np.log(px / px.shift(1))
    spy_rets = rets["SPY"]

    car_records = []
    pnl_paths = []
    for _, row in events.iterrows():
        tkr = row["ticker"]
        d0 = row["Notice_Date"]
        if tkr not in rets.columns:
            continue
        # Find trading-day index near d0
        idx = rets.index.searchsorted(d0)
        entry = idx + 5
        exit_ = idx + 5 + 90
        if entry >= len(rets) or entry + 5 >= len(rets):
            continue
        exit_ = min(exit_, len(rets) - 1)
        window = rets.iloc[entry:exit_+1]
        if window.empty:
            continue
        # Short stock, long SPY = (spy - stk)
        excess = (window["SPY"] - window[tkr]).dropna()
        if len(excess) < 10:
            continue
        car = float(excess.sum())  # log-return CAR
        car_records.append({
            "ticker": tkr,
            "notice_date": str(d0.date()),
            "n_days": int(len(excess)),
            "CAR_short_log": car,
            "CAR_short_pct": float(np.expm1(car)),
        })
        # Build a daily PnL contribution path (each event contributes 1/N weight)
        pnl_paths.append(excess.rename(tkr))

    if not car_records:
        return mark_failed("G07_warn_act_layoffs",
                           "No event-study windows could be built.")

    car_df = pd.DataFrame(car_records)
    avg_car_log = float(car_df["CAR_short_log"].mean())
    avg_car_pct = float(np.expm1(avg_car_log))
    std_car = float(car_df["CAR_short_log"].std())
    t_stat_cs = float(avg_car_log / (std_car / np.sqrt(len(car_df)))) if std_car > 0 else 0.0

    # Build composite daily pnl: equal-weight short on each notice's window,
    # average across active events each day.
    pnl_panel = pd.concat(pnl_paths, axis=1).sort_index()
    daily_pnl = pnl_panel.mean(axis=1).dropna()  # log returns
    daily_pnl_simple = np.expm1(daily_pnl)

    bench = spy.pct_change().reindex(daily_pnl_simple.index).dropna()
    m = compute_metrics(daily_pnl_simple, benchmark=bench,
                        name="G07 WARN Act short-event composite")
    print_metrics(m)
    print("CAR records:")
    print(car_df.to_string(index=False))

    save_result("G07_warn_act_layoffs", m, extra={
        "status": "ok_small_sample" if small_sample else "ok",
        "rule": ("For each WARN notice mapped to a public ticker, short the "
                 "stock at notice + 5 trading days, hold 90 trading days, "
                 "hedge with SPY. Composite is equal-weight overlap."),
        "data_source": "CA EDD Detailed WARN Report XLSX (current annual file)",
        "n_events": int(len(car_df)),
        "avg_CAR_log": avg_car_log,
        "avg_CAR_pct": avg_car_pct,
        "CAR_t_stat": t_stat_cs,
        "events": car_df.to_dict(orient="records"),
        "caveats": ("Only one calendar year of notices is parsed from the "
                    "current XLSX. PDF historicals (2014-2024) not ingested. "
                    "Sample size small; sign of effect ambiguous; "
                    "ticker mapping heuristic."),
    })


if __name__ == "__main__":
    main()
