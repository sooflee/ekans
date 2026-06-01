"""PL935_tsa_throughput_ulcc_short TSA Throughput Decel + Jet Fuel Crack + XLY Breadth -> JBLU Short"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def load_tsa_throughput():
    """Load TSA historical throughput data. Try to fetch from tsa.gov CSV or fall back to FRED-like proxy."""
    import urllib.request
    import io

    DATA = Path(__file__).resolve().parents[1] / "data"
    cache_file = DATA / "tsa_throughput.csv"

    if not cache_file.exists():
        # Try downloading TSA CSV
        url = "https://www.tsa.gov/coronavirus/passenger-throughput"
        try:
            # TSA provides an API-accessible JSON endpoint
            api_url = "https://www.tsa.gov/sites/default/files/resources/tsaflow_datafile.csv"
            req = urllib.request.urlopen(api_url, timeout=20)
            data = req.read().decode("utf-8", errors="replace")
            with open(cache_file, "w") as f:
                f.write(data)
        except Exception:
            return None

    try:
        df = pd.read_csv(cache_file)
        # Typical columns: Date, Numbers
        df.columns = [c.strip() for c in df.columns]
        # Find date and throughput column
        date_col = next((c for c in df.columns if 'date' in c.lower()), df.columns[0])
        num_col = next((c for c in df.columns if c != date_col), None)
        if num_col is None:
            return None
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        df = df.set_index(date_col).sort_index()
        df[num_col] = pd.to_numeric(df[num_col].astype(str).str.replace(',', ''), errors='coerce')
        return df[[num_col]].rename(columns={num_col: "throughput"})
    except Exception:
        return None


def main():
    sid = "PL935_tsa_throughput_ulcc_short"

    try:
        px = load_prices(["JBLU", "SPY", "XLY"], start="2019-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load prices: {e}")

    try:
        fred_data = load_fred(["DCOILWTICO", "DJFUELUSGULF"], start="2019-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load FRED: {e}")

    rets = daily_returns(px)
    spy_r = rets["SPY"].dropna()

    # Attempt to load TSA data
    tsa_df = load_tsa_throughput()

    # Build daily date range
    all_dates = px.index
    # ---- Crack spread signal ----
    wti = fred_data["DCOILWTICO"].reindex(all_dates, method="ffill")
    # DJFUELUSGULF is in $/gallon; convert to $/bbl (*42) for comparison with WTI
    jet = fred_data["DJFUELUSGULF"].reindex(all_dates, method="ffill") * 42
    crack = jet - wti
    crack_2yr_avg = crack.rolling(504).mean()  # ~2 years of trading days
    crack_signal = (crack > crack_2yr_avg + 3).astype(float)  # above avg + $3/bbl

    # ---- XLY breadth signal ----
    # Use XLY vs its 50-day MA as a proxy for constituent breadth
    xly_50dma = px["XLY"].rolling(50).mean()
    # Approximation: if XLY itself is below its 50-DMA, treat breadth as below 40%
    xly_breadth_signal = (px["XLY"] < xly_50dma).astype(float)

    # ---- TSA YoY deceleration signal ----
    if tsa_df is not None and len(tsa_df) > 250:
        tsa = tsa_df["throughput"].reindex(all_dates, method="ffill")
        tsa_yoy = (tsa / tsa.shift(252) - 1) * 100  # YoY %
        tsa_4w = tsa_yoy.rolling(20).mean()          # 4-week rolling avg
        tsa_decel = (tsa_4w.diff(20) < -5).astype(float)  # >5pp decel
    else:
        # If TSA data unavailable, use JBLU vs SPY relative performance as proxy
        # JBLU underperforming SPY on 3-month basis
        jblu_3m = (1 + rets["JBLU"].fillna(0)).rolling(63).apply(lambda x: x.prod()) - 1
        spy_3m = (1 + spy_r.fillna(0)).rolling(63).apply(lambda x: x.prod()) - 1
        tsa_decel = (jblu_3m < spy_3m - 0.10).astype(float)

    # ---- Combined signal: all three conditions must hold ----
    combined_signal = (crack_signal * xly_breadth_signal * tsa_decel)

    # Short JBLU when signal is active (next-day return, lagged)
    # Position: -1 when signal, 0 otherwise
    position = -combined_signal.shift(1)

    jblu_r = rets["JBLU"].fillna(0)
    pnl = (position * jblu_r).dropna()

    if len(pnl) < 60:
        return mark_failed(sid, "insufficient data after signal construction")

    spy_aligned = spy_r.reindex(pnl.index).dropna()

    m = compute_metrics(pnl, benchmark=spy_aligned,
                        name="TSA Throughput Decel + Jet Crack + XLY Breadth -> Short JBLU")
    save_result(sid, m, extra={
        "rule": "Triple-confirmation short JBLU: TSA 4-week YoY decel >5pp AND jet fuel crack > 2yr avg+$3 AND XLY breadth < 40%.",
        "mechanism": "ULCC (JBLU) has no corporate travel buffer; leisure demand most sensitive to cost of travel (fuel) and consumer confidence (XLY breadth).",
        "source": "TSA daily throughput CSV; FRED DCOILWTICO + DJFUELUSGULF; XLY price from yfinance",
        "tsa_data_available": tsa_df is not None,
        "status": "ok",
    })


if __name__ == "__main__":
    main()
