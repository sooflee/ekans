"""PL965_howey_ruling_cluster_bkch_short_ibit_long
SEC Howey-Test Win Cluster -> Short BKCH / Long IBIT (Crypto-Equity vs Crypto-Spot Pair)

Event study: On SEC enforcement cluster dates (3+ Howey-test rulings in 60 days),
enter Short BKCH / Long IBIT (or BTC-USD pre-2024) for 25 trading days.
Strategy thesis: SEC enforcement against crypto-token issuers punishes blockchain-equity
plays (BKCH) more than spot BTC, as BTC is increasingly treated as commodity.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL965_howey_ruling_cluster_bkch_short_ibit_long"

    try:
        px_main = load_prices(["BKCH", "BLOK", "SPY"], start="2020-01-01")
        px_btc = load_prices(["BTC-USD"], start="2020-01-01")
        px_ibit = load_prices(["IBIT"], start="2024-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    # Retry on failure
    missing_main = [t for t in ["BKCH", "BLOK", "SPY"] if t not in px_main.columns]
    if missing_main:
        try:
            px_main = load_prices(["BKCH", "BLOK", "SPY"], start="2020-01-01")
            missing_main = [t for t in ["BKCH", "BLOK", "SPY"] if t not in px_main.columns]
        except Exception as e:
            return mark_failed(sid, f"data reload: {e}")
    if missing_main:
        return mark_failed(sid, f"missing tickers: {missing_main}")

    ret_main = daily_returns(px_main)
    spy_r = ret_main["SPY"].dropna()

    # Build the "spot BTC" leg: use IBIT from 2024, BTC-USD before that
    btc_r = daily_returns(px_btc).rename(columns={"BTC-USD": "BTC"}).get("BTC", pd.Series(dtype=float))

    if "IBIT" in px_ibit.columns:
        ibit_r = daily_returns(px_ibit[["IBIT"]]).iloc[:, 0]
    else:
        ibit_r = pd.Series(dtype=float)

    # Combine: IBIT where available, BTC-USD otherwise
    if len(ibit_r) > 0:
        spot_r = ibit_r.reindex(btc_r.index.union(ibit_r.index))
        spot_r = spot_r.fillna(btc_r.reindex(spot_r.index))
    else:
        spot_r = btc_r

    # BKCH (from Jul 2021) and BLOK (from Jan 2020)
    bkch_r = ret_main.get("BKCH", pd.Series(dtype=float)).dropna() if "BKCH" in ret_main.columns else pd.Series(dtype=float)
    blok_r = ret_main.get("BLOK", pd.Series(dtype=float)).dropna() if "BLOK" in ret_main.columns else pd.Series(dtype=float)

    # Build crypto-equity short leg: BKCH where available, else BLOK
    if len(bkch_r) > 0:
        crypto_eq_r = bkch_r.reindex(blok_r.index.union(bkch_r.index))
        crypto_eq_r = crypto_eq_r.fillna(blok_r.reindex(crypto_eq_r.index))
    else:
        crypto_eq_r = blok_r

    # Known SEC enforcement cluster event dates
    # Each entry is the date of the triggering enforcement action
    # (the 3rd in a rolling 60-day cluster - approximated from known_events)
    known_event_dates = pd.to_datetime([
        "2022-11-07",   # SEC v. LBRY ruling (LBC security)
        "2023-06-06",   # SEC v. Coinbase + SEC v. Binance same week
        "2023-11-20",   # SEC v. Kraken complaint
    ])

    # Find nearest trading day in our universe
    all_dates = px_main.index.sort_values()
    deduped_events = []
    for d in known_event_dates:
        if d < all_dates[0] or d > all_dates[-1]:
            continue
        diffs = abs(all_dates - d)
        nearest = all_dates[diffs.argmin()]
        deduped_events.append(nearest)

    if len(deduped_events) < 2:
        return mark_failed(sid, f"too few event dates: {len(deduped_events)}")

    hold_days = 25
    stop_loss_pct = 0.08     # exit if pair moves against us by 8%
    take_profit_pct = 0.15   # exit if pair moves for us by 15%

    pnl_series = pd.Series(0.0, index=all_dates)
    n_events = 0

    for entry_date in deduped_events:
        if entry_date not in all_dates:
            continue
        entry_iloc = all_dates.get_loc(entry_date)
        end_iloc = min(entry_iloc + hold_days, len(all_dates) - 1)
        n_events += 1
        cumulative_pair = 0.0

        for j in range(entry_iloc + 1, end_iloc + 1):
            date_j = all_dates[j]
            short_r_j = crypto_eq_r.get(date_j, 0.0) if date_j in crypto_eq_r.index else 0.0
            long_r_j = spot_r.get(date_j, 0.0) if date_j in spot_r.index else 0.0

            # Short crypto-equity, Long spot BTC proxy
            day_pnl = long_r_j - short_r_j
            pnl_series[date_j] = pnl_series[date_j] + day_pnl
            cumulative_pair += day_pnl

            if cumulative_pair < -stop_loss_pct:
                break
            if cumulative_pair > take_profit_pct:
                break

    first_nonzero = pnl_series[pnl_series != 0].first_valid_index()
    if first_nonzero is None:
        return mark_failed(sid, "no trades executed")
    pnl = pnl_series.loc[first_nonzero:]
    spy_r_aligned = spy_r.reindex(pnl.index).fillna(0)

    if len(pnl) < 30:
        return mark_failed(sid, f"too few trading days: {len(pnl)}")

    m = compute_metrics(pnl, benchmark=spy_r_aligned,
                        name="SEC Howey Cluster: Short BKCH / Long BTC-IBIT")
    m["n_events"] = n_events

    save_result(sid, m, extra={
        "rule": "Short BKCH / Long IBIT-or-BTC for 25 trading days on SEC enforcement cluster (3+ Howey rulings in 60 days)",
        "mechanism": "SEC enforcement targeting token issuers punishes blockchain-equity stocks (BKCH) while spot BTC increasingly classified as commodity, creating divergence",
        "source": "SEC litigation releases: SEC v. LBRY (Nov 2022), SEC v. Coinbase/Binance (Jun 2023), SEC v. Kraken (Nov 2023)",
        "status": "ok",
    }, pnl=pnl)


if __name__ == "__main__":
    main()
