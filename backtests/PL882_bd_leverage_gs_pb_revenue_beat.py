"""PL882 — Fed Z.1 Broker-Dealer Leverage Expansion -> Long GS vs Short MS Pre-Earnings"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics, save_result, mark_failed, daily_returns)


# GS earnings dates (approximate: ~10-15th of Jan/Apr/Jul/Oct each year)
# Mid-month of each quarter earnings month: Q1 Apr, Q2 Jul, Q3 Oct, Q4 Jan
GS_EARNINGS_DATES = [
    # Format: quarterly earnings release dates (approximate; GS typically 2nd Mon Jan/Apr/Jul/Oct)
    "2000-01-18", "2000-04-18", "2000-07-18", "2000-10-17",
    "2001-01-16", "2001-04-17", "2001-07-17", "2001-10-16",
    "2002-01-15", "2002-04-16", "2002-07-16", "2002-10-15",
    "2003-01-14", "2003-04-15", "2003-07-15", "2003-10-14",
    "2004-01-13", "2004-04-13", "2004-07-13", "2004-10-19",
    "2005-01-18", "2005-04-19", "2005-07-19", "2005-10-18",
    "2006-01-17", "2006-04-18", "2006-07-18", "2006-10-17",
    "2007-01-16", "2007-04-17", "2007-07-17", "2007-10-16",
    "2008-01-15", "2008-04-15", "2008-07-15", "2008-10-14",
    "2009-01-13", "2009-04-14", "2009-07-14", "2009-10-15",
    "2010-01-19", "2010-04-20", "2010-07-20", "2010-10-19",
    "2011-01-19", "2011-04-19", "2011-07-19", "2011-10-18",
    "2012-01-17", "2012-04-17", "2012-07-17", "2012-10-16",
    "2013-01-16", "2013-04-16", "2013-07-16", "2013-10-15",
    "2014-01-16", "2014-04-15", "2014-07-15", "2014-10-16",
    "2015-01-21", "2015-04-16", "2015-07-16", "2015-10-15",
    "2016-01-20", "2016-04-19", "2016-07-19", "2016-10-18",
    "2017-01-18", "2017-04-18", "2017-07-18", "2017-10-17",
    "2018-01-17", "2018-04-17", "2018-07-17", "2018-10-16",
    "2019-01-16", "2019-04-15", "2019-07-16", "2019-10-15",
    "2020-01-15", "2020-04-15", "2020-07-15", "2020-10-14",
    "2021-01-19", "2021-04-14", "2021-07-13", "2021-10-15",
    "2022-01-18", "2022-04-14", "2022-07-18", "2022-10-18",
    "2023-01-17", "2023-04-18", "2023-07-19", "2023-10-17",
    "2024-01-16", "2024-04-15", "2024-07-15", "2024-10-15",
    "2025-01-15", "2025-04-14", "2025-07-14", "2025-10-14",
]


def main():
    sid = "PL882_bd_leverage_gs_pb_revenue_beat"
    try:
        # Load FRED Z.1 broker-dealer total financial assets
        bd = load_fred("BOGZ1FL664090005Q", start="1999-01-01")
        bd = bd.squeeze()
        bd = bd.dropna()
    except Exception as e:
        try:
            bd = load_fred("BOGZ1FL664090005Q", start="1999-01-01", cache=False)
            bd = bd.squeeze()
            bd = bd.dropna()
        except Exception as e2:
            return mark_failed(sid, f"FRED load: {e2}")

    if len(bd) < 20:
        return mark_failed(sid, f"Insufficient Z.1 data: {len(bd)} quarters")

    # Compute rolling 12-quarter percentile rank
    def rolling_pct_rank(series, window=12):
        ranks = []
        for i in range(len(series)):
            if i < window - 1:
                ranks.append(np.nan)
            else:
                window_vals = series.iloc[i - window + 1:i + 1]
                current = window_vals.iloc[-1]
                rank = (window_vals < current).sum() / (window - 1)
                ranks.append(float(rank))
        return pd.Series(ranks, index=series.index)

    bd_rank = rolling_pct_rank(bd, window=12)
    HIGH_LEVERAGE = 0.75

    # Mark which quarters have high leverage
    # Z.1 released ~4 weeks after quarter-end; build a lookup: date -> rank
    # For trading: if Q ends Mar 31, Z.1 released ~end of April -> use from May 1
    # Approximate: Z.1 quarter end dates are Mar/Jun/Sep/Dec
    # Add 6 weeks to quarter end to approximate release date
    bd_release = pd.Series(bd_rank.values, index=bd.index + pd.DateOffset(weeks=6))

    def get_leverage_rank_at(date):
        """Return the most recently available Z.1 leverage rank as of a given date."""
        past = bd_release[bd_release.index <= date]
        if len(past) == 0:
            return np.nan
        return past.iloc[-1]

    # Load prices
    try:
        px = load_prices(["GS", "MS", "XLF", "SPY"], start="2000-01-01")
        spy_r = daily_returns(px[["SPY"]]).iloc[:, 0]
        gs_r = daily_returns(px[["GS"]]).iloc[:, 0]
        ms_r = daily_returns(px[["MS"]]).iloc[:, 0]
    except Exception as e:
        try:
            px = load_prices(["GS", "MS", "XLF", "SPY"], start="2000-01-01", cache=False)
            spy_r = daily_returns(px[["SPY"]]).iloc[:, 0]
            gs_r = daily_returns(px[["GS"]]).iloc[:, 0]
            ms_r = daily_returns(px[["MS"]]).iloc[:, 0]
        except Exception as e2:
            return mark_failed(sid, f"price load: {e2}")

    HOLD = 10  # trading days before earnings

    events = []
    pnl_dates = []
    pnl_rets = []

    # Exclude GFC quarters (2008 Q3 to 2009 Q2)
    GFC_EXCLUDE_START = pd.Timestamp("2008-07-01")
    GFC_EXCLUDE_END = pd.Timestamp("2009-06-30")

    for earn_date_str in GS_EARNINGS_DATES:
        earn_date = pd.Timestamp(earn_date_str)

        # Skip GFC period
        if GFC_EXCLUDE_START <= earn_date <= GFC_EXCLUDE_END:
            continue

        # Check if earnings date is in price index
        future_idx = gs_r.index[gs_r.index <= earn_date]
        if len(future_idx) < HOLD:
            continue

        # Entry: 10 trading days before earnings
        earn_pos = gs_r.index.searchsorted(earn_date)
        entry_pos = max(0, earn_pos - HOLD)
        entry_date = gs_r.index[entry_pos]

        # Check leverage signal at entry date
        lev_rank = get_leverage_rank_at(entry_date)
        if np.isnan(lev_rank) or lev_rank < HIGH_LEVERAGE:
            continue

        # Compute pair return over window (entry to day before earnings)
        end_pos = min(earn_pos, len(gs_r))
        gs_window = gs_r.iloc[entry_pos:end_pos]
        ms_window = ms_r.reindex(gs_window.index).fillna(0)

        pair_r = gs_window - ms_window
        gs_cum = float((1 + gs_window).prod() - 1)
        ms_cum = float((1 + ms_window).prod() - 1)
        pair_cum = float((1 + pair_r).prod() - 1)

        # Check that GS didn't already outperform by >8% in prior 20 days (no chasing)
        pre_start = max(0, entry_pos - 20)
        prior_gs = gs_r.iloc[pre_start:entry_pos]
        prior_ms = ms_r.reindex(prior_gs.index).fillna(0)
        prior_outperf = float((1 + prior_gs).prod() - (1 + prior_ms).prod())
        if prior_outperf > 0.08:
            continue

        events.append({
            "earnings_date": earn_date_str,
            "entry_date": str(entry_date.date()),
            "leverage_rank": round(lev_rank, 3),
            "hold_days": end_pos - entry_pos,
            "gs_return": round(gs_cum, 4),
            "ms_return": round(ms_cum, 4),
            "pair_return": round(pair_cum, 4),
        })
        pnl_dates.extend(pair_r.index.tolist())
        pnl_rets.extend(pair_r.values.tolist())

    print(f"Z.1 data: {len(bd)} quarters, range {bd.index[0].date()} to {bd.index[-1].date()}")
    print(f"Events with high leverage (>75th pct): {len(events)}")
    if events:
        win_rate = np.mean([1 if e["pair_return"] > 0 else 0 for e in events])
        print(f"Win rate: {win_rate:.1%}, mean pair return: {np.mean([e['pair_return'] for e in events]):.2%}")

    if len(events) < 5:
        return mark_failed(sid, f"Too few events: {len(events)}")

    combined_pnl = pd.Series(pnl_rets, index=pd.DatetimeIndex(pnl_dates)).sort_index()
    if len(combined_pnl) < 30:
        return mark_failed(sid, f"insufficient combined days: {len(combined_pnl)}")

    m = compute_metrics(combined_pnl, benchmark=spy_r, name="Z.1 BD Leverage → Long GS / Short MS Pre-Earnings")
    win_rates = [1 if e["pair_return"] > 0 else 0 for e in events]

    save_result(sid, m, extra={
        "rule": "Long GS / Short MS for 10 trading days before GS quarterly earnings when Fed Z.1 broker-dealer financial assets rolling 12Q pct rank > 75%",
        "mechanism": "Elevated dealer balance sheets → GS prime brokerage revenue outperforms vs MS; pre-earnings drift captures expectation revision",
        "source": "FRED BOGZ1FL664090005Q (Fed Z.1 Flow of Funds, quarterly); yfinance GS/MS/SPY",
        "n_events": len(events),
        "event_win_rate": round(float(np.mean(win_rates)), 4),
        "events": events[:20],  # truncate for storage
        "caveats": "Z.1 has 4-6 week publication lag; earnings dates are approximate (mid-quarter); GFC period excluded; GS/MS business mix convergence may reduce edge post-2020",
    })
    print(f"Done: {len(events)} events, Sharpe={m.get('sharpe', 'N/A'):.3f}, CAGR={m.get('cagr', 0)*100:.1f}%")


if __name__ == "__main__":
    main()
