"""
AN-2: Forced Remote → Long Suburban Builder / Short Urban Luxury
Regime trade: long TMHC vs short AIV from Mar 2022 (remote work stabilized +
rate hikes began crushing urban luxury) through present.
"""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import pandas as pd
import numpy as np
from backtests.harness import (
    load_prices, daily_returns, compute_metrics,
    save_result, mark_failed, print_metrics,
)

SIGNAL_ID = "AN-2"
ENTRY_DATE = "2022-03-01"
LONG_TICKER = "TMHC"
SHORT_TICKER = "AIV"


def run():
    print("=== AN-2: Remote Work Regime → Long TMHC / Short AIV ===\n")
    print(f"Entry: {ENTRY_DATE} (remote work stabilized + rate hikes begin)")
    print(f"Long: {LONG_TICKER} (suburban builder)")
    print(f"Short: {SHORT_TICKER} (urban luxury apartments)\n")

    # --- Load prices ---
    tickers = [LONG_TICKER, SHORT_TICKER, "SPY"]
    prices = load_prices(tickers, start="2020-01-01")
    rets = daily_returns(prices)

    entry = pd.Timestamp(ENTRY_DATE)
    mask = rets.index >= entry
    window_rets = rets.loc[mask]

    if len(window_rets) < 30:
        mark_failed(SIGNAL_ID, "Insufficient data after entry date",
                     extra={"rule": "Long TMHC / Short AIV from Mar 2022",
                            "mechanism": "Remote work → suburban housing demand, urban luxury apartments suffer",
                            "source": "yfinance"})
        return

    # Long TMHC, short AIV, equal weight each leg (50/50)
    port_ret = 0.5 * window_rets[LONG_TICKER].fillna(0) - 0.5 * window_rets[SHORT_TICKER].fillna(0)

    # Individual legs
    long_cum = (1 + window_rets[LONG_TICKER].fillna(0)).cumprod()
    short_cum = (1 + window_rets[SHORT_TICKER].fillna(0)).cumprod()
    spy_cum = (1 + window_rets["SPY"].fillna(0)).cumprod()

    print(f"Period: {window_rets.index[0].date()} → {window_rets.index[-1].date()} ({len(window_rets)} days)")
    print(f"  TMHC total return: {(long_cum.iloc[-1]-1)*100:+.1f}%")
    print(f"  AIV total return:  {(short_cum.iloc[-1]-1)*100:+.1f}%")
    print(f"  SPY total return:  {(spy_cum.iloc[-1]-1)*100:+.1f}%")

    spy_rets = rets["SPY"].reindex(port_ret.index)
    metrics = compute_metrics(port_ret, benchmark=spy_rets, name="AN-2 Long TMHC / Short AIV")
    print_metrics(metrics)

    trade_detail = {
        "entry": ENTRY_DATE,
        "exit": str(window_rets.index[-1].date()),
        "tmhc_total_return": float(long_cum.iloc[-1] - 1),
        "aiv_total_return": float(short_cum.iloc[-1] - 1),
        "spy_total_return": float(spy_cum.iloc[-1] - 1),
        "n_days": len(window_rets),
    }

    save_result(SIGNAL_ID, metrics, extra={
        "rule": "Long TMHC / Short AIV from Mar 2022 onward (regime trade)",
        "mechanism": "Remote work permanence → suburban housing demand up; rate hikes + WFH → urban luxury apartment vacancy/distress",
        "source": "yfinance",
        "trades": [trade_detail],
        "status": "ok",
    })
    print(f"\nSaved → results/{SIGNAL_ID}.json")


if __name__ == "__main__":
    run()
