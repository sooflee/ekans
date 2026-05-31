"""PL576_leveraged_etf_rebalance_overnight_fade - 3x ETF MOC Rebalance Overnight Fade
On +2% QQQ day: short QQQ at close, cover next-day close. Symmetric for -2% days (long).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from harness import load_prices, compute_metrics, save_result, mark_failed, daily_returns


def main():
    sid = "PL576_leveraged_etf_rebalance_overnight_fade"
    try:
        px = load_prices(["QQQ", "SPY"], start="2010-01-01")
    except Exception as e:
        return mark_failed(sid, f"price load: {e}")
    ret = daily_returns(px)
    qqq = ret["QQQ"].dropna()
    spy_r = ret["SPY"].dropna()

    # Signal: position for next-day = -sign(qqq_t) when |qqq_t| >= 2%
    pos = pd.Series(0.0, index=qqq.index)
    big_up = qqq >= 0.02
    big_dn = qqq <= -0.02
    pos[big_up] = -1.0
    pos[big_dn] = +1.0
    # PnL = pos_t * next-day return
    next_ret = qqq.shift(-1)
    pnl = (pos * next_ret).dropna()
    # Only include days where we had a position
    pnl = pnl[pos.shift(0).reindex(pnl.index).abs() > 0]
    if len(pnl) < 30:
        return mark_failed(sid, f"insufficient signal days ({len(pnl)})")

    m = compute_metrics(pnl, benchmark=spy_r, name="3x ETF MOC Rebalance Overnight Fade")
    save_result(sid, m, extra={
        "rule": "On QQQ +/- 2% day, take opposite position at close, exit next-day close.",
        "mechanism": "Leveraged ETF MOC rebalance flow imbalance -> overnight mean reversion",
        "source": "yfinance QQQ",
        "n_events": int(pos.abs().sum()),
    })
    print(f"Done {sid}: events={int(pos.abs().sum())} Sharpe={m.get('sharpe',0):.2f} CAGR={m.get('cagr',0)*100:.1f}%")


if __name__ == "__main__":
    main()
