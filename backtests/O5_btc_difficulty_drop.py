"""
O5 BTC mining difficulty drop. When the Bitcoin network difficulty falls month-over-
month by more than 3%, long an equal-weighted miner basket {MARA, RIOT, CLSK} for
14 trading days.

Source: blockchain.info /charts/difficulty endpoint (free, all-history).
Mechanism: Difficulty drops follow hash-rate capitulation (miner shutdowns). The
survivors capture a larger share of constant block rewards; profitability per TH/s
spikes -> equity revaluation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json
import urllib.request

import pandas as pd
import numpy as np
from harness import (
    load_prices, daily_returns,
    compute_metrics, print_metrics, save_result, mark_failed,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)


def fetch_difficulty():
    fp = DATA / "blockchain_info_difficulty.parquet"
    if fp.exists():
        return pd.read_parquet(fp).iloc[:, 0]
    url = "https://api.blockchain.info/charts/difficulty?format=json&timespan=all"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = urllib.request.urlopen(req, timeout=60).read()
    j = json.loads(raw)
    vals = j["values"]
    df = pd.DataFrame(vals)
    df["date"] = pd.to_datetime(df["x"], unit="s")
    df = df.set_index("date")[["y"]].rename(columns={"y": "difficulty"})
    df.to_parquet(fp)
    return df.iloc[:, 0]


def main():
    try:
        diff = fetch_difficulty()
        # MARA, RIOT, CLSK basket; will adjust for availability
        miners = load_prices(["MARA", "RIOT", "CLSK"], start="2017-01-01")
    except Exception as e:
        return mark_failed("O5_btc_difficulty_drop", f"data load failed: {e}")

    # Difficulty adjusts every ~2 weeks. Compute 14-day change (one adjustment epoch).
    # Resample to daily forward-fill (so each calendar day has latest difficulty)
    diff_d = diff.resample("D").ffill().dropna()
    # 14-day change (one difficulty epoch)
    mom = diff_d.pct_change(14)

    # Trigger when MoM drop < -3%, only count each adjustment once
    trig_mask = mom < -0.03
    # Detect first day of consecutive trigger spell
    rising = trig_mask & ~trig_mask.shift(1, fill_value=False)
    triggers = mom.index[rising]
    n_events = len(triggers)

    # Build basket returns: equal-weight average where ticker is alive
    basket = miners.pct_change()
    basket_ret = basket.mean(axis=1)  # average of available
    pos = pd.Series(0.0, index=basket_ret.index)

    hold = 14
    for d in triggers:
        loc = basket_ret.index.searchsorted(d)
        for k in range(1, hold + 1):
            if loc + k < len(pos):
                pos.iloc[loc + k] = 1.0

    if n_events < 3:
        return mark_failed("O5_btc_difficulty_drop", f"only {n_events} events", extra={"n_events": int(n_events)})

    pnl = (pos * basket_ret).dropna()
    # benchmark: BTC-USD
    try:
        btc = load_prices(["BTC-USD"], start="2017-01-01").iloc[:, 0].pct_change()
        bench = btc.reindex(pnl.index)
    except Exception:
        bench = basket_ret.reindex(pnl.index)
    m = compute_metrics(pnl, benchmark=bench, name="O5 BTC difficulty drop -> long miners 14d")
    m["n_events"] = int(n_events)
    print(f"Triggers: {n_events}; first/last: {triggers[0].date()} ... {triggers[-1].date()}")
    print_metrics(m)
    save_result("O5_btc_difficulty_drop", m, extra={
        "status": "ok",
        "rule": "When BTC network difficulty 14-day change < -3% (one adjustment epoch), long {MARA,RIOT,CLSK} equal-weight for 14 sessions.",
        "mechanism": "Difficulty drops follow hash-rate capitulation -> survivors capture constant rewards, profitability per TH spikes.",
        "universe": "MARA, RIOT, CLSK",
        "source": "blockchain.info /charts/difficulty (free, full history)",
        "n_events": int(n_events),
    })


if __name__ == "__main__":
    main()
