"""PL721_bis_entity_list_short_avgo_mrvl — BIS Entity List China Semi Adds -> Short AVGO/MRVL (Counter)"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "PL721_bis_entity_list_short_avgo_mrvl"
    tickers = ["AVGO", "MRVL", "SPY"]
    basket = ["AVGO", "MRVL"]

    try:
        px = load_prices(tickers, start="2018-01-01")
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    px = px.sort_index().ffill(limit=2)
    missing = [t for t in tickers if t not in px.columns]
    if missing:
        return mark_failed(sid, f"missing tickers: {missing}")

    ret = daily_returns(px)
    spy_r = ret["SPY"]

    # Hand-coded BIS Entity List major China semiconductor-related updates
    # Events with >=3 China semi supply-chain entity additions
    events = [
        # 2018-08 ZTE/Huawei-adjacent entities
        ("2018-08-01", 15),
        # 2019-05 Huawei Entity List
        ("2019-05-16", 15),
        # 2019-10 China surveillance tech (Hikvision, etc.)
        ("2019-10-07", 15),
        # 2020-05 Huawei expansion
        ("2020-05-15", 15),
        # 2020-08 SMIC addition
        ("2020-09-25", 15),
        # 2021-12 China semiconductor/AI companies
        ("2021-12-16", 15),
        # 2022-10 BIS October rules - major China semi export controls
        ("2022-10-07", 15),
        # 2023-10 BIS update - additional Chinese chip designers
        ("2023-10-17", 15),
        # 2024-12 BIS update
        ("2024-12-02", 15),
    ]

    all_dates = ret.index
    pnl = pd.Series(0.0, index=all_dates)
    n_events = 0

    for event_date_str, hold_days in events:
        event_date = pd.Timestamp(event_date_str)
        valid = all_dates[all_dates >= event_date]
        if len(valid) == 0:
            continue
        entry_date = valid[0]
        idx_start = all_dates.get_loc(entry_date)
        idx_end = min(idx_start + hold_days, len(all_dates))
        window = all_dates[idx_start:idx_end]

        # Short equal-weight basket: negative of returns
        basket_ret = ret[basket].reindex(window).fillna(0).mean(axis=1)
        pnl.loc[window] += -basket_ret  # short
        n_events += 1

    if n_events == 0:
        return mark_failed(sid, "no valid events found")

    m = compute_metrics(pnl, benchmark=spy_r, name="BIS Entity List -> Short AVGO/MRVL")
    m["n_events"] = n_events

    save_result(sid, m, extra={
        "rule": "When BIS adds >=3 China semi supply-chain entities in a single update, short equal-weight AVGO+MRVL for 15 trading days",
        "mechanism": "Export control escalations reduce addressable market for high China-revenue semis; AVGO/MRVL face direct revenue risk from restricted China customers and supply chain exposure",
        "source": "BIS Entity List Federal Register notices; AVGO/MRVL annual reports for China revenue disclosure",
        "events": [{"date": d, "hold_days": h} for d, h in events],
    })


if __name__ == "__main__":
    main()
