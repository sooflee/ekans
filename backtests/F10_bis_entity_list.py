"""
F10 BIS Entity List China-tech additions.

BIS (Bureau of Industry and Security) publishes Entity List additions on Fridays
in the Federal Register. China-tech targeted additions tend to benefit US
semiconductor capital equipment (KLAC, AMAT, LRCX) and test/measurement (KEYS).

For each event: long equal-weight basket KLAC+AMAT+LRCX+KEYS at next open,
hold 3 trading days.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from harness import load_prices, compute_metrics, print_metrics, save_result, mark_failed

# Curated BIS Entity List addition dates (Federal Register publication) targeting
# Chinese-tech entities. Sources: BIS press releases / Federal Register.
EVENTS = [
    "2018-08-01",  # 44 PLA-linked entities
    "2018-10-30",  # Fujian Jinhua
    "2019-05-16",  # Huawei + 68 affiliates
    "2019-08-21",  # Huawei add 46 more affiliates
    "2019-10-09",  # 28 Chinese AI/surveillance (Hikvision, iFlyTek, Megvii, etc.)
    "2020-05-22",  # 24 entities (Qihoo 360, CloudMinds)
    "2020-06-05",  # 33 China affiliates
    "2020-07-22",  # 11 China human-rights additions
    "2020-08-26",  # 24 China islands construction
    "2020-12-18",  # SMIC, DJI, +60 others (major event)
    "2021-04-08",  # 7 China supercomputing
    "2021-06-24",  # 5 China solar-poly silicon
    "2021-07-09",  # 23 China Xinjiang surveillance
    "2021-11-24",  # 27 entities incl. China quantum
    "2022-02-07",  # 33 entities Russia + China items
    "2022-08-23",  # YMTC + others to UVL/Entity
    "2022-10-07",  # Sweeping semiconductor export controls (major)
    "2022-12-15",  # YMTC + 35 China (CXMT, Pengxinwei)
    "2023-03-02",  # 28 China (BGI, Inspur, etc.)
    "2023-10-17",  # Updated AI chip rules + entities (major)
    "2024-03-25",  # 4 China advanced computing
    "2024-05-07",  # 37 China & Russia
    "2024-09-23",  # 9 China quantum
    "2024-12-02",  # 140 China semi-equipment (major)
    "2024-12-13",  # +20 China
    "2025-01-15",  # 14 China AI/biotech
    "2025-03-25",  # 70+ entities (Inspur subs, AI)
    "2025-05-13",  # 13 China advanced computing
    "2025-07-25",  # additional list update
    "2025-10-17",  # 50/50 affiliates rule entities
]


def main():
    sid = "F10_bis_entity_list"
    try:
        tickers = ["KLAC", "AMAT", "LRCX", "KEYS"]
        px = load_prices(tickers + ["SPY", "SOXX"], start="2017-06-01")
        rets = px.pct_change()
        idx = rets.index

        pos = pd.DataFrame(0.0, index=idx, columns=tickers)
        used = []
        for d in EVENTS:
            Dts = pd.Timestamp(d)
            loc = idx.searchsorted(Dts, side="right")
            if loc >= len(idx):
                continue
            start = loc
            end = min(loc + 3, len(idx) - 1)
            for t in tickers:
                pos.iloc[start:end + 1, pos.columns.get_loc(t)] = 1.0 / len(tickers)
            used.append(d)

        port = (pos.shift(1) * rets[tickers]).sum(axis=1)
        active = (pos.abs().sum(axis=1) > 0)
        port_active = port[active.shift(1).fillna(False)].dropna()
        if len(port_active) < 10:
            return mark_failed(sid, "Too few active days")

        m = compute_metrics(port_active, benchmark=rets["SPY"].reindex(port_active.index),
                            name="F10 BIS Entity List -> US semi-equip")
        print_metrics(m)
        save_result(sid, m, extra={
            "status": "ok",
            "rule": "On each BIS Entity List China-tech addition (Federal Register pub date), "
                    "long equal-weight KLAC+AMAT+LRCX+KEYS at next open, hold 3 trading days.",
            "mechanism": "Tightening export controls force China to substitute domestic supply, "
                         "but also slow tech transfer; market initially reads US capex moat / "
                         "service revenue protection.",
            "universe": "KLAC, AMAT, LRCX, KEYS",
            "n_events": len(used),
            "events": used,
            "source": "BIS press releases / Federal Register Entity List notices (curated)",
        })
    except Exception as e:
        import traceback; traceback.print_exc()
        return mark_failed(sid, f"unhandled exception: {e}")


if __name__ == "__main__":
    main()
