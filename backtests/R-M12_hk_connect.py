"""
R-M12 HK Stock Connect Northbound flows (retry).

Original spec: long FXI 30d when 10d cumulative Northbound > RMB 50B.
Attempted sources:
1) hkex.com.hk/eng/csm/DailyStat/* JSON endpoints -- all 404.
2) Stock Connect HKEX landing -- HTML loads JS components; no embedded
   CSV / JSON time series. Daily statistics rendered client-side from
   un-discoverable XHR endpoints.
3) Mainland mirrors sse.com.cn / szse.cn -- 404/500.
4) chinamoney.com.cn -- 404.

No multi-year daily Northbound-flow CSV reachable.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import mark_failed


SIGNAL_ID = "R-M12_hk_connect"


def main():
    mark_failed(
        SIGNAL_ID,
        ("HKEX Stock Connect statistics rendered client-side, no public daily CSV; "
         "alternative mainland sources (SSE / SZSE / Chinamoney) returned 404/500."),
        extra={
            "rule": "Long FXI 30d when 10d cumulative Northbound > RMB 50B.",
            "mechanism": "Foreign-buy surges through Connect program signal sentiment improvement on A-shares; spillover to H-shares.",
            "source_attempted": "hkex.com.hk csm/DailyStat JSON; sse.com.cn; szse.cn; chinamoney.com.cn.",
            "data_substitution": "None successful.",
        },
    )


if __name__ == "__main__":
    main()
