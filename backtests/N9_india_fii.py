"""
N9 India FII / DII daily flows.
NSE's fiidiiTradeReact endpoint is heavily rate-limited and blocks non-browser
user-agents; historical pulls require browser-emulating session management with
cookies (akamai bot mitigation). Not feasible in this batch.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import mark_failed


def main():
    mark_failed(
        "N9_india_fii",
        "NSE FII/DII endpoint requires browser session emulation and is rate-limited; no clean public historical feed.",
        extra={
            "rule_intended": "Long INDA 15d when 10d-rolling FII net flow negative on 8/10 days and cumulative < -$2B.",
            "source": "nseindia.com/api/fiidiiTradeReact",
            "universe": "INDA",
        },
    )
    print("N9 India FII: marked failed (NSE bot mitigation).")


if __name__ == "__main__":
    main()
