# Loop 3 — Backtester

You implement and run backtests from the strategy queue.

## Your task on each loop iteration

1. **Read the strategy queue**: Load `pipeline/strategies_queue.json`. Find items with `status: "ready"`. Pick the oldest (FIFO).

2. **Claim the strategy**: Set `status: "in_progress"`. Write back.

3. **Implement the backtest**: Create `backtests/<signal_id>.py` following project conventions:
   - Import from `harness` (save_result, mark_failed, compute_metrics, load_prices, load_fred, daily_returns)
   - Build a daily PnL series
   - Compare against SPY benchmark
   - Call `save_result()` or `mark_failed()` at the end
   - Must be standalone-runnable: `.venv/bin/python backtests/<signal_id>.py`

4. **Run the backtest**: `.venv/bin/python backtests/<signal_id>.py`

5. **Check the result**: Read `results/<signal_id>.json`.

6. **Update the strategy queue**: Set `status: "done"` or `"failed"`. Copy key metrics into `backtest_result`:
   ```json
   {"status": "ok|fail", "sharpe": <N>, "cagr": <N>, "max_dd": <N>, "t_stat": <N>}
   ```
   Set `backtested_at` to current ISO timestamp.

7. **If a winner** (sharpe > 0.5 AND cagr > 0.10): Run `.venv/bin/python build_report.py` to regenerate `full_catalog.html`. Print "WINNER FOUND: <signal_id> — Sharpe <X>, CAGR <Y>%"

8. **If backtest errors**: Call `mark_failed(signal_id, reason)`. Set status to `"failed"`.

9. **If no ready strategies**: Print "No strategies ready for backtesting." and exit.

## Backtest file template

```python
"""<signal_id> <Name>"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from harness import (load_prices, load_fred, compute_metrics,
                     save_result, mark_failed, daily_returns)


def main():
    sid = "<signal_id>"
    try:
        px = load_prices(["TICKER1", "TICKER2", "SPY"], start="YYYY-01-01")
        spy_r = daily_returns(px[["SPY"]]).iloc[:, 0]
    except Exception as e:
        return mark_failed(sid, f"data load: {e}")

    # ... signal logic: build a positions Series (1=long, -1=short, 0=flat) ...
    # ... compute pnl from positions * returns ...

    m = compute_metrics(pnl, benchmark=spy_r, name="<Name>")
    save_result(sid, m, extra={
        "rule": "<plain-English rule>",
        "mechanism": "<causal mechanism>",
        "source": "<citation>",
    })


if __name__ == "__main__":
    main()
```

## Heartbeat
After each iteration, update `pipeline/status.json`: set `loops.backtester.last_run` to current ISO 8601 timestamp, set `status` to `"running"`.

## Rules
- Process at most 1 strategy per iteration (backtests can be slow)
- Do NOT commit or push — user reviews first
- Only modify: `backtests/<signal_id>.py` (create), `results/` (via harness), `pipeline/strategies_queue.json` (status updates), `full_catalog.html` (via build_report.py for winners)
- Never modify existing backtest files
- Never delete result files
- Always use the shared harness
