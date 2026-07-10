# Loop 3 — Backtester

You implement and run backtests from the strategy queue.

## Your task on each loop iteration

All queue mutations MUST go through `pipeline/queue_io.py`. Other agents (idea scouts, strategy developers, and possibly other backtesters) may be writing to the queues in parallel — naive `json.load` / `json.dump` will race. Start every script with:

```python
import sys
sys.path.insert(0, "/Users/benson/Projects/ekans/pipeline")
from queue_io import claim_ready_strategy, update_strategy_status, heartbeat
```

1. **Atomically claim a strategy**: Call `claim_ready_strategy()`. The helper holds `LOCK_EX` on `strategies_queue.json`, picks the oldest `ready` strategy by `created_at`, sets status to `"in_progress"`, and returns the entry (or `None` if nothing ready).

   ```python
   strat = claim_ready_strategy()
   if strat is None:
       print("No strategies ready for backtesting.")
       heartbeat("backtester")
       sys.exit(0)
   sid = strat["signal_id"]
   ```

2. **Implement the backtest**: Create `backtests/<signal_id>.py` following project conventions:
   - Import from `harness` (save_result, mark_failed, compute_metrics, load_prices, load_fred, daily_returns)
   - Build a daily PnL series
   - Compare against SPY benchmark
   - Call `save_result()` or `mark_failed()` at the end
   - Must be standalone-runnable: `.venv/bin/python backtests/<signal_id>.py`

3. **Run the backtest**: `.venv/bin/python backtests/<signal_id>.py`

4. **Check the result**: Read `results/<signal_id>.json`.

5. **Update the strategy queue via the helper**:

   ```python
   from datetime import datetime, timezone
   update_strategy_status(
       strat["strategy_id"],
       "done",  # or "failed"
       backtest_result={"status": "ok", "sharpe": ..., "cagr": ..., "max_dd": ..., "t_stat": ...},
       backtested_at=datetime.now(timezone.utc).isoformat(),
   )
   ```

6. **Check if it's a winner** using the canonical gate (do NOT hand-roll
   `sharpe > 0.5 and cagr > 0.10` — that gate is intentionally loose enough that
   ~30% of all backtests pass, which is statistically meaningless given the
   ~800-test corpus). The gate requires BH-significance + positive OOS Sharpe +
   sample-size floor on top of the Sharpe/CAGR thresholds:

   ```python
   import sys
   sys.path.insert(0, "/Users/benson/Projects/ekans/pipeline")
   from winner_gate import is_winner, load_mt_data

   result = json.load(open(f"results/{sid}.json"))
   if not result.get("signal_id"):
       result["signal_id"] = sid

   # Refresh BH correction across the whole catalog before checking.
   # This is cheap (~1s) and required because BH thresholds depend on the
   # total number of tested signals — a stale table will mis-classify.
   import subprocess
   subprocess.run([".venv/bin/python", "pipeline/refresh_bh.py"], check=True)

   if is_winner(result, load_mt_data()):
       # Promote to High Return tab + regenerate full_catalog.html.
       ...
   ```

   **If it passes the gate**: Run `.venv/bin/python build_report.py` to regenerate
   `full_catalog.html`. Also add a `<article class="card">` block to the High
   Return tab in `index.html` before the appendix comment, with Sharpe / CAGR /
   MaxDD / t-stat / OOS Sharpe metrics and rule / mechanism / caveats. Print
   `"WINNER FOUND: <signal_id> — Sharpe <X>, CAGR <Y>%, OOS <Z>, BH-sig"`.

   **If it doesn't pass the gate**: don't add anything to `index.html`. The
   demote sweep (`.venv/bin/python pipeline/demote_index.py`) will visually mark
   any previously-promoted card that no longer qualifies. Note in your output
   which criterion failed (`winner_reasons(result, mt)` returns a per-criterion
   bool dict for diagnostics).

6b. **Check if it's a sleeper** — a signal that fails the winner gate but has a
   strong *conditional* edge (pays off hard in rare regimes, flat the rest of the
   time: tail trades, counter-signals, secular inflections). The winner gate
   rejects these because their blended Sharpe is diluted by long flat stretches.
   Conditional metrics are written automatically by `compute_metrics`
   (`cond_sharpe`, `cond_oos_sharpe`, `n_episodes`, `mean_episode_ret`,
   `active_frac`), so no extra backtest work is needed — just gate on them:

   ```python
   import sys
   sys.path.insert(0, "/Users/benson/Projects/ekans/pipeline")
   from sleeper_gate import is_sleeper
   from winner_gate import is_winner, load_mt_data

   mt = load_mt_data()  # reuse from step 6
   if is_sleeper(result) and not is_winner(result, mt):
       # Hidden sleeper: promote to the Sleepers tab (NOT High Return).
       ...
   ```

   **If it's a hidden sleeper** (passes `is_sleeper`, fails `is_winner`): add an
   `<article class="card">` block to the **Sleepers tab** in `index.html`,
   inserted directly above the `<!-- SLEEPERS-INSERT ... -->` marker comment, and
   bump the `#tab-sleepers-count` badge. Mirror the existing sleeper cards: the
   metrics row shows **Cond. Sharpe / Cond. OOS / Episodes / Avg-per-episode /
   Active%** (NOT the blended winner metrics), and the caveat is a "Why it's
   hidden" note that contrasts the diluted blended Sharpe against the conditional
   edge and lists which `winner_reasons` it fails. Do NOT run `build_report.py`
   or touch the High Return tab — sleepers are a separate surface. Print
   `"SLEEPER FOUND: <signal_id> — cond.Sharpe <X>, OOS <Y>, <N> episodes, active <Z>%"`.
   If a signal qualifies as BOTH a winner and a sleeper, treat it as a winner only
   (step 6) — `is_hidden_sleeper` already excludes winners.

   **If it's neither**: nothing to add. You can confirm the full picture with
   `.venv/bin/python pipeline/find_sleepers.py`, which ranks all hidden sleepers.

7. **If backtest errors**: Call `mark_failed(signal_id, reason)` from `harness`, then `update_strategy_status(..., "failed", backtest_result={...})`.

8. **If no ready strategies**: handled in step 1.

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

    # Pass positions= so conditional/sleeper metrics are exact (active days =
    # position != 0). Pass pnl= so the series is saved for find_sleepers backfill.
    m = compute_metrics(pnl, benchmark=spy_r, name="<Name>", positions=positions)
    save_result(sid, m, extra={
        "rule": "<plain-English rule>",
        "mechanism": "<causal mechanism>",
        "source": "<citation>",
    }, pnl=pnl)


if __name__ == "__main__":
    main()
```

## Heartbeat
After each iteration, call `heartbeat("backtester")` from `queue_io`.

## Rules
- Process at most 1 strategy per iteration (backtests can be slow)
- Do NOT commit or push — user reviews first
- Only modify: `backtests/<signal_id>.py` (create), `results/` (via harness), `pipeline/strategies_queue.json` (via `queue_io` only), `full_catalog.html` (via build_report.py for winners), `index.html` (add card for winners in the High Return tab, OR for hidden sleepers in the Sleepers tab — see steps 6 / 6b)
- Never modify existing backtest files
- Never delete result files
- Always use the shared harness — and pass `positions=` to `compute_metrics` whenever you have a positions Series, so the conditional / sleeper metrics are exact (`positions`-based) rather than the `pnl != 0` proxy
- Do NOT do raw `json.load` / `json.dump` on the queue files — always go through `queue_io` helpers
