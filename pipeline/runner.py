"""
Pipeline runner — manages the 3-loop signal research pipeline.

Usage:
  python pipeline/runner.py              # Run one cycle of all 3 loops
  python pipeline/runner.py scout        # Run Loop 1 only (needs ANTHROPIC_API_KEY)
  python pipeline/runner.py develop      # Run Loop 2 only (needs ANTHROPIC_API_KEY)
  python pipeline/runner.py backtest     # Run Loop 3 only (mechanical, no API needed)
  python pipeline/runner.py daemon       # Run all 3 in a loop (Ctrl+C to stop)
  python pipeline/runner.py status       # Print queue status

Loop 3 (backtester) is fully mechanical — it reads strategy specs and runs
existing backtest scripts. Loops 1 and 2 require ANTHROPIC_API_KEY to generate
ideas and develop strategies via Claude API.
"""

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIPELINE = ROOT / "pipeline"
IDEAS_Q = PIPELINE / "ideas_queue.json"
STRATS_Q = PIPELINE / "strategies_queue.json"
STATUS_F = PIPELINE / "status.json"
BACKTESTS = ROOT / "backtests"
RESULTS = ROOT / "results"
VENV_PY = ROOT / ".venv" / "bin" / "python"

sys.path.insert(0, str(PIPELINE))
from queue_io import (  # noqa: E402
    append_ideas, append_strategies, claim_new_ideas, claim_ready_strategy,
    update_idea_status, update_strategy_status, heartbeat,
)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def _read_queue(path):
    """Read-only snapshot of a queue file. Safe under the atomic-rename writer."""
    if not path.exists():
        return [] if path.suffix == ".json" and path.name.endswith("queue.json") else {}
    with open(path) as f:
        return json.load(f)


def print_status():
    ideas = _read_queue(IDEAS_Q)
    strats = _read_queue(STRATS_Q)
    status = _read_queue(STATUS_F) if STATUS_F.exists() else {}

    count = lambda arr, s: sum(1 for x in arr if x.get("status") == s)

    print("=== Pipeline Status ===")
    print(f"\nIdeas Queue ({len(ideas)} total):")
    print(f"  new: {count(ideas, 'new')}  claimed: {count(ideas, 'claimed')}  "
          f"developed: {count(ideas, 'developed')}  rejected: {count(ideas, 'rejected')}")

    print(f"\nStrategies Queue ({len(strats)} total):")
    print(f"  ready: {count(strats, 'ready')}  in_progress: {count(strats, 'in_progress')}  "
          f"done: {count(strats, 'done')}  failed: {count(strats, 'failed')}  "
          f"needs_implementation: {count(strats, 'needs_implementation')}")

    # Winners per the canonical gate (BH + OOS + sample-size, not the loose
    # legacy sharpe>0.5 AND cagr>0.10 check).
    from winner_gate import is_winner, load_mt_data
    mt = load_mt_data()
    bt_done = [s for s in strats if s.get("backtest_result")]
    winners = []
    for s in bt_done:
        sid = s.get("signal_id")
        if not sid:
            continue
        rp = RESULTS / f"{sid}.json"
        if not rp.exists():
            continue
        try:
            res = json.loads(rp.read_text())
        except Exception:
            continue
        res.setdefault("signal_id", sid)
        if is_winner(res, mt):
            winners.append(s)
    print(f"\nBacktest Results: {len(bt_done)} done, {len(winners)} winners (canonical gate)")

    loops = status.get("loops", {})
    print("\nLoop Heartbeats:")
    for name in ["idea_scout", "strategy_developer", "backtester"]:
        info = loops.get(name, {})
        last = info.get("last_run", "never")
        st = info.get("status", "inactive")
        print(f"  {name}: {st} (last: {last})")


def run_backtest_loop():
    """Loop 3 — fully mechanical. Pick oldest 'ready' strategy, run its backtest.

    Uses the canonical winner gate (winner_gate.is_winner), which adds
    BH-significance + positive OOS Sharpe + sample-size floor on top of the
    raw Sharpe/CAGR thresholds. Refreshes the BH table before checking.
    """
    from winner_gate import is_winner, load_mt_data, winner_reasons

    strat = claim_ready_strategy()
    if strat is None:
        print("Backtester: No strategies ready for backtesting.")
        heartbeat("backtester", "idle")
        return False

    sid = strat.get("signal_id", strat.get("strategy_id", "unknown"))
    strategy_id = strat.get("strategy_id", sid)
    print(f"Backtester: Claimed {sid} — {strat.get('name', '')}")

    bt_file = BACKTESTS / f"{sid}.py"
    result_file = RESULTS / f"{sid}.json"

    if not bt_file.exists():
        print(f"Backtester: No backtest script at {bt_file} — needs Claude to write it.")
        print(f"  Run: claude 'Read pipeline/backtester.md. Implement backtest for {sid}'")
        # Park it as 'needs_implementation' so claim_ready_strategy (which only
        # picks 'ready') skips it on the next iteration. Without this, the
        # daemon would re-claim the same item every cycle and never make
        # progress on other ready strategies.
        update_strategy_status(strategy_id, "needs_implementation", claimed_at=None)
        heartbeat("backtester", "waiting_for_implementation")
        return False

    print(f"Backtester: Running {bt_file}...")
    try:
        result = subprocess.run(
            [str(VENV_PY), str(bt_file)],
            capture_output=True, text=True, timeout=300, cwd=str(ROOT)
        )
        print(result.stdout[-500:] if result.stdout else "(no stdout)")
        if result.returncode != 0:
            print(f"Backtester: Script errored: {result.stderr[-300:]}")
    except subprocess.TimeoutExpired:
        print(f"Backtester: Timeout after 5 minutes")
        update_strategy_status(
            strategy_id, "failed",
            backtest_result={"status": "fail", "reason": "timeout"},
            backtested_at=now_iso(),
        )
        heartbeat("backtester")
        return True

    if not result_file.exists():
        update_strategy_status(
            strategy_id, "failed",
            backtest_result={"status": "fail", "reason": "no result file produced"},
            backtested_at=now_iso(),
        )
        heartbeat("backtester")
        return True

    with open(result_file) as f:
        res = json.load(f)
    res.setdefault("signal_id", sid)

    new_status = "done" if res.get("status") != "fail" else "failed"
    update_strategy_status(
        strategy_id, new_status,
        backtest_result={
            "status": res.get("status", "ok"),
            "sharpe": res.get("sharpe"),
            "cagr": res.get("cagr"),
            "max_dd": res.get("max_dd"),
            "t_stat": res.get("t_stat"),
        },
        backtested_at=now_iso(),
    )

    if new_status == "done":
        # Refresh BH table across the whole catalog so winner_gate has fresh
        # thresholds (cheap, ~1s).
        subprocess.run([str(VENV_PY), str(PIPELINE / "refresh_bh.py")],
                       capture_output=True, cwd=str(ROOT))
        mt = load_mt_data()
        sharpe = res.get("sharpe") or 0
        cagr = res.get("cagr") or 0
        if is_winner(res, mt):
            print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% (canonical gate) ***")
            subprocess.run([str(VENV_PY), str(ROOT / "build_report.py")],
                           capture_output=True, cwd=str(ROOT))
        else:
            failed = [k for k, v in winner_reasons(res, mt).items() if not v]
            print(f"Backtester: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% "
                  f"(not a winner; failed: {failed})")

    heartbeat("backtester")
    return True


CLAUDE_MODEL = "claude-sonnet-4-5"


def run_scout_loop():
    """Loop 1 — needs Claude API. Generates ideas via append_ideas (which
    assigns idea_id under lock, so concurrent scouts don't collide)."""
    try:
        import anthropic
    except ImportError:
        print("Scout: pip install anthropic to use this loop, or run manually via Claude Code.")
        return False

    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Scout: Set ANTHROPIC_API_KEY env var, or run manually via Claude Code:")
        print("  claude 'Read pipeline/idea_scout.md and follow its instructions exactly'")
        heartbeat("idea_scout", "needs_api_key")
        return False

    instructions = (PIPELINE / "idea_scout.md").read_text()
    ideas = _read_queue(IDEAS_Q)
    existing_keys = sorted({i.get("dedup_key", "") for i in ideas if i.get("dedup_key")})
    backtests_list = "\n".join(sorted(p.name for p in BACKTESTS.glob("*.py")))

    prompt = f"""{instructions}

Current ideas_queue.json has {len(ideas)} items. Existing dedup_keys: {existing_keys}

Existing backtest files:
{backtests_list}

Generate 2-3 new ideas and return ONLY the JSON array of new idea objects (no markdown, no explanation).
DO NOT set idea_id on the returned objects — it is assigned inside the queue lock by `append_ideas`."""

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        text = msg.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        new_ideas = json.loads(text)
        if not isinstance(new_ideas, list):
            new_ideas = [new_ideas]

        # Dedup before appending (the lock allocates IDs but doesn't dedup).
        existing_set = set(existing_keys)
        filtered = []
        for idea in new_ideas:
            dk = idea.get("dedup_key", "")
            if dk and dk in existing_set:
                print(f"Scout: Skipping duplicate {dk}")
                continue
            # Strip any idea_id Claude included — append_ideas reassigns.
            idea.pop("idea_id", None)
            filtered.append(idea)

        if filtered:
            assigned = append_ideas(filtered)
            for sid, idea in zip(assigned, filtered):
                print(f"Scout: Added {sid} — {idea.get('name')}")
            print(f"Scout: Added {len(filtered)} ideas")
        else:
            print("Scout: No new ideas after dedup.")
    except (json.JSONDecodeError, IndexError, KeyError) as e:
        print(f"Scout: Failed to parse API response: {e}")

    heartbeat("idea_scout")
    return True


def run_develop_loop():
    """Loop 2 — needs Claude API. Develops strategies from ideas via the
    locking claim helper, so concurrent developers don't double-claim."""
    try:
        import anthropic
    except ImportError:
        print("Developer: pip install anthropic to use this loop, or run manually via Claude Code.")
        return False

    import os
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Developer: Set ANTHROPIC_API_KEY env var, or run manually via Claude Code:")
        print("  claude 'Read pipeline/strategy_developer.md and follow its instructions exactly'")
        heartbeat("strategy_developer", "needs_api_key")
        return False

    claimed = claim_new_ideas(1)
    if not claimed:
        print("Developer: No new ideas in queue.")
        heartbeat("strategy_developer", "idle")
        return False

    idea = claimed[0]
    instructions = (PIPELINE / "strategy_developer.md").read_text()
    prompt = f"""{instructions}

Here is the idea to develop:
{json.dumps(idea, indent=2)}

Return ONLY the JSON object for the developed strategy (no markdown, no explanation).
Use signal_id: PL{idea['idea_id'].replace('PL','')}_{ idea.get('dedup_key','unknown')[:30] }"""

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        text = msg.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        strat = json.loads(text)
        strat.setdefault("idea_id", idea["idea_id"])
        strat.setdefault("strategy_id", idea["idea_id"])
        if idea.get("counter_signal"):
            strat["counter_signal"] = True
            if "counters" in idea:
                strat["counters"] = idea["counters"]
        append_strategies([strat])
        update_idea_status(idea["idea_id"], "developed")
        print(f"Developer: Developed {strat.get('signal_id')} — {strat.get('name')}")
    except (json.JSONDecodeError, IndexError, KeyError) as e:
        print(f"Developer: Failed to parse API response: {e}")
        # Put it back so the next iteration can retry.
        update_idea_status(idea["idea_id"], "new", claimed_at=None)

    heartbeat("strategy_developer")
    return True


def _sweep_stale(timeout_minutes: int = 30) -> None:
    """Recover stuck claims at the top of every daemon cycle.

    If a backtester subprocess crashes or a strategy_developer is killed
    after claim but before status update, the entry is stranded in
    'in_progress' / 'claimed' until something releases it. Sweeping every
    cycle keeps the FIFO moving.
    """
    from queue_io import release_stale_claims, release_stale_idea_claims
    released_strats = release_stale_claims(timeout_minutes)
    released_ideas = release_stale_idea_claims(timeout_minutes)
    if released_strats:
        print(f"  swept {len(released_strats)} stale in_progress strategies back to ready: {released_strats[:5]}{'...' if len(released_strats) > 5 else ''}")
    if released_ideas:
        print(f"  swept {len(released_ideas)} stale claimed ideas back to new: {released_ideas[:5]}{'...' if len(released_ideas) > 5 else ''}")


def daemon_loop(interval=300, stale_timeout=30):
    """Run all 3 loops continuously, sweeping stale claims first each cycle."""
    print(f"Pipeline daemon starting (interval={interval}s, stale_timeout={stale_timeout}min). Ctrl+C to stop.")
    while True:
        print(f"\n--- Cycle at {now_iso()} ---")
        _sweep_stale(stale_timeout)
        run_scout_loop()
        run_develop_loop()
        run_backtest_loop()
        print(f"Sleeping {interval}s...")
        time.sleep(interval)


if __name__ == "__main__":
    args = sys.argv[1:]
    cmd = args[0] if args else "all"

    if cmd == "status":
        print_status()
    elif cmd == "scout":
        run_scout_loop()
    elif cmd == "develop":
        run_develop_loop()
    elif cmd == "backtest":
        run_backtest_loop()
    elif cmd == "daemon":
        interval = int(args[1]) if len(args) > 1 else 300
        try:
            daemon_loop(interval)
        except KeyboardInterrupt:
            print("\nDaemon stopped.")
    elif cmd == "all":
        run_scout_loop()
        run_develop_loop()
        run_backtest_loop()
    else:
        print(__doc__)
