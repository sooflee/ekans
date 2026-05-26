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


def load_json(path):
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def update_heartbeat(loop_name, status="running"):
    s = load_json(STATUS_F) if STATUS_F.exists() else {"loops": {}}
    if "loops" not in s:
        s["loops"] = {}
    if loop_name not in s["loops"]:
        s["loops"][loop_name] = {}
    s["loops"][loop_name]["last_run"] = now_iso()
    s["loops"][loop_name]["status"] = status
    save_json(STATUS_F, s)


def print_status():
    ideas = load_json(IDEAS_Q)
    strats = load_json(STRATS_Q)
    status = load_json(STATUS_F) if STATUS_F.exists() else {}

    count = lambda arr, s: sum(1 for x in arr if x.get("status") == s)

    print("=== Pipeline Status ===")
    print(f"\nIdeas Queue ({len(ideas)} total):")
    print(f"  new: {count(ideas, 'new')}  claimed: {count(ideas, 'claimed')}  "
          f"developed: {count(ideas, 'developed')}  rejected: {count(ideas, 'rejected')}")

    print(f"\nStrategies Queue ({len(strats)} total):")
    print(f"  ready: {count(strats, 'ready')}  in_progress: {count(strats, 'in_progress')}  "
          f"done: {count(strats, 'done')}  failed: {count(strats, 'failed')}")

    bt_done = [s for s in strats if s.get("backtest_result")]
    winners = [s for s in bt_done if s["backtest_result"].get("status") == "ok"
               and (s["backtest_result"].get("sharpe") or 0) > 0.5
               and (s["backtest_result"].get("cagr") or 0) > 0.10]
    print(f"\nBacktest Results: {len(bt_done)} done, {len(winners)} winners")

    loops = status.get("loops", {})
    print("\nLoop Heartbeats:")
    for name in ["idea_scout", "strategy_developer", "backtester"]:
        info = loops.get(name, {})
        last = info.get("last_run", "never")
        st = info.get("status", "inactive")
        print(f"  {name}: {st} (last: {last})")


def run_backtest_loop():
    """Loop 3 — fully mechanical. Pick oldest 'ready' strategy, run its backtest."""
    strats = load_json(STRATS_Q)
    ready = [s for s in strats if s.get("status") == "ready"]

    if not ready:
        print("Backtester: No strategies ready for backtesting.")
        update_heartbeat("backtester", "idle")
        return False

    strat = ready[0]
    sid = strat.get("signal_id", strat.get("strategy_id", "unknown"))
    print(f"Backtester: Claiming {sid} — {strat.get('name', '')}")

    # Claim it
    strat["status"] = "in_progress"
    save_json(STRATS_Q, strats)

    bt_file = BACKTESTS / f"{sid}.py"
    result_file = RESULTS / f"{sid}.json"

    # Check if backtest script exists
    if not bt_file.exists():
        print(f"Backtester: No backtest script at {bt_file} — needs Claude to write it.")
        print(f"  Run: claude 'Read pipeline/backtester.md. Implement backtest for {sid}'")
        strat["status"] = "ready"  # put it back
        save_json(STRATS_Q, strats)
        update_heartbeat("backtester", "waiting_for_implementation")
        return False

    # Run it
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
        strat["status"] = "failed"
        strat["backtest_result"] = {"status": "fail", "reason": "timeout"}
        strat["backtested_at"] = now_iso()
        save_json(STRATS_Q, strats)
        update_heartbeat("backtester")
        return True

    # Check result
    if result_file.exists():
        with open(result_file) as f:
            res = json.load(f)
        strat["status"] = "done" if res.get("status") == "ok" else "failed"
        strat["backtest_result"] = {
            "status": res.get("status", "ok"),
            "sharpe": res.get("sharpe"),
            "cagr": res.get("cagr"),
            "max_dd": res.get("max_dd"),
            "t_stat": res.get("t_stat"),
        }
        strat["backtested_at"] = now_iso()

        sharpe = res.get("sharpe") or 0
        cagr = res.get("cagr") or 0
        if sharpe > 0.5 and cagr > 0.10:
            print(f"*** WINNER FOUND: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% ***")
            # Regenerate catalog
            subprocess.run([str(VENV_PY), str(ROOT / "build_report.py")],
                           capture_output=True, cwd=str(ROOT))
        else:
            print(f"Backtester: {sid} — Sharpe {sharpe:.2f}, CAGR {cagr*100:.1f}% (not a winner)")
    else:
        strat["status"] = "failed"
        strat["backtest_result"] = {"status": "fail", "reason": "no result file produced"}
        strat["backtested_at"] = now_iso()

    save_json(STRATS_Q, strats)
    update_heartbeat("backtester")
    return True


def run_scout_loop():
    """Loop 1 — needs Claude API. Generates ideas."""
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
        update_heartbeat("idea_scout", "needs_api_key")
        return False

    instructions = (PIPELINE / "idea_scout.md").read_text()
    ideas = load_json(IDEAS_Q)
    existing_keys = [i.get("dedup_key", "") for i in ideas]
    backtests_list = "\n".join(sorted(p.name for p in BACKTESTS.glob("*.py")))

    prompt = f"""{instructions}

Current ideas_queue.json has {len(ideas)} items. Existing dedup_keys: {existing_keys}

Existing backtest files:
{backtests_list}

Generate 2-3 new ideas and return ONLY the JSON array of new idea objects (no markdown, no explanation).
The next idea_id should be PL{len(ideas)+1:03d}."""

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
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

        added = 0
        for idea in new_ideas:
            dk = idea.get("dedup_key", "")
            if dk in existing_keys:
                print(f"Scout: Skipping duplicate {dk}")
                continue
            ideas.append(idea)
            existing_keys.append(dk)
            added += 1
            print(f"Scout: Added {idea.get('idea_id')} — {idea.get('name')}")

        save_json(IDEAS_Q, ideas)
        print(f"Scout: Added {added} ideas, queue now has {len(ideas)}")
    except (json.JSONDecodeError, IndexError, KeyError) as e:
        print(f"Scout: Failed to parse API response: {e}")

    update_heartbeat("idea_scout")
    return True


def run_develop_loop():
    """Loop 2 — needs Claude API. Develops strategies from ideas."""
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
        update_heartbeat("strategy_developer", "needs_api_key")
        return False

    ideas = load_json(IDEAS_Q)
    strats = load_json(STRATS_Q)
    new_ideas = [i for i in ideas if i.get("status") == "new"]

    if not new_ideas:
        print("Developer: No new ideas in queue.")
        update_heartbeat("strategy_developer", "idle")
        return False

    idea = new_ideas[0]
    idea["status"] = "claimed"
    save_json(IDEAS_Q, ideas)

    instructions = (PIPELINE / "strategy_developer.md").read_text()
    prompt = f"""{instructions}

Here is the idea to develop:
{json.dumps(idea, indent=2)}

Return ONLY the JSON object for the developed strategy (no markdown, no explanation).
Use signal_id: PL{idea['idea_id'].replace('PL','')}_{ idea.get('dedup_key','unknown')[:30] }"""

    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        text = msg.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        strat = json.loads(text)
        strat["status"] = "ready"
        strat["created_at"] = now_iso()
        strats.append(strat)
        save_json(STRATS_Q, strats)
        idea["status"] = "developed"
        save_json(IDEAS_Q, ideas)
        print(f"Developer: Developed {strat.get('signal_id')} — {strat.get('name')}")
    except (json.JSONDecodeError, IndexError, KeyError) as e:
        print(f"Developer: Failed to parse API response: {e}")
        idea["status"] = "new"  # put it back
        save_json(IDEAS_Q, ideas)

    update_heartbeat("strategy_developer")
    return True


def daemon_loop(interval=300):
    """Run all 3 loops continuously."""
    print(f"Pipeline daemon starting (interval={interval}s). Ctrl+C to stop.")
    while True:
        print(f"\n--- Cycle at {now_iso()} ---")
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
