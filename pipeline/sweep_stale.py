"""
Release stuck queue entries back to the live FIFO.

- Strategies stuck in 'in_progress' past the timeout go back to 'ready'.
- Ideas stuck in 'claimed' past the timeout go back to 'new'.

Recovers from scout / strategy-developer / backtester loops that crashed or
were killed mid-claim, otherwise those entries block FIFO processing forever.

Safe to run alongside live loops — both sweeps use fcntl.LOCK_EX via queue_io.

Usage:
    .venv/bin/python pipeline/sweep_stale.py              # dry-run
    .venv/bin/python pipeline/sweep_stale.py --apply      # release
    .venv/bin/python pipeline/sweep_stale.py --apply --timeout 60   # 60-min cutoff
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from queue_io import (
    IDEAS, STRATEGIES,
    release_stale_claims, release_stale_idea_claims,
)


def _parse_claimed_at(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _age_str(claimed_at: str | None) -> str:
    dt = _parse_claimed_at(claimed_at)
    if dt is None:
        return "no/bad claimed_at field"
    return str(datetime.now(timezone.utc) - dt).split(".")[0]


def preview_strategies(timeout_minutes: int) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=timeout_minutes)
    data = json.loads(STRATEGIES.read_text())
    stale = []
    for d in data:
        if d.get("status") != "in_progress":
            continue
        ts = _parse_claimed_at(d.get("claimed_at"))
        is_stale = ts is None or ts < cutoff
        if is_stale:
            stale.append({
                "id": d.get("strategy_id"),
                "name": d.get("name", "")[:60],
                "age": _age_str(d.get("claimed_at")),
            })
    return stale


def preview_ideas(timeout_minutes: int) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=timeout_minutes)
    data = json.loads(IDEAS.read_text())
    stale = []
    for d in data:
        if d.get("status") != "claimed":
            continue
        ts = _parse_claimed_at(d.get("claimed_at"))
        is_stale = ts is None or ts < cutoff
        if is_stale:
            stale.append({
                "id": d.get("idea_id"),
                "name": d.get("name", "")[:60],
                "age": _age_str(d.get("claimed_at")),
            })
    return stale


def _print_section(title: str, entries: list[dict]) -> None:
    print(f"  {title}: {len(entries)}")
    for e in entries:
        print(f"    {e['id']:8s}  age={e['age']:20s}  {e['name']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true", help="actually release stale claims")
    p.add_argument("--timeout", type=int, default=30,
                   help="minutes before in_progress / claimed is stale (default 30)")
    args = p.parse_args()

    stale_strats = preview_strategies(args.timeout)
    stale_ideas = preview_ideas(args.timeout)
    mode = "APPLIED" if args.apply else "DRY-RUN"
    print(f"=== Stale claim sweep [{mode}] (timeout={args.timeout}min) ===")
    _print_section("stale in_progress strategies", stale_strats)
    _print_section("stale claimed ideas", stale_ideas)

    if not args.apply:
        print("\n(dry run; pass --apply to release)")
        return

    released_strats = release_stale_claims(args.timeout)
    released_ideas = release_stale_idea_claims(args.timeout)
    print(f"\nReleased {len(released_strats)} strategies back to 'ready'.")
    print(f"Released {len(released_ideas)} ideas back to 'new'.")


if __name__ == "__main__":
    main()
