"""Concurrency-safe IO for pipeline queues.

Wraps reads/writes to ideas_queue.json, strategies_queue.json, and status.json
with fcntl.LOCK_EX so multiple agent loops can run in parallel without
clobbering each other's writes.

Usage from an agent script:

    import sys
    sys.path.insert(0, "/Users/benson/Projects/ekans/pipeline")
    from queue_io import append_ideas, claim_new_ideas, append_strategies, \
        claim_ready_strategy, update_idea_status, update_strategy_status, \
        heartbeat
"""
from __future__ import annotations

import fcntl
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
IDEAS = PIPELINE_DIR / "ideas_queue.json"
STRATEGIES = PIPELINE_DIR / "strategies_queue.json"
STATUS = PIPELINE_DIR / "status.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _locked_update(path: Path, default, updater):
    """Hold LOCK_EX on a sidecar lockfile, read `path`, apply updater(data),
    write the result to `<path>.tmp` and atomically `os.replace` it into place.

    Crash-safe: a crash before `os.replace` leaves the original file intact;
    `os.replace` itself is atomic on POSIX. Non-cooperative readers never see
    partial content — they see either the old file or the new one.

    The lock is held on a sidecar file (`<path>.lock`) rather than on `path`
    itself, because `os.replace` swaps the underlying inode and any fcntl lock
    held on the old inode would not protect a waiting writer who had already
    opened the path before the rename.
    """
    lock_path = Path(str(path) + ".lock")
    lock_path.touch(exist_ok=True)
    with open(lock_path, "r+") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        try:
            if path.exists():
                with open(path, "r") as f:
                    raw = f.read()
                data = json.loads(raw) if raw.strip() else default
            else:
                data = default
            new_data = updater(data)
            tmp_path = Path(str(path) + ".tmp")
            with open(tmp_path, "w") as f:
                json.dump(new_data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
            return new_data
        finally:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)


def _max_pl_id(items, key="idea_id") -> int:
    n = 0
    pat = re.compile(r"^PL(\d+)$")
    for it in items:
        v = it.get(key, "")
        m = pat.match(v)
        if m:
            try:
                n = max(n, int(m.group(1)))
            except ValueError:
                pass
    return n


def append_ideas(new_ideas: list[dict]) -> list[str]:
    """Append ideas with status='new'; re-allocate idea_id from current max under lock.

    Existing idea_id field on each input is ignored — IDs are assigned fresh
    inside the lock so concurrent scouts can't collide.
    Returns the list of assigned IDs in input order.
    """
    assigned: list[str] = []

    def upd(data):
        nonlocal assigned
        assigned = []
        max_n = _max_pl_id(data, key="idea_id")
        out = list(data)
        for idea in new_ideas:
            max_n += 1
            new_id = f"PL{max_n:03d}"
            entry = {**idea, "idea_id": new_id, "status": "new"}
            entry.setdefault("created_at", _now())
            out.append(entry)
            assigned.append(new_id)
        return out

    _locked_update(IDEAS, [], upd)
    return assigned


def claim_new_ideas(n: int) -> list[dict]:
    """Atomically claim the n oldest 'new' ideas (FIFO by created_at).

    Sets status='claimed' in the queue and returns the (post-claim) entries.
    Returns [] if no new ideas are available.
    """
    claimed: list[dict] = []

    def upd(data):
        nonlocal claimed
        claimed = []
        new_items = [(i, d) for i, d in enumerate(data) if d.get("status") == "new"]
        new_items.sort(key=lambda x: x[1].get("created_at", ""))
        for i, d in new_items[:n]:
            updated = {**d, "status": "claimed", "claimed_at": _now()}
            data[i] = updated
            claimed.append(updated)
        return data

    _locked_update(IDEAS, [], upd)
    return claimed


def update_idea_status(idea_id: str, new_status: str, **extras) -> bool:
    """Set status (and any extra fields) on a specific idea_id. Returns True if found."""
    found = [False]

    def upd(data):
        for i, d in enumerate(data):
            if d.get("idea_id") == idea_id:
                data[i] = {**d, "status": new_status, **extras}
                found[0] = True
                break
        return data

    _locked_update(IDEAS, [], upd)
    return found[0]


def append_strategies(new_strategies: list[dict]) -> None:
    """Append developed strategies to the strategies queue (status=ready)."""

    def upd(data):
        out = list(data)
        for s in new_strategies:
            entry = {**s, "status": "ready"}
            entry.setdefault("created_at", _now())
            out.append(entry)
        return out

    _locked_update(STRATEGIES, [], upd)


def claim_ready_strategy() -> dict | None:
    """Atomically claim the oldest 'ready' strategy (FIFO by created_at).

    Sets status='in_progress' and returns the entry, or None if nothing ready.
    """
    claimed: list = [None]

    def upd(data):
        ready = [(i, d) for i, d in enumerate(data) if d.get("status") == "ready"]
        ready.sort(key=lambda x: x[1].get("created_at", ""))
        if not ready:
            return data
        i, d = ready[0]
        updated = {**d, "status": "in_progress", "claimed_at": _now()}
        data[i] = updated
        claimed[0] = updated
        return data

    _locked_update(STRATEGIES, [], upd)
    return claimed[0]


def update_strategy_status(strategy_id: str, new_status: str, **extras) -> bool:
    """Set status (and any extra fields) on a specific strategy. Returns True if found.

    Matches the unique `signal_id` first. `strategy_id` mirrors the idea_id,
    and idea_ids are reused across scout rounds (PL742 can map to two different
    strategies), so it is NOT a safe key — a first-match-by-strategy_id update
    can land on the wrong (often already-resolved) row and strand the real one
    in 'in_progress'. We therefore prefer the unique `signal_id` and fall back
    to `strategy_id` only for legacy callers that pass the short id.
    """
    found = [False]

    def upd(data):
        idx = next((i for i, d in enumerate(data)
                    if d.get("signal_id") == strategy_id), None)
        if idx is None:
            idx = next((i for i, d in enumerate(data)
                        if d.get("strategy_id") == strategy_id), None)
        if idx is not None:
            data[idx] = {**data[idx], "status": new_status, **extras}
            found[0] = True
        return data

    _locked_update(STRATEGIES, [], upd)
    return found[0]


def _is_stale(claimed_at: str | None, cutoff: datetime) -> bool:
    """True if a claim should be released. Missing/malformed claimed_at = stale."""
    if not claimed_at:
        return True
    try:
        dt = datetime.fromisoformat(claimed_at)
    except (ValueError, TypeError):
        return True
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt < cutoff


def release_stale_claims(timeout_minutes: int = 30) -> list[str]:
    """Release strategies stuck in 'in_progress' past the timeout, back to 'ready'.

    Used to recover after a backtester loop crashes or is killed mid-claim.
    A strategy is considered stale when:
      - status == 'in_progress', AND
      - claimed_at is missing (legacy claim before that field existed) OR
        claimed_at is more than `timeout_minutes` ago.

    Returns the list of strategy_ids that were released.
    """
    from datetime import timedelta

    released: list[str] = []
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=timeout_minutes)

    def upd(data):
        for i, d in enumerate(data):
            if d.get("status") != "in_progress":
                continue
            if _is_stale(d.get("claimed_at"), cutoff):
                data[i] = {**d, "status": "ready", "claimed_at": None,
                           "stale_released_at": _now()}
                released.append(d.get("strategy_id", "?"))
        return data

    _locked_update(STRATEGIES, [], upd)
    return released


def release_stale_idea_claims(timeout_minutes: int = 30) -> list[str]:
    """Release ideas stuck in 'claimed' past the timeout, back to 'new'.

    Mirror of `release_stale_claims` for the ideas queue. Used to recover
    after a strategy-developer loop crashes or is killed after `claim_new_ideas`
    but before `update_idea_status(..., "developed" | "rejected")`.

    Returns the list of idea_ids that were released.
    """
    from datetime import timedelta

    released: list[str] = []
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=timeout_minutes)

    def upd(data):
        for i, d in enumerate(data):
            if d.get("status") != "claimed":
                continue
            if _is_stale(d.get("claimed_at"), cutoff):
                data[i] = {**d, "status": "new", "claimed_at": None,
                           "stale_released_at": _now()}
                released.append(d.get("idea_id", "?"))
        return data

    _locked_update(IDEAS, [], upd)
    return released


def heartbeat(loop_name: str, status: str = "running") -> None:
    """Update pipeline/status.json with last_run + status for the named loop.

    Preserves any existing fields (job_id, interval, etc.) on the loop entry.
    """

    def upd(data):
        if not isinstance(data, dict):
            data = {}
        loops = data.setdefault("loops", {})
        entry = loops.setdefault(loop_name, {})
        entry["last_run"] = _now()
        entry["status"] = status
        return data

    _locked_update(STATUS, {"loops": {}}, upd)
