"""
Resolve duplicate PL-prefix collisions in results/ + backtests/ + strategies_queue.

Two strategy-developer agents running in parallel can each produce a strategy
under the same idea_id, with different signal_id slugs. The artifacts (result
JSON, backtest .py, strategies_queue entries) collide at the PL prefix even
though their slugs differ.

For each collision we identify the canonical signal_id by matching against
`ideas_queue.dedup_key` (or `name` substring as fallback). The other file in
the pair is the orphan. We rename the orphan's PL id to the next free one
across all artifacts so no work is lost.

If one of the pair is a placeholder (no metrics AND no backtest .py file) it
is deleted instead of renamed.

The strategies_queue is also deduped: duplicate strategy_id entries are
collapsed to the canonical one (the one matching the renamed/kept signal_id).

Usage:
    .venv/bin/python pipeline/dedupe_pl_collisions.py            # dry-run
    .venv/bin/python pipeline/dedupe_pl_collisions.py --apply    # mutate
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
from queue_io import _locked_update, STRATEGIES as STRATEGIES_PATH

RESULTS = ROOT / "results"
BACKTESTS = ROOT / "backtests"
IDEAS = ROOT / "pipeline" / "ideas_queue.json"
STRATEGIES = STRATEGIES_PATH
INDEX_HTML = ROOT / "index.html"

PL_PREFIX_RE = re.compile(r"^(PL\d+)_(.+)$")


def load_ideas_by_pl() -> dict[str, dict]:
    return {it["idea_id"]: it for it in json.loads(IDEAS.read_text())}


def collisions() -> dict[str, list[str]]:
    from collections import defaultdict
    g: dict[str, list[str]] = defaultdict(list)
    for fp in sorted(RESULTS.glob("PL*.json")):
        m = PL_PREFIX_RE.match(fp.stem)
        if m:
            g[m.group(1)].append(fp.stem)
    return {k: v for k, v in g.items() if len(v) > 1}


def is_placeholder(sid: str) -> bool:
    """A placeholder has no metrics AND no backtest .py."""
    rp = RESULTS / f"{sid}.json"
    if not rp.exists():
        return True
    d = json.loads(rp.read_text())
    no_metrics = d.get("sharpe") is None and d.get("cagr") is None
    no_backtest = not (BACKTESTS / f"{sid}.py").exists()
    return no_metrics and no_backtest


def pick_canonical(pl: str, candidates: list[str], idea: dict | None) -> tuple[str, str]:
    """Return (canonical_sid, orphan_sid).

    Priority:
      1. The candidate whose slug overlaps MORE tokens of ideas_queue.dedup_key.
      2. Tiebreaker: larger n_days (more rigorous backtest, less curve-fit).
      3. Tiebreaker: higher Sharpe.
      4. Tiebreaker: lexicographic.
    """
    dk_tokens: list[str] = []
    if idea:
        dk = (idea.get("dedup_key") or "").lower()
        dk_tokens = [t for t in dk.split("_") if len(t) > 2]

    def score(sid: str) -> tuple:
        d = json.loads((RESULTS / f"{sid}.json").read_text())
        slug = sid.lower()
        dk_overlap = sum(1 for t in dk_tokens if t in slug)
        return (
            -dk_overlap,  # more overlap is better (negative for sort-ascending)
            -(d.get("n_days") or 0),
            -(d.get("sharpe") if d.get("sharpe") is not None else -1e9),
            sid,
        )
    ordered = sorted(candidates, key=score)
    return ordered[0], ordered[1]


def next_pl_id(used: set[int]) -> int:
    n = max(used) + 1
    while n in used:
        n += 1
    return n


def all_used_pl_ids() -> set[int]:
    used: set[int] = set()
    for fp in RESULTS.glob("PL*.json"):
        m = re.match(r"PL(\d+)_", fp.stem)
        if m:
            used.add(int(m.group(1)))
    for fp in BACKTESTS.glob("PL*.py"):
        m = re.match(r"PL(\d+)_", fp.stem)
        if m:
            used.add(int(m.group(1)))
    for path in (IDEAS, STRATEGIES):
        for it in json.loads(path.read_text()):
            for key in ("idea_id", "strategy_id", "signal_id"):
                v = it.get(key, "")
                m = re.match(r"PL(\d+)", v or "")
                if m:
                    used.add(int(m.group(1)))
    return used


def rename_artifacts(old_sid: str, new_sid: str, apply: bool) -> list[str]:
    actions: list[str] = []
    old_result = RESULTS / f"{old_sid}.json"
    new_result = RESULTS / f"{new_sid}.json"
    old_bt = BACKTESTS / f"{old_sid}.py"
    new_bt = BACKTESTS / f"{new_sid}.py"

    if old_result.exists():
        actions.append(f"mv results/{old_sid}.json -> results/{new_sid}.json")
        if apply:
            d = json.loads(old_result.read_text())
            d["signal_id"] = new_sid
            new_result.write_text(json.dumps(d, indent=2))
            old_result.unlink()
    if old_bt.exists():
        actions.append(f"mv backtests/{old_sid}.py -> backtests/{new_sid}.py")
        if apply:
            content = old_bt.read_text()
            content = content.replace(old_sid, new_sid)
            new_bt.write_text(content)
            old_bt.unlink()
    return actions


def delete_artifacts(sid: str, apply: bool) -> list[str]:
    actions: list[str] = []
    rp = RESULTS / f"{sid}.json"
    bp = BACKTESTS / f"{sid}.py"
    if rp.exists():
        actions.append(f"rm results/{sid}.json")
        if apply:
            rp.unlink()
    if bp.exists():
        actions.append(f"rm backtests/{sid}.py")
        if apply:
            bp.unlink()
    return actions


def fix_index_html(orphan_map: dict[str, str | None], apply: bool) -> list[str]:
    """Update or remove index.html card references for renamed/deleted signal_ids.

    For renames: rewrite the badge text and `<code>backtests/<sid>.py</code>`
    references in-place. For deletes: leave the card alone but report it (we
    don't auto-delete cards because they may contain hand-written prose).
    """
    actions: list[str] = []
    if not INDEX_HTML.exists():
        return actions
    html = INDEX_HTML.read_text()
    new_html = html
    for old_sid, new_sid in orphan_map.items():
        if old_sid not in new_html:
            continue
        if new_sid is None:
            n = new_html.count(old_sid)
            actions.append(f"NOTE: {n} reference(s) to deleted {old_sid} remain in index.html (manual review)")
            continue
        n = new_html.count(old_sid)
        actions.append(f"rewrite {n} reference(s): {old_sid} -> {new_sid}")
        new_html = new_html.replace(old_sid, new_sid)
    if apply and new_html != html:
        INDEX_HTML.write_text(new_html)
    return actions


def fix_strategies_queue(pl_to_orphan_to_new: dict[str, dict[str, str | None]], apply: bool) -> list[str]:
    """For each colliding PL: deduplicate queue entries, renaming/dropping as needed.

    pl_to_orphan_to_new[pl][orphan_sid] = new_sid_or_None_if_deleted
    """
    actions: list[str] = []

    # Build orphan-by-old-sid lookup
    orphan_map: dict[str, str | None] = {}
    for pl, sub in pl_to_orphan_to_new.items():
        for old_sid, new_sid in sub.items():
            orphan_map[old_sid] = new_sid

    def upd(data):
        nonlocal actions
        actions = []
        out: list[dict] = []
        seen: set[str] = set()
        for entry in data:
            sid = entry.get("signal_id", "")
            strat_id = entry.get("strategy_id", "")
            if sid in orphan_map:
                new_sid = orphan_map[sid]
                if new_sid is None:
                    actions.append(f"drop strategies_queue entry strategy_id={strat_id} signal_id={sid}")
                    continue
                actions.append(f"rename strategies_queue entry strategy_id={strat_id}: signal_id {sid} -> {new_sid}")
                m = re.match(r"PL(\d+)", new_sid)
                new_pl = m.group(0) if m else strat_id
                if new_pl in seen:
                    actions.append(f"  (collapsed: duplicate strategy_id={new_pl} already present)")
                    continue
                seen.add(new_pl)
                out.append({**entry, "signal_id": new_sid, "strategy_id": new_pl, "idea_id": new_pl})
                continue
            if strat_id in seen:
                actions.append(f"drop duplicate strategies_queue entry strategy_id={strat_id} signal_id={sid}")
                continue
            seen.add(strat_id)
            out.append(entry)
        return out

    if apply:
        _locked_update(STRATEGIES, [], upd)
    else:
        # Compute actions for the dry-run preview without touching the queue.
        # The atomic-rename writer makes a non-locked read safe (you always see
        # a complete file), and `upd` only mutates the list passed in so the
        # on-disk file is untouched.
        import copy
        upd(copy.deepcopy(json.loads(STRATEGIES.read_text())))
    return actions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()

    ideas = load_ideas_by_pl()
    cols = collisions()
    used = all_used_pl_ids()

    print(f"=== PL collision dedup [{'APPLY' if args.apply else 'DRY-RUN'}] ===")
    print(f"  collisions: {len(cols)}")
    print()

    plan: dict[str, dict[str, str | None]] = {}
    all_actions: list[str] = []

    for pl in sorted(cols, key=lambda x: int(x[2:])):
        candidates = cols[pl]
        idea = ideas.get(pl)
        idea_label = f'queue dedup_key="{idea.get("dedup_key","")}"' if idea else "no ideas_queue entry"

        # If one is a placeholder, delete it (regardless of canonical match)
        placeholders = [c for c in candidates if is_placeholder(c)]
        if placeholders and len(placeholders) < len(candidates):
            survivor = [c for c in candidates if c not in placeholders][0]
            plan[pl] = {ph: None for ph in placeholders}
            print(f"{pl}  KEEP {survivor}")
            print(f"     {idea_label}")
            for ph in placeholders:
                acts = delete_artifacts(ph, args.apply)
                print(f"  DELETE placeholder {ph}")
                for a in acts:
                    print(f"    - {a}")
                all_actions.extend(acts)
            print()
            continue

        # Both are placeholders — drop one arbitrarily
        if len(placeholders) == len(candidates):
            keep, *drop = sorted(candidates)
            plan[pl] = {d: None for d in drop}
            print(f"{pl}  both placeholders; KEEP {keep}")
            for d in drop:
                acts = delete_artifacts(d, args.apply)
                print(f"  DELETE placeholder {d}")
                for a in acts:
                    print(f"    - {a}")
                all_actions.extend(acts)
            print()
            continue

        # Both real: pick canonical, rename orphan to next free PL id
        canonical, orphan = pick_canonical(pl, candidates, idea)
        new_n = next_pl_id(used)
        used.add(new_n)
        orphan_suffix = PL_PREFIX_RE.match(orphan).group(2)
        new_sid = f"PL{new_n:03d}_{orphan_suffix}"
        plan[pl] = {orphan: new_sid}
        print(f"{pl}  KEEP {canonical}")
        print(f"     {idea_label}")
        print(f"  RENAME {orphan} -> {new_sid}")
        acts = rename_artifacts(orphan, new_sid, args.apply)
        for a in acts:
            print(f"    - {a}")
        all_actions.extend(acts)
        print()

    print()
    # Flatten orphan_map for queue + index.html fixes
    orphan_map: dict[str, str | None] = {}
    for sub in plan.values():
        orphan_map.update(sub)

    queue_actions = fix_strategies_queue(plan, args.apply)
    print(f"=== strategies_queue fixes ({len(queue_actions)}) ===")
    for a in queue_actions[:40]:
        print(f"  {a}")
    if len(queue_actions) > 40:
        print(f"  ... +{len(queue_actions)-40} more")
    print()

    index_actions = fix_index_html(orphan_map, args.apply)
    print(f"=== index.html fixes ({len(index_actions)}) ===")
    for a in index_actions:
        print(f"  {a}")
    print()
    print(f"Total artifact actions: {len(all_actions)}")
    if not args.apply:
        print("(dry run; pass --apply to mutate)")


if __name__ == "__main__":
    main()
