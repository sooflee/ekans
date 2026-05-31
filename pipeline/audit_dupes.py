"""
Audit duplicate signal IDs in results/ and report — never deletes.

Two scout agents running in parallel can pick the same dedup_key and produce two
result files sharing the same PL-prefix (e.g. `PL301_retail_is_ratio_low_…` and
`PL301_census_retail_is_ratio_…`). Both files contain real backtest output and
might disagree, so this script reports them for manual review.

Usage:
    .venv/bin/python pipeline/audit_dupes.py
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
BACKTESTS_DIR = ROOT / "backtests"

PL_PREFIX_RE = re.compile(r"^(PL\d+)_(.+)$")


def main():
    groups: dict[str, list[str]] = defaultdict(list)
    for fp in sorted(RESULTS_DIR.glob("*.json")):
        if fp.stem.startswith("_"):
            continue
        m = PL_PREFIX_RE.match(fp.stem)
        if m:
            groups[m.group(1)].append(fp.stem)

    dupes = {k: v for k, v in groups.items() if len(v) > 1}
    print(f"=== Duplicate PL-prefix audit ===")
    print(f"  total PL signals: {sum(len(v) for v in groups.values())}")
    print(f"  colliding prefixes: {len(dupes)}")

    if not dupes:
        return

    print()
    for prefix in sorted(dupes, key=lambda x: int(x[2:])):
        files = dupes[prefix]
        print(f"  {prefix}")
        for sid in files:
            r_path = RESULTS_DIR / f"{sid}.json"
            b_path = BACKTESTS_DIR / f"{sid}.py"
            try:
                d = json.loads(r_path.read_text())
                sharpe = d.get("sharpe")
                cagr = d.get("cagr")
                n = d.get("n_days") or d.get("n_events")
                sharpe_str = f"{sharpe:+.2f}" if sharpe is not None else "  —  "
                cagr_str = f"{(cagr or 0)*100:+.1f}%".rjust(7) if cagr is not None else "  —  "
                print(f"    sharpe={sharpe_str} cagr={cagr_str} n={n}  {sid}")
                if not b_path.exists():
                    print(f"      ! no backtest file at backtests/{sid}.py")
            except Exception as e:
                print(f"    [read error: {e}]  {sid}")
        print()

    print("Review each pair and decide:")
    print("  - keep both: they're actually different signals that happened to share a PL id")
    print("  - delete one: it's a true duplicate from a scout race condition")
    print("  - rename one: the underlying idea differs but PL id should be unique")
    print()
    print("Deletion is manual — do not auto-delete results/ files.")


if __name__ == "__main__":
    main()
