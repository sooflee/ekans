"""
Canonical winner gate for ekans signals.

A single function `is_winner(result, mt_data)` is the source of truth for whether
a backtested signal qualifies for promotion to the index.html High Return tab and
inclusion in build_portfolio.py.

Gate criteria (all must hold):
  1. status not failed             (any value that isn't "fail"/"failed"/"error";
                                    older results omit the field entirely)
  2. sharpe   > 0.5                (raw absolute Sharpe)
  3. cagr     > 0.10               (raw CAGR > 10%)
  4. bh_significant == True        (survives Benjamini-Hochberg multiple-testing
                                    correction across the whole catalog; loaded
                                    from results/_multiple_testing.json)
  5. oos_sharpe > 0                (positive out-of-sample Sharpe on the 2nd half
                                    of the in-sample period)
  6. sample size floor:            n_days >= 252  OR  n_events >= 20
                                   (rejects ~1-year-of-data daily PnLs and event
                                    studies with too few events to be meaningful)

If `_multiple_testing.json` has no entry for a signal, criterion 4 fails closed:
the BH run must include the signal before it can be promoted. This is intentional —
without BH it's just one of N tests and the front page should not advertise it.

Usage:
    from pipeline.winner_gate import is_winner, winner_reasons, load_mt_data
    mt = load_mt_data()
    if is_winner(result_json_dict, mt):
        ...
    # debugging why a candidate failed:
    reasons = winner_reasons(result_json_dict, mt)
    # -> {"status_ok": True, "sharpe": True, "cagr": True,
    #     "bh_significant": False, "oos_sharpe": True, "sample_size": True}
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
_MT_PATH = _ROOT / "results" / "_multiple_testing.json"

MIN_SHARPE = 0.5
MIN_CAGR = 0.10
MIN_N_DAYS = 252
MIN_N_EVENTS = 20


def load_mt_data() -> dict[str, dict[str, Any]]:
    if not _MT_PATH.exists():
        return {}
    with open(_MT_PATH) as f:
        return json.load(f)


def _signal_id(result: dict[str, Any]) -> str | None:
    return result.get("signal_id") or result.get("id")


def winner_reasons(result: dict[str, Any], mt_data: dict[str, dict[str, Any]]) -> dict[str, bool]:
    sid = _signal_id(result)
    mt_entry = mt_data.get(sid or "", {})

    sharpe = result.get("sharpe")
    cagr = result.get("cagr")
    oos_sharpe = result.get("oos_sharpe")
    n_days = result.get("n_days") or 0
    n_events = result.get("n_events") or 0

    status = (result.get("status") or "").lower()
    status_ok = status not in {"fail", "failed", "error"}

    return {
        "status_ok": status_ok,
        "sharpe": sharpe is not None and sharpe > MIN_SHARPE,
        "cagr": cagr is not None and cagr > MIN_CAGR,
        "bh_significant": bool(mt_entry.get("bh_significant")),
        "oos_sharpe": oos_sharpe is not None and oos_sharpe > 0,
        "sample_size": n_days >= MIN_N_DAYS or n_events >= MIN_N_EVENTS,
    }


def is_winner(result: dict[str, Any], mt_data: dict[str, dict[str, Any]]) -> bool:
    return all(winner_reasons(result, mt_data).values())
