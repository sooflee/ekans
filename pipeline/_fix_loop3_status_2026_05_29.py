"""Fix loop3 status updates: 48 strategies got stuck as in_progress because
update_strategy_status keys on strategy_id alone, and these strategies share
strategy_ids with older records. Disambiguate by signal_id and update via
queue_io's _locked_update.
"""
import sys, json
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, "/Users/benson/Projects/ekans/pipeline")
from queue_io import _locked_update, STRATEGIES

ROOT = Path("/Users/benson/Projects/ekans")
RESDIR = ROOT / "results"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fix():
    fixed = []
    failed_no_res = []

    def upd(data):
        for i, d in enumerate(data):
            if d.get("status") != "in_progress":
                continue
            sid = d.get("signal_id")
            if not sid:
                continue
            res_path = RESDIR / f"{sid}.json"
            if not res_path.exists():
                failed_no_res.append(sid)
                continue
            try:
                res = json.load(open(res_path))
            except Exception as e:
                data[i] = {**d, "status": "failed",
                           "backtest_result": {"status": "fail",
                                               "reason": f"parse: {e}"},
                           "backtested_at": _now()}
                failed_no_res.append(sid)
                continue
            if res.get("status") == "fail":
                data[i] = {**d, "status": "failed",
                           "backtest_result": {"status": "fail",
                                               "reason": res.get("reason", "")[:200]},
                           "backtested_at": _now()}
                fixed.append((sid, "failed"))
            else:
                br = {"status": "ok",
                      "sharpe": res.get("sharpe"),
                      "cagr": res.get("cagr"),
                      "max_dd": res.get("max_dd"),
                      "t_stat": res.get("t_stat")}
                data[i] = {**d, "status": "done",
                           "backtest_result": br,
                           "backtested_at": _now()}
                fixed.append((sid, "done"))
        return data

    _locked_update(STRATEGIES, [], upd)

    out = {"fixed_count": len(fixed),
           "no_result": failed_no_res,
           "fixed": fixed}
    print(json.dumps({"fixed_count": len(fixed),
                      "done": sum(1 for _, s in fixed if s == "done"),
                      "failed": sum(1 for _, s in fixed if s == "failed"),
                      "no_result": len(failed_no_res)}, indent=2))
    return out


if __name__ == "__main__":
    fix()
