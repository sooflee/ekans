"""
Sleeper gate for ekans signals — the counterpart to winner_gate.

winner_gate scores signals on *blended* Sharpe/CAGR over the whole sample, plus
a sample-size floor (n_days >= 252). That bias is correct for "yields steadily
now" strategies but structurally rejects the opposite kind: signals that sit
flat most of the time and pay off hard in rare regimes/triggers (long-vol, tail
hedges, counter-signals, secular-inflection trades). Their big episodes get
diluted into noise, so blended Sharpe < 0.5 and they never get promoted.

A "sleeper" is defined by its *conditional* profile — the payoff GIVEN the
signal is on — produced by harness.conditional_metrics() and stored on each
result as cond_sharpe / cond_oos_sharpe / cond_cagr / n_episodes /
mean_episode_ret / active_frac.

Gate criteria (all must hold):
  1. status not failed
  2. cond_sharpe      > 0.8     (strong edge while deployed)
  3. n_episodes       >= 5      (enough distinct firings; episode-count analog of
                                the winner-gate sample floor — rejects one-lucky-trade)
  4. cond_oos_sharpe  > 0       (the conditional edge persists out of sample)
  5. mean_episode_ret > 0       (positive expected payoff per firing)
  6. active_frac      < 0.60    (it IS dormant — otherwise it's just a normal
                                always-on strategy and winner_gate already covers it)

A *hidden* sleeper is one that passes this gate but FAILS winner_gate — i.e. a
real future-yielder the standard pipeline is throwing away. `is_hidden_sleeper`
combines the two; `sleeper_score` ranks them for triage.

Usage:
    from sleeper_gate import is_sleeper, sleeper_reasons, sleeper_score
    if is_sleeper(result):
        ...
    # find the ones winner_gate misses:
    from winner_gate import is_winner, load_mt_data
    mt = load_mt_data()
    hidden = is_sleeper(r) and not is_winner(r, mt)
"""
from __future__ import annotations

import math
from typing import Any

MIN_COND_SHARPE = 0.8
MIN_EPISODES = 5
MAX_ACTIVE_FRAC = 0.60


def _signal_id(result: dict[str, Any]) -> str | None:
    return result.get("signal_id") or result.get("id")


def sleeper_reasons(result: dict[str, Any]) -> dict[str, bool]:
    status = (result.get("status") or "").lower()
    cond_sharpe = result.get("cond_sharpe")
    cond_oos = result.get("cond_oos_sharpe")
    n_eps = result.get("n_episodes") or 0
    mean_ep = result.get("mean_episode_ret")
    active_frac = result.get("active_frac")

    return {
        "status_ok": status not in {"fail", "failed", "error"},
        "cond_sharpe": cond_sharpe is not None and cond_sharpe > MIN_COND_SHARPE,
        "episodes": n_eps >= MIN_EPISODES,
        "cond_oos_sharpe": cond_oos is not None and cond_oos > 0,
        "mean_episode_ret": mean_ep is not None and mean_ep > 0,
        "dormant": active_frac is not None and active_frac < MAX_ACTIVE_FRAC,
    }


def is_sleeper(result: dict[str, Any]) -> bool:
    return all(sleeper_reasons(result).values())


def is_hidden_sleeper(result: dict[str, Any], mt_data: dict[str, Any]) -> bool:
    """A sleeper that winner_gate rejects — the future-yielder the pipeline drops."""
    from winner_gate import is_winner
    return is_sleeper(result) and not is_winner(result, mt_data)


def sleeper_score(result: dict[str, Any]) -> float:
    """Triage rank: reward strong conditional edge, many firings, fat per-episode
    payoff. Returns 0.0 if it isn't a sleeper. Not a gate — just an ordering."""
    if not is_sleeper(result):
        return 0.0
    cond_sharpe = result.get("cond_sharpe") or 0.0
    n_eps = result.get("n_episodes") or 0
    mean_ep = result.get("mean_episode_ret") or 0.0
    return float(cond_sharpe * math.sqrt(n_eps) * max(mean_ep, 0.0))
