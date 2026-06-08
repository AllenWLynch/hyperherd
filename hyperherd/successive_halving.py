"""Stateless successive-halving (SH) pruning planner.

This module is a **pure** function of the current sweep state: given each
trial's status and logged metric stream plus the sweep's SH parameters, it
returns the action that should accompany each trial (SUBMIT / PRUNE / PAUSE /
NONE). It holds no state between calls and performs no I/O — the `herd sh`
command (in `cli.py`) is responsible for loading inputs, applying the returned
actions (manifest writes, SLURM submits, signal files), and re-invoking this
each tick.

## The algorithm

Trials are evaluated at geometrically-spaced step *rungs*:
``r_k = min_steps * eta**k`` for ``r_k <= budget`` (eta=2 → halving). At each
rung the surviving *cohort* is split: the better half is promoted to the next
rung, the worse half is pruned, and trials whose standing can't yet be decided
(because not enough of the field has reached the rung) are paused until it can.

The key to acting before every trial has been evaluated: a trial at a rung is
decided as soon as its relative standing is *certain regardless of how the
not-yet-arrived trials turn out*.

For a cohort of size ``m`` we keep the top ``K = ceil(m/2)`` and prune the
bottom ``floor(m/2)``. For a trial ``t`` that has reached the rung, let

* ``ahead_definite`` = number of cohort members that have reached the rung and
  are *certainly* ranked above ``t`` — strictly better by the objective, or
  equal-valued but with a smaller (tie-breaking) trial index;
* ``unreached``      = number of cohort members that have not reached the rung
  yet (each could turn out better than ``t``).

Then:

* **PRUNE**   iff ``ahead_definite >= K`` — already out of the top ``K``, and
  late arrivals can only push ``t`` further down.
* **PROMOTE** iff ``ahead_definite + unreached < K`` — in the top ``K`` even if
  every late arrival beats ``t``.
* **PAUSE**   otherwise — genuinely undecidable until more of the field arrives.

Ties are broken by trial index (immutable), so the partition is deterministic
and recomputation is reproducible/idempotent.

## Statelessness, stickiness, liveness

* ``pruned`` is sticky and authoritative: a pruned trial is excluded from every
  cohort and never reconsidered.
* ``failed`` / ``cancelled`` trials are excluded from cohorts too (their compute
  is gone — they can't compete or arrive). Excluding them is also what keeps the
  algorithm live: every remaining cohort member is either already at the rung or
  still *advancing* toward it (running/queued/ready) or *resumable*
  (paused-but-promoted), so a PAUSE always eventually resolves — no deadlock.
* ``paused`` is **not** authoritative: it is recomputed from metrics. A paused
  trial that is now provably top-half becomes SUBMIT (resume); now provably
  bottom-half becomes PRUNE; still ambiguous stays paused.

## Resume-thrash guards

* A trial is only judged at the highest rung it has *data* for; a trial promoted
  past a rung but sitting exactly at that rung's step (e.g. just resumed from a
  checkpoint, no new data yet) is left to run, never re-paused at the rung it
  already cleared.
* Only ``running`` trials are PRUNE/PAUSE-able. A ``submitted``/``queued`` trial
  (just (re)launched, not yet producing) is left alone until it is running and
  has logged past the rung.
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence


# Trial statuses that are out of the competition entirely: pruned is sticky;
# failed/cancelled trials can no longer produce values or advance.
_EXCLUDED_STATUSES = frozenset({"pruned", "failed", "cancelled"})

# Statuses we never issue an active PRUNE/PAUSE/SUBMIT(resume) against beyond
# the explicit rules in `verdict_to_action`.
_TERMINAL_STATUSES = frozenset({"pruned", "failed", "cancelled", "completed"})


class Verdict(Enum):
    """The algorithm's ranking judgement for a trial at its decision rung."""

    PROMOTE = "promote"        # provably top-half at its reached rung
    PRUNE = "prune"            # provably bottom-half at a rung
    PAUSE = "pause"            # reached a rung but standing not yet decidable
    RUN_FREE = "run_free"      # promoted past the final rung → run to budget
    NOT_AT_RUNG = "not_at_rung"  # hasn't reached its first decision rung yet
    NONE = "none"              # excluded (terminal/sticky); no judgement


class Action(Enum):
    """The side effect `herd sh` should apply for a trial."""

    SUBMIT = "submit"   # launch a ready trial, or resume a promoted paused one
    PRUNE = "prune"     # stop + mark pruned (terminal)
    PAUSE = "pause"     # stop + mark paused (resumable)
    NONE = "none"       # leave as-is


@dataclass(frozen=True)
class TrialState:
    """One trial's input to the planner.

    `stream` is the raw per-step records as written by `log_result(step=...)`
    and read back by `logging.load_metric_stream` — a sequence of mappings with
    `step`, `value`, and (optionally) `ts` keys. Duplicate steps are deduped
    keeping the largest `ts`, matching `logging.collect_step_rows`.
    """

    index: int
    status: str
    stream: Sequence[dict] = ()


@dataclass(frozen=True)
class SweepConfig:
    """Sweep-level SH parameters (mirrors `config.SuccessiveHalving`)."""

    metric: str
    direction: str   # "min" | "max"
    min_steps: int
    budget: int
    eta: int = 2


@dataclass(frozen=True)
class TrialAction:
    """The planner's decision for one trial."""

    index: int
    action: Action
    verdict: Verdict
    rung: Optional[int]   # decision rung index (None when not at a rung)
    reason: str


# --- pure helpers ----------------------------------------------------------

def rung_schedule(min_steps: int, budget: int, eta: int) -> List[int]:
    """Geometric rung ladder: [min_steps, min_steps*eta, ...] capped at budget.

    Returns [] when min_steps > budget (no pruning possible). `eta` must be
    >= 2 (validated upstream in config); guarded here to avoid a non-terminating
    loop if called directly.
    """
    if eta < 2 or min_steps > budget:
        return []
    rungs = []
    r = min_steps
    while r <= budget:
        rungs.append(r)
        r *= eta
    return rungs


def dedup_stream(stream: Sequence[dict]) -> Dict[int, float]:
    """Collapse a metric stream to {step: value}, keeping the max-`ts` record
    per step. Mirrors the dedup rule in `logging.collect_step_rows` so a trial
    that re-logged a step after a restart resolves to its newest value."""
    by_step: Dict[int, tuple] = {}
    for rec in stream:
        step = rec.get("step")
        if step is None:
            continue
        step = int(step)
        ts = rec.get("ts") or 0
        if step not in by_step or ts >= by_step[step][1]:
            by_step[step] = (rec.get("value"), ts)
    return {s: v for s, (v, _ts) in by_step.items()}


def reached_rung_index(steps: Dict[int, float], rungs: Sequence[int]) -> int:
    """Highest rung index k such that the trial has trained to rungs[k]
    (its max logged step >= rungs[k]). -1 if it hasn't reached rung 0."""
    if not steps:
        return -1
    mx = max(steps)
    idx = -1
    for k, r in enumerate(rungs):
        if mx >= r:
            idx = k
        else:
            break
    return idx


def value_at_rung(steps: Dict[int, float], r: int) -> Optional[float]:
    """The trial's objective value as of rung step `r`.

    The value at the largest logged step <= r. If the trial logs sparsely and
    has no step at or before `r` (its first record is already past the rung),
    fall back to the earliest logged value — a reached trial always has a value.
    None only for an empty stream.
    """
    if not steps:
        return None
    eligible = [s for s in steps if s <= r]
    if eligible:
        return steps[max(eligible)]
    return steps[min(steps)]


def _worst(direction: str) -> float:
    return math.inf if direction == "min" else -math.inf


def _normalize(value, direction: str) -> float:
    """Map a metric value to a comparable float, sending None/NaN/inf to the
    worst possible value so a blown-up trial ranks last (and gets pruned)
    rather than spuriously appearing to beat everyone."""
    if value is None:
        return _worst(direction)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return _worst(direction)
    if not math.isfinite(v):
        return _worst(direction)
    return v


def better(a, b, direction: str) -> bool:
    """Strict objective comparison: is `a` strictly better than `b`?"""
    av, bv = _normalize(a, direction), _normalize(b, direction)
    return av < bv if direction == "min" else av > bv


# --- core ------------------------------------------------------------------

def _rung_verdict(
    idx: int,
    cohort: Sequence[int],
    rung_idx: int,
    rungs: Sequence[int],
    vals: Dict[int, Dict[int, float]],
    reached: Dict[int, int],
    direction: str,
) -> Verdict:
    """PRUNE / PROMOTE / PAUSE for a trial that has reached `rung_idx`.

    `vals[i][k]` is trial i's value at rung k (only populated for reached
    rungs). `reached[i]` is trial i's reached rung index.
    """
    m = len(cohort)
    K = (m + 1) // 2  # ceil(m/2)
    nv_idx = _normalize(vals[idx][rung_idx], direction)

    ahead_definite = 0
    unreached = 0
    for other in cohort:
        if other == idx:
            continue
        if reached[other] < rung_idx:
            unreached += 1
            continue
        nv_other = _normalize(vals[other][rung_idx], direction)
        if nv_other == nv_idx:
            # tie → smaller index ranks ahead (deterministic, recomputable)
            if other < idx:
                ahead_definite += 1
        elif (nv_other < nv_idx) if direction == "min" else (nv_other > nv_idx):
            ahead_definite += 1

    if ahead_definite >= K:
        return Verdict.PRUNE
    if ahead_definite + unreached < K:
        return Verdict.PROMOTE
    return Verdict.PAUSE


def plan_successive_halving(
    trials: Sequence[TrialState],
    cfg: SweepConfig,
) -> List[TrialAction]:
    """Compute the SH action for every trial. Pure and idempotent."""
    rungs = rung_schedule(cfg.min_steps, cfg.budget, cfg.eta)

    # Precompute deduped streams, reached-rung index, and per-rung values.
    reached: Dict[int, int] = {}
    vals: Dict[int, Dict[int, float]] = {}
    for t in trials:
        steps = dedup_stream(t.stream)
        rj = reached_rung_index(steps, rungs)
        reached[t.index] = rj
        vals[t.index] = {k: value_at_rung(steps, rungs[k]) for k in range(rj + 1)}

    # Cohort pool: everything that can still compete (excludes pruned/failed/
    # cancelled). `completed` trials stay in — they are full results and remain
    # valid competitors at every rung they reached.
    pool = [t.index for t in trials if t.status not in _EXCLUDED_STATUSES]

    # Build cohorts rung-by-rung, recording each trial's verdict at each rung
    # it participates in. C_{k+1} = trials PROMOTEd at rung k.
    cohorts: List[List[int]] = []
    rung_verdicts: Dict[tuple, Verdict] = {}
    C = pool
    for k in range(len(rungs)):
        if not C:
            break
        cohorts.append(C)
        promoted = []
        for idx in C:
            if reached[idx] >= k:
                v = _rung_verdict(idx, C, k, rungs, vals, reached, cfg.direction)
                if v == Verdict.PROMOTE:
                    promoted.append(idx)
            else:
                v = Verdict.NOT_AT_RUNG
            rung_verdicts[(idx, k)] = v
        C = promoted

    last_rung = len(rungs) - 1

    # Resolve each trial's final verdict at its *reached* rung (the highest rung
    # it has data for), respecting where it dropped out of the cohort.
    actions: List[TrialAction] = []
    for t in trials:
        idx, status = t.index, t.status
        rj = reached[idx]

        if status in _EXCLUDED_STATUSES:
            actions.append(_action(idx, status, Verdict.NONE, None,
                                   f"{status}: excluded from SH"))
            continue

        if not rungs:
            # No rungs (min_steps > budget): SH never prunes; just launch readies.
            verdict = Verdict.NOT_AT_RUNG
            actions.append(_action(idx, status, verdict, None,
                                   "no rung schedule (min_steps > budget)"))
            continue

        if rj < 0:
            actions.append(_action(idx, status, Verdict.NOT_AT_RUNG, None,
                                   "not yet at first rung"))
            continue

        if rj < len(cohorts) and idx in cohorts[rj]:
            # Still in the cohort at its reached rung → decision is made there.
            verdict = rung_verdicts[(idx, rj)]
            if verdict == Verdict.PROMOTE and rj == last_rung:
                verdict = Verdict.RUN_FREE
            decision_rung = rj
        else:
            # Dropped out earlier: find the highest rung it was a member of and
            # use the verdict there (PRUNE, or PAUSE if it was paused there).
            member_rungs = [k for k in range(len(cohorts)) if idx in cohorts[k]]
            jtop = member_rungs[-1] if member_rungs else 0
            verdict = rung_verdicts.get((idx, jtop), Verdict.PRUNE)
            if verdict == Verdict.PROMOTE:
                # Promoted at jtop but reached beyond without staying in cohort —
                # only happens if cohorts list ended (everyone else gone); treat
                # as run-free.
                verdict = Verdict.RUN_FREE
            decision_rung = jtop

        actions.append(_action(idx, status, verdict, decision_rung,
                               _reason(verdict, decision_rung, rungs)))

    return actions


def verdict_to_action(status: str, verdict: Verdict) -> Action:
    """Map (current status, ranking verdict) → side effect. See module docstring."""
    if status == "ready":
        return Action.SUBMIT  # initial launch
    if status in _TERMINAL_STATUSES:
        return Action.NONE
    if status in ("submitted", "queued"):
        return Action.NONE  # just (re)launched — thrash guard, don't judge yet
    if status == "running":
        if verdict == Verdict.PRUNE:
            return Action.PRUNE
        if verdict == Verdict.PAUSE:
            return Action.PAUSE
        return Action.NONE  # PROMOTE / RUN_FREE / NOT_AT_RUNG → keep running
    if status == "paused":
        if verdict in (Verdict.PROMOTE, Verdict.RUN_FREE):
            return Action.SUBMIT  # provably top-half → resume
        if verdict == Verdict.PRUNE:
            return Action.PRUNE
        return Action.NONE  # still ambiguous → stay paused
    return Action.NONE


def _action(idx: int, status: str, verdict: Verdict, rung: Optional[int],
            reason: str) -> TrialAction:
    return TrialAction(
        index=idx,
        action=verdict_to_action(status, verdict),
        verdict=verdict,
        rung=rung,
        reason=reason,
    )


def _reason(verdict: Verdict, rung: Optional[int], rungs: Sequence[int]) -> str:
    if rung is None:
        step = None
    else:
        step = rungs[rung] if rung < len(rungs) else None
    at = f" at rung {rung} (step {step})" if step is not None else ""
    return {
        Verdict.PROMOTE: f"top-half{at}",
        Verdict.PRUNE: f"bottom-half{at}",
        Verdict.PAUSE: f"undecidable{at} — pausing until field catches up",
        Verdict.RUN_FREE: f"cleared final rung{at} — running to budget",
        Verdict.NOT_AT_RUNG: "not yet at a rung",
        Verdict.NONE: "excluded",
    }[verdict]
