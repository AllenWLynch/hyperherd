"""Stateless successive-halving (SH) pruning planner.

This module is a **pure** function of the current sweep state: given each
trial's status and logged metric stream plus the sweep's SH parameters, it
returns the action that should accompany each trial (SUBMIT / PRUNE / PAUSE /
NONE). SUBMIT is only ever a *resume* of a trial SH itself paused — initial
launching of never-submitted (`ready`) trials is the user's job (`herd run`),
not SH's. It holds no state between calls and performs no I/O — the `herd sh`
command (in `cli.py`) is responsible for loading inputs, applying the returned
actions (manifest writes, SLURM submits, signal files), and re-invoking this
each tick.

## Rungs

Trials are evaluated at geometrically-spaced step *rungs*:
``r_k = min_steps * eta**k`` for ``r_k <= budget`` (eta=2 → halving). A trial is
"at" rung ``k`` once its max logged step reaches ``r_k``.

## Two schedulers (`cfg.mode`)

**`"sync"`** — the conservative bracket. The surviving *cohort* at each rung is
split: keep the top ``K = ceil(m/2)``, prune the bottom, and *pause* trials
whose standing can't yet be decided because not enough of the field has arrived.
A trial at a rung is decided as soon as its standing is certain *regardless of
how the not-yet-arrived trials turn out*. For a trial ``t`` with

* ``ahead_definite`` = cohort members that reached the rung and are certainly
  above ``t`` (strictly better, or tied with a smaller trial index);
* ``unreached``      = cohort members not yet at the rung (each could beat ``t``);

  * **PRUNE**   iff ``ahead_definite >= K``;
  * **PROMOTE** iff ``ahead_definite + unreached < K``;
  * **PAUSE**   otherwise.

Never makes an early-stop mistake, but stalls (pauses) when trials are launched
unevenly, since `unreached` keeps decisions open.

**`"asha"`** — asynchronous halving (Li et al. 2020). At each rung, rank only
the trials that have *arrived* (reached the rung) and keep the top
``floor(n/eta)`` — keep all while fewer than ``eta`` have arrived (the gate that
prevents over-eager cuts on a tiny field). There is no waiting and no PAUSE: a
trial is pruned at the lowest rung it failed to survive, else it keeps running.
Always makes progress; in exchange it can occasionally stop a trial that a
slower, worse-but-later arrival would have spared.

Ties are broken by trial index (immutable) in both modes, so decisions are
deterministic and recomputation is reproducible/idempotent.

## Statelessness, stickiness, liveness

* ``pruned`` is sticky and authoritative: a pruned trial is excluded from every
  cohort and never reconsidered.
* ``failed`` / ``cancelled`` trials are excluded from cohorts too (their compute
  is gone — they can't compete or arrive). ``ready`` (never-launched) trials are
  *also* excluded: SH never launches them (starting trials is the user's job, via
  `herd run`), so a `ready` member could never arrive on its own. Excluding all
  of these keeps the cohort to the *committed* field. (Consequence: if the user
  launches trials in waves, early arrivals are judged against only the launched
  subset. Under `"sync"` this means more PAUSEs until the field fills in; under
  `"asha"` it means earlier — occasionally premature — cuts.)
* ``paused`` is **not** authoritative: it is recomputed from metrics. A paused
  trial now provably top-of-its-rung becomes SUBMIT (resume); now provably out
  becomes PRUNE; still ambiguous stays paused (only possible under `"sync"`).

## Resume-thrash guards

* A trial is only judged at the highest rung it has *data* for; a trial promoted
  past a rung but sitting exactly at that rung's step (e.g. just resumed from a
  checkpoint, no new data yet) is left to run, never re-cut at a rung it cleared.
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

# Statuses excluded from the cohort (the denominator that sets K = ceil(m/2)
# and the `unreached` phantom count). Terminal trials plus `ready`: SH never
# launches a never-submitted trial (starting trials is the user's job), so a
# `ready` member could never arrive on its own — counting it would let a paused
# trial wait forever on a phantom. See the liveness note in the module docstring.
_NON_COHORT_STATUSES = _EXCLUDED_STATUSES | frozenset({"ready"})

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

    SUBMIT = "submit"   # resume a promoted trial that SH had paused
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
    mode: str = "sync"   # "sync" (conservative bracket) | "asha" (asynchronous)


@dataclass(frozen=True)
class RungStanding:
    """The cohort arithmetic behind a trial's verdict at its decision rung.

    This is what `herd sh --reason` surfaces to explain *why* a trial was
    pruned/paused/kept: at the rung, the cohort has `cohort_size` members and
    keeps the top `keep` (= ceil(cohort_size/2)); `ahead_definite` of them are
    certainly ranked above this trial and `unreached` have not reached the rung
    yet (each could still turn out better). The verdict follows directly:
    PRUNE iff ``ahead_definite >= keep``; PROMOTE iff
    ``ahead_definite + unreached < keep``; PAUSE otherwise.
    """

    rung: int             # decision rung index
    step: int             # rungs[rung] — the step threshold
    value: Optional[float]  # this trial's objective value at the rung
    cohort_size: int      # m — trials still competing at this rung
    keep: int             # K = ceil(m/2) — how many survive
    ahead_definite: int   # cohort members certainly ranked above this trial
    unreached: int        # cohort members not yet at the rung


@dataclass(frozen=True)
class TrialAction:
    """The planner's decision for one trial."""

    index: int
    action: Action
    verdict: Verdict
    rung: Optional[int]   # decision rung index (None when not at a rung)
    reason: str
    standing: Optional[RungStanding] = None  # cohort arithmetic (None off-rung)
    status: str = ""           # the trial's manifest status (for labelling)
    max_step: Optional[int] = None  # highest logged step (None if nothing logged)


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
) -> tuple:
    """PRUNE / PROMOTE / PAUSE for a trial that has reached `rung_idx`.

    `vals[i][k]` is trial i's value at rung k (only populated for reached
    rungs). `reached[i]` is trial i's reached rung index. Returns
    ``(verdict, RungStanding)`` — the standing carries the cohort arithmetic so
    callers (e.g. `herd sh --reason`) can explain the decision.
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
        verdict = Verdict.PRUNE
    elif ahead_definite + unreached < K:
        verdict = Verdict.PROMOTE
    else:
        verdict = Verdict.PAUSE

    standing = RungStanding(
        rung=rung_idx,
        step=rungs[rung_idx],
        value=vals[idx][rung_idx],
        cohort_size=m,
        keep=K,
        ahead_definite=ahead_definite,
        unreached=unreached,
    )
    return verdict, standing


def _precompute(trials, rungs):
    """Per-trial: reached-rung index, value at each reached rung, and the highest
    logged step (the last shown by `herd sh --reason` for not-yet-at-rung trials)."""
    reached: Dict[int, int] = {}
    vals: Dict[int, Dict[int, float]] = {}
    max_steps: Dict[int, Optional[int]] = {}
    for t in trials:
        steps = dedup_stream(t.stream)
        rj = reached_rung_index(steps, rungs)
        reached[t.index] = rj
        vals[t.index] = {k: value_at_rung(steps, rungs[k]) for k in range(rj + 1)}
        max_steps[t.index] = max(steps) if steps else None
    return reached, vals, max_steps


def plan_successive_halving(
    trials: Sequence[TrialState],
    cfg: SweepConfig,
) -> List[TrialAction]:
    """Compute the SH action for every trial. Pure and idempotent.

    Two schedulers share everything but the per-rung decision (`cfg.mode`):

    * ``"sync"``  — the conservative bracket: wait (PAUSE) until a trial's
      standing is certain regardless of who hasn't arrived yet. Never makes an
      early-stop mistake; can stall when the field is launched unevenly.
    * ``"asha"``  — asynchronous halving: at each rung rank only the trials that
      have *arrived* and keep the top ``floor(n/eta)`` (all of them while fewer
      than ``eta`` have arrived). Never waits; accepts the occasional early-stop
      mistake in exchange for always making progress.
    """
    rungs = rung_schedule(cfg.min_steps, cfg.budget, cfg.eta)
    reached, vals, max_steps = _precompute(trials, rungs)

    # Cohort pool: everything that can still compete. Excludes terminal trials
    # (pruned/failed/cancelled) and `ready` (never-launched — SH won't start it,
    # so it can't arrive). `completed` trials stay in — they are full results and
    # remain valid competitors at every rung they reached.
    pool = [t.index for t in trials if t.status not in _NON_COHORT_STATUSES]

    # Algorithm-specific: decide (verdict, decision_rung, standing) for every
    # pool trial that has reached at least rung 0. Off-rung/excluded trials are
    # handled uniformly by the emit loop below.
    if cfg.mode == "asha":
        decided = _decide_asha(pool, rungs, vals, reached, cfg)
    else:
        decided = _decide_sync(pool, rungs, vals, reached, cfg)

    actions: List[TrialAction] = []
    for t in trials:
        idx, status = t.index, t.status
        rj = reached[idx]
        ms = max_steps.get(idx)

        if status in _EXCLUDED_STATUSES:
            actions.append(_action(idx, status, Verdict.NONE, None,
                                   f"{status}: excluded from SH", max_step=ms))
            continue

        if status == "ready":
            # Never-launched and out of the cohort: SH doesn't start trials
            # (the user's job), so there is nothing to decide here.
            actions.append(_action(idx, status, Verdict.NOT_AT_RUNG, None,
                                   "not launched — SH never launches trials",
                                   max_step=ms))
            continue

        if not rungs:
            # No rungs (min_steps > budget): SH never prunes.
            actions.append(_action(idx, status, Verdict.NOT_AT_RUNG, None,
                                   "no rung schedule (min_steps > budget)",
                                   max_step=ms))
            continue

        if rj < 0:
            actions.append(_action(
                idx, status, Verdict.NOT_AT_RUNG, None,
                f"still training toward rung 0 (step {rungs[0]})", max_step=ms))
            continue

        verdict, decision_rung, standing = decided[idx]
        actions.append(_action(idx, status, verdict, decision_rung,
                               _reason(verdict, decision_rung, rungs),
                               standing, max_step=ms))

    return actions


def _decide_sync(pool, rungs, vals, reached, cfg):
    """Conservative bracket with early, provably-correct decisions + PAUSE.

    Returns ``{idx: (verdict, decision_rung, standing)}`` for every pool trial
    that has reached at least rung 0.
    """
    # Build cohorts rung-by-rung, recording each trial's verdict at each rung
    # it participates in. C_{k+1} = trials PROMOTEd at rung k.
    cohorts: List[List[int]] = []
    rung_verdicts: Dict[tuple, Verdict] = {}
    rung_standings: Dict[tuple, RungStanding] = {}
    C = pool
    for k in range(len(rungs)):
        if not C:
            break
        cohorts.append(C)
        promoted = []
        for idx in C:
            if reached[idx] >= k:
                v, standing = _rung_verdict(
                    idx, C, k, rungs, vals, reached, cfg.direction)
                rung_standings[(idx, k)] = standing
                if v == Verdict.PROMOTE:
                    promoted.append(idx)
            else:
                v = Verdict.NOT_AT_RUNG
            rung_verdicts[(idx, k)] = v
        C = promoted

    last_rung = len(rungs) - 1
    decided = {}
    for idx in pool:
        rj = reached[idx]
        if rj < 0:
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
        decided[idx] = (verdict, decision_rung,
                        rung_standings.get((idx, decision_rung)))
    return decided


def _decide_asha(pool, rungs, vals, reached, cfg):
    """Asynchronous halving: at each rung rank only the trials that have arrived
    and keep the top ``floor(n/eta)`` — no waiting, no PAUSE.

    Returns ``{idx: (verdict, decision_rung, standing)}`` for every pool trial
    that has reached at least rung 0.
    """
    last_rung = len(rungs) - 1

    # Per rung: who survives the cut, and each arrived trial's standing there.
    survivors: Dict[tuple, bool] = {}
    standings: Dict[tuple, RungStanding] = {}
    for k in range(len(rungs)):
        arrived = [idx for idx in pool if reached[idx] >= k]
        n = len(arrived)
        keep = n if n < cfg.eta else n // cfg.eta  # ASHA gate: cut once n >= eta
        order = sorted(
            arrived,
            key=lambda i: (_normalize(vals[i][k], cfg.direction), i),
        )
        for rank, idx in enumerate(order):
            survivors[(idx, k)] = rank < keep
            standings[(idx, k)] = RungStanding(
                rung=k, step=rungs[k], value=vals[idx][k],
                cohort_size=n, keep=keep, ahead_definite=rank, unreached=0)

    decided = {}
    for idx in pool:
        rj = reached[idx]
        if rj < 0:
            continue
        # Prune at the lowest reached rung it failed to survive; otherwise it
        # cleared every rung it reached → keep running (or run to budget).
        cut = next((k for k in range(rj + 1) if not survivors[(idx, k)]), None)
        if cut is not None:
            decided[idx] = (Verdict.PRUNE, cut, standings[(idx, cut)])
        elif rj == last_rung:
            decided[idx] = (Verdict.RUN_FREE, rj, standings[(idx, rj)])
        else:
            decided[idx] = (Verdict.PROMOTE, rj, standings[(idx, rj)])
    return decided


def verdict_to_action(status: str, verdict: Verdict) -> Action:
    """Map (current status, ranking verdict) → side effect. See module docstring."""
    if status == "ready":
        # Initial launch of a never-submitted trial is the *user's* job (via
        # `herd run`), not SH's. SH only ever (re)starts trials it paused.
        return Action.NONE
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
            reason: str, standing: Optional[RungStanding] = None,
            max_step: Optional[int] = None) -> TrialAction:
    return TrialAction(
        index=idx,
        action=verdict_to_action(status, verdict),
        verdict=verdict,
        rung=rung,
        reason=reason,
        standing=standing,
        status=status,
        max_step=max_step,
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


# Human-facing verb for a decision: the *effect* the user sees, derived from the
# trial's status together with the (verdict, action) pair rather than the raw
# Action enum — so a finished trial reads "complete" (not a verdict-derived verb
# like "stay-paused"), a kept-running trial "continue", a resumed one "resume".
def decision_label(ta: TrialAction) -> str:
    # Terminal/just-launched statuses describe themselves — never a rung verb.
    if ta.status == "completed":
        return "complete"
    if ta.status in ("submitted", "queued"):
        return "just-launched"
    if ta.action == Action.PRUNE:
        return "prune"
    if ta.action == Action.PAUSE:
        return "pause"
    if ta.action == Action.SUBMIT:
        return "resume"
    # action is NONE — distinguish "left running" verdicts from inert ones.
    if ta.verdict == Verdict.RUN_FREE:
        return "run-to-budget"
    if ta.verdict == Verdict.PROMOTE:
        return "continue"
    if ta.status == "paused":
        return "stay-paused"   # paused and not (yet) promoted
    if ta.verdict == Verdict.NOT_AT_RUNG:
        return "pre-rung"      # running, still training toward the first rung
    return "no-op"


def explain(ta: TrialAction) -> str:
    """A one-line, plain-language account of *why* a trial got its verdict.

    When the trial reached a rung this spells out the arithmetic behind the
    decision (this is what `herd sh --reason` shows); off-rung it falls back to
    the short `reason` string. With no stragglers (`unreached == 0`, always the
    case under ASHA) it uses a simple rank-of-arrived phrasing; otherwise it
    shows the sync scheduler's ahead/unreached breakdown.
    """
    s = ta.standing
    if s is None:
        return ta.reason
    rank = s.ahead_definite + 1

    if ta.status == "completed":
        # Terminal — SH takes no action; describe where its final result landed
        # without action language ("prune"/"continue" would be misleading).
        return (f"completed — final result ranks {rank} of {s.cohort_size} "
                f"at rung {s.rung} (keep top {s.keep})")

    if s.unreached == 0:
        head = f"rank {rank} of {s.cohort_size} arrived, keep top {s.keep}: "
        if ta.verdict == Verdict.PRUNE:
            return head + "below the cut → prune"
        if ta.verdict == Verdict.RUN_FREE:
            return head + "survived the final rung → running to budget"
        if ta.verdict == Verdict.PROMOTE:
            return head + "survives → keeps training toward the next rung"
        return ta.reason

    head = f"cohort of {s.cohort_size}, keep top {s.keep}: "
    if ta.verdict == Verdict.PRUNE:
        return (head + f"{s.ahead_definite} trial(s) already ranked ahead "
                f"(≥ {s.keep}) → bottom half")
    if ta.verdict in (Verdict.PROMOTE, Verdict.RUN_FREE):
        tail = " (final rung → run to budget)" if ta.verdict == Verdict.RUN_FREE else ""
        return (head + f"{s.ahead_definite} ahead + {s.unreached} not yet at rung "
                f"< {s.keep} → top half{tail}")
    if ta.verdict == Verdict.PAUSE:
        return (head + f"{s.ahead_definite} ahead, {s.unreached} not yet at this "
                f"rung → can't decide until the field catches up")
    return ta.reason
