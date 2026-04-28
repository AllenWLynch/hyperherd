"""Apply constraints to filter and modify hyperparameter combinations."""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List

from hyperwhip.config import Constraint


@dataclass
class Trial:
    """A parameter combination plus any constraint-injected extra Hydra overrides.

    `params` are the swept parameter values (subject to dedup).
    `extras` are arbitrary Hydra overrides accumulated from constraint `set`
    blocks. Keys are Hydra paths (not parameter names) and never participate
    in dedup — two trials with identical `params` are guaranteed to receive
    identical `extras` because `extras` is a deterministic function of
    `params` and the constraint list.
    """
    params: Dict[str, Any]
    extras: Dict[str, Any] = field(default_factory=dict)


def _floats_close(a: Any, b: Any) -> bool:
    return (
        isinstance(a, (int, float))
        and isinstance(b, (int, float))
        and math.isclose(float(a), float(b), rel_tol=1e-9)
    )


def _eq(actual: Any, expected: Any) -> bool:
    if _floats_close(actual, expected):
        return True
    return actual == expected


def _value_in(actual: Any, choices: List[Any]) -> bool:
    for c in choices:
        if _eq(actual, c):
            return True
    return False


def _match_one(actual: Any, matcher: Any) -> bool:
    """Match a single combo value against a `when` matcher.

    Matcher forms:
      - scalar: exact equality (with float tolerance)
      - list:   OR — actual matches any element
      - {op: value}: comparison operator (eq/ne/gt/ge/lt/le/in/not_in)
    """
    if isinstance(matcher, dict):
        # Operator map — schema validator already ensures exactly one valid key
        op, target = next(iter(matcher.items()))
        if op == "eq":
            return _eq(actual, target)
        if op == "ne":
            return not _eq(actual, target)
        if op == "in":
            return _value_in(actual, target)
        if op == "not_in":
            return not _value_in(actual, target)
        # Numeric comparisons
        if not isinstance(actual, (int, float)) or not isinstance(target, (int, float)):
            return False
        if op == "gt":
            return actual > target
        if op == "ge":
            return actual >= target
        if op == "lt":
            return actual < target
        if op == "le":
            return actual <= target
        return False

    if isinstance(matcher, list):
        return _value_in(actual, matcher)

    return _eq(actual, matcher)


def _match_when(combo: Dict[str, Any], when: Dict[str, Any]) -> bool:
    """All `when` clauses must match (AND across params)."""
    for param, matcher in when.items():
        if param not in combo:
            return False
        if not _match_one(combo[param], matcher):
            return False
    return True


def apply_constraints(
    combinations: List[Dict[str, Any]], constraints: List[Constraint]
) -> List[Trial]:
    """Apply constraints to filter/modify combinations and accumulate extras.

    Returns a list of Trial objects. The dedup key is built from `params`
    only — `extras` is determined by `params` and the constraint list, so
    identical params produce identical extras.
    """
    trials: List[Trial] = [Trial(params=dict(c)) for c in combinations]

    for constraint in constraints:
        next_trials: List[Trial] = []
        for trial in trials:
            if not _match_when(trial.params, constraint.when):
                next_trials.append(trial)
                continue

            # Apply exclude: drop trials where target param has excluded values
            if constraint.exclude:
                excluded = False
                for param, exc_values in constraint.exclude.items():
                    if param in trial.params and _value_in(trial.params[param], exc_values):
                        excluded = True
                        break
                if excluded:
                    continue

            # Apply force: override param values
            if constraint.force:
                new_params = dict(trial.params)
                for param, forced_val in constraint.force.items():
                    new_params[param] = forced_val
                trial = Trial(params=new_params, extras=dict(trial.extras))

            # Apply set: accumulate extra Hydra overrides
            if constraint.set:
                if trial.extras is trial.params or not isinstance(trial.extras, dict):
                    trial = Trial(params=trial.params, extras={})
                else:
                    trial = Trial(params=trial.params, extras=dict(trial.extras))
                for k, v in constraint.set.items():
                    trial.extras[k] = v

            next_trials.append(trial)

        trials = next_trials

    # Deduplicate on params only
    seen = set()
    deduped: List[Trial] = []
    for trial in trials:
        key = _combo_key(trial.params)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(trial)

    return deduped


def _combo_key(combo: Dict[str, Any]) -> str:
    """Hashable key for a parameter combination (dedup)."""
    parts = []
    for k in sorted(combo.keys()):
        v = combo[k]
        if isinstance(v, float):
            parts.append(f"{k}={v:.10g}")
        else:
            parts.append(f"{k}={v}")
    return "|".join(parts)
