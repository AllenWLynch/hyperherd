"""Apply constraints to filter and modify hyperparameter combinations."""

import math
from typing import Any, Dict, List

from hyperwhip.config import Constraint


def _match_when(combo: Dict[str, Any], when: Dict[str, Any]) -> bool:
    """Check if a combination matches all 'when' conditions."""
    for param, expected in when.items():
        actual = combo.get(param)
        if actual is None:
            return False
        if isinstance(actual, float) and isinstance(expected, (int, float)):
            if not math.isclose(actual, float(expected), rel_tol=1e-9):
                return False
        elif actual != expected:
            return False
    return True


def _is_excluded(value: Any, excluded_values: List[Any]) -> bool:
    """Check if a value is in the exclusion list."""
    for ev in excluded_values:
        if isinstance(value, float) and isinstance(ev, (int, float)):
            if math.isclose(value, float(ev), rel_tol=1e-9):
                return True
        elif value == ev:
            return True
    return False


def apply_constraints(
    combinations: List[Dict[str, Any]], constraints: List[Constraint]
) -> List[Dict[str, Any]]:
    """Apply constraints to filter/modify combinations, then deduplicate."""
    result = list(combinations)

    for constraint in constraints:
        filtered = []
        for combo in result:
            if not _match_when(combo, constraint.when):
                filtered.append(combo)
                continue

            # Apply exclude: drop combos where target param has excluded values
            if constraint.exclude:
                excluded = False
                for param, exc_values in constraint.exclude.items():
                    if param in combo and _is_excluded(combo[param], exc_values):
                        excluded = True
                        break
                if excluded:
                    continue

            # Apply force: override param values
            if constraint.force:
                combo = dict(combo)
                for param, forced_val in constraint.force.items():
                    combo[param] = forced_val

            filtered.append(combo)

        result = filtered

    # Deduplicate (force constraints can create duplicates)
    seen = []
    deduped = []
    for combo in result:
        key = _combo_key(combo)
        if key not in seen:
            seen.append(key)
            deduped.append(combo)

    return deduped


def _combo_key(combo: Dict[str, Any]) -> str:
    """Create a hashable key for a combination for deduplication."""
    parts = []
    for k in sorted(combo.keys()):
        v = combo[k]
        if isinstance(v, float):
            parts.append(f"{k}={v:.10g}")
        else:
            parts.append(f"{k}={v}")
    return "|".join(parts)
