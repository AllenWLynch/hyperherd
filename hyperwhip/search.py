"""Generate hyperparameter combinations for grid and axes search modes."""

import itertools
import math
from typing import Any, Dict, List

from hyperwhip.config import Config, ParameterSpec


def _discretize_continuous(param: ParameterSpec) -> List[Any]:
    """Turn a continuous parameter into a list of discrete values."""
    low = param.low
    high = param.high
    steps = param.steps

    if steps < 2:
        return [low]

    if param.scale == "log":
        if low <= 0 or high <= 0:
            raise ValueError(
                f"Parameter '{param.name}': log scale requires positive low/high"
            )
        log_low = math.log10(low)
        log_high = math.log10(high)
        values = [10 ** (log_low + i * (log_high - log_low) / (steps - 1)) for i in range(steps)]
    else:
        values = [low + i * (high - low) / (steps - 1) for i in range(steps)]

    return values


def _get_param_values(param: ParameterSpec) -> List[Any]:
    if param.type == "discrete":
        return list(param.values)
    else:
        return _discretize_continuous(param)


def generate_grid(config: Config) -> List[Dict[str, Any]]:
    """Generate all combinations (Cartesian product) of parameter values."""
    param_names = [p.name for p in config.parameters]
    param_values = [_get_param_values(p) for p in config.parameters]

    combinations = []
    for combo in itertools.product(*param_values):
        combinations.append(dict(zip(param_names, combo)))

    return combinations


def generate_axes(config: Config) -> List[Dict[str, Any]]:
    """Generate one-at-a-time combinations: vary each param while others stay at default."""
    defaults = config.search.defaults
    if not defaults:
        raise ValueError("axes mode requires search.defaults to be set")

    combinations = []
    # Start with the default combination
    base = {p.name: defaults[p.name] for p in config.parameters}
    combinations.append(dict(base))

    # For each parameter, vary it through its values (skip the default)
    for param in config.parameters:
        values = _get_param_values(param)
        default_val = defaults[param.name]
        for val in values:
            if _values_equal(val, default_val):
                continue
            combo = dict(base)
            combo[param.name] = val
            combinations.append(combo)

    return combinations


def _values_equal(a: Any, b: Any) -> bool:
    """Compare values, handling float precision for continuous params."""
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-9)
    return a == b


def generate_combinations(config: Config) -> List[Dict[str, Any]]:
    if config.search.mode == "grid":
        return generate_grid(config)
    elif config.search.mode == "axes":
        return generate_axes(config)
    else:
        raise ValueError(f"Unknown search mode: {config.search.mode}")
