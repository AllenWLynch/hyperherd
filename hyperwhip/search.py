"""Generate hyperparameter combinations based on grid configuration."""

import itertools
import math
from typing import Any, Dict, List

from hyperwhip.config import Config, ContinuousParameter, DiscreteParameter, ParameterSpec


def _discretize_continuous(param: ContinuousParameter) -> List[Any]:
    """Turn a continuous parameter into a list of discrete values."""
    low = param.low
    high = param.high
    steps = param.steps

    if steps < 2:
        return [low]

    if param.scale == "log":
        log_low = math.log10(low)
        log_high = math.log10(high)
        values = [10 ** (log_low + i * (log_high - log_low) / (steps - 1)) for i in range(steps)]
    else:
        values = [low + i * (high - low) / (steps - 1) for i in range(steps)]

    return values


def _get_param_values(param: ParameterSpec) -> List[Any]:
    if isinstance(param, DiscreteParameter):
        return list(param.values)
    else:
        return _discretize_continuous(param)


def _values_equal(a: Any, b: Any) -> bool:
    """Compare values, handling float precision for continuous params."""
    if isinstance(a, float) and isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-9)
    return a == b


def generate_combinations(config: Config) -> List[Dict[str, Any]]:
    """Generate parameter combinations based on the grid config.

    - grid: "all" -> Cartesian product of all parameters
    - grid: [subset] -> Cartesian product of listed params, defaults for the rest
    - grid: None -> one-at-a-time from defaults
    """
    if config.grid == "all":
        return _generate_full_grid(config)
    elif isinstance(config.grid, list):
        return _generate_partial_grid(config, config.grid)
    else:
        return _generate_one_at_a_time(config)


def _generate_full_grid(config: Config) -> List[Dict[str, Any]]:
    """Cartesian product of all parameter values."""
    names = config.param_names
    values = [_get_param_values(config.parameters[n]) for n in names]

    combinations = []
    for combo in itertools.product(*values):
        combinations.append(dict(zip(names, combo)))

    return combinations


def _generate_partial_grid(config: Config, grid_params: List[str]) -> List[Dict[str, Any]]:
    """Grid over listed params, hold others at defaults."""
    defaults = config.defaults

    grid_names = grid_params
    grid_values = [_get_param_values(config.parameters[n]) for n in grid_names]

    combinations = []
    for combo in itertools.product(*grid_values):
        row = dict(defaults)  # start from defaults
        row.update(dict(zip(grid_names, combo)))
        combinations.append(row)

    return combinations


def _generate_one_at_a_time(config: Config) -> List[Dict[str, Any]]:
    """Vary each parameter independently while others stay at default."""
    defaults = config.defaults
    base = dict(defaults)

    combinations = [dict(base)]  # default combination first

    for name in config.param_names:
        values = _get_param_values(config.parameters[name])
        default_val = defaults[name]
        for val in values:
            if _values_equal(val, default_val):
                continue
            combo = dict(base)
            combo[name] = val
            combinations.append(combo)

    return combinations
