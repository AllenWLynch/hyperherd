"""HyperHerd - SLURM hyperparameter optimization job array manager."""

__version__ = "0.1.0"

from hyperherd.logging import TrialPruned, log_result, parse_overrides

__all__ = ["log_result", "parse_overrides", "TrialPruned"]
