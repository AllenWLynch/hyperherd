"""Pre-flight validation checks run before launch and dry-run."""

import os
import subprocess
from typing import List

from hyperwhip.config import Config, ContinuousParameter, DiscreteParameter


class PreflightError(Exception):
    """Raised when a preflight check fails."""


class PreflightWarning:
    """A non-fatal preflight issue."""

    def __init__(self, message: str):
        self.message = message

    def __str__(self):
        return self.message


def run_preflight(config: Config, strict: bool = False) -> List[PreflightWarning]:
    """Run all preflight checks. Raises PreflightError on fatal issues.

    Returns a list of non-fatal warnings.
    """
    warnings = []

    _check_launcher(config)
    _check_workspace_writable(config)
    _check_defaults(config)

    warnings.extend(_check_partition(config))

    return warnings


def _check_launcher(config: Config) -> None:
    """Verify the launcher script exists and is executable."""
    if not config.launcher:
        raise PreflightError(
            "No launcher script specified. Set the 'launcher' field in your config."
        )
    if not os.path.isfile(config.launcher):
        raise PreflightError(
            f"Launcher script not found: {config.launcher}\n"
            f"  Create it or fix the 'launcher' path in your config."
        )
    if not os.access(config.launcher, os.X_OK):
        raise PreflightError(
            f"Launcher script is not executable: {config.launcher}\n"
            f"  Run: chmod +x {config.launcher}"
        )


def _check_workspace_writable(config: Config) -> None:
    """Verify we can write to the workspace parent directory."""
    ws = config.workspace
    if os.path.isdir(ws):
        if not os.access(ws, os.W_OK):
            raise PreflightError(
                f"Workspace directory is not writable: {ws}"
            )
        return

    parent = os.path.dirname(ws)
    if not parent:
        parent = "."
    if not os.path.isdir(parent):
        raise PreflightError(
            f"Workspace parent directory does not exist: {parent}\n"
            f"  Create it or change the 'workspace' path in your config."
        )
    if not os.access(parent, os.W_OK):
        raise PreflightError(
            f"Workspace parent directory is not writable: {parent}"
        )


def _check_defaults(config: Config) -> None:
    """Verify default values are valid for their parameters (when defaults are used)."""
    if not config.defaults:
        return

    for name, spec in config.parameters.items():
        default = config.defaults.get(name)
        if default is None:
            continue
        if isinstance(spec, DiscreteParameter) and default not in spec.values:
            raise PreflightError(
                f"Default value '{default}' for parameter '{name}' "
                f"is not in its values list: {spec.values}"
            )
        if isinstance(spec, ContinuousParameter):
            try:
                val = float(default)
            except (TypeError, ValueError):
                raise PreflightError(
                    f"Default value '{default}' for continuous parameter "
                    f"'{name}' is not a number."
                )
            if val < spec.low or val > spec.high:
                raise PreflightError(
                    f"Default value {val} for parameter '{name}' "
                    f"is outside range [{spec.low}, {spec.high}]."
                )


def _check_partition(config: Config) -> List[PreflightWarning]:
    """Check if the SLURM partition exists. Returns warnings (non-fatal)."""
    warnings = []
    try:
        result = subprocess.run(
            ["sinfo", "-h", "-p", config.slurm.partition, "--format=%P"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0 or not result.stdout.strip():
            warnings.append(PreflightWarning(
                f"SLURM partition '{config.slurm.partition}' not found or sinfo failed. "
                f"Verify the partition name is correct."
            ))
    except FileNotFoundError:
        warnings.append(PreflightWarning(
            "sinfo not found. Cannot verify SLURM partition. "
            "This is expected if you're not on a SLURM login node."
        ))
    except subprocess.TimeoutExpired:
        warnings.append(PreflightWarning(
            "sinfo timed out. Cannot verify SLURM partition."
        ))

    return warnings
