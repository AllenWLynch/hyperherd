"""Terminal output formatting for monitor and dry-run commands."""

from typing import Any, Dict, List, Optional


# ANSI color/style codes
_RESET = "\033[0m"
_BOLD = "\033[1m"
_DIM = "\033[2m"

# Status colors
_STATUS_COLORS = {
    "COMPLETED": "\033[32m",  # green
    "RUNNING": "\033[34m",    # blue
    "READY": "\033[90m",      # gray (never submitted)
    "QUEUED": "\033[33m",     # yellow (SLURM PENDING — waiting in queue)
    "SUBMITTED": "\033[33m",  # yellow
    "FAILED": "\033[31m",     # red
    "CANCELLED": "\033[90m",  # gray
    "TIMEOUT": "\033[31m",    # red
}

# Parameter display colors
_PARAM_NAME = "\033[36m"   # cyan for parameter names
_PARAM_VALUE = "\033[33m"  # yellow for values
_TRIAL_HEADER = "\033[1;37m"  # bold white for trial headers
_EXP_NAME = "\033[35m"    # magenta for experiment name
_BG_HIGHLIGHT = "\033[47m"  # light grey background for non-default values
_GREEN = "\033[32m"
_BOLD_GREEN = "\033[1;32m"
_CYAN = "\033[36m"


def _colorize_status(text: str, status: str) -> str:
    color = _STATUS_COLORS.get(status.upper(), "")
    reset = _RESET if color else ""
    return f"{color}{text}{reset}"


def _format_param_kv(name: str, value: Any, is_non_default: bool = False) -> str:
    """Format a single param as colored name=value.

    If is_non_default, adds a light grey background to make it stand out.
    """
    if isinstance(value, float):
        val_str = f"{value:.4g}"
    else:
        val_str = str(value)
    bg = _BG_HIGHLIGHT if is_non_default else ""
    return f"{bg}{_PARAM_NAME}{name}{_RESET}{bg}={_PARAM_VALUE}{val_str}{_RESET}"


def format_params_compact(params: Dict[str, Any], max_width: int = 50) -> str:
    """Format parameter dict into a compact string (no colors)."""
    parts = []
    for k, v in params.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4g}")
        else:
            parts.append(f"{k}={v}")
    result = " ".join(parts)
    if len(result) > max_width:
        result = result[: max_width - 3] + "..."
    return result


def format_params_colored(params: Dict[str, Any]) -> str:
    """Format parameter dict with colored names and values."""
    parts = [_format_param_kv(k, v) for k, v in params.items()]
    return "  ".join(parts)


def print_status_table(trials: List[dict], log_tails: Dict[int, str]) -> None:
    """Print a formatted status table of all trials."""
    if not trials:
        print("No trials found.")
        return

    idx_width = max(len(str(t["index"])) for t in trials)
    idx_width = max(idx_width, 5)

    param_strs = {t["index"]: format_params_compact(t["params"]) for t in trials}
    param_width = max(len(s) for s in param_strs.values())
    param_width = max(min(param_width, 50), 6)

    status_width = max(len(t.get("status", "")) for t in trials)
    status_width = max(status_width, 6)

    header = (
        f"{'Trial':>{idx_width}}  "
        f"{'Params':<{param_width}}  "
        f"{'Status':<{status_width}}  "
        f"Last Log"
    )
    print(header)
    print("-" * len(header.expandtabs()))

    for trial in trials:
        idx = trial["index"]
        status = trial.get("status", "unknown").upper()
        params = format_params_compact(trial["params"], max_width=param_width)
        log_tail = log_tails.get(idx, "")
        if len(log_tail) > 60:
            log_tail = log_tail[:57] + "..."

        status_str = _colorize_status(f"{status:<{status_width}}", status)

        print(
            f"{idx:>{idx_width}}  "
            f"{params:<{param_width}}  "
            f"{status_str}  "
            f"{log_tail}"
        )


def print_summary(trials: List[dict]) -> None:
    """Print a summary of trial statuses."""
    counts: Dict[str, int] = {}
    for t in trials:
        status = t.get("status", "unknown").upper()
        counts[status] = counts.get(status, 0) + 1

    total = len(trials)
    parts = []
    for status in ["COMPLETED", "RUNNING", "QUEUED", "SUBMITTED", "READY", "FAILED", "CANCELLED"]:
        if status in counts:
            parts.append(_colorize_status(f"{status}: {counts[status]}", status))

    print(f"\nTotal: {total}  |  {'  '.join(parts)}")


def _is_non_default(name: str, value: Any, defaults: Optional[Dict[str, Any]]) -> bool:
    """Check if a parameter value differs from its default."""
    if defaults is None:
        return False
    default = defaults.get(name)
    if default is None:
        return False
    if isinstance(value, float) and isinstance(default, (int, float)):
        import math
        return not math.isclose(value, float(default), rel_tol=1e-9)
    return value != default


def print_dry_run(
    trials: List[dict],
    sbatch_script: str,
    defaults: Optional[Dict[str, Any]] = None,
) -> None:
    """Print dry-run output with verbose, colored trial listing.

    Non-default parameter values are highlighted with a grey background.
    """
    print(f"{_BOLD}{'=' * 60}{_RESET}")
    print(f"{_BOLD}DRY RUN — No jobs will be submitted{_RESET}")
    print(f"{_BOLD}{'=' * 60}{_RESET}")
    print()

    # Sbatch script
    print(f"{_DIM}Generated sbatch script:{_RESET}")
    print(f"{_DIM}{'-' * 40}{_RESET}")
    for line in sbatch_script.rstrip().split("\n"):
        print(f"{_DIM}{line}{_RESET}")
    print(f"{_DIM}{'-' * 40}{_RESET}")
    print()

    # Trial listing
    print(f"{_BOLD}Trials: {len(trials)}{_RESET}")
    print()

    for trial in trials:
        idx = trial["index"]
        exp_name = trial.get("experiment_name", "")
        params = trial["params"]

        # Trial header line
        header = f"{_TRIAL_HEADER}[{idx}]{_RESET}"
        if exp_name:
            header += f"  {_EXP_NAME}{exp_name}{_RESET}"
        print(header)

        # Parameter lines, one per param, indented
        for name, value in params.items():
            highlight = _is_non_default(name, value, defaults)
            print(f"    {_format_param_kv(name, value, is_non_default=highlight)}")

        # Constraint-injected extras, if any
        extras = trial.get("extras") or {}
        if extras:
            print(f"    {_DIM}# constraint set:{_RESET}")
            for name, value in extras.items():
                print(f"    {_format_param_kv(name, value, is_non_default=True)}")
        print()


def print_launch_success(
    job_id: str,
    n_trials: int,
    workspace: str,
    logs: str,
) -> None:
    """Celebratory banner shown after a successful sbatch submission."""
    bar = f"{_BOLD_GREEN}{'━' * 60}{_RESET}"
    print()
    print(bar)
    print(
        f"{_BOLD_GREEN}  ✓ Launched {n_trials} trial"
        f"{'s' if n_trials != 1 else ''} as SLURM job array "
        f"{_CYAN}{job_id}{_RESET}"
    )
    print(bar)
    print(f"  {_DIM}workspace:{_RESET} {workspace}")
    print(f"  {_DIM}logs:     {_RESET} {logs}")
    print(f"  {_DIM}monitor:  {_RESET} {_GREEN}herd status{_RESET}")
    print()
