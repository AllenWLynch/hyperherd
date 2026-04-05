"""Terminal output formatting for the monitor command."""

from typing import Any, Dict, List


# ANSI color codes
_COLORS = {
    "COMPLETED": "\033[32m",  # green
    "RUNNING": "\033[34m",    # blue
    "PENDING": "\033[33m",    # yellow
    "SUBMITTED": "\033[33m",  # yellow
    "FAILED": "\033[31m",     # red
    "CANCELLED": "\033[90m",  # gray
    "TIMEOUT": "\033[31m",    # red
    "RESET": "\033[0m",
}


def _colorize(text: str, status: str) -> str:
    color = _COLORS.get(status.upper(), "")
    reset = _COLORS["RESET"] if color else ""
    return f"{color}{text}{reset}"


def format_params(params: Dict[str, Any], max_width: int = 50) -> str:
    """Format parameter dict into a compact string."""
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


def print_status_table(trials: List[dict], log_tails: Dict[int, str]) -> None:
    """Print a formatted status table of all trials."""
    if not trials:
        print("No trials found.")
        return

    # Compute column widths
    idx_width = max(len(str(t["index"])) for t in trials)
    idx_width = max(idx_width, 5)  # "Trial"

    param_strs = {t["index"]: format_params(t["params"]) for t in trials}
    param_width = max(len(s) for s in param_strs.values())
    param_width = max(min(param_width, 50), 6)  # "Params"

    status_width = max(len(t.get("status", "")) for t in trials)
    status_width = max(status_width, 6)  # "Status"

    # Header
    header = (
        f"{'Trial':>{idx_width}}  "
        f"{'Params':<{param_width}}  "
        f"{'Status':<{status_width}}  "
        f"Last Log"
    )
    print(header)
    print("-" * len(header.expandtabs()))

    # Rows
    for trial in trials:
        idx = trial["index"]
        status = trial.get("status", "unknown").upper()
        params = format_params(trial["params"], max_width=param_width)
        log_tail = log_tails.get(idx, "")
        # Truncate log tail for display
        if len(log_tail) > 60:
            log_tail = log_tail[:57] + "..."

        status_str = _colorize(f"{status:<{status_width}}", status)

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
    for status in ["COMPLETED", "RUNNING", "PENDING", "SUBMITTED", "FAILED", "CANCELLED"]:
        if status in counts:
            parts.append(_colorize(f"{status}: {counts[status]}", status))

    print(f"\nTotal: {total}  |  {'  '.join(parts)}")


def print_dry_run(trials: List[dict], sbatch_script: str) -> None:
    """Print dry-run output: the generated sbatch script and trial listing."""
    print("=" * 60)
    print("DRY RUN - No jobs will be submitted")
    print("=" * 60)
    print()
    print("Generated sbatch script:")
    print("-" * 40)
    print(sbatch_script)
    print("-" * 40)
    print()
    print(f"Total trials: {len(trials)}")
    print()
    for trial in trials:
        idx = trial["index"]
        params = format_params(trial["params"])
        print(f"  [{idx:>4}] {params}")
