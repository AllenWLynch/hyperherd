"""Terminal output formatting for monitor and dry-run commands."""

import re
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
    "PRUNED": "\033[35m",     # magenta — algorithmic kill, distinct from cancelled
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


def format_short_value(value: Any) -> str:
    """Format a value for short human-facing display (`.4g` for floats, str otherwise).

    Used in compact param strings, experiment-name tokens, and dry-run output —
    anywhere the rendered value is for reading, not round-tripping. For
    Hydra-override emission and dedup keys, see `manifest._format_override_value`
    and `constraints._combo_key`, which keep more precision.
    """
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _colorize_status(text: str, status: str) -> str:
    color = _STATUS_COLORS.get(status.upper(), "")
    reset = _RESET if color else ""
    return f"{color}{text}{reset}"


def _format_param_kv(name: str, value: Any, is_non_default: bool = False) -> str:
    """Format a single param as colored name=value.

    If is_non_default, adds a light grey background to make it stand out.
    """
    val_str = format_short_value(value)
    bg = _BG_HIGHLIGHT if is_non_default else ""
    return f"{bg}{_PARAM_NAME}{name}{_RESET}{bg}={_PARAM_VALUE}{val_str}{_RESET}"


def format_params_compact(params: Dict[str, Any], max_width: int = 50) -> str:
    """Format parameter dict into a compact string (no colors)."""
    parts = [f"{k}={format_short_value(v)}" for k, v in params.items()]
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


_CASE_OPEN_RE = re.compile(r'^\s*case\s+"\$SLURM_ARRAY_TASK_ID"\s+in\s*$')
_CASE_NUMERIC_ARM_RE = re.compile(r"^\s*\d+\)\s*$")
_CASE_WILDCARD_ARM_RE = re.compile(r"^\s*\*\)\s*$")


def _condense_case_block(script: str, keep_first: int = 1) -> str:
    """Collapse the sbatch script's per-trial case body for display.

    Show the first `keep_first` numeric arms + an elision marker + the
    wildcard arm. Real script content is never printed by SLURM, so this
    only affects what `herd run --dry-run` shows the user — submission
    still uses the full script.
    """
    lines = script.splitlines()
    open_idx = next(
        (i for i, l in enumerate(lines) if _CASE_OPEN_RE.match(l)), None
    )
    if open_idx is None:
        return script
    close_idx = next(
        (i for i in range(open_idx + 1, len(lines)) if lines[i].strip() == "esac"),
        None,
    )
    if close_idx is None:
        return script

    arms: List[List[str]] = []
    current: List[str] = []
    for line in lines[open_idx + 1 : close_idx]:
        if _CASE_NUMERIC_ARM_RE.match(line) or _CASE_WILDCARD_ARM_RE.match(line):
            if current:
                arms.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        arms.append(current)

    numeric = [a for a in arms if a and _CASE_NUMERIC_ARM_RE.match(a[0])]
    wildcard = [a for a in arms if a and _CASE_WILDCARD_ARM_RE.match(a[0])]
    if len(numeric) <= keep_first + 1:
        return script  # not worth condensing — show as-is

    elided = len(numeric) - keep_first
    new_body: List[str] = []
    for arm in numeric[:keep_first]:
        new_body.extend(arm)
    new_body.append(
        f"  # ... [{elided} more trial arm(s) elided in dry-run; full script "
        f"is submitted] ..."
    )
    for arm in wildcard:
        new_body.extend(arm)

    return "\n".join(lines[: open_idx + 1] + new_body + lines[close_idx:])


def print_dry_run(
    trials: List[dict],
    sbatch_script: str,
    defaults: Optional[Dict[str, Any]] = None,
) -> None:
    """Print dry-run output with verbose, colored trial listing.

    Non-default parameter values are highlighted with a grey background.
    The baked per-trial `case` block is condensed for readability — the
    full block is what actually gets submitted, and the per-trial details
    are shown explicitly below in the Trials section.
    """
    print(f"{_BOLD}{'=' * 60}{_RESET}")
    print(f"{_BOLD}DRY RUN — No jobs will be submitted{_RESET}")
    print(f"{_BOLD}{'=' * 60}{_RESET}")
    print()

    # Sbatch script
    print(f"{_DIM}Generated sbatch script (per-trial lookup elided for brevity):{_RESET}")
    print(f"{_DIM}{'-' * 40}{_RESET}")
    for line in _condense_case_block(sbatch_script).rstrip().split("\n"):
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


_MEM_UNIT_BYTES = {
    "K": 1024,
    "M": 1024 ** 2,
    "G": 1024 ** 3,
    "T": 1024 ** 4,
}


def _format_mem_gb(value: str) -> str:
    """Convert a sacct memory string (e.g. '382648K', '4G', '1024') to 'X.XXG'.

    sacct emits values with a unit suffix (K/M/G/T, base 1024) — or no suffix
    (raw bytes) for some fields. Returns '-' for empty/unparseable input.
    """
    if not value:
        return "-"
    s = value.strip()
    if not s:
        return "-"
    suffix = s[-1].upper()
    multiplier = _MEM_UNIT_BYTES.get(suffix)
    num_str = s[:-1] if multiplier is not None else s
    if multiplier is None:
        multiplier = 1  # raw bytes
    try:
        bytes_val = float(num_str) * multiplier
    except ValueError:
        return value  # fall back to raw string if we can't parse
    return f"{bytes_val / _MEM_UNIT_BYTES['G']:.2f}G"


def print_stats_table(rows) -> None:
    """Print a per-trial table of runtime + memory accounting from sacct.

    `rows` is an iterable of (index, trial_dict, JobStats).
    """
    rows = list(rows)
    if not rows:
        return

    headers = ["idx", "state", "elapsed", "max_rss", "ave_rss", "req_mem", "name"]
    table = [
        [
            str(idx),
            st.state or "-",
            st.elapsed or "-",
            _format_mem_gb(st.max_rss),
            _format_mem_gb(st.ave_rss),
            _format_mem_gb(st.req_mem),
            trial.get("experiment_name", ""),
        ]
        for idx, trial, st in rows
    ]
    widths = [max(len(h), max(len(r[i]) for r in table)) for i, h in enumerate(headers)]

    def _join(cells):
        return "  ".join(c.ljust(w) for c, w in zip(cells, widths))

    print(f"{_BOLD}{_join(headers)}{_RESET}")
    print(f"{_DIM}{'  '.join('-' * w for w in widths)}{_RESET}")
    for cells in table:
        line = _join(cells)
        color = _STATUS_COLORS.get(cells[1].upper(), "")
        if color:
            # Recolor the state cell in-place; widths already padded above.
            line = line.replace(
                cells[1].ljust(widths[1]),
                f"{color}{cells[1].ljust(widths[1])}{_RESET}",
                1,
            )
        print(line)


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
