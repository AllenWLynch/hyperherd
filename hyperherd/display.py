"""Terminal output formatting for monitor and dry-run commands."""

import re
from typing import Any, Dict, List, Optional


# ANSI color/style codes
_RESET = "\033[0m"
_BOLD = "\033[1m"

# Matches any ANSI CSI escape (color/style/cursor). Used to strip foreign codes
# out of borrowed text (e.g. a wandb-colored log tail) so they don't bleed into
# the rest of the table — a truncated tail can otherwise cut off mid-sequence,
# leaving an open color that paints everything until the next reset.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
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
    "PAUSED": "\033[36m",     # cyan — intentionally paused by SH, resumable (calm, not alarming)
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


# Per-status display priority (smaller = more important). The Discord
# dashboard sorts by the same key, so a truncated view drops the boring tail
# (completed/ready) instead of the trials the user cares about. Single source
# of truth — `monitor_agent.commands` re-exports these.
STATUS_PRIORITY = {
    "running":   0,
    "queued":    1,
    "submitted": 2,
    "paused":    3,
    "failed":    4,
    "pruned":    5,
    "cancelled": 6,
    "completed": 7,
    "ready":     8,
}

# Statuses hidden by the status table's brief view (`herd status --brief`):
# never-ran and user-cancelled trials are noise when triaging a live sweep.
_BRIEF_HIDDEN_STATUSES = frozenset({"ready", "cancelled"})


def trial_sort_key(trial: dict) -> tuple:
    """Sort by status priority then index — active trials first, then problems,
    then settled, then never-ran. Index breaks ties so the order is
    deterministic within each status bucket."""
    status = (trial.get("status") or "").lower()
    return (STATUS_PRIORITY.get(status, 99), trial.get("index", 0))


def _trial_name(trial: dict) -> str:
    """A trial's compact experiment name, falling back to a compact param
    string for legacy manifests written before `experiment_name` existed."""
    return trial.get("experiment_name") or format_params_compact(
        trial.get("params", {})
    )


def _fmt_step(step) -> str:
    """Render a current-step value for the status table (blank if unknown)."""
    return "" if step is None else f"{int(step):,}"


def _fmt_rate(spm) -> str:
    """Render a steps/min rate compactly (blank if unknown)."""
    if spm is None:
        return ""
    if spm >= 100:
        return f"{spm:,.0f}/m"
    if spm >= 10:
        return f"{spm:.0f}/m"
    if spm >= 1:
        return f"{spm:.1f}/m"
    return f"{spm:.2f}/m"


def print_status_table(
    trials: List[dict],
    log_tails: Dict[int, str],
    *,
    progress: Optional[Dict[int, tuple]] = None,
    brief: bool = False,
) -> None:
    """Print a formatted status table of all trials.

    Shows each trial's compact experiment **name** (the swept-param values get
    truncated unhelpfully). With ``brief=True``, sort by status (active first,
    like the dashboard) and hide READY/CANCELLED trials so the view focuses on
    what's in flight.

    ``progress`` maps trial index → ``(current_step, steps_per_min)``. When any
    trial has a step value, a ``Step`` (and, if any rate is known, ``Steps/min``)
    column is inserted; otherwise the layout is unchanged.
    """
    if brief:
        trials = sorted(
            (t for t in trials
             if (t.get("status") or "").lower() not in _BRIEF_HIDDEN_STATUSES),
            key=trial_sort_key,
        )
    if not trials:
        print("No trials found.")
        return

    progress = progress or {}
    step_strs = {t["index"]: _fmt_step(progress.get(t["index"], (None, None))[0])
                 for t in trials}
    rate_strs = {t["index"]: _fmt_rate(progress.get(t["index"], (None, None))[1])
                 for t in trials}
    show_step = any(step_strs.values())
    show_rate = any(rate_strs.values())

    idx_width = max(len(str(t["index"])) for t in trials)
    idx_width = max(idx_width, 5)

    name_strs = {t["index"]: _trial_name(t) for t in trials}
    name_width = max(len(s) for s in name_strs.values())
    name_width = max(min(name_width, 60), 6)

    status_width = max(len(t.get("status", "")) for t in trials)
    status_width = max(status_width, 6)

    step_width = max([len("Step")] + [len(s) for s in step_strs.values()]) if show_step else 0
    rate_width = max([len("Steps/min")] + [len(s) for s in rate_strs.values()]) if show_rate else 0

    header = (
        f"{'Trial':>{idx_width}}  "
        f"{'Name':<{name_width}}  "
        f"{'Status':<{status_width}}  "
    )
    if show_step:
        header += f"{'Step':>{step_width}}  "
    if show_rate:
        header += f"{'Steps/min':>{rate_width}}  "
    header += "Last Log"
    print(header)
    print("-" * len(header.expandtabs()))

    for trial in trials:
        idx = trial["index"]
        status = trial.get("status", "unknown").upper()
        name = name_strs[idx]
        if len(name) > name_width:
            name = name[: name_width - 1] + "…"
        # Strip ANSI from the borrowed log tail so its colors don't bleed into
        # the table, and so truncation counts visible chars (not escape bytes).
        log_tail = _ANSI_RE.sub("", log_tails.get(idx, ""))
        if len(log_tail) > 60:
            log_tail = log_tail[:57] + "..."

        status_str = _colorize_status(f"{status:<{status_width}}", status)

        row = (
            f"{idx:>{idx_width}}  "
            f"{name:<{name_width}}  "
            f"{status_str}  "
        )
        if show_step:
            row += f"{step_strs[idx]:>{step_width}}  "
        if show_rate:
            row += f"{rate_strs[idx]:>{rate_width}}  "
        row += log_tail
        print(row)


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


def print_trial_listing(
    trials: List[dict],
    defaults: Optional[Dict[str, Any]] = None,
    show_status: bool = False,
    title: Optional[str] = None,
) -> None:
    """Print a verbose per-trial parameter dump.

    One stanza per trial: header (`[idx] experiment_name`) plus the
    swept params (one per line, non-defaults highlighted) plus any
    constraint-injected `extras`. With `show_status=True` the header
    also carries the trial's status emoji + label.

    This is the canonical "what does the sweep look like" view —
    `herd ls` calls it for the full sweep; the run dry-run no longer
    uses it (the dry-run is now a narrow submission preview).
    """
    if title:
        print(f"{_BOLD}{title}{_RESET}")
    print(f"{_BOLD}Trials: {len(trials)}{_RESET}")
    print()

    for trial in trials:
        idx = trial["index"]
        exp_name = trial.get("experiment_name", "")
        params = trial.get("params", {})

        header = f"{_TRIAL_HEADER}[{idx}]{_RESET}"
        if exp_name:
            header += f"  {_EXP_NAME}{exp_name}{_RESET}"
        if show_status:
            status = trial.get("status", "?")
            color = _STATUS_COLORS.get(status.upper(), "")
            if color:
                header += f"  {color}{status}{_RESET}"
            else:
                header += f"  {status}"
        print(header)

        for name, value in params.items():
            highlight = _is_non_default(name, value, defaults)
            print(f"    {_format_param_kv(name, value, is_non_default=highlight)}")

        extras = trial.get("extras") or {}
        if extras:
            print(f"    {_DIM}# constraint set:{_RESET}")
            for name, value in extras.items():
                print(f"    {_format_param_kv(name, value, is_non_default=True)}")
        print()


def print_dry_run(
    sbatch_script: str,
    pending_indices: List[int],
    total_trials: int,
    filter_summary: Optional[str] = None,
) -> None:
    """Print the submission preview: sbatch script + pending summary.

    Narrow scope by design — what `herd run` *would* submit right now,
    given current status, --indices, --pin, and config-reconciliation
    state. The full per-trial parameter table belongs to `herd ls`,
    which is status-agnostic and meant for sweep-shape inspection.

    `filter_summary` is an optional one-liner describing how the
    pending set was narrowed (e.g. "--pin lr=0.001 narrowed from 6
    to 1 trial"); printed if provided.
    """
    print(f"{_BOLD}{'=' * 60}{_RESET}")
    print(f"{_BOLD}DRY RUN — No jobs will be submitted{_RESET}")
    print(f"{_BOLD}{'=' * 60}{_RESET}")
    print()

    print(f"{_DIM}Generated sbatch script (per-trial lookup elided for brevity):{_RESET}")
    print(f"{_DIM}{'-' * 40}{_RESET}")
    for line in _condense_case_block(sbatch_script).rstrip().split("\n"):
        print(f"{_DIM}{line}{_RESET}")
    print(f"{_DIM}{'-' * 40}{_RESET}")
    print()

    print(f"{_BOLD}Submission plan{_RESET}")
    print(f"  Pending: {len(pending_indices)} of {total_trials} trial(s)")
    if pending_indices:
        # Render as a compact range spec when possible.
        try:
            from hyperherd.slurm import _indices_to_array_spec
            spec = _indices_to_array_spec(pending_indices)
            print(f"  Indices: {spec}")
        except Exception:
            print(f"  Indices: {pending_indices}")
    if filter_summary:
        print(f"  Filter:  {filter_summary}")
    print(f"  Use {_BOLD}herd ls{_RESET} to see every trial in the sweep.")
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
