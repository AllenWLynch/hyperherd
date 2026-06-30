"""JSON output helpers for `--json` (agent) mode.

The CLI's default human-readable output bakes in ANSI color codes, fixed-width
table padding, and "1.50G"-style memory formatting — fine for a terminal,
expensive for an agent that has to parse it back into structured fields.

This module mirrors the read-style commands' outputs as plain dicts the
caller can hand to `json.dump`. Three rules across all payloads:

  - memory in raw **bytes** (numeric, not formatted)
  - elapsed time in **seconds** (numeric, not "01:30:00")
  - empty / unknown values as `null`, never an empty string masquerading as data

Errors are not JSON-wrapped: they keep going to stderr with a non-zero exit
code, on the assumption the agent reads exit codes anyway. Mixing
success-JSON and failure-JSON on stdout is more fragile than just keeping the
two channels separate.
"""

import json
import sys
from typing import Any, Dict, Iterable, List, Optional, Tuple


# --- output sink -------------------------------------------------------------

def emit(payload: Any) -> None:
    """Write `payload` to stdout as a single JSON document with a trailing
    newline. Indented for human-readable diffs; agents that care about size
    can re-encode."""
    json.dump(payload, sys.stdout, indent=2, default=str, sort_keys=False)
    sys.stdout.write("\n")


# --- numeric coercion --------------------------------------------------------

_MEM_UNIT_BYTES = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}


def parse_mem_bytes(value: str) -> Optional[int]:
    """Parse SLURM-style memory ('1500M', '1.5G', '512K') to integer bytes.

    Returns None on empty/unparseable so JSON consumers see `null` rather
    than 0 (which would be indistinguishable from a true zero reading).
    """
    if not value:
        return None
    s = value.strip()
    if not s:
        return None
    suffix = s[-1].upper()
    multiplier = _MEM_UNIT_BYTES.get(suffix)
    num_str = s[:-1] if multiplier is not None else s
    if multiplier is None:
        multiplier = 1
    try:
        return int(float(num_str) * multiplier)
    except ValueError:
        return None


def parse_elapsed_seconds(elapsed: str) -> Optional[int]:
    """Parse SLURM-style elapsed ('HH:MM:SS' or 'D-HH:MM:SS') to seconds."""
    if not elapsed:
        return None
    s = elapsed.strip()
    if not s:
        return None
    days = 0
    if "-" in s:
        d, _, s = s.partition("-")
        try:
            days = int(d)
        except ValueError:
            return None
    parts = s.split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    if len(nums) == 3:
        h, m, sec = nums
    elif len(nums) == 2:
        h, m, sec = 0, nums[0], nums[1]
    elif len(nums) == 1:
        h, m, sec = 0, 0, nums[0]
    else:
        return None
    return days * 86400 + h * 3600 + m * 60 + sec


# --- per-command payloads ----------------------------------------------------

def _trial_dict(trial: dict) -> Dict[str, Any]:
    return {
        "index": trial["index"],
        "status": trial.get("status", "unknown"),
        "experiment_name": trial.get("experiment_name") or None,
        "params": trial.get("params", {}),
    }


def status_payload(
    trials: List[dict],
    log_tails: Optional[Dict[int, str]] = None,
    progress: Optional[Dict[int, tuple]] = None,
) -> Dict[str, Any]:
    """`herd status --json`: totals + one entry per trial.

    The `last_log_line` field is what the human table shows in its rightmost
    column — convenient for an agent that wants to skim without `herd tail`.
    `progress` maps index → `(current_step, steps_per_min)`; when present, each
    trial gains `step` and `steps_per_min` fields (either may be null).
    """
    by_status: Dict[str, int] = {}
    for t in trials:
        s = t.get("status", "unknown")
        by_status[s] = by_status.get(s, 0) + 1

    out_trials = []
    for t in trials:
        d = _trial_dict(t)
        if log_tails is not None:
            tail = log_tails.get(t["index"], "")
            d["last_log_line"] = tail or None
        if progress is not None:
            step, spm = progress.get(t["index"], (None, None))
            d["step"] = step
            d["steps_per_min"] = round(spm, 2) if spm is not None else None
        out_trials.append(d)

    return {
        "totals": {"total": len(trials), **by_status},
        "trials": out_trials,
    }


def stats_payload(rows: Iterable[Tuple[int, dict, Any]]) -> Dict[str, Any]:
    """`herd stats --json`. `rows` is `(index, trial, JobStats)` — same shape
    `display.print_stats_table` consumes. Memory in bytes, elapsed in seconds,
    plus the original SLURM strings so callers that care about formatting
    don't have to re-derive them."""
    out_trials = []
    for idx, trial, stats in rows:
        out_trials.append({
            "index": idx,
            "experiment_name": (trial or {}).get("experiment_name") or None,
            "slurm_state": stats.state or None,
            "elapsed": stats.elapsed or None,
            "elapsed_seconds": parse_elapsed_seconds(stats.elapsed),
            "max_rss_bytes": parse_mem_bytes(stats.max_rss),
            "ave_rss_bytes": parse_mem_bytes(stats.ave_rss),
            "req_mem_bytes": parse_mem_bytes(stats.req_mem),
        })
    return {"trials": out_trials}


def launch_payload(
    *,
    dry_run: bool,
    submitted_indices: List[int],
    slurm_job_id: Optional[str],
    sbatch_path: Optional[str],
    trials: List[dict],
    sbatch_script: Optional[str] = None,
) -> Dict[str, Any]:
    """`herd run --json` and `herd run --dry-run --json`.

    Dry-run sets `slurm_job_id=None` and includes the full sbatch script
    contents under `sbatch_script`. A real run instead populates
    `slurm_job_id` and `sbatch_path` (the on-disk path the script was
    written to). `submitted_indices` is what the array would carry — empty
    on a no-op (e.g. nothing pending)."""
    return {
        "dry_run": dry_run,
        "slurm_job_id": slurm_job_id,
        "sbatch_path": sbatch_path,
        "submitted_indices": list(submitted_indices),
        "sbatch_script": sbatch_script,
        "trials": [_trial_dict(t) for t in trials],
    }


def stop_payload(cancelled: List[Dict[str, Any]]) -> Dict[str, Any]:
    """`herd stop --json`. `cancelled` is a list of
    `{index, slurm_job_id, previous_status}` dicts — empty on a no-op."""
    return {"cancelled": cancelled}


def tail_payload(
    *,
    index: int,
    status: str,
    experiment_name: str,
    streams: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """`herd tail --json`. `streams` maps `'stdout' | 'stderr'` to a dict
    `{path, lines, requested}`. A stream that doesn't exist on disk is
    represented as `{path, lines: null, requested}` — distinguishes "no
    file" from "file exists but is empty"."""
    return {
        "index": index,
        "status": status,
        "experiment_name": experiment_name or None,
        "streams": streams,
    }


def snapshot_payload(
    *,
    sweep_name: str,
    workspace_path: str,
    trials: List[dict],
    stats_by_idx: Dict[int, Any],
    metrics_by_idx: Dict[int, Dict[str, Any]],
    log_tails: Dict[int, str],
    failed_stderr: Dict[int, Dict[str, Any]],
    job_id_by_idx: Dict[int, Optional[str]],
) -> Dict[str, Any]:
    """`herd snapshot`: status + sacct + logged metrics + last-line tails +
    most-recent-failure stderr in one document. Sized so an agent loop can
    afford to read the whole thing every tick.

    `failed_stderr` is keyed by trial index; each value is `{path, lines,
    truncated}` (`lines` may be `None` if the file didn't exist or couldn't
    be read).
    """
    by_status: Dict[str, int] = {}
    for t in trials:
        s = t.get("status", "unknown")
        by_status[s] = by_status.get(s, 0) + 1

    out_trials = []
    for t in trials:
        idx = t["index"]
        stats = stats_by_idx.get(idx)
        out_trials.append({
            "index": idx,
            "status": t.get("status", "unknown"),
            "experiment_name": t.get("experiment_name") or None,
            "params": t.get("params", {}),
            "slurm_job_id": job_id_by_idx.get(idx),
            "slurm_state": (stats.state if stats else None) or None,
            "elapsed": (stats.elapsed if stats else "") or None,
            "elapsed_seconds": parse_elapsed_seconds(stats.elapsed) if stats else None,
            "max_rss_bytes": parse_mem_bytes(stats.max_rss) if stats else None,
            "ave_rss_bytes": parse_mem_bytes(stats.ave_rss) if stats else None,
            "req_mem_bytes": parse_mem_bytes(stats.req_mem) if stats else None,
            "metrics": dict(metrics_by_idx.get(idx, {})),
            "last_log_line": log_tails.get(idx) or None,
        })

    failed_blocks = []
    for idx in sorted(failed_stderr):
        info = failed_stderr[idx]
        failed_blocks.append({
            "index": idx,
            "stderr_path": info.get("path"),
            "stderr_lines": info.get("lines"),
            "stderr_truncated": bool(info.get("truncated")),
        })

    return {
        "sweep_name": sweep_name,
        "workspace": workspace_path,
        "totals": {"total": len(trials), **by_status},
        "trials": out_trials,
        "failed_stderr": failed_blocks,
    }


def results_payload(
    trials: List[dict],
    results: Dict[int, Dict[str, Any]],
    param_names: List[str],
) -> Dict[str, Any]:
    """`herd res --json`. Per-trial: declared params + logged metrics. Trials
    without any logged metrics still appear with an empty `metrics` dict so
    the agent can see them."""
    out = []
    for trial in trials:
        idx = trial["index"]
        params = trial.get("params", {})
        out.append({
            "index": idx,
            "experiment_name": trial.get("experiment_name") or None,
            "params": {p: params.get(p) for p in param_names if p in params},
            "metrics": dict(results.get(idx, {})),
        })
    return {"trials": out}
