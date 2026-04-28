"""Manage the .hyperwhip/ workspace: trial manifest, status, and job tracking."""

import json
import os
from typing import Any, Dict, List, Optional, Union

from hyperwhip.constraints import Trial


WORKSPACE_DIR = ".hyperwhip"
MANIFEST_FILE = "manifest.json"
JOB_IDS_FILE = "job_ids.json"
SBATCH_FILE = "job.sbatch"
LOGS_DIR = "logs"



def workspace_path(base: str) -> str:
    return os.path.join(base, WORKSPACE_DIR)


def manifest_path(base: str) -> str:
    return os.path.join(workspace_path(base), MANIFEST_FILE)


def job_ids_path(base: str) -> str:
    return os.path.join(workspace_path(base), JOB_IDS_FILE)


def sbatch_path(base: str) -> str:
    return os.path.join(workspace_path(base), SBATCH_FILE)


def logs_path(base: str) -> str:
    return os.path.join(workspace_path(base), LOGS_DIR)


def init_workspace(base: str) -> None:
    """Create the .hyperwhip/ workspace directory structure."""
    ws = workspace_path(base)
    os.makedirs(ws, exist_ok=True)
    os.makedirs(logs_path(base), exist_ok=True)


def workspace_exists(base: str) -> bool:
    return os.path.isdir(workspace_path(base)) and os.path.isfile(manifest_path(base))


# --- Manifest operations ---

def build_experiment_name(
    params: Dict[str, Any], abbrevs: Dict[str, str]
) -> str:
    """Construct a deterministic experiment name from parameter abbreviations and values.

    Uses '-' as key-value separator (not '=') to avoid conflicts with Hydra's
    override syntax when experiment_name is passed as an override.

    Example: with abbrevs={"learning_rate": "lr", "optimizer": "opt"} and
    params={"learning_rate": 0.001, "optimizer": "adam"},
    returns "lr-0.001_opt-adam".
    """
    parts = []
    for param_name, value in params.items():
        abbr = abbrevs.get(param_name, param_name)
        if isinstance(value, float):
            parts.append(f"{abbr}-{value:.4g}")
        else:
            parts.append(f"{abbr}-{value}")
    return "_".join(parts)


def create_manifest(
    base: str,
    trials: List[Union[Trial, Dict[str, Any]]],
    abbrevs: Optional[Dict[str, str]] = None,
) -> List[dict]:
    """Create a new manifest from a list of Trials (or bare param dicts).

    Bare dicts are accepted for back-compat and treated as trials with no
    extras.
    """
    if abbrevs is None:
        abbrevs = {}
    records = []
    for i, item in enumerate(trials):
        if isinstance(item, Trial):
            params = item.params
            extras = item.extras
        else:
            params = item
            extras = {}
        records.append({
            "index": i,
            "params": params,
            "extras": extras,
            "experiment_name": build_experiment_name(params, abbrevs),
            "status": "pending",
        })
    _write_manifest(base, records)
    return records


def load_manifest(base: str) -> List[dict]:
    path = manifest_path(base)
    if not os.path.isfile(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def _write_manifest(base: str, trials: List[dict]) -> None:
    with open(manifest_path(base), "w") as f:
        json.dump(trials, f, indent=2, default=_json_default)


def _json_default(obj):
    if isinstance(obj, float):
        return obj
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def update_trial_status(base: str, index: int, status: str) -> None:
    trials = load_manifest(base)
    for trial in trials:
        if trial["index"] == index:
            trial["status"] = status
            break
    _write_manifest(base, trials)


def bulk_update_status(base: str, updates: Dict[int, str]) -> None:
    """Update status for multiple trials at once."""
    if not updates:
        return
    trials = load_manifest(base)
    for trial in trials:
        idx = trial["index"]
        if idx in updates:
            trial["status"] = updates[idx]
    _write_manifest(base, trials)


def get_trials_by_status(base: str, status: str) -> List[dict]:
    return [t for t in load_manifest(base) if t["status"] == status]


def get_pending_indices(base: str) -> List[int]:
    """Get indices that need (re)submission: pending, failed, or cancelled."""
    trials = load_manifest(base)
    return [t["index"] for t in trials if t["status"] in ("pending", "failed", "cancelled")]


# --- Job ID tracking ---

def record_job_submission(base: str, slurm_job_id: str, indices: List[int]) -> None:
    """Record a SLURM job array submission."""
    records = _load_job_ids(base)
    records.append({
        "slurm_job_id": slurm_job_id,
        "indices": indices,
    })
    _write_job_ids(base, records)


def get_job_ids(base: str) -> List[dict]:
    return _load_job_ids(base)


def _load_job_ids(base: str) -> List[dict]:
    path = job_ids_path(base)
    if not os.path.isfile(path):
        return []
    with open(path, "r") as f:
        return json.load(f)


def _write_job_ids(base: str, records: List[dict]) -> None:
    with open(job_ids_path(base), "w") as f:
        json.dump(records, f, indent=2)


# --- Hydra override resolution ---

def _format_override_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.10g}"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def resolve_overrides(
    base: str, task_id: int, static_overrides: Optional[List[str]] = None
) -> str:
    """Build the Hydra override string for a given array task ID.

    Order (Hydra applies left-to-right, last wins):
      1. experiment_name=<name>
      2. swept parameter overrides
      3. hydra.static_overrides
      4. constraint `set` extras (last → wins over statics)
    """
    trials = load_manifest(base)
    trial = None
    for t in trials:
        if t["index"] == task_id:
            trial = t
            break

    if trial is None:
        raise ValueError(f"No trial found for task ID {task_id}")

    parts = []

    exp_name = trial.get("experiment_name", "")
    if exp_name:
        parts.append(f"experiment_name={exp_name}")

    for param, value in trial["params"].items():
        parts.append(f"{param}={_format_override_value(value)}")

    if static_overrides:
        parts.extend(static_overrides)

    # Extras are emitted last so constraint `set` values override statics.
    extras = trial.get("extras") or {}
    for k, v in extras.items():
        parts.append(f"{k}={_format_override_value(v)}")

    return " ".join(parts)
