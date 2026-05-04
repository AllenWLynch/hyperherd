"""Manage the .hyperherd/ workspace: trial manifest, status, and job tracking."""

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from hyperherd.constraints import Trial
from hyperherd.display import format_short_value


WORKSPACE_DIR = ".hyperherd"
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
    """Create the .hyperherd/ workspace directory structure."""
    ws = workspace_path(base)
    os.makedirs(ws, exist_ok=True)
    os.makedirs(logs_path(base), exist_ok=True)


def workspace_exists(base: str) -> bool:
    return os.path.isdir(workspace_path(base)) and os.path.isfile(manifest_path(base))


# --- Manifest operations ---

def build_experiment_name(
    params: Dict[str, Any],
    abbrevs: Dict[str, str],
    labels: Optional[Dict[str, Dict[Any, str]]] = None,
) -> str:
    """Construct a deterministic experiment name from parameter abbreviations and values.

    Uses '-' as key-value separator (not '=') to avoid conflicts with Hydra's
    override syntax when experiment_name is passed as an override.

    `labels`, if provided, maps parameter_name -> {value: display_label}. When
    a value has a registered label, the label is used in place of the raw
    value (useful for paths or other long discrete values).

    Slashes in raw values would produce unsafe experiment names; the config
    validator rejects discrete values containing '/' unless an explicit
    `labels:` list is provided (and labels themselves may not contain '/').

    Example: with abbrevs={"learning_rate": "lr", "optimizer": "opt"} and
    params={"learning_rate": 0.001, "optimizer": "adam"},
    returns "lr-0.001_opt-adam".
    """
    parts = []
    for param_name, value in params.items():
        abbr = abbrevs.get(param_name, param_name)
        label = None
        if labels is not None:
            label = labels.get(param_name, {}).get(value)
        token = label if label is not None else format_short_value(value)
        parts.append(f"{abbr}-{token}")
    return "_".join(parts)


def trial_hash(params: Dict[str, Any], extras: Optional[Dict[str, Any]] = None) -> str:
    """Stable identity hash for a trial, derived from its swept params + constraint extras.

    Two trials with the same hash are considered the same trial across
    config edits — this is how reconciliation distinguishes
    "kept" trials from "added"/"removed" ones.
    """
    blob = json.dumps(
        {"params": params, "extras": extras or {}},
        sort_keys=True,
        default=_json_default,
    )
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _trial_record(
    index: int,
    params: Dict[str, Any],
    extras: Dict[str, Any],
    abbrevs: Dict[str, str],
    labels: Optional[Dict[str, Dict[Any, str]]],
) -> dict:
    return {
        "index": index,
        "hash": trial_hash(params, extras),
        "params": params,
        "extras": extras,
        "experiment_name": build_experiment_name(params, abbrevs, labels),
        "status": "ready",
    }


def create_manifest(
    base: str,
    trials: List[Union[Trial, Dict[str, Any]]],
    abbrevs: Optional[Dict[str, str]] = None,
    labels: Optional[Dict[str, Dict[Any, str]]] = None,
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
        records.append(_trial_record(i, params, extras, abbrevs, labels))
    _write_manifest(base, records)
    return records


def load_manifest(base: str) -> List[dict]:
    path = manifest_path(base)
    if not os.path.isfile(path):
        return []
    with open(path, "r") as f:
        trials = json.load(f)
    # Backfill `hash` on legacy manifests written before reconciliation existed.
    # Avoids a forced migration; the hash is fully derivable from params+extras.
    for t in trials:
        if "hash" not in t:
            t["hash"] = trial_hash(t.get("params", {}), t.get("extras") or {})
    return trials


@dataclass
class Reconciled:
    """Diff between the on-disk manifest and a freshly-generated combo list."""
    kept: List[dict] = field(default_factory=list)        # trials present in both
    added: List[Trial] = field(default_factory=list)      # in combos, not in manifest
    removed: List[dict] = field(default_factory=list)     # in manifest, not in combos

    @property
    def is_clean(self) -> bool:
        return not self.added and not self.removed


def reconcile_manifest(
    existing: List[dict],
    combos: List[Trial],
) -> Reconciled:
    """Diff an on-disk manifest against a freshly-generated combo list by trial hash.

    Identity is `trial_hash(params, extras)` — independent of index, abbrevs,
    or experiment_name. Kept trials retain their existing record (frozen
    experiment_name, status, etc.); added trials are returned as Trial objects
    awaiting index assignment.
    """
    by_hash = {t["hash"]: t for t in existing}
    new_hashes = {trial_hash(c.params, c.extras): c for c in combos}

    kept = [t for t in existing if t["hash"] in new_hashes]
    removed = [t for t in existing if t["hash"] not in new_hashes]
    added = [c for h, c in new_hashes.items() if h not in by_hash]
    return Reconciled(kept=kept, added=added, removed=removed)


def _next_index(base: str, existing: List[dict]) -> int:
    """High-water mark across the live manifest AND historical job_ids.

    Even after a trial is dropped from the manifest, its index may still
    appear in job_ids.json — recycling it would let _sync_slurm_status
    apply a stale SLURM state to a new trial.
    """
    max_live = max((t["index"] for t in existing), default=-1)
    max_hist = -1
    for record in _load_job_ids(base):
        for idx in record.get("indices", []):
            if idx > max_hist:
                max_hist = idx
    return max(max_live, max_hist) + 1


def append_trials(
    base: str,
    new_combos: List[Trial],
    abbrevs: Dict[str, str],
    labels: Optional[Dict[str, Dict[Any, str]]],
) -> List[dict]:
    """Append new trials to an existing manifest, assigning fresh indices.

    Indices are never reused — they extend past the high-water mark across
    both the live manifest and `job_ids.json`, so SLURM job_ids referencing
    old indices can never collide with new trials.
    """
    if not new_combos:
        return load_manifest(base)
    existing = load_manifest(base)
    next_idx = _next_index(base, existing)
    for combo in new_combos:
        existing.append(_trial_record(next_idx, combo.params, combo.extras, abbrevs, labels))
        next_idx += 1
    _write_manifest(base, existing)
    return existing


def drop_trials(base: str, indices: List[int]) -> List[dict]:
    """Remove trials by index. Caller is responsible for ensuring it's safe
    (e.g., no live SLURM jobs reference these indices)."""
    if not indices:
        return load_manifest(base)
    drop = set(indices)
    existing = load_manifest(base)
    kept = [t for t in existing if t["index"] not in drop]
    _write_manifest(base, kept)
    return kept


def _write_manifest(base: str, trials: List[dict]) -> None:
    """Atomically write the manifest. The previous `open("w")` path
    truncated the destination before any bytes were written, so any
    concurrent reader (the SLURM poller's `herd snapshot`, the
    dashboard's `cmd_status`, the agent's `state.compute`) saw an
    empty file and crashed on `json.load`. Write to a sibling temp
    file in the same directory and `os.replace` — the rename is
    atomic on POSIX, so readers always see either the old contents
    or the new contents, never an in-between."""
    target = manifest_path(base)
    tmp = target + ".tmp"
    with open(tmp, "w") as f:
        json.dump(trials, f, indent=2, default=_json_default)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, target)


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
    """Get indices that need (re)submission: ready (never submitted), failed, or cancelled."""
    trials = load_manifest(base)
    # "pending" kept as a legacy alias for manifests written before the rename to "ready".
    return [t["index"] for t in trials if t["status"] in ("ready", "pending", "failed", "cancelled")]


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
    target = job_ids_path(base)
    tmp = target + ".tmp"
    with open(tmp, "w") as f:
        json.dump(records, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, target)


# --- Hydra override resolution ---

def _format_override_value(value: Any) -> str:
    if value is None:
        # Hydra reads bare `None` as the string "None"; `null` is the YAML
        # null literal that resolves to Python None.
        return "null"
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
      3. static_overrides
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
