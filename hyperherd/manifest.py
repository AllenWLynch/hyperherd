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


def coerce_scalar(raw: str) -> Any:
    """Coerce a user-typed string to int → float → str, in that order.

    Trial params keep their original YAML types, so a CLI token `32` must become
    the int `32` to compare (or name) equal to a stored `32`.
    """
    try:
        return int(raw)
    except (TypeError, ValueError):
        pass
    try:
        return float(raw)
    except (TypeError, ValueError):
        return raw


def effective_params(
    params: Dict[str, Any], overrides: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """`params` with any override that *shadows a swept param* folded in.

    This is the trial's configuration **as actually trained**, and it's what any
    code that *interprets* a trial must read: which bracket it belongs in
    (`bracket_by`), what hyperparameters to attribute its metrics to (`herd
    res`), whether it matches a `--where` filter. Reading raw `params` there
    would report — and rank — the trial under a config it didn't run.

    Overrides on keys that aren't swept params (`ckpt=...`, `debug=true`) aren't
    folded in: they don't move the trial to a different point in the search
    space.

    NOT used for naming — see `experiment_name_for`, which must stay collision-
    free and therefore can't simply substitute the new value into the name.

    Values are coerced so `labels:` lookups still resolve (a label map is keyed
    by the declared YAML value, e.g. int 32, not the string "32").
    """
    out = dict(params)
    for key, raw in (overrides or {}).items():
        if key in out:
            out[key] = coerce_scalar(raw) if isinstance(raw, str) else raw
    return out


def _override_token(value: Any) -> str:
    """A filename-safe token for an override value in an experiment name."""
    return format_short_value(value).replace("/", "_")


def experiment_name_for(
    params: Dict[str, Any],
    overrides: Optional[Dict[str, str]],
    abbrevs: Dict[str, str],
    labels: Optional[Dict[str, Dict[Any, str]]] = None,
) -> str:
    """The trial's output-directory name, accounting for CLI overrides.

    An override that shadows a swept parameter **appends a suffix** to the
    original name rather than substituting the new value into it::

        trial 0: params {lr: 0.1, bs: 32}  ->  lr-0.1_bs-32
        herd run 0 bs=64                   ->  lr-0.1_bs-32_ov_bs-64

    Substituting would produce `lr-0.1_bs-64` — which is *already some other
    trial's name* whenever the override value is one the sweep covers. Trial 0
    would then write straight into trial 1's output directory and destroy its
    results, and if trial 1 were running, two array tasks would race over the
    same files. Appending keeps the base (the original params, which are unique
    by construction), so the full name stays unique too.

    Overrides on non-swept keys (`ckpt=`, `debug=`) leave the name alone: they
    don't identify a different point in the search space, so the trial re-runs
    **in place**, which is what a checkpoint resume wants.
    """
    base = build_experiment_name(params, abbrevs, labels)
    shadowing = [(k, v) for k, v in (overrides or {}).items() if k in params]
    if not shadowing:
        return base
    suffix = "_".join(
        f"{abbrevs.get(k, k)}-{_override_token(v)}" for k, v in shadowing
    )
    return f"{base}_ov_{suffix}"


def trial_hash(
    params: Dict[str, Any],
    extras: Optional[Dict[str, Any]] = None,
    derived_overrides: Optional[Dict[str, Any]] = None,
) -> str:
    """Stable identity hash for a trial: swept params + constraint extras + any
    `derived:` overrides.

    Two trials with the same hash are considered the same trial across
    config edits — this is how reconciliation distinguishes
    "kept" trials from "added"/"removed" ones. A `derived:` trial shares its
    base's params/extras, so its `derived_overrides` are what make its identity
    distinct.

    `derived_overrides` is folded in ONLY when non-empty, so an ordinary trial's
    hash is byte-identical to the pre-derived-feature formula — existing
    manifests reconcile cleanly without a rehash/migration.
    """
    payload = {"params": params, "extras": extras or {}}
    if derived_overrides:
        payload["derived"] = derived_overrides
    blob = json.dumps(payload, sort_keys=True, default=_json_default)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def _derived_name_suffix(
    derived_overrides: Optional[Dict[str, Any]],
    abbrevs: Dict[str, str],
) -> str:
    """Name suffix distinguishing a `derived:` trial from its base grid trial.

    Mirrors the `_ov_` convention `experiment_name_for` uses for swept-param
    overrides, so a derivation reads `<base>_ov_<key>-<value>`. Without it a
    derived trial (identical base params, its override living in `extras`) would
    share the base trial's experiment_name and clobber its checkpoint.
    """
    if not derived_overrides:
        return ""
    parts = [
        f"{abbrevs.get(k, k)}-{_override_token(v)}"
        for k, v in derived_overrides.items()
    ]
    return "_ov_" + "_".join(parts)


def _trial_record(
    index: int,
    params: Dict[str, Any],
    extras: Dict[str, Any],
    abbrevs: Dict[str, str],
    labels: Optional[Dict[str, Dict[Any, str]]],
    overrides: Optional[Dict[str, str]] = None,
    derived_from: Optional[int] = None,
    derived_overrides: Optional[Dict[str, Any]] = None,
) -> dict:
    overrides = overrides or {}
    derived_overrides = derived_overrides or {}
    experiment_name = experiment_name_for(params, overrides, abbrevs, labels)
    # A `derived:` trial shares its base's params (and thus base name); the
    # `_ov_…` suffix keeps its output path distinct. See `_derived_name_suffix`.
    experiment_name += _derived_name_suffix(derived_overrides, abbrevs)
    record = {
        "index": index,
        # NOTE: `overrides` is deliberately NOT hashed. The hash is the trial's
        # reconciliation identity (which point in the search space it is); an
        # ad-hoc CLI override doesn't move it, so folding overrides in here
        # would make every override look like a config edit that replaced the
        # trial. `derived_overrides`, by contrast, DEFINE a new trial, so they
        # ARE hashed.
        "hash": trial_hash(params, extras, derived_overrides),
        "params": params,
        "extras": extras,
        "overrides": overrides,
        "experiment_name": experiment_name,
        "status": "ready",
    }
    if derived_overrides:
        record["derived_from"] = derived_from
        record["derived_overrides"] = derived_overrides
    return record


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
            records.append(_trial_record(
                i, item.params, item.extras, abbrevs, labels,
                derived_from=item.derived_from,
                derived_overrides=item.derived_overrides,
            ))
        else:
            records.append(_trial_record(i, item, {}, abbrevs, labels))
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
        # Manifests written before per-trial CLI overrides existed have no key.
        t.setdefault("overrides", {})
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
    new_hashes = {
        trial_hash(c.params, c.extras, c.derived_overrides): c for c in combos
    }

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
        existing.append(_trial_record(
            next_idx, combo.params, combo.extras, abbrevs, labels,
            derived_from=combo.derived_from,
            derived_overrides=combo.derived_overrides,
        ))
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


def set_trial_overrides(
    base: str,
    indices: List[int],
    overrides: Dict[str, str],
    abbrevs: Dict[str, str],
    labels: Optional[Dict[str, Dict[Any, str]]] = None,
    clear: bool = False,
) -> List[dict]:
    """Merge per-trial CLI overrides into the given trials; return those records.

    `clear` drops any stored overrides first, so `--clear-overrides` alone resets
    a trial and `--clear-overrides k=v` replaces rather than merges.

    Rewrites `experiment_name` (see `experiment_name_for`) so an override that
    shadows a swept parameter sends the trial to a new output directory instead
    of overwriting the original run's results. `hash` is untouched — an override
    doesn't change which point in the search space the trial is, and making it
    the reconciliation identity would turn every override into a config edit
    that replaced the trial.
    """
    trials = load_manifest(base)
    targets = set(indices)
    touched = []
    for trial in trials:
        if trial["index"] not in targets:
            continue
        current = {} if clear else dict(trial.get("overrides") or {})
        current.update(overrides)
        trial["overrides"] = current
        trial["experiment_name"] = experiment_name_for(
            trial.get("params", {}), current, abbrevs, labels)
        touched.append(trial)
    if touched:
        _write_manifest(base, trials)
    return touched


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
      4. constraint `set` extras (wins over statics)
      5. `derived:` overrides (wins over constraint extras)
      6. per-trial CLI overrides from `herd run <idx> k=v` (last → wins over all)
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

    # Extras are emitted after statics so constraint `set` values override them.
    extras = trial.get("extras") or {}
    for k, v in extras.items():
        parts.append(f"{k}={_format_override_value(v)}")

    # `derived:` overrides define this variant, so they win over params/statics/
    # constraint extras (but not the user's explicit per-trial CLI overrides).
    for k, v in (trial.get("derived_overrides") or {}).items():
        parts.append(f"{k}={_format_override_value(v)}")

    # Per-trial CLI overrides win over everything. Emitted VERBATIM, not through
    # `_format_override_value` — the user typed these, and round-tripping them
    # through the float formatter would silently rewrite `lr=1e-3` to `lr=0.001`.
    for k, v in (trial.get("overrides") or {}).items():
        parts.append(f"{k}={v}")

    return " ".join(parts)
