"""HyperHerd CLI: launch, monitor, and clean SLURM hyperparameter job arrays."""

import argparse
import json
import os
import shutil
import subprocess
import sys
from typing import Dict, Optional

from hyperherd import agent_output
from hyperherd.config import ConfigError, load_config
from hyperherd.constraints import apply_constraints
from hyperherd.display import (
    _DIM,
    _RESET,
    print_dry_run,
    print_launch_success,
    print_stats_table,
    print_status_table,
    print_summary,
    print_trial_listing,
)
from hyperherd.init import scaffold
from hyperherd.preflight import PreflightError, run_preflight
from hyperherd.search import generate_combinations
from hyperherd import manifest
from hyperherd import slurm
from hyperherd.logging import load_all_results


_ACTIVE_STATUSES = ("running", "queued", "submitted", "completed")


def _apply_reconciliation(config, diff, force: bool) -> bool:
    """Apply a manifest/config diff: drop config-removed trials, append new ones.

    Returns False if the user must intervene (active trials removed without --force).
    """
    active_removed = [t for t in diff.removed if t.get("status") in _ACTIVE_STATUSES]
    droppable = [t for t in diff.removed if t.get("status") not in _ACTIVE_STATUSES]

    if active_removed and not force:
        print(
            f"Config edit removes {len(active_removed)} trial(s) that are "
            f"running, queued, or completed:",
            file=sys.stderr,
        )
        for t in active_removed[:5]:
            print(
                f"  idx {t['index']} [{t.get('status')}] {t.get('experiment_name','')}",
                file=sys.stderr,
            )
        if len(active_removed) > 5:
            print(f"  ... and {len(active_removed) - 5} more", file=sys.stderr)
        print(
            "Refusing to drop them. Pass --force to keep them in the manifest "
            "as orphans, or revert your config edit.",
            file=sys.stderr,
        )
        return False

    if droppable:
        manifest.drop_trials(config.workspace, [t["index"] for t in droppable])
        print(f"  Dropped {len(droppable)} trial(s) no longer in config.")

    if active_removed and force:
        print(
            f"  Keeping {len(active_removed)} active trial(s) as orphans (--force)."
        )

    if diff.added:
        manifest.append_trials(
            config.workspace, diff.added, config.abbrevs, config.labels
        )
        print(f"  Added {len(diff.added)} new trial(s) from config edits.")

    return True


def _parse_pin_args(pin_args, config):
    """Parse `--pin name=value` strings into a {name: coerced_value} dict.

    Raises ValueError with a user-facing message on:
      - malformed entries (no `=`)
      - names not in config.parameters (incl. static_overrides keys —
        those aren't sweep parameters and pinning to them is meaningless)

    Coerces the RHS by trying int → float → str, in that order. Trial
    params are stored with their original YAML types so a stored `32`
    (int) still matches a `--pin batch_size=32` (coerced int 32).
    """
    if not pin_args:
        return {}

    sweep_params = set(config.parameters.keys())
    static_keys = {
        s.split("=", 1)[0] for s in (config.static_overrides or [])
    }

    pins = {}
    for entry in pin_args:
        if "=" not in entry:
            raise ValueError(
                f"--pin {entry!r}: expected 'name=value', got no '='"
            )
        name, raw = entry.split("=", 1)
        name = name.strip()
        raw = raw.strip()
        if not name:
            raise ValueError(f"--pin {entry!r}: empty parameter name")
        if name in static_keys:
            raise ValueError(
                f"--pin {name!r}: that's a static_overrides key, not a "
                f"sweep parameter. --pin only accepts sweep parameter "
                f"names. Sweep parameters: {sorted(sweep_params)}"
            )
        if name not in sweep_params:
            raise ValueError(
                f"--pin {name!r}: unknown parameter. "
                f"Sweep parameters: {sorted(sweep_params)}"
            )

        # Type coercion: int → float → str.
        coerced: object = raw
        try:
            coerced = int(raw)
        except ValueError:
            try:
                coerced = float(raw)
            except ValueError:
                coerced = raw

        pins[name] = coerced

    return pins


def _filter_trials_by_pins(trials, pins):
    """Return the subset of trials whose `params` match every pin.

    Comparison uses `==`, which gives loose numeric equality (int 32
    matches float 32.0) but strict string equality. Trials are
    expected to be dicts with a `params` sub-dict, as stored in the
    manifest.
    """
    if not pins:
        return list(trials)

    out = []
    for t in trials:
        params = t.get("params", {})
        if all(params.get(k) == v for k, v in pins.items()):
            out.append(t)
    return out


def cmd_launch(args):
    """Launch (or re-launch) a hyperparameter sweep as a SLURM job array."""
    # Apply `.env` BEFORE load_config so any `${VAR}` interpolation in
    # the YAML can resolve from values in the file.
    _apply_workspace_env(args.workspace)
    config = load_config(args.workspace)
    json_mode = getattr(args, "json_output", False)

    def _say(*a, **kw):
        if not json_mode:
            print(*a, **kw)

    # Parse --pin early so config validation errors surface before any
    # SLURM/manifest work.
    try:
        pins = _parse_pin_args(getattr(args, "pin", None), config)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Preflight checks
    try:
        warnings = run_preflight(config)
    except PreflightError as e:
        print(f"Preflight check failed: {e}", file=sys.stderr)
        return 1
    for w in warnings:
        print(f"Warning: {w}", file=sys.stderr)

    # Generate parameter combinations
    combinations = generate_combinations(config)
    combinations = apply_constraints(combinations, config.conditions)

    if not combinations:
        print("No valid parameter combinations after applying conditions.", file=sys.stderr)
        return 1

    # Initialize workspace
    manifest.init_workspace(config.workspace)

    # Build abbreviation mapping for experiment names
    abbrevs = config.abbrevs

    # Create or load manifest
    existing = manifest.load_manifest(config.workspace) if manifest.workspace_exists(config.workspace) else []
    if existing:
        _say(f"Existing workspace found with {len(existing)} trials.")
        # Refresh status from SLURM before deciding what to resubmit
        _sync_slurm_status(config.workspace)
        existing = manifest.load_manifest(config.workspace)

        diff = manifest.reconcile_manifest(existing, combinations)
        if json_mode:
            # Reconciliation chatter would be on stdout; skip it in JSON mode.
            # The agent can diff trials before/after if it cares.
            active_removed = [t for t in diff.removed if t.get("status") in _ACTIVE_STATUSES]
            if active_removed and not args.force:
                print(
                    f"Config edit removes {len(active_removed)} active trial(s). "
                    "Pass --force to keep them as orphans.",
                    file=sys.stderr,
                )
                return 1
            droppable = [t for t in diff.removed if t.get("status") not in _ACTIVE_STATUSES]
            if droppable:
                manifest.drop_trials(config.workspace, [t["index"] for t in droppable])
            if diff.added:
                manifest.append_trials(
                    config.workspace, diff.added, config.abbrevs, config.labels
                )
        else:
            if not _apply_reconciliation(config, diff, args.force):
                return 1

        trials = manifest.load_manifest(config.workspace)
        pending = manifest.get_pending_indices(config.workspace)
        if not pending and not args.indices:
            if json_mode:
                agent_output.emit(agent_output.launch_payload(
                    dry_run=bool(args.dry_run),
                    submitted_indices=[],
                    slurm_job_id=None,
                    sbatch_path=None,
                    trials=trials,
                ))
                return 0
            print("All trials are completed or currently running. Nothing to submit.")
            return 0
        if not args.indices:
            _say(f"  {len(pending)} trials need (re)submission.")
    else:
        trials = manifest.create_manifest(config.workspace, combinations, abbrevs, config.labels)
        pending = [t["index"] for t in trials]

    # Narrow to a user-specified subset of indices, if given.
    if args.indices:
        try:
            requested = sorted(set(slurm._parse_array_range(args.indices)))
        except ValueError as e:
            print(f"Invalid --indices spec {args.indices!r}: {e}", file=sys.stderr)
            return 1
        valid = {t["index"] for t in trials}
        unknown = [i for i in requested if i not in valid]
        if unknown:
            print(
                f"Indices out of range: {unknown} (valid: 0-{max(valid)})",
                file=sys.stderr,
            )
            return 1
        if not args.force:
            status_by_idx = {t["index"]: t["status"] for t in trials}
            blocked = [
                i for i in requested
                if status_by_idx[i] in ("running", "queued", "submitted", "completed")
            ]
            if blocked:
                rows = ", ".join(f"{i}={status_by_idx[i]}" for i in blocked)
                print(
                    f"Refusing to resubmit indices already running/completed: {rows}.\n"
                    f"Pass --force to override, or `herd clean` to reset.",
                    file=sys.stderr,
                )
                return 1
        pending = requested
        _say(f"  Submitting {len(pending)} requested trial(s): {args.indices}")

    # Apply --pin filter (after status- and indices-based filtering), so
    # the user's pinned params narrow the candidate pool to matching
    # trials only. An empty result is a hard error: the user expected
    # specific trials and we shouldn't silently submit nothing.
    pin_filter_summary: Optional[str] = None
    if pins:
        all_trials = manifest.load_manifest(config.workspace)
        matching = _filter_trials_by_pins(all_trials, pins)
        matching_indices = {t["index"] for t in matching}
        before = len(pending)
        pending = [i for i in pending if i in matching_indices]
        pin_desc = ", ".join(f"{k}={v}" for k, v in pins.items())
        if not pending:
            print(
                f"Error: no submittable trials match --pin {pin_desc}. "
                f"({len(matching)} total trials match the pin, but none "
                f"were in the {before} pending after status/indices "
                f"filtering — they may already be running or completed.)",
                file=sys.stderr,
            )
            return 1
        pin_filter_summary = (
            f"--pin {pin_desc} narrowed from {before} to {len(pending)} trial(s)"
        )
        _say(f"  {pin_filter_summary}.")

    # Generate sbatch script
    script = slurm.generate_sbatch_script(config, pending, args.max_concurrent)

    if args.dry_run:
        if json_mode:
            agent_output.emit(agent_output.launch_payload(
                dry_run=True,
                submitted_indices=pending,
                slurm_job_id=None,
                sbatch_path=None,
                trials=trials,
                sbatch_script=script,
            ))
            return 0
        print_dry_run(
            sbatch_script=script,
            pending_indices=pending,
            total_trials=len(trials),
            filter_summary=pin_filter_summary,
        )
        return 0

    # Submit
    _say(f"Submitting {len(pending)} trials as SLURM job array...")
    job_id = slurm.submit_job(config, script, dry_run=False)
    assert job_id is not None

    # Record submission and update statuses
    manifest.record_job_submission(config.workspace, job_id, pending)
    manifest.bulk_update_status(config.workspace, {i: "submitted" for i in pending})

    if json_mode:
        agent_output.emit(agent_output.launch_payload(
            dry_run=False,
            submitted_indices=pending,
            slurm_job_id=job_id,
            sbatch_path=manifest.sbatch_path(config.workspace),
            trials=manifest.load_manifest(config.workspace),
        ))
        return 0

    print_launch_success(
        job_id=job_id,
        n_trials=len(pending),
        workspace=manifest.workspace_path(config.workspace),
        logs=manifest.logs_path(config.workspace),
    )
    return 0


def cmd_ls(args):
    """List every trial in the sweep with its swept parameters.

    Status-agnostic: shows the *shape* of the sweep, not what `herd
    run` would do next. Reads the manifest if present; otherwise
    materializes the combinations from the config so users can `herd
    ls` BEFORE the first `herd run` to sanity-check the YAML.

    Supports the same `--pin name=value` filter as `herd run` for
    inspecting a slice of the grid.
    """
    config = load_config(args.workspace)

    try:
        pins = _parse_pin_args(getattr(args, "pin", None), config)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if manifest.workspace_exists(config.workspace):
        trials = manifest.load_manifest(config.workspace)
        show_status = True
        title = None
    else:
        # Pre-`herd run`: render combinations on the fly so this still
        # works as a YAML sanity check.
        try:
            warnings = run_preflight(config)
        except PreflightError as e:
            print(f"Preflight check failed: {e}", file=sys.stderr)
            return 1
        for w in warnings:
            print(f"Warning: {w}", file=sys.stderr)
        combos = apply_constraints(generate_combinations(config), config.conditions)
        if not combos:
            print(
                "No valid parameter combinations after applying conditions.",
                file=sys.stderr,
            )
            return 1
        # Render to manifest-shaped dicts without persisting anything.
        from hyperherd.manifest import _trial_record
        trials = [
            _trial_record(i, c.params, c.extras, config.abbrevs, config.labels)
            for i, c in enumerate(combos)
        ]
        show_status = False
        title = (
            "(no manifest yet — showing combinations from hyperherd.yaml)"
        )

    if pins:
        before = len(trials)
        trials = _filter_trials_by_pins(trials, pins)
        if not trials:
            pin_desc = ", ".join(f"{k}={v}" for k, v in pins.items())
            print(
                f"No trials match --pin {pin_desc} "
                f"(of {before} total).",
                file=sys.stderr,
            )
            return 1

    print_trial_listing(
        trials,
        defaults=config.defaults,
        show_status=show_status,
        title=title,
    )
    return 0


def cmd_status(args):
    """Show the status of all trials in a hyperparameter sweep."""
    config = load_config(args.workspace)

    if not manifest.workspace_exists(config.workspace):
        print("No workspace found. Run 'hyperherd launch' first.", file=sys.stderr)
        return 1

    # Sync status from SLURM
    _sync_slurm_status(config.workspace)

    trials = manifest.load_manifest(config.workspace)

    # Gather log tails
    log_tails = {}
    for trial in trials:
        log_tails[trial["index"]] = slurm.get_log_tail(config.workspace, trial["index"])

    if getattr(args, "json_output", False):
        agent_output.emit(agent_output.status_payload(trials, log_tails))
        return 0

    print_status_table(trials, log_tails)
    print_summary(trials)
    return 0


def cmd_stats(args):
    """Print SLURM accounting (runtime, max/ave RSS, requested mem) per trial."""
    config = load_config(args.workspace)

    if not manifest.workspace_exists(config.workspace):
        print("No workspace found. Run 'herd run' first.", file=sys.stderr)
        return 1

    job_records = manifest.get_job_ids(config.workspace)
    job_ids = [r["slurm_job_id"] for r in job_records]
    if not job_ids:
        print("No SLURM jobs recorded for this workspace.", file=sys.stderr)
        return 1

    try:
        all_stats = slurm.query_job_stats(job_ids)
    except Exception as e:
        print(f"Could not query SLURM accounting: {e}", file=sys.stderr)
        return 1

    # Pick the most recent JobStats record per index (a trial may have been
    # resubmitted). Records are processed in submission order.
    by_index = {}
    for record in job_records:
        jid = record["slurm_job_id"]
        for idx in record.get("indices", []):
            stats = all_stats.get((jid, idx))
            if stats is not None:
                by_index[idx] = stats

    trials = manifest.load_manifest(config.workspace)
    trial_by_idx = {t["index"]: t for t in trials}

    if args.index is None:
        rows = [
            (idx, trial_by_idx.get(idx, {}), by_index[idx])
            for idx in sorted(by_index)
        ]
        if getattr(args, "json_output", False):
            agent_output.emit(agent_output.stats_payload(rows))
            return 0
        if not rows:
            print("No accounting data available.")
            return 0
        print_stats_table(rows)
        return 0

    idx = args.index
    if idx not in trial_by_idx:
        print(f"No trial found with index {idx}.", file=sys.stderr)
        return 1
    if idx not in by_index:
        print(f"No SLURM accounting data for trial {idx}.", file=sys.stderr)
        return 1
    rows = [(idx, trial_by_idx[idx], by_index[idx])]
    if getattr(args, "json_output", False):
        agent_output.emit(agent_output.stats_payload(rows))
        return 0
    print_stats_table(rows)
    return 0


def cmd_tail(args):
    """Print the last N lines of a trial's log files (stdout and stderr).

    `--stdout` / `--stderr` (mutually exclusive) restrict the output to one
    stream; without either flag, both streams are shown. `--json` returns a
    structured payload with each requested stream's path and lines."""
    config = load_config(args.workspace)

    if not manifest.workspace_exists(config.workspace):
        print("No workspace found. Run 'hyperherd launch' first.", file=sys.stderr)
        return 1

    index = args.index
    lines = args.lines
    log_dir = manifest.logs_path(config.workspace)

    out_file = os.path.join(log_dir, f"{index}.out")
    err_file = os.path.join(log_dir, f"{index}.err")

    # Which streams the user wants. Default (no flag) is both.
    stream_filter = getattr(args, "stream", None)
    streams_to_show = (
        [(stream_filter, out_file if stream_filter == "stdout" else err_file)]
        if stream_filter
        else [("stdout", out_file), ("stderr", err_file)]
    )

    if not getattr(args, "json_output", False):
        if not any(os.path.isfile(p) for _, p in streams_to_show):
            print(f"No log files found for trial {index}", file=sys.stderr)
            return 1

    # Trial header info (status + experiment_name) — used by both modes.
    trials = manifest.load_manifest(config.workspace)
    trial = next((t for t in trials if t["index"] == index), None)
    exp_name = trial.get("experiment_name", "") if trial else ""
    status = trial.get("status", "unknown") if trial else "unknown"

    if getattr(args, "json_output", False):
        streams_payload: dict = {}
        for label, path in streams_to_show:
            if not os.path.isfile(path):
                streams_payload[label] = {
                    "path": path,
                    "lines": None,
                    "requested": lines,
                }
                continue
            try:
                with open(path, "r") as f:
                    all_lines = f.readlines()
            except (OSError, UnicodeDecodeError) as e:
                streams_payload[label] = {
                    "path": path,
                    "lines": None,
                    "requested": lines,
                    "error": str(e),
                }
                continue
            tail = all_lines[-lines:] if len(all_lines) > lines else all_lines
            streams_payload[label] = {
                "path": path,
                "lines": [ln.rstrip("\n") for ln in tail],
                "requested": lines,
            }
        agent_output.emit(agent_output.tail_payload(
            index=index,
            status=status,
            experiment_name=exp_name,
            streams=streams_payload,
        ))
        return 0

    print(f"{_DIM}Trial {index} [{status}] {exp_name}{_RESET}")
    print(f"{_DIM}{'-' * 60}{_RESET}")

    for label, log_file in streams_to_show:
        if not os.path.isfile(log_file):
            continue
        try:
            with open(log_file, "r") as f:
                all_lines = f.readlines()
        except (OSError, UnicodeDecodeError) as e:
            print(f"Could not read {label} log: {e}", file=sys.stderr)
            continue
        if not all_lines:
            continue
        tail = all_lines[-lines:] if len(all_lines) > lines else all_lines
        print(f"\n{_DIM}[{label}] {log_file} (last {len(tail)} lines){_RESET}")
        for line in tail:
            print(line, end="")
        if tail and not tail[-1].endswith("\n"):
            print()

    return 0


def cmd_test(args):
    """Run a single trial locally via the launcher (no SLURM).

    The default invocation runs the trial end-to-end exactly as the SLURM
    array would, and refuses any index that has ever been submitted to SLURM
    (so the launcher's outputs/logs aren't clobbered).

    With `--cfg-job`, appends `--cfg job` to the override string. Hydra
    trainers interpret this as "print the resolved config and exit without
    running" — useful as a quick config-validation step. In this mode no
    real outputs are produced, so the previously-submitted guard is skipped.
    """
    import subprocess

    # Apply `.env` so the trainer sees the same secrets it would under
    # `herd run` / `herd monitor` (WANDB_API_KEY, etc.).
    _apply_workspace_env(args.workspace)
    config = load_config(args.workspace)

    try:
        run_preflight(config)
    except PreflightError as e:
        print(f"Preflight check failed: {e}", file=sys.stderr)
        return 1

    combinations = generate_combinations(config)
    combinations = apply_constraints(combinations, config.conditions)

    if not combinations:
        print("No valid parameter combinations after applying conditions.", file=sys.stderr)
        return 1

    manifest.init_workspace(config.workspace)

    if not manifest.workspace_exists(config.workspace):
        manifest.create_manifest(config.workspace, combinations, config.abbrevs, config.labels)

    trials = manifest.load_manifest(config.workspace)

    index = args.index
    if index < 0 or index >= len(trials):
        print(f"Trial index {index} out of range (0-{len(trials) - 1}).", file=sys.stderr)
        return 1

    cfg_job = getattr(args, "cfg_job", False)

    # End-to-end runs would clobber a trial that's already been launched.
    # `--cfg-job` validates without running, so it's safe to skip.
    if not cfg_job:
        for record in manifest.get_job_ids(config.workspace):
            if index in record.get("indices", []):
                print(
                    f"Trial {index} was previously submitted to SLURM "
                    f"(job {record['slurm_job_id']}). Refusing to run locally — "
                    f"running would clobber its outputs/logs. Pick a different "
                    f"index, pass --cfg-job for a Hydra config-only check, or "
                    f"`herd clean --all` first.",
                    file=sys.stderr,
                )
                return 1

    trial = trials[index]
    exp_name = trial.get("experiment_name", "")
    overrides = manifest.resolve_overrides(
        config.workspace, index, config.static_overrides or None
    )
    if cfg_job:
        overrides = f"{overrides} --cfg job"

    if cfg_job:
        print(f"Validating Hydra config for trial {index}")
    else:
        print(f"Running trial {index} locally")
    if exp_name:
        print(f"  experiment_name: {exp_name}")
    print(f"  overrides: {overrides}")
    print(f"  launcher: {config.launcher}")
    print("-" * 60)
    print()

    env = os.environ.copy()
    env["HYPERHERD_WORKSPACE"] = config.workspace
    env["HYPERHERD_SWEEP_NAME"] = config.name
    env["HYPERHERD_TRIAL_ID"] = str(index)
    env["HYPERHERD_TRIAL_NAME"] = exp_name
    # Legacy alias — `experiment` here means "this trial's identifier",
    # which is confusing with the broader use of "experiment" for the
    # whole sweep. Kept for backward compat with trainer code that
    # reads it. New code should use HYPERHERD_TRIAL_NAME.
    env["HYPERHERD_EXPERIMENT_NAME"] = exp_name

    result = subprocess.run(
        ["bash", config.launcher, overrides],
        cwd=config.workspace,
        env=env,
    )

    print()
    print("-" * 60)
    if result.returncode == 0:
        msg = "Hydra config is valid." if cfg_job else "local run completed successfully."
        print(f"Trial {index}: {msg}")
    else:
        msg = (
            "Hydra config validation failed"
            if cfg_job
            else "local run failed"
        )
        print(f"Trial {index}: {msg} (exit code {result.returncode}).")

    return result.returncode


def cmd_results(args):
    """Print a TSV of trial parameters and logged metrics."""
    config = load_config(args.workspace)

    if not manifest.workspace_exists(config.workspace):
        print("No workspace found. Run 'hyperherd launch' first.", file=sys.stderr)
        return 1

    trials = manifest.load_manifest(config.workspace)
    results = load_all_results(config.workspace)

    if getattr(args, "json_output", False):
        agent_output.emit(agent_output.results_payload(
            trials, results, config.param_names
        ))
        return 0

    if not results:
        print("No results logged yet. Use hyperherd.log_result() from your training script.", file=sys.stderr)
        return 1

    # Collect all metric names across trials
    metric_names = []
    seen = set()
    for trial_results in results.values():
        for k in trial_results:
            if k not in seen:
                metric_names.append(k)
                seen.add(k)

    # Build header: trial_id, experiment_name, param1, param2, ..., metric1, metric2, ...
    param_names = config.param_names
    header = ["trial_id", "experiment_name"] + param_names + metric_names

    sep = "\t"
    print(sep.join(header))

    for trial in trials:
        idx = trial["index"]
        exp_name = trial.get("experiment_name", "")
        params = trial["params"]
        trial_results = results.get(idx, {})

        row = [str(idx), exp_name]
        for p in param_names:
            v = params.get(p, "")
            if isinstance(v, float):
                row.append(f"{v:.6g}")
            else:
                row.append(str(v))
        for m in metric_names:
            v = trial_results.get(m, "")
            if isinstance(v, float):
                row.append(f"{v:.6g}")
            else:
                row.append(str(v))

        print(sep.join(row))

    return 0


_LIVE_STATUSES = ("running", "queued", "submitted")


def _latest_job_id_for(records, index: int):
    """Return the most recent SLURM job_id that included this index, or None."""
    for record in reversed(records):
        if index in record.get("indices", []):
            return record["slurm_job_id"]
    return None


def cmd_stop(args):
    """Cancel one or all running/queued trials via scancel <jobid>_<index>."""
    config = load_config(args.workspace)

    if not manifest.workspace_exists(config.workspace):
        print("No workspace found.", file=sys.stderr)
        return 1

    if (args.index is None) == (not args.all):
        print(
            "Pass either an index or --all (not both, not neither).",
            file=sys.stderr,
        )
        return 1

    _sync_slurm_status(config.workspace)
    trials = manifest.load_manifest(config.workspace)
    records = manifest.get_job_ids(config.workspace)

    if args.all:
        targets = [t for t in trials if t.get("status") in _LIVE_STATUSES]
        cancelled: list = []
        for t in targets:
            jid = _latest_job_id_for(records, t["index"])
            if jid is None:
                continue
            slurm.cancel_array_task(jid, t["index"])
            cancelled.append({
                "index": t["index"],
                "slurm_job_id": jid,
                "previous_status": t.get("status"),
            })
        if cancelled:
            manifest.bulk_update_status(
                config.workspace,
                {row["index"]: "cancelled" for row in cancelled},
            )
        if getattr(args, "json_output", False):
            agent_output.emit(agent_output.stop_payload(cancelled))
            return 0
        if not targets:
            print("No live trials to cancel.")
            return 0
        idxs = sorted(row["index"] for row in cancelled)
        print(f"Cancelled {len(cancelled)} trial(s): {idxs}")
        return 0

    index = args.index
    trial = next((t for t in trials if t["index"] == index), None)
    if trial is None:
        print(f"No trial found with index {index}.", file=sys.stderr)
        return 1

    status = trial.get("status", "unknown")
    if status not in _LIVE_STATUSES:
        print(
            f"Trial {index} is {status!r}, not running/queued — nothing to cancel.",
            file=sys.stderr,
        )
        return 1

    job_id = _latest_job_id_for(records, index)
    if job_id is None:
        print(f"No SLURM job ID recorded for trial {index}.", file=sys.stderr)
        return 1

    if not getattr(args, "json_output", False):
        print(f"Cancelling trial {index} (job {job_id}_{index})...")
    slurm.cancel_array_task(job_id, index)
    manifest.update_trial_status(config.workspace, index, "cancelled")

    if getattr(args, "json_output", False):
        agent_output.emit(agent_output.stop_payload([{
            "index": index,
            "slurm_job_id": job_id,
            "previous_status": status,
        }]))
    return 0


def cmd_clean(args):
    """Cancel running jobs and clean up workspace."""
    config = load_config(args.workspace)
    ws = manifest.workspace_path(config.workspace)

    if not os.path.isdir(ws):
        print("No workspace to clean.", file=sys.stderr)
        return 1

    # Cancel running jobs
    job_records = manifest.get_job_ids(config.workspace)
    job_ids = [r["slurm_job_id"] for r in job_records]
    if job_ids:
        print(f"Cancelling {len(job_ids)} job(s)...")
        slurm.cancel_jobs(job_ids)

    if args.logs:
        log_dir = manifest.logs_path(config.workspace)
        if os.path.isdir(log_dir):
            shutil.rmtree(log_dir)
            print(f"Removed logs: {log_dir}")

    if args.all:
        shutil.rmtree(ws)
        print(f"Removed workspace: {ws}")
    else:
        print(f"Workspace preserved at: {ws}")
        print("  Use --all to remove the entire workspace.")

    return 0


def _load_workspace_env(workspace: str) -> Dict[str, str]:
    """If `<workspace>/.env` exists, read it and apply to `os.environ` —
    but **only** for keys not already set (so `FOO=bar herd monitor`
    still overrides a `FOO=baz` line in the file).

    Per-workspace `.env` files are how users pin sweep-specific env
    vars (DISCORD_BOT_TOKEN, ANTHROPIC_API_KEY, EXPERIMENT, WANDB_*,
    cluster module settings, etc.) without retyping them on every
    daemon launch and without leaking them into committed YAML.

    Format:

        # comments are skipped
        FOO=bar
        export QUX=quux        # leading `export ` stripped
        TOKEN="value with spaces"   # matched quotes stripped

    Returns the dict of keys actually applied (for the startup log).
    Values are NOT echoed — they're often secrets.
    """
    path = os.path.join(workspace, ".env")
    if not os.path.isfile(path):
        return {}
    loaded: Dict[str, str] = {}
    try:
        with open(path) as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].lstrip()
                k, sep, v = line.partition("=")
                if not sep:
                    continue  # malformed line; skip silently
                k = k.strip()
                v = v.strip()
                if (
                    (v.startswith('"') and v.endswith('"'))
                    or (v.startswith("'") and v.endswith("'"))
                ):
                    v = v[1:-1]
                if k and k not in os.environ:
                    os.environ[k] = v
                    loaded[k] = v
    except OSError as e:
        print(f"Warning: couldn't read {path}: {e}", file=sys.stderr)
    return loaded


def _apply_workspace_env(workspace: str) -> Dict[str, str]:
    """Apply `<workspace>/.env` to os.environ and log what was loaded.

    Shared by `herd run`, `herd test`, and `herd monitor` so the same
    `.env` controls every entry point that touches the workspace
    (trainer-side secrets like WANDB_API_KEY, daemon-side ones like
    DISCORD_BOT_TOKEN). The log goes to stderr so it doesn't corrupt
    `--json` stdout.
    """
    loaded = _load_workspace_env(workspace)
    if loaded:
        keys = ", ".join(sorted(loaded.keys()))
        print(
            f"Loaded {len(loaded)} env var(s) from {workspace}/.env: {keys}",
            file=sys.stderr,
        )
    return loaded


def _monitor_preflight(workspace: str) -> bool:
    """Print a startup checklist to stderr and hard-fail on
    configured-but-broken cases. Returns False if the daemon should not
    start.

    The intent is to bark loudly when the user has wired something up
    in `hyperherd.yaml` but the matching env var or extras aren't in
    place — silent fall-through to "no Discord, no chat surface" is
    almost never what they wanted."""
    try:
        config = load_config(workspace)
    except Exception as e:
        print(f"Could not load config: {e}", file=sys.stderr)
        return False

    print("herd monitor — startup check:", file=sys.stderr)

    # Auth: env-var or subscription OAuth — both work, just informational.
    if os.environ.get("ANTHROPIC_API_KEY"):
        print("  [✓] ANTHROPIC_API_KEY set (API console billing)",
              file=sys.stderr)
    else:
        print("  [·] ANTHROPIC_API_KEY unset — falling back to "
              "claude /login subscription credentials",
              file=sys.stderr)

    # Discord: configured-but-broken is a hard fail.
    fatal = []
    if config.discord.guild_id:
        if os.environ.get("DISCORD_BOT_TOKEN"):
            print(f"  [✓] Discord channel configured "
                  f"(guild_id={config.discord.guild_id})", file=sys.stderr)
        else:
            print(f"  [✗] Discord guild_id is set in hyperherd.yaml "
                  f"but DISCORD_BOT_TOKEN is not in the environment.",
                  file=sys.stderr)
            print(f"      Either:", file=sys.stderr)
            print(f"        export DISCORD_BOT_TOKEN=...   "
                  f"(or `source .env`)", file=sys.stderr)
            print(f"        OR remove the `discord:` block from "
                  f"hyperherd.yaml.", file=sys.stderr)
            fatal.append("missing DISCORD_BOT_TOKEN")
    else:
        print("  [·] No `discord:` block in hyperherd.yaml — daemon "
              "will run with no chat surface.", file=sys.stderr)

    # External MCP servers: list them; warn (not fatal) on env-var refs
    # that don't resolve, since the agent can still get useful work
    # done without one MCP among many.
    if config.mcp_servers:
        for server in config.mcp_servers:
            unresolved = [
                k for k, v in server.env.items()
                if "${" in v and not _expand_env_check(v)
            ]
            if unresolved:
                print(f"  [!] MCP server '{server.name}' references "
                      f"unset env var(s): {', '.join(unresolved)}",
                      file=sys.stderr)
            else:
                print(f"  [✓] MCP server: {server.name}", file=sys.stderr)

    if fatal:
        print(f"\nRefusing to start: {'; '.join(fatal)}.", file=sys.stderr)
        return False

    print("", file=sys.stderr)  # blank line before the loop's logs
    return True


def _expand_env_check(value: str) -> bool:
    """Return True if every `${VAR}` reference in `value` resolves to a
    non-empty env var. Used for the preflight check, not the actual
    expansion path."""
    import re
    for match in re.finditer(r"\$\{([A-Z_][A-Z0-9_]*)\}", value):
        if not os.environ.get(match.group(1)):
            return False
    return True


def cmd_monitor(args):
    """Run the autonomous monitor daemon.

    Connects to Discord (if configured), runs the boot interview, and
    operates the sweep until it halts. Supports `--once` (run a single
    tick and exit) and `--dry-run` (assemble state and render the prompt
    without calling the model, for verifying the deterministic path).
    """
    import asyncio

    workspace = args.workspace
    if not manifest.workspace_exists(workspace):
        # Auto-initialize so the cold-start UX is seamless: drop a
        # hyperherd.yaml, `herd monitor`, daemon launches, and the
        # agent runs the boot interview against a freshly-materialized
        # greenfield workspace. Equivalent to `herd run --dry-run`.
        print(f"No workspace at {workspace} — initializing manifest "
              f"(equivalent to `herd run --dry-run`).", file=sys.stderr)
        proc = subprocess.run(
            [sys.executable, "-m", "hyperherd.cli", "run", "--dry-run", workspace],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(f"Workspace init failed:\n{proc.stderr}", file=sys.stderr)
            return 1

    if args.dry_run:
        from hyperherd.monitor_agent import tick as tick_mod
        try:
            result = tick_mod.dry_run(workspace, trigger=args.trigger)
        except Exception as e:
            print(f"Dry run failed: {e}", file=sys.stderr)
            return 1
        print("=== System prompt ({} chars) ===".format(result["system_prompt_chars"]))
        print("(skill markdown — not displayed; cached after first tick)")
        print()
        print("=== TickState ===")
        print(json.dumps(result["state"], indent=2, default=str))
        print()
        print("=== User message (the per-tick turn) ===")
        print(result["user_message"])
        return 0

    # Auto-load <workspace>/.env BEFORE the preflight so a token
    # configured in the file is visible to the preflight's check.
    # CLI-prefixed env vars (`FOO=bar herd monitor`) still win because
    # the loader only fills in keys not already set.
    _apply_workspace_env(workspace)

    # Live paths only — preflight the env so misconfigurations fail
    # before the daemon spends API tokens or sits silently with no
    # chat surface.
    if not _monitor_preflight(workspace):
        return 1

    if args.once:
        from hyperherd.monitor_agent import tick as tick_mod
        try:
            result = asyncio.run(tick_mod.run_tick(workspace, trigger=args.trigger))
        except RuntimeError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1

        print(f"Tick complete. cost=${result.cost_usd:.4f} turns={result.turns}")
        if result.halted:
            print(f"Agent halted: {result.halt_reason or '(no reason)'}")
        elif result.next_delay_seconds is not None:
            print(f"Next tick in {result.next_delay_seconds}s")
        return 0

    # Daemon mode: loop until agent halts or signal arrives.
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    from hyperherd.monitor_agent import daemon as daemon_mod
    try:
        outcome = asyncio.run(
            daemon_mod.run_daemon(
                workspace,
                max_ticks=args.max_ticks,
                agent_enabled=not args.no_agent,
            )
        )
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(
        f"Daemon stopped. ticks={outcome.ticks} "
        f"total_cost=${outcome.total_cost_usd:.4f} "
        f"halted={outcome.halted} signaled={outcome.stopped_by_signal}"
    )
    if outcome.halted:
        print(f"Agent halted: {outcome.halt_reason or '(no reason)'}")
    return 0


def cmd_snapshot(args):
    """Bundle status + sacct + logged metrics + per-trial last-log + recent
    failed-trial stderr into a single JSON document.

    Designed for an agent loop: one CLI call per tick instead of three or
    four (status / stats / res / tail). Emits JSON unconditionally — the
    command has no human-formatted form."""
    config = load_config(args.workspace)
    if not manifest.workspace_exists(config.workspace):
        print("No workspace found. Run 'herd run' first.", file=sys.stderr)
        return 1

    _sync_slurm_status(config.workspace)
    trials = manifest.load_manifest(config.workspace)
    job_records = manifest.get_job_ids(config.workspace)

    # sacct accounting is best-effort: missing binary, parse failure, or no
    # submitted jobs all leave stats_by_idx empty rather than aborting the
    # snapshot. The agent can detect the empty dict and decide what to do.
    stats_by_idx: dict = {}
    job_id_by_idx: dict = {}
    if job_records:
        try:
            all_stats = slurm.query_job_stats([r["slurm_job_id"] for r in job_records])
        except FileNotFoundError:
            all_stats = {}
        except Exception as e:
            print(f"Warning: stats query failed: {e}", file=sys.stderr)
            all_stats = {}
        for record in job_records:
            jid = record["slurm_job_id"]
            for idx in record.get("indices", []):
                s = all_stats.get((jid, idx))
                if s is not None:
                    stats_by_idx[idx] = s
                # Most recent submission wins — record_job_submission appends.
                job_id_by_idx[idx] = jid

    metrics_by_idx = load_all_results(config.workspace)

    log_tails = {
        t["index"]: slurm.get_log_tail(config.workspace, t["index"])
        for t in trials
    }

    failed_stderr: dict = {}
    failed_indices = [t["index"] for t in trials if t.get("status") == "failed"]
    failed_indices = failed_indices[-args.max_failed:]
    log_dir = manifest.logs_path(config.workspace)
    for idx in failed_indices:
        err_file = os.path.join(log_dir, f"{idx}.err")
        if not os.path.isfile(err_file):
            failed_stderr[idx] = {"path": err_file, "lines": None, "truncated": False}
            continue
        try:
            with open(err_file, "r", errors="replace") as f:
                all_lines = f.readlines()
        except OSError:
            failed_stderr[idx] = {"path": err_file, "lines": None, "truncated": False}
            continue
        truncated = len(all_lines) > args.lines
        tail = all_lines[-args.lines:] if truncated else all_lines
        failed_stderr[idx] = {
            "path": err_file,
            "lines": [ln.rstrip("\n") for ln in tail],
            "truncated": truncated,
        }

    agent_output.emit(agent_output.snapshot_payload(
        sweep_name=config.name,
        workspace_path=os.path.abspath(config.workspace),
        trials=trials,
        stats_by_idx=stats_by_idx,
        metrics_by_idx=metrics_by_idx,
        log_tails=log_tails,
        failed_stderr=failed_stderr,
        job_id_by_idx=job_id_by_idx,
    ))
    return 0


def cmd_init(args):
    """Scaffold a new hyperherd.yaml and launch.sh."""
    directory = args.directory or "."
    try:
        config_path, launcher_path = scaffold(
            directory=directory,
            overwrite=args.force,
            from_config=args.config,
            from_launcher=args.launcher,
        )
    except (FileExistsError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Created {config_path}")
    print(f"Created {launcher_path}")
    print()
    print("Next steps:")
    print("  1. Edit hyperherd.yaml to define your parameters and SLURM resources")
    print("  2. Edit launch.sh to set up your container/environment")
    print(f"  3. Run: herd run {directory} --dry-run")
    return 0


_DOG_ASCII = r"""
        __
   (___()'`;       Woof! Ready to herd some hyperparameters.
   /,    /`
   \\"--\\
"""


def cmd_dog(args):
    print(_DOG_ASCII)
    return 0


def _packaged_skills_dir() -> str:
    return os.path.join(os.path.dirname(__file__), "skills")


def _list_packaged_skills() -> list[tuple[str, str]]:
    """Return [(name, src_path), ...] for every `<pkg>/skills/<name>/SKILL.md`."""
    base = _packaged_skills_dir()
    if not os.path.isdir(base):
        return []
    out: list[tuple[str, str]] = []
    for name in sorted(os.listdir(base)):
        src = os.path.join(base, name, "SKILL.md")
        if os.path.isfile(src):
            out.append((name, src))
    return out


def cmd_install_skill(args):
    """Install bundled Claude Code skills into ~/.claude/skills/ (or
    ./.claude/skills/ with `--scope project`).

    By default installs all skills under `hyperherd/skills/*/`. Use
    `--name X` to install just one (matches the directory name)."""
    skills = _list_packaged_skills()
    if not skills:
        print(
            f"Error: no packaged skills found under {_packaged_skills_dir()}.\n"
            "This usually means the install is broken — try `pip install -e .` "
            "from a checkout, or reinstall the package.",
            file=sys.stderr,
        )
        return 1

    if args.name:
        skills = [(n, p) for n, p in skills if n == args.name]
        if not skills:
            print(
                f"Error: no skill named {args.name!r}. "
                f"Available: {', '.join(n for n, _ in _list_packaged_skills())}",
                file=sys.stderr,
            )
            return 1

    if args.scope == "project":
        base = os.path.abspath(".claude/skills")
    else:
        base = os.path.expanduser("~/.claude/skills")

    failures = 0
    for name, src in skills:
        dest_dir = os.path.join(base, name)
        dest = os.path.join(dest_dir, "SKILL.md")

        if os.path.exists(dest):
            if os.path.realpath(dest) == os.path.realpath(src):
                print(f"  {name}: already linked to packaged source — skipping.")
                continue
            if not args.force:
                print(
                    f"  {name}: already installed at {dest}. Use --force to overwrite.",
                    file=sys.stderr,
                )
                failures += 1
                continue

        os.makedirs(dest_dir, exist_ok=True)
        shutil.copyfile(src, dest)
        print(f"  {name}: installed to {dest}")

    if failures:
        return 1
    return 0


def _sync_slurm_status(workspace: str):
    """Sync trial statuses from SLURM job tracking data."""
    job_records = manifest.get_job_ids(workspace)
    if not job_records:
        return

    job_ids = [r["slurm_job_id"] for r in job_records]

    try:
        statuses = slurm.query_job_status(job_ids)
    except FileNotFoundError:
        # sacct/squeue not on PATH (e.g., local development without SLURM).
        return
    except Exception as e:
        # Surface parse/logic bugs instead of leaving statuses silently stale.
        print(f"Warning: failed to sync SLURM status: {e}", file=sys.stderr)
        return

    # Map SLURM states to our status values
    state_map = {
        "COMPLETED": "completed",
        "RUNNING": "running",
        "PENDING": "queued",     # SLURM queued (waiting to start) != "ready" (never submitted)
        "FAILED": "failed",
        "CANCELLED": "cancelled",
        "TIMEOUT": "failed",
        "NODE_FAIL": "failed",
        "OUT_OF_MEMORY": "failed",
    }

    # Trials marked `pruned` are algorithmic decisions made by the
    # autonomous monitor; they're typically also CANCELLED in SLURM
    # (because the agent calls `scancel`), but we must not let SLURM's
    # status sync flip them back to `cancelled`. The `pruned` label is
    # the source of truth for those trials.
    trials = manifest.load_manifest(workspace)
    pruned_set = {t["index"] for t in trials if t.get("status") == "pruned"}

    updates = {}
    for (_, array_idx), slurm_state in statuses.items():
        if array_idx in pruned_set:
            continue
        our_status = state_map.get(slurm_state, slurm_state.lower())
        updates[array_idx] = our_status

    manifest.bulk_update_status(workspace, updates)


def main():
    parser = argparse.ArgumentParser(
        prog="herd",
        description="Launch and monitor hyperparameter optimization job arrays on SLURM.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --json flag, attached via parent to commands an agent typically drives.
    # The agent flag is a small surface (one parser parent) so adding a new
    # JSON-capable command is a one-liner: `parents=[json_parent]`.
    json_parent = argparse.ArgumentParser(add_help=False)
    json_parent.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help=(
            "Emit machine-readable JSON to stdout instead of the human table. "
            "Memory in bytes, elapsed in seconds, status as a stable enum. "
            "Errors still go to stderr with a non-zero exit code."
        ),
    )

    # init
    p_init = subparsers.add_parser("init", help="Scaffold a new config and launcher script")
    p_init.add_argument("directory", nargs="?", default=".", help="Directory to create files in (default: current)")
    p_init.add_argument("--config", default=None, help="Copy this file as hyperherd.yaml instead of generating a template")
    p_init.add_argument("--launcher", default=None, help="Copy this file as launch.sh instead of generating a template")
    p_init.add_argument("-f", "--force", action="store_true", help="Overwrite existing files")

    # launch
    p_launch = subparsers.add_parser("run", help="Submit a hyperparameter sweep", parents=[json_parent])
    p_launch.add_argument("workspace", nargs="?", default=".", help="Workspace directory (default: current dir)")
    p_launch.add_argument(
        "-n", "--dry-run", action="store_true",
        help="Print the sbatch script and trial list without submitting"
    )
    p_launch.add_argument(
        "-j", "--max-concurrent", type=int, default=None,
        help="Cap concurrent running array tasks (overrides slurm.max_concurrent)"
    )
    p_launch.add_argument(
        "-i", "--indices", default=None,
        help="Submit only these trial indices (SLURM-style spec, e.g. '0-3,5,7-9')"
    )
    p_launch.add_argument(
        "-p", "--pin", nargs="+", default=None, metavar="NAME=VALUE",
        help=(
            "Only submit trials whose swept parameters match every pin "
            "(e.g. `--pin batch_size=32 optimizer=adam`). Names must be "
            "sweep parameters; values are coerced to int/float/str. "
            "Combines with --indices and the standard status filter."
        ),
    )
    p_launch.add_argument(
        "-f", "--force", action="store_true",
        help=(
            "Override safety checks: with --indices, resubmit trials that are "
            "already running or completed; without --indices, allow config "
            "edits that drop running/completed trials (kept as orphans)"
        ),
    )

    # ls — pure trial listing (status-agnostic), works pre- or post-`herd run`
    p_ls = subparsers.add_parser(
        "ls",
        help="List every trial in the sweep with its swept parameters",
    )
    p_ls.add_argument(
        "workspace", nargs="?", default=".",
        help="Workspace directory (default: current dir)",
    )
    p_ls.add_argument(
        "-p", "--pin", nargs="+", default=None, metavar="NAME=VALUE",
        help=(
            "Filter to trials whose swept params match every pin "
            "(e.g. `--pin batch_size=32 optimizer=adam`)"
        ),
    )

    # monitor
    p_monitor = subparsers.add_parser("status", help="Show status of all trials", parents=[json_parent])
    p_monitor.add_argument("workspace", nargs="?", default=".", help="Workspace directory (default: current dir)")

    # stats — SLURM accounting (sacct)
    p_stats = subparsers.add_parser("stats", help="Print runtime/memory accounting from sacct", parents=[json_parent])
    p_stats.add_argument("workspace", nargs="?", default=".", help="Workspace directory (default: current dir)")
    p_stats.add_argument("index", nargs="?", type=int, default=None, help="Trial index (omit to show every trial)")

    # test — run a single trial locally (with optional Hydra --cfg job validation)
    p_test = subparsers.add_parser(
        "test",
        help="Run a single trial locally via the launcher (no SLURM)",
    )
    p_test.add_argument("workspace", nargs="?", default=".", help="Workspace directory (default: current dir)")
    p_test.add_argument("index", nargs="?", type=int, default=0, help="Trial index to run (default: 0)")
    p_test.add_argument(
        "--cfg-job",
        action="store_true",
        help=(
            "Append `--cfg job` to the override string. For Hydra trainers this "
            "prints the resolved config and exits without running training "
            "(safe to use on indices already submitted to SLURM)."
        ),
    )

    # tail
    p_tail = subparsers.add_parser("tail", help="Print last N lines of a trial's log", parents=[json_parent])
    p_tail.add_argument("workspace", nargs="?", default=".", help="Workspace directory (default: current dir)")
    p_tail.add_argument("index", type=int, help="Trial index to tail")
    p_tail.add_argument("-n", "--lines", type=int, default=20, help="Number of lines to show (default: 20)")
    p_tail_streams = p_tail.add_mutually_exclusive_group()
    p_tail_streams.add_argument(
        "--stdout", dest="stream", action="store_const", const="stdout",
        help="Show only stdout (.out)",
    )
    p_tail_streams.add_argument(
        "--stderr", dest="stream", action="store_const", const="stderr",
        help="Show only stderr (.err)",
    )

    # results
    p_results = subparsers.add_parser("res", help="Print TSV of trial parameters and logged metrics", parents=[json_parent])
    p_results.add_argument("workspace", nargs="?", default=".", help="Workspace directory (default: current dir)")

    # stop
    p_stop = subparsers.add_parser("stop", help="Cancel a running/queued trial (or all of them)", parents=[json_parent])
    p_stop.add_argument("workspace", nargs="?", default=".", help="Workspace directory (default: current dir)")
    p_stop.add_argument("index", nargs="?", type=int, default=None, help="Trial index to cancel")
    p_stop.add_argument("-a", "--all", action="store_true", help="Cancel every running/queued trial in the workspace")

    # monitor — autonomous monitor daemon (Claude Agent SDK + Discord)
    p_monitor = subparsers.add_parser(
        "monitor",
        help="Run the autonomous monitor daemon",
    )
    p_monitor.add_argument(
        "workspace", nargs="?", default=".",
        help="Workspace directory (default: current dir)",
    )
    p_monitor.add_argument(
        "--once", action="store_true",
        help="Run exactly one tick and exit (live — calls the model)",
    )
    p_monitor.add_argument(
        "--dry-run", action="store_true",
        help="Assemble state + render the prompt; no model call. Use this to verify the deterministic path before paying for tokens.",
    )
    p_monitor.add_argument(
        "--trigger", default="boot",
        choices=["scheduled", "failure", "completion", "user_message", "boot"],
        help="Trigger for --once / --dry-run (daemon mode picks its own)",
    )
    p_monitor.add_argument(
        "--max-ticks", type=int, default=None,
        help="Stop the daemon after N ticks (safety cap for testing)",
    )
    p_monitor.add_argument(
        "--no-agent", action="store_true",
        help=(
            "Passive mode: keep the chat channel + dashboard + event posts "
            "alive but skip the agent loop entirely (no model calls, no "
            "token spend). Useful for very long sweeps where the daemon's "
            "transport surface is enough."
        ),
    )

    # snapshot — bundled JSON for agent loops (status + stats + failed stderr)
    p_snapshot = subparsers.add_parser(
        "snapshot",
        help="Bundled JSON snapshot (status + sacct + metrics + last-log + failed stderr) for agents",
    )
    p_snapshot.add_argument(
        "workspace", nargs="?", default=".",
        help="Workspace directory (default: current dir)",
    )
    p_snapshot.add_argument(
        "-n", "--lines", type=int, default=20,
        help="Max stderr lines per failed trial (default: 20)",
    )
    p_snapshot.add_argument(
        "--max-failed", type=int, default=20,
        help="Max number of failed trials to include stderr for (default: 20)",
    )

    # clean
    p_clean = subparsers.add_parser("clean", help="Cancel jobs and clean up workspace")
    p_clean.add_argument("workspace", nargs="?", default=".", help="Workspace directory (default: current dir)")
    p_clean.add_argument("-l", "--logs", action="store_true", help="Remove log files")
    p_clean.add_argument("-a", "--all", action="store_true", help="Remove entire .hyperherd state")

    # install-skill
    p_skill = subparsers.add_parser(
        "install-skill",
        help="Install the bundled hyperherd-config Claude Code skill",
    )
    p_skill.add_argument(
        "--scope",
        choices=("user", "project"),
        default="user",
        help="user: ~/.claude/skills (default); project: ./.claude/skills",
    )
    p_skill.add_argument(
        "--name", default=None,
        help="Install only this skill (default: install all bundled skills)",
    )
    p_skill.add_argument(
        "-f", "--force", action="store_true", help="Overwrite an existing install",
    )

    # dog — easter egg, hidden from --help.
    subparsers.add_parser("dog", help=argparse.SUPPRESS)

    args = parser.parse_args()

    # Disambiguate `<workspace?> <index?>` positionals when only one was given.
    # `herd stop 5` from inside a workspace binds "5" to `workspace` because it
    # comes first; if it looks like an int and isn't a directory, treat it as
    # `index` instead. Affects: stop, stats, test.
    if (
        getattr(args, "index", "<missing>") is None
        and isinstance(getattr(args, "workspace", None), str)
        and not os.path.isdir(args.workspace)
    ):
        try:
            args.index = int(args.workspace)
            args.workspace = "."
        except ValueError:
            pass

    handlers = {
        "init": cmd_init,
        "run": cmd_launch,
        "ls": cmd_ls,
        "test": cmd_test,
        "status": cmd_status,
        "stats": cmd_stats,
        "tail": cmd_tail,
        "res": cmd_results,
        "stop": cmd_stop,
        "clean": cmd_clean,
        "monitor": cmd_monitor,
        "snapshot": cmd_snapshot,
        "install-skill": cmd_install_skill,
        "dog": cmd_dog,
    }

    try:
        rc = handlers[args.command](args)
    except (ConfigError, PreflightError, FileNotFoundError, FileExistsError) as e:
        print(f"Error: {e}", file=sys.stderr)
        if os.environ.get("HYPERHERD_DEBUG"):
            raise
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
