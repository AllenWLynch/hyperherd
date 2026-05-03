"""HyperHerd CLI: launch, monitor, and clean SLURM hyperparameter job arrays."""

import argparse
import json
import os
import shutil
import subprocess
import sys

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


def cmd_launch(args):
    """Launch (or re-launch) a hyperparameter sweep as a SLURM job array."""
    config = load_config(args.workspace)
    json_mode = getattr(args, "json_output", False)

    def _say(*a, **kw):
        if not json_mode:
            print(*a, **kw)

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
        print_dry_run(trials, script, defaults=config.defaults)
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
    env["HYPERHERD_TRIAL_ID"] = str(index)
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


def cmd_watch(args):
    """Poll the manifest and post trial state changes to a webhook.

    Settings come from the `watch:` block in hyperherd.yaml (webhook URL,
    format, interval, heartbeat, events). Foreground process — wrap in
    `nohup`, `tmux`, or `screen` to outlive your shell.
    """
    from hyperherd import watch

    config = load_config(args.workspace)

    if not manifest.workspace_exists(config.workspace):
        print("No workspace found. Run 'herd run' first.", file=sys.stderr)
        return 1

    try:
        watch.run(config, once=args.once, pidfile=args.pidfile)
    except watch.WatchError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


_MONITOR_INITIAL_PROMPT_TEMPLATE = (
    "Use the hyperherd-monitor skill to manage the sweep at {workspace}.\n"
    "Run the setup interview, kick off the phased rollout, then `/loop` "
    "(no interval — self-pace via ScheduleWakeup per the skill's cadence "
    "selection table; tight during rollout, long when the sweep is stable)."
)


# Allow-rules `herd monitor` writes to <workspace>/.claude/settings.local.json
# so the agent's per-tick toolbox doesn't trigger a permission prompt every
# few minutes. Scope is the agent's full toolbox by design — herd
# subcommands, JSON-extraction helpers (jq + python the agent will inevitably
# reach for), and files inside this workspace.
_MONITOR_ALLOW_RULES = [
    # First-class HyperHerd surface
    "Bash(herd *)",
    "Bash(jq *)",
    # Common shell utilities the agent naturally reaches for. Erring broad
    # here on purpose — the alternative is the agent stalling on a permission
    # prompt mid-tick because it pipelined through `tee` or `head`. The skill
    # tells the agent that off-list commands require a `herd msg` warning to
    # the user first so the session never blocks silently.
    "Bash(cat *)",
    "Bash(head *)",
    "Bash(tail *)",
    "Bash(tee *)",
    "Bash(grep *)",
    "Bash(sed *)",
    "Bash(awk *)",
    "Bash(cut *)",
    "Bash(tr *)",
    "Bash(sort *)",
    "Bash(uniq *)",
    "Bash(wc *)",
    "Bash(ls *)",
    "Bash(find *)",
    "Bash(echo *)",
    "Bash(printf *)",
    "Bash(date *)",
    "Bash(diff *)",
    "Bash(comm *)",
    "Bash(mkdir *)",
    "Bash(cp *)",
    "Bash(mv *)",
    "Bash(touch *)",
    "Bash(test *)",
    # Workspace files
    "Edit(/hyperherd.yaml)",
    "Read(.hyperherd/**)",
    "Write(.hyperherd/**)",
    "Edit(.hyperherd/**)",
]


def _ensure_monitor_settings(workspace: str) -> tuple[str, list[str]]:
    """Merge `_MONITOR_ALLOW_RULES` into `<workspace>/.claude/settings.local.json`.

    Returns (settings_path, newly_added_rules). Existing rules are preserved
    by string-equality dedup; the rest of the file (other keys, deny lists,
    etc.) is left alone so this is safe to run on a workspace that already
    has settings.
    """
    settings_dir = os.path.join(workspace, ".claude")
    settings_path = os.path.join(settings_dir, "settings.local.json")
    os.makedirs(settings_dir, exist_ok=True)

    if os.path.isfile(settings_path):
        try:
            with open(settings_path, "r") as f:
                data = json.load(f)
        except (OSError, ValueError):
            # Corrupt / unreadable — back it up rather than overwrite blindly.
            backup = settings_path + ".bak"
            try:
                shutil.copyfile(settings_path, backup)
            except OSError:
                pass
            data = {}
    else:
        data = {}

    perms = data.setdefault("permissions", {})
    allow = perms.setdefault("allow", [])
    if not isinstance(allow, list):
        allow = []
        perms["allow"] = allow

    added = []
    for rule in _MONITOR_ALLOW_RULES:
        if rule not in allow:
            allow.append(rule)
            added.append(rule)

    with open(settings_path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    return settings_path, added


def _spawn_watch_background(workspace: str) -> tuple[int, str, str]:
    """Spawn `herd watch` as a background daemon.

    Uses the same nohup+pidfile recipe documented for the standalone command.
    Returns (pid, pidfile_path, log_path). Raises FileExistsError if a watch
    is already running for this workspace.
    """
    import subprocess

    ws_dir = manifest.workspace_path(workspace)
    pidfile = os.path.join(ws_dir, "watch.pid")
    log_path = os.path.join(ws_dir, "watch.log")

    if os.path.isfile(pidfile):
        try:
            with open(pidfile) as f:
                stale_pid = int(f.read().strip())
            os.kill(stale_pid, 0)  # signal 0 = "are you alive?"
        except (OSError, ValueError):
            # Stale pidfile (process gone or unreadable). Reclaim it.
            try:
                os.unlink(pidfile)
            except OSError:
                pass
        else:
            raise FileExistsError(
                f"watch already running (PID {stale_pid}, pidfile {pidfile})"
            )

    os.makedirs(ws_dir, exist_ok=True)
    log_fp = open(log_path, "ab")
    proc = subprocess.Popen(
        ["herd", "watch", "--pidfile", pidfile, workspace],
        stdout=log_fp,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,  # detach from this controlling terminal
    )
    log_fp.close()
    return proc.pid, pidfile, log_path


def cmd_monitor(args):
    """Start the agent-driven sweep monitor.

    Spawns `herd watch` in the background (unless `--no-watch`), prints the
    initial prompt the user should paste into Claude Code, and then `exec`s
    `claude` so the user lands directly in an interactive session. The user
    wraps the whole thing in tmux/screen if they want it to outlive their
    shell — same recipe as `herd watch`.

    `--stop` is the inverse: kills the background watch (if any) and exits.
    The Claude Code session is the user's terminal, so it dies when they
    close it; this command doesn't try to track or kill claude PIDs.
    """
    config = load_config(args.workspace)

    if args.stop:
        ws_dir = manifest.workspace_path(config.workspace)
        pidfile = os.path.join(ws_dir, "watch.pid")
        if not os.path.isfile(pidfile):
            print("No background watch is running for this workspace.")
            return 0
        try:
            with open(pidfile) as f:
                pid = int(f.read().strip())
            os.kill(pid, 15)  # SIGTERM
            print(f"Sent SIGTERM to watch (PID {pid}).")
        except (OSError, ValueError) as e:
            print(f"Could not stop watch: {e}", file=sys.stderr)
            try:
                os.unlink(pidfile)
            except OSError:
                pass
            return 1
        # The watch daemon removes its own pidfile on shutdown.
        return 0

    if not manifest.workspace_exists(config.workspace):
        print(
            "No workspace found. Run 'herd run' (even just '--dry-run') first "
            "so there's a manifest to monitor.",
            file=sys.stderr,
        )
        return 1

    # Resolve (and persist) the ntfy fallback topic NOW, before spawning
    # watch or exec'ing claude. Without this, watch's subprocess-cold-start
    # delay means the agent's first `herd msg` could race in and mint a
    # different topic — and you'd end up with watch posting to one channel
    # and the agent posting to another.
    if not config.watch.webhook:
        from hyperherd import watch
        url, _ = watch.resolve_default_webhook(config.workspace, config.name)
        print(f"Webhook (ntfy fallback): {url}")
        print("  Subscribe by pasting that URL into the ntfy iOS/Android app,")
        print("  or open it in a browser to watch live.")
        print()

    if not args.no_watch:
        try:
            pid, pidfile, log_path = _spawn_watch_background(config.workspace)
        except FileExistsError as e:
            print(f"Note: {e}. Skipping watch start.", file=sys.stderr)
        else:
            print(f"Started herd watch in background (PID {pid}).")
            print(f"  pidfile: {pidfile}")
            print(f"  log:     {log_path}")
            print(f"  stop:    herd monitor --stop  (or kill $(cat {pidfile}))")
            print()

    if not args.no_auto_allow:
        settings_path, added = _ensure_monitor_settings(config.workspace)
        if added:
            print(f"Added {len(added)} permission rule(s) to {settings_path}:")
            for rule in added:
                print(f"  + {rule}")
            print(
                "These pre-approve the agent's tool calls so it can run "
                "unattended. Pass --no-auto-allow to skip."
            )
        else:
            print(f"Workspace {settings_path} already has the monitor allow-rules.")
        print()

    prompt = _MONITOR_INITIAL_PROMPT_TEMPLATE.format(
        workspace=os.path.abspath(config.workspace),
    )

    # Stash the prompt as an audit trail — useful for debugging what the
    # agent was actually told to do on a given run, even though the user
    # doesn't need to paste it (`claude "<prompt>"` injects it directly).
    prompt_path = os.path.join(manifest.workspace_path(config.workspace),
                               "monitor-prompt.txt")
    try:
        with open(prompt_path, "w") as f:
            f.write(prompt + "\n")
    except OSError:
        prompt_path = None

    print("Starting Claude Code with the monitor agent...")
    print(f"  prompt: {prompt_path}" if prompt_path else "")
    print(
        "  detach: close the terminal (watch keeps running); "
        "wrap in `tmux new -s monitor 'herd monitor'` to keep the session alive."
    )
    print()

    # `claude "<prompt>"` (without -p) starts an interactive session with
    # the prompt already running as the first user turn. -p is one-shot and
    # would defeat the persistent /loop pattern.
    try:
        os.execvp("claude", ["claude", prompt])
    except FileNotFoundError:
        print(
            f"Error: `claude` not on PATH. Install Claude Code "
            f"(https://claude.com/claude-code) or run "
            f"`claude \"$(cat {prompt_path or '<prompt>'})\"` from a host that has it.",
            file=sys.stderr,
        )
        return 1


def cmd_monitor_v2(args):
    """Experimental: agent-SDK-based monitor daemon.

    Phase-1 entrypoint. Supports `--once` (run a single tick and exit) and
    `--dry-run` (assemble state and render the prompt without calling the
    Anthropic API, for verifying the deterministic path).
    """
    import asyncio

    workspace = args.workspace
    if not manifest.workspace_exists(workspace):
        # Auto-initialize so the cold-start UX is seamless: drop a
        # hyperherd.yaml, `herd monitor-v2`, daemon launches, and the
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
            daemon_mod.run_daemon(workspace, max_ticks=args.max_ticks)
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


def cmd_msg(args):
    """Post a free-text message to the watch webhook configured for this
    workspace (or its zero-config ntfy fallback). Lets you announce things
    like 'sweep started' or 'manual restart' alongside the daemon's events.
    """
    from hyperherd import watch

    config = load_config(args.workspace)

    if not manifest.workspace_exists(config.workspace):
        print("No workspace found. Run 'herd run' first.", file=sys.stderr)
        return 1

    text = " ".join(args.message).strip()
    if not text:
        print("Error: empty message.", file=sys.stderr)
        return 1

    webhook = config.watch.webhook
    fmt = config.watch.format
    if not webhook:
        webhook, _ = watch.resolve_default_webhook(config.workspace, config.name)
        fmt = "ntfy"

    try:
        watch.post_message(webhook, fmt, text, config.name)
    except OSError as e:
        print(f"Error: webhook POST failed: {e}", file=sys.stderr)
        return 1

    print(f"Posted to {webhook}")
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

    updates = {}
    for (_, array_idx), slurm_state in statuses.items():
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
        "-f", "--force", action="store_true",
        help=(
            "Override safety checks: with --indices, resubmit trials that are "
            "already running or completed; without --indices, allow config "
            "edits that drop running/completed trials (kept as orphans)"
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

    # watch — polling daemon that posts trial state changes to a webhook
    p_watch = subparsers.add_parser(
        "watch",
        help="Poll trial state and post events to the webhook from hyperherd.yaml",
    )
    p_watch.add_argument(
        "workspace", nargs="?", default=".",
        help="Workspace directory (default: current dir)",
    )
    p_watch.add_argument(
        "--once", action="store_true",
        help="Run a single poll and exit (for cron-driven setups)",
    )
    p_watch.add_argument(
        "--pidfile", default=None,
        help="Write the daemon PID here (for external supervisors / kill scripts)",
    )

    # monitor — start the agent-driven sweep monitor (watch + Claude Code)
    p_monitor = subparsers.add_parser(
        "monitor",
        help="Start the agent-driven sweep monitor (background watch + Claude Code)",
    )
    p_monitor.add_argument(
        "workspace", nargs="?", default=".",
        help="Workspace directory (default: current dir)",
    )
    p_monitor.add_argument(
        "--no-watch", action="store_true",
        help="Don't spawn `herd watch` in the background (assume one is already running)",
    )
    p_monitor.add_argument(
        "--no-auto-allow", action="store_true",
        help=(
            "Don't write monitor allow-rules to <workspace>/.claude/settings.local.json. "
            "Without these rules Claude Code will prompt before each tool call, "
            "which defeats unattended operation."
        ),
    )
    p_monitor.add_argument(
        "--stop", action="store_true",
        help="Stop the background watch for this workspace and exit (does not touch Claude Code sessions)",
    )

    # monitor-v2 — experimental: agent-SDK-based monitor (Phases 1-2 of PLAN.md).
    p_mv2 = subparsers.add_parser(
        "monitor-v2",
        help="(experimental) Agent-SDK-based monitor daemon",
    )
    p_mv2.add_argument(
        "workspace", nargs="?", default=".",
        help="Workspace directory (default: current dir)",
    )
    p_mv2.add_argument(
        "--once", action="store_true",
        help="Run exactly one tick and exit (live — calls the model)",
    )
    p_mv2.add_argument(
        "--dry-run", action="store_true",
        help="Assemble state + render the prompt; no model call. Use this to verify the deterministic path before paying for tokens.",
    )
    p_mv2.add_argument(
        "--trigger", default="boot",
        choices=["scheduled", "failure", "completion", "user_message", "boot"],
        help="Trigger for --once / --dry-run (daemon mode picks its own)",
    )
    p_mv2.add_argument(
        "--max-ticks", type=int, default=None,
        help="Stop the daemon after N ticks (safety cap for testing)",
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

    # msg — post a free-text message to the same webhook
    p_msg = subparsers.add_parser(
        "msg",
        help="Post a free-text message to the watch webhook",
    )
    p_msg.add_argument(
        "-w", "--workspace", default=".",
        help="Workspace directory (default: current dir)",
    )
    p_msg.add_argument(
        "message", nargs="+",
        help="Message text (quote multi-word messages)",
    )

    # clean
    p_clean = subparsers.add_parser("clean", help="Cancel jobs and clean up workspace")
    p_clean.add_argument("workspace", nargs="?", default=".", help="Workspace directory (default: current dir)")
    p_clean.add_argument("-l", "--logs", action="store_true", help="Remove log files")
    p_clean.add_argument("-a", "--all", action="store_true", help="Remove entire .hyperherd state")

    # install-skill
    p_skill = subparsers.add_parser(
        "install-skill",
        help="Install bundled Claude Code skills (hyperherd-config, hyperherd-monitor)",
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
        "test": cmd_test,
        "status": cmd_status,
        "stats": cmd_stats,
        "tail": cmd_tail,
        "res": cmd_results,
        "stop": cmd_stop,
        "clean": cmd_clean,
        "watch": cmd_watch,
        "monitor": cmd_monitor,
        "monitor-v2": cmd_monitor_v2,
        "snapshot": cmd_snapshot,
        "msg": cmd_msg,
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
