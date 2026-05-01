"""HyperHerd CLI: launch, monitor, and clean SLURM hyperparameter job arrays."""

import argparse
import os
import shutil
import sys

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
        print(f"Existing workspace found with {len(existing)} trials.")
        # Refresh status from SLURM before deciding what to resubmit
        _sync_slurm_status(config.workspace)
        existing = manifest.load_manifest(config.workspace)

        diff = manifest.reconcile_manifest(existing, combinations)
        if not _apply_reconciliation(config, diff, args.force):
            return 1

        trials = manifest.load_manifest(config.workspace)
        pending = manifest.get_pending_indices(config.workspace)
        if not pending and not args.indices:
            print("All trials are completed or currently running. Nothing to submit.")
            return 0
        if not args.indices:
            print(f"  {len(pending)} trials need (re)submission.")
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
        print(f"  Submitting {len(pending)} requested trial(s): {args.indices}")

    # Generate sbatch script
    script = slurm.generate_sbatch_script(config, pending, args.max_concurrent)

    if args.dry_run:
        print_dry_run(trials, script, defaults=config.defaults)
        return 0

    # Submit
    print(f"Submitting {len(pending)} trials as SLURM job array...")
    job_id = slurm.submit_job(config, script, dry_run=False)
    assert job_id is not None

    # Record submission and update statuses
    manifest.record_job_submission(config.workspace, job_id, pending)
    manifest.bulk_update_status(config.workspace, {i: "submitted" for i in pending})

    print_launch_success(
        job_id=job_id,
        n_trials=len(pending),
        workspace=manifest.workspace_path(config.workspace),
        logs=manifest.logs_path(config.workspace),
    )
    return 0


def cmd_monitor(args):
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
    print_stats_table([(idx, trial_by_idx[idx], by_index[idx])])
    return 0


def cmd_tail(args):
    """Print the last N lines of a trial's log files (stdout and stderr)."""
    config = load_config(args.workspace)

    if not manifest.workspace_exists(config.workspace):
        print("No workspace found. Run 'hyperherd launch' first.", file=sys.stderr)
        return 1

    index = args.index
    lines = args.lines
    log_dir = manifest.logs_path(config.workspace)

    out_file = os.path.join(log_dir, f"{index}.out")
    err_file = os.path.join(log_dir, f"{index}.err")

    has_out = os.path.isfile(out_file)
    has_err = os.path.isfile(err_file)

    if not has_out and not has_err:
        print(f"No log files found for trial {index}", file=sys.stderr)
        return 1

    # Print trial info header
    trials = manifest.load_manifest(config.workspace)
    for t in trials:
        if t["index"] == index:
            exp_name = t.get("experiment_name", "")
            status = t.get("status", "unknown")
            print(f"{_DIM}Trial {index} [{status}] {exp_name}{_RESET}")
            print(f"{_DIM}{'-' * 60}{_RESET}")
            break

    for log_file, label in [(out_file, "stdout"), (err_file, "stderr")]:
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
        if not targets:
            print("No live trials to cancel.")
            return 0
        cancelled = []
        for t in targets:
            jid = _latest_job_id_for(records, t["index"])
            if jid is None:
                continue
            slurm.cancel_array_task(jid, t["index"])
            cancelled.append(t["index"])
        if cancelled:
            manifest.bulk_update_status(
                config.workspace, {i: "cancelled" for i in cancelled}
            )
        print(f"Cancelled {len(cancelled)} trial(s): {sorted(cancelled)}")
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

    print(f"Cancelling trial {index} (job {job_id}_{index})...")
    slurm.cancel_array_task(job_id, index)
    manifest.update_trial_status(config.workspace, index, "cancelled")
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


_SKILL_NAME = "hyperherd-config"


def _packaged_skill_path() -> str:
    return os.path.join(os.path.dirname(__file__), "skill", "SKILL.md")


def cmd_install_skill(args):
    """Install the hyperherd-config Claude Code skill into ~/.claude/skills/."""
    src = _packaged_skill_path()
    if not os.path.isfile(src):
        print(
            f"Error: packaged skill not found at {src}.\n"
            "This usually means the install is broken — try `pip install -e .` "
            "from a checkout, or reinstall the package.",
            file=sys.stderr,
        )
        return 1

    if args.scope == "project":
        base = os.path.abspath(".claude/skills")
    else:
        base = os.path.expanduser("~/.claude/skills")

    dest_dir = os.path.join(base, _SKILL_NAME)
    dest = os.path.join(dest_dir, "SKILL.md")

    if os.path.exists(dest):
        if os.path.realpath(dest) == os.path.realpath(src):
            print(f"Skill already linked to packaged source at {dest} — nothing to do.")
            return 0
        if not args.force:
            print(
                f"Skill already installed at {dest}. Use --force to overwrite.",
                file=sys.stderr,
            )
            return 1

    os.makedirs(dest_dir, exist_ok=True)
    shutil.copyfile(src, dest)
    print(f"Installed {_SKILL_NAME} skill to {dest}")
    print(
        "Open a new Claude Code session in any directory to use it — invoke by "
        "asking Claude to author or edit a hyperherd.yaml."
    )
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

    # init
    p_init = subparsers.add_parser("init", help="Scaffold a new config and launcher script")
    p_init.add_argument("directory", nargs="?", default=".", help="Directory to create files in (default: current)")
    p_init.add_argument("--config", default=None, help="Copy this file as hyperherd.yaml instead of generating a template")
    p_init.add_argument("--launcher", default=None, help="Copy this file as launch.sh instead of generating a template")
    p_init.add_argument("-f", "--force", action="store_true", help="Overwrite existing files")

    # launch
    p_launch = subparsers.add_parser("run", help="Submit a hyperparameter sweep")
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
    p_monitor = subparsers.add_parser("status", help="Show status of all trials")
    p_monitor.add_argument("workspace", nargs="?", default=".", help="Workspace directory (default: current dir)")

    # stats — SLURM accounting (sacct)
    p_stats = subparsers.add_parser("stats", help="Print runtime/memory accounting from sacct")
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
    p_tail = subparsers.add_parser("tail", help="Print last N lines of a trial's log")
    p_tail.add_argument("workspace", nargs="?", default=".", help="Workspace directory (default: current dir)")
    p_tail.add_argument("index", type=int, help="Trial index to tail")
    p_tail.add_argument("-n", "--lines", type=int, default=20, help="Number of lines to show (default: 20)")

    # results
    p_results = subparsers.add_parser("res", help="Print TSV of trial parameters and logged metrics")
    p_results.add_argument("workspace", nargs="?", default=".", help="Workspace directory (default: current dir)")

    # stop
    p_stop = subparsers.add_parser("stop", help="Cancel a running/queued trial (or all of them)")
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
        help="Install the hyperherd-config Claude Code skill",
    )
    p_skill.add_argument(
        "--scope",
        choices=("user", "project"),
        default="user",
        help="user: ~/.claude/skills (default); project: ./.claude/skills",
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
        "status": cmd_monitor,
        "stats": cmd_stats,
        "tail": cmd_tail,
        "res": cmd_results,
        "stop": cmd_stop,
        "clean": cmd_clean,
        "watch": cmd_watch,
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
