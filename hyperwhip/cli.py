"""HyperWhip CLI: launch, monitor, and clean SLURM hyperparameter job arrays."""

import argparse
import json
import os
import shutil
import sys

from hyperwhip.config import ConfigError, load_config
from hyperwhip.constraints import apply_constraints
from hyperwhip.display import print_dry_run, print_status_table, print_summary
from hyperwhip.search import generate_combinations
from hyperwhip import manifest
from hyperwhip import slurm


def cmd_launch(args):
    """Launch (or re-launch) a hyperparameter sweep as a SLURM job array."""
    config = load_config(args.config)

    # Generate parameter combinations
    combinations = generate_combinations(config)
    combinations = apply_constraints(combinations, config.constraints)

    if not combinations:
        print("No valid parameter combinations after applying constraints.", file=sys.stderr)
        return 1

    # Initialize workspace
    manifest.init_workspace(config.workspace)

    # Create or load manifest
    if manifest.workspace_exists(config.workspace):
        existing = manifest.load_manifest(config.workspace)
        if existing:
            print(f"Existing workspace found with {len(existing)} trials.")
            # Refresh status from SLURM before deciding what to resubmit
            _sync_slurm_status(config.workspace)
            pending = manifest.get_pending_indices(config.workspace)
            if not pending:
                print("All trials are completed or currently running. Nothing to submit.")
                return 0
            print(f"  {len(pending)} trials need (re)submission.")
            trials = existing
        else:
            trials = manifest.create_manifest(config.workspace, combinations)
            pending = [t["index"] for t in trials]
    else:
        trials = manifest.create_manifest(config.workspace, combinations)
        pending = [t["index"] for t in trials]

    # Generate sbatch script
    script = slurm.generate_sbatch_script(config, pending)

    if args.dry_run:
        print_dry_run(trials, script)
        return 0

    # Submit
    print(f"Submitting {len(pending)} trials as SLURM job array...")
    job_id = slurm.submit_job(config, script, dry_run=False)
    print(f"Submitted job array: {job_id}")

    # Record submission and update statuses
    manifest.record_job_submission(config.workspace, job_id, pending)
    manifest.bulk_update_status(config.workspace, {i: "submitted" for i in pending})

    print(f"Workspace: {manifest.workspace_path(config.workspace)}")
    print(f"Logs: {manifest.logs_path(config.workspace)}")
    return 0


def cmd_monitor(args):
    """Show the status of all trials in a hyperparameter sweep."""
    config = load_config(args.config)

    if not manifest.workspace_exists(config.workspace):
        print("No workspace found. Run 'hyperwhip launch' first.", file=sys.stderr)
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


def cmd_clean(args):
    """Cancel running jobs and clean up workspace."""
    config = load_config(args.config)
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


def cmd_resolve_overrides(args):
    """Internal subcommand: resolve Hydra overrides for a SLURM array task."""
    manifest_file = args.manifest
    task_id = int(args.task_id)

    # Load manifest directly from the given path
    base = os.path.dirname(os.path.dirname(manifest_file))  # .hyperwhip/../
    static = args.static.split() if args.static else None
    overrides = manifest.resolve_overrides(base, task_id, static)
    print(overrides)
    return 0


def _sync_slurm_status(workspace: str):
    """Sync trial statuses from SLURM job tracking data."""
    job_records = manifest.get_job_ids(workspace)
    if not job_records:
        return

    job_ids = [r["slurm_job_id"] for r in job_records]

    try:
        statuses = slurm.query_job_status(job_ids)
    except Exception:
        # SLURM commands might not be available (e.g., local development)
        return

    # Map SLURM states to our status values
    state_map = {
        "COMPLETED": "completed",
        "RUNNING": "running",
        "PENDING": "pending",
        "FAILED": "failed",
        "CANCELLED": "cancelled",
        "TIMEOUT": "failed",
        "NODE_FAIL": "failed",
        "OUT_OF_MEMORY": "failed",
    }

    updates = {}
    for (jid, array_idx), slurm_state in statuses.items():
        our_status = state_map.get(slurm_state, slurm_state.lower())
        updates[array_idx] = our_status

    manifest.bulk_update_status(workspace, updates)


def main():
    parser = argparse.ArgumentParser(
        prog="hyperwhip",
        description="Launch and monitor hyperparameter optimization job arrays on SLURM.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # launch
    p_launch = subparsers.add_parser("launch", help="Submit a hyperparameter sweep")
    p_launch.add_argument("config", help="Path to hyperwhip YAML config file")
    p_launch.add_argument(
        "--dry-run", action="store_true",
        help="Print the sbatch script and trial list without submitting"
    )

    # monitor
    p_monitor = subparsers.add_parser("monitor", help="Show status of all trials")
    p_monitor.add_argument("config", help="Path to hyperwhip YAML config file")

    # clean
    p_clean = subparsers.add_parser("clean", help="Cancel jobs and clean up workspace")
    p_clean.add_argument("config", help="Path to hyperwhip YAML config file")
    p_clean.add_argument("--logs", action="store_true", help="Remove log files")
    p_clean.add_argument("--all", action="store_true", help="Remove entire workspace")

    # Internal: resolve-overrides (called from within sbatch script)
    p_resolve = subparsers.add_parser("resolve-overrides", help=argparse.SUPPRESS)
    p_resolve.add_argument("manifest", help="Path to manifest.json")
    p_resolve.add_argument("task_id", help="SLURM_ARRAY_TASK_ID")
    p_resolve.add_argument("--static", default="", help="Static Hydra overrides")

    args = parser.parse_args()

    handlers = {
        "launch": cmd_launch,
        "monitor": cmd_monitor,
        "clean": cmd_clean,
        "resolve-overrides": cmd_resolve_overrides,
    }

    rc = handlers[args.command](args)
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
