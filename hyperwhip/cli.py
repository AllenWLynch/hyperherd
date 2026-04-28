"""HyperWhip CLI: launch, monitor, and clean SLURM hyperparameter job arrays."""

import argparse
import os
import shutil
import sys

from hyperwhip.config import load_config
from hyperwhip.constraints import apply_constraints
from hyperwhip.display import print_dry_run, print_status_table, print_summary
from hyperwhip.init import scaffold
from hyperwhip.preflight import PreflightError, run_preflight
from hyperwhip.search import generate_combinations
from hyperwhip import manifest
from hyperwhip import slurm
from hyperwhip.logging import load_all_results


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
            trials = manifest.create_manifest(config.workspace, combinations, abbrevs)
            pending = [t["index"] for t in trials]
    else:
        trials = manifest.create_manifest(config.workspace, combinations, abbrevs)
        pending = [t["index"] for t in trials]

    # Generate sbatch script
    script = slurm.generate_sbatch_script(config, pending)

    if args.dry_run:
        print_dry_run(trials, script, defaults=config.defaults)
        return 0

    # Submit
    print(f"Submitting {len(pending)} trials as SLURM job array...")
    job_id = slurm.submit_job(config, script, dry_run=False)
    assert job_id is not None
    print(f"Submitted job array: {job_id}")

    # Record submission and update statuses
    manifest.record_job_submission(config.workspace, job_id, pending)
    manifest.bulk_update_status(config.workspace, {i: "submitted" for i in pending})

    print(f"Workspace: {manifest.workspace_path(config.workspace)}")
    print(f"Logs: {manifest.logs_path(config.workspace)}")
    return 0


def cmd_monitor(args):
    """Show the status of all trials in a hyperparameter sweep."""
    config = load_config(args.workspace)

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


def cmd_tail(args):
    """Print the last N lines of a trial's log files (stdout and stderr)."""
    config = load_config(args.workspace)

    if not manifest.workspace_exists(config.workspace):
        print("No workspace found. Run 'hyperwhip launch' first.", file=sys.stderr)
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

    _DIM = "\033[2m"
    _RESET = "\033[0m"

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
    """Run a single trial locally via the launcher script (no SLURM)."""
    import subprocess

    config = load_config(args.workspace)

    # Preflight checks
    try:
        run_preflight(config)
    except PreflightError as e:
        print(f"Preflight check failed: {e}", file=sys.stderr)
        return 1

    # Generate combinations and ensure manifest exists
    combinations = generate_combinations(config)
    combinations = apply_constraints(combinations, config.conditions)

    if not combinations:
        print("No valid parameter combinations after applying conditions.", file=sys.stderr)
        return 1

    manifest.init_workspace(config.workspace)

    if not manifest.workspace_exists(config.workspace):
        abbrevs = config.abbrevs
        manifest.create_manifest(config.workspace, combinations, abbrevs)

    trials = manifest.load_manifest(config.workspace)

    index = args.index
    if index < 0 or index >= len(trials):
        print(f"Trial index {index} out of range (0-{len(trials) - 1}).", file=sys.stderr)
        return 1

    trial = trials[index]
    exp_name = trial.get("experiment_name", "")
    overrides = manifest.resolve_overrides(
        config.workspace, index, config.hydra.static_overrides or None
    )

    # Append --cfg job to validate Hydra config without running training
    overrides_with_cfg = f"{overrides} --cfg job"

    print(f"Validating Hydra config for trial {index}")
    if exp_name:
        print(f"  experiment_name: {exp_name}")
    print(f"  overrides: {overrides}")
    print(f"  launcher: {config.launcher}")
    print(f"  (appending --cfg job for Hydra config validation)")
    print("-" * 60)
    print()

    # Set env vars to match what the sbatch script would export
    env = os.environ.copy()
    env["HYPERWHIP_WORKSPACE"] = config.workspace
    env["HYPERWHIP_TRIAL_ID"] = str(index)
    env["HYPERWHIP_EXPERIMENT_NAME"] = exp_name

    result = subprocess.run(
        ["bash", config.launcher, overrides_with_cfg],
        cwd=config.workspace,
        env=env,
    )

    print()
    print("-" * 60)
    if result.returncode == 0:
        print(f"Trial {index}: Hydra config is valid.")
    else:
        print(f"Trial {index}: Hydra config validation failed (exit code {result.returncode}).")

    return result.returncode


def cmd_results(args):
    """Print a TSV of trial parameters and logged metrics."""
    config = load_config(args.workspace)

    if not manifest.workspace_exists(config.workspace):
        print("No workspace found. Run 'hyperwhip launch' first.", file=sys.stderr)
        return 1

    trials = manifest.load_manifest(config.workspace)
    results = load_all_results(config.workspace)

    if not results:
        print("No results logged yet. Use hyperwhip.log_result() from your training script.", file=sys.stderr)
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


def cmd_init(args):
    """Scaffold a new hyperwhip.yaml and launch.sh."""
    directory = args.directory or "."
    try:
        config_path, launcher_path = scaffold(
            directory=directory,
            name=args.name,
            grid=args.grid,
            partition=args.partition,
            time=args.time,
            mem=args.mem,
            cpus=args.cpus,
            gres=args.gres,
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
    print("  1. Edit hyperwhip.yaml to define your parameters")
    print("  2. Edit launch.sh to set up your container/environment")
    print(f"  3. Run: hyperwhip launch {directory} --dry-run")
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


def cmd_resolve_name(args):
    """Internal subcommand: print the experiment_name for a SLURM array task."""
    manifest_file = args.manifest
    task_id = int(args.task_id)

    base = os.path.dirname(os.path.dirname(manifest_file))  # .hyperwhip/../
    trials = manifest.load_manifest(base)
    for t in trials:
        if t["index"] == task_id:
            print(t.get("experiment_name", ""))
            return 0
    print(f"No trial found for task ID {task_id}", file=sys.stderr)
    return 1


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
        "PENDING": "queued",     # SLURM queued != never submitted
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
        prog="hyperwhip",
        description="Launch and monitor hyperparameter optimization job arrays on SLURM.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # init
    p_init = subparsers.add_parser("init", help="Scaffold a new config and launcher script")
    p_init.add_argument("directory", nargs="?", default=".", help="Directory to create files in (default: current)")
    p_init.add_argument("--name", default=None, help="Experiment name (default: directory name)")
    p_init.add_argument("--grid", default="all", help="Grid mode: 'all' for full grid, or omit for one-at-a-time (default: all)")
    p_init.add_argument("--partition", default="default", help="SLURM partition (default: 'default')")
    p_init.add_argument("--time", default="04:00:00", help="Wall time limit (default: 04:00:00)")
    p_init.add_argument("--mem", default="8G", help="Memory per node (default: 8G)")
    p_init.add_argument("--cpus", type=int, default=1, help="CPUs per task (default: 1)")
    p_init.add_argument("--gres", default=None, help="Generic resources (e.g. gpu:1)")
    p_init.add_argument("--config", default=None, help="Copy this file as hyperwhip.yaml instead of generating a template")
    p_init.add_argument("--launcher", default=None, help="Copy this file as launch.sh instead of generating a template")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing files")

    # launch
    p_launch = subparsers.add_parser("run", help="Submit a hyperparameter sweep")
    p_launch.add_argument("workspace", nargs="?", default=".", help="Workspace directory (default: current dir)")
    p_launch.add_argument(
        "--dry-run", action="store_true",
        help="Print the sbatch script and trial list without submitting"
    )

    # monitor
    p_monitor = subparsers.add_parser("status", help="Show status of all trials")
    p_monitor.add_argument("workspace", nargs="?", default=".", help="Workspace directory (default: current dir)")

    # test
    p_test = subparsers.add_parser("test", help="Validate Hydra config by running a trial with --cfg job")
    p_test.add_argument("workspace", nargs="?", default=".", help="Workspace directory (default: current dir)")
    p_test.add_argument("index", nargs="?", type=int, default=0, help="Trial index to test (default: 0)")

    # tail
    p_tail = subparsers.add_parser("tail", help="Print last N lines of a trial's log")
    p_tail.add_argument("index", type=int, help="Trial index to tail")
    p_tail.add_argument("workspace", nargs="?", default=".", help="Workspace directory (default: current dir)")
    p_tail.add_argument("-n", "--lines", type=int, default=20, help="Number of lines to show (default: 20)")

    # results
    p_results = subparsers.add_parser("res", help="Print TSV of trial parameters and logged metrics")
    p_results.add_argument("workspace", nargs="?", default=".", help="Workspace directory (default: current dir)")

    # clean
    p_clean = subparsers.add_parser("clean", help="Cancel jobs and clean up workspace")
    p_clean.add_argument("workspace", nargs="?", default=".", help="Workspace directory (default: current dir)")
    p_clean.add_argument("--logs", action="store_true", help="Remove log files")
    p_clean.add_argument("--all", action="store_true", help="Remove entire .hyperwhip state")

    # Internal: resolve-overrides (called from within sbatch script)
    p_resolve = subparsers.add_parser("resolve-overrides", help=argparse.SUPPRESS)
    p_resolve.add_argument("manifest", help="Path to manifest.json")
    p_resolve.add_argument("task_id", help="SLURM_ARRAY_TASK_ID")
    p_resolve.add_argument("--static", default="", help="Static Hydra overrides")

    # Internal: resolve-name (called from within sbatch script)
    p_name = subparsers.add_parser("resolve-name", help=argparse.SUPPRESS)
    p_name.add_argument("manifest", help="Path to manifest.json")
    p_name.add_argument("task_id", help="SLURM_ARRAY_TASK_ID")

    args = parser.parse_args()

    handlers = {
        "init": cmd_init,
        "run": cmd_launch,
        "test": cmd_test,
        "status": cmd_monitor,
        "tail": cmd_tail,
        "res": cmd_results,
        "clean": cmd_clean,
        "resolve-overrides": cmd_resolve_overrides,
        "resolve-name": cmd_resolve_name,
    }

    rc = handlers[args.command](args)
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
