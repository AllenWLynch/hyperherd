# Command reference

Every `herd` subcommand is documented below. Most subcommands take an optional `workspace` directory as their first positional argument (default: `.`); when run from inside the workspace you can usually omit it.

## Agent (`--json`) mode

The read- and run-style commands (`run`, `status`, `stats`, `tail`, `res`, `stop`) accept a `--json` flag that emits a structured JSON document to stdout instead of the human-formatted table. Use it when an agent or other automation is driving HyperHerd:

- Numeric memory in **bytes** (not `1.50G`); elapsed time in **seconds** (not `01:30:00`).
- Status uses the stable internal enum: `ready`, `submitted`, `queued`, `running`, `completed`, `failed`, `cancelled`.
- Empty / unknown values come through as `null`, never an empty string.
- Errors still go to stderr with a non-zero exit code; stdout in JSON mode is always a single valid JSON document or empty.
- Warnings (preflight, partition checks) print to stderr as in normal mode and **do not** corrupt the stdout JSON, so `herd run --dry-run --json | jq ...` is always safe.

The JSON shape for each command is documented inline below.

## `herd init`

Scaffold a new sweep workspace.

```bash
herd init [DIRECTORY] [--config FILE] [--launcher FILE] [-f]
```

Creates `DIRECTORY/hyperherd.yaml` and `DIRECTORY/launch.sh` with template content; the experiment name is taken from the directory name. If `DIRECTORY` is omitted, files are written to the current directory.

The templates have placeholder SLURM resource fields (`partition`, `time`, `mem`, `cpus_per_task`) you'll edit to match your cluster — `herd init` doesn't try to be a substitute for opening the YAML.

| Flag | Description |
|------|-------------|
| `--config FILE` | Copy `FILE` in as `hyperherd.yaml` instead of generating a template — useful for cloning an existing sweep |
| `--launcher FILE` | Copy `FILE` in as `launch.sh` instead of generating a template |
| `-f, --force` | Overwrite existing files in the target directory |

**Example output**

--8<-- "_outputs/init.html"

## `herd run`

Submit (or resubmit) the sweep.

```bash
herd run [WORKSPACE] [flags]
```

Generates the trial manifest, runs preflight checks, writes the sbatch script to `.hyperherd/job.sbatch`, submits it, and records the SLURM job ID.

`herd run` is idempotent: it only submits trials whose status is `ready`, `failed`, or `cancelled`. Trials that are `submitted`, `queued`, `running`, or `completed` are skipped unless you opt in with `--force`.

| Flag | Description |
|------|-------------|
| `-n, --dry-run` | Print the submission plan (sbatch script + pending indices); don't submit. Use [`herd ls`](#herd-ls) for the full trial list. |
| `-j, --max-concurrent N` | Cap concurrent running tasks (overrides `slurm.max_concurrent`) |
| `-i, --indices SPEC` | Submit only these trial indices, e.g. `0-3,5,7-9` |
| `-p, --pin NAME=VALUE [NAME=VALUE ...]` | Submit only trials whose swept params match every pin (e.g. `--pin batch_size=32 optimizer=adam`). Names must be sweep parameters; values are coerced to int/float/str. |
| `-f, --force` | With `--indices`, allow resubmitting running/completed trials. Without, allow config edits that drop running/completed trials (kept as orphans). |

**Editing the config mid-sweep is supported.** If you edit `hyperherd.yaml` between runs, `herd run` reconciles the new manifest against the old one: new trials are appended, removed trials are dropped (or kept as orphans with `-f` if they were already running/completed). See [Re-running and reconciliation](workspace.md#re-running-and-reconciliation) for the rules.

**Example output** — successful submission

--8<-- "_outputs/launch.html"

**Example output** — `--dry-run`

--8<-- "_outputs/dry-run.html"

**Agent mode** — `herd run --dry-run --json` emits the submission plan (the indices + sbatch script that would actually be submitted right now, given current status / `--pin` / `--indices`). For the full sweep enumeration regardless of status, use `herd ls` (or its JSON variant when added). The intended workflow for an agent is to inspect the trials, then call `herd run --json` to submit.

```json
{
  "dry_run": true,
  "slurm_job_id": null,
  "sbatch_path": null,
  "submitted_indices": [0, 1, 2, 3],
  "sbatch_script": "#!/bin/bash\n#SBATCH --array=0-3\n...",
  "trials": [
    {"index": 0, "status": "ready", "experiment_name": "lr-0.01_bs-32",
     "params": {"lr": 0.01, "bs": 32}},
    {"index": 1, "status": "ready", "experiment_name": "lr-0.01_bs-64",
     "params": {"lr": 0.01, "bs": 64}}
  ]
}
```

A real (non-dry-run) `herd run --json` returns the same shape with `dry_run: false`, `slurm_job_id` populated, `sbatch_path` set to where the script was written (`.hyperherd/job.sbatch`), and `sbatch_script: null`.

## `herd ls`

List every trial in the sweep with its swept parameters.

```bash
herd ls [WORKSPACE] [-p NAME=VALUE ...]
```

Status-agnostic — shows the *shape* of the sweep, not what `herd run` would do next. Reads the manifest if present; otherwise materializes the combinations from `hyperherd.yaml` so you can `herd ls` BEFORE the first `herd run` to sanity-check the YAML.

| Flag | Description |
|------|-------------|
| `-p, --pin NAME=VALUE [NAME=VALUE ...]` | Filter to trials whose swept params match every pin (e.g. `--pin batch_size=32 optimizer=adam`). |

Use `herd status` for the SLURM-synced status table (this command does not touch SLURM); use `herd run --dry-run` for a submission preview.

**Example output**

--8<-- "_outputs/ls.html"

## `herd status`

Show the current status table for every trial.

```bash
herd status [WORKSPACE]
```

Status values:

| Status | Meaning |
|--------|---------|
| `ready` | Never submitted |
| `submitted` | Sent to SLURM, not yet picked up |
| `queued` | SLURM `PENDING` |
| `running` | SLURM `RUNNING` |
| `completed` | SLURM `COMPLETED` |
| `failed` | SLURM `FAILED` / `TIMEOUT` / `OUT_OF_MEMORY` / `NODE_FAIL` |
| `cancelled` | SLURM `CANCELLED` (or via `herd stop`) |

`herd status` syncs from SLURM each time it runs (via `sacct`).

**Example output**

--8<-- "_outputs/status.html"

**Agent mode** — `herd status --json`:

```json
{
  "totals": {"total": 11, "running": 4, "completed": 5, "failed": 1, "queued": 1},
  "trials": [
    {"index": 0, "status": "completed", "experiment_name": "lr-0.001_opt-adam",
     "params": {"lr": 0.001, "optimizer": "adam"},
     "last_log_line": "Test acc: 0.978"}
  ]
}
```

## `herd stats`

Print runtime + memory accounting for one or all trials, sourced from `sacct`.

```bash
herd stats [WORKSPACE] [INDEX]
```

Columns: index, state, elapsed, max RSS (GB), avg RSS (GB), requested mem (GB), experiment name. Memory values are converted from sacct's raw units to gigabytes.

**Example output**

--8<-- "_outputs/stats.html"

**Agent mode** — `herd stats --json` emits memory in bytes and elapsed time in seconds, with the SLURM state and the original sacct strings preserved so callers don't have to re-derive them:

```json
{
  "trials": [
    {"index": 0, "experiment_name": "lr-0.001_opt-adam",
     "slurm_state": "COMPLETED",
     "elapsed": "00:01:30", "elapsed_seconds": 90,
     "max_rss_bytes": 1610612736, "ave_rss_bytes": 858993459,
     "req_mem_bytes": 1610612736}
  ]
}
```

## `herd tail`

Print the last N lines of a trial's logs.

```bash
herd tail [WORKSPACE] INDEX [-n LINES] [--stdout | --stderr]
```

By default `herd tail` prints both `.hyperherd/logs/<index>.out` (stdout) and `.err` (stderr), each prefixed by a labelled header. Use `--stdout` or `--stderr` (mutually exclusive) to restrict to one stream. `-n` (default 20) controls how many lines per stream.

**Agent mode** — `herd tail --json` returns each requested stream's path and lines as a structured payload. A stream that doesn't exist on disk shows up with `lines: null` so an agent can distinguish "no log file" from "empty log file":

```json
{
  "index": 3,
  "status": "failed",
  "experiment_name": "lr-0.1_opt-sgd",
  "streams": {
    "stdout": {"path": ".hyperherd/logs/3.out", "lines": ["epoch 1", "..."], "requested": 20},
    "stderr": {"path": ".hyperherd/logs/3.err", "lines": ["RuntimeError: CUDA OOM"], "requested": 20}
  }
}
```

## `herd res`

Print a TSV of every trial's parameters and logged metrics.

```bash
herd res [WORKSPACE]
```

Combines `manifest.json` (parameters, experiment name) with `.hyperherd/results/*.json` (metrics written by [`log_result()`](results.md)). Trials without results show empty cells.

**Agent mode** — `herd res --json` emits one entry per trial (including those without logged metrics, with `metrics: {}`):

```json
{
  "trials": [
    {"index": 0, "experiment_name": "lr-0.001_opt-adam",
     "params": {"lr": 0.001, "optimizer": "adam"},
     "metrics": {"test_acc": 0.978, "test_loss": 0.071}}
  ]
}
```

## `herd test`

Run a single trial locally (no SLURM) via the configured launcher.

```bash
herd test [WORKSPACE] [INDEX] [--cfg-job]
```

Default `INDEX` is 0. The launcher is invoked exactly as the SLURM array would invoke it, so this is the right place to debug the launcher script itself, exercise the trainer end-to-end on a login node, or verify a fix before resubmitting the array.

For safety, `herd test` refuses any index that has previously been submitted to SLURM — running again would clobber its outputs and logs. Pick a different index, or `herd clean --all` first.

| Flag | Description |
|------|-------------|
| `--cfg-job` | Append `--cfg job` to the override string. For Hydra trainers, this prints the fully resolved config and exits without running training — handy for catching unknown parameter names, type mismatches, or missing required fields. Because nothing real runs, the previously-submitted guard is skipped in this mode. **Hydra-specific** — has no effect on launchers whose trainers don't recognize `--cfg job`. |

`herd test` runs on the login node, so your launcher's environment must be accessible there. If your launcher requires a GPU container that isn't available on the login node, adapt it to gate the heavy parts on a `HYPERHERD_TEST` flag, or test manually.

## `herd stop`

Cancel a running/queued trial.

```bash
herd stop [WORKSPACE] INDEX
herd stop [WORKSPACE] --all
```

Calls `scancel <jobid>_<index>` and updates the manifest to `cancelled`. Pass either an `INDEX` or `--all`, not both. With `--all`, every trial whose status is in (`submitted`, `queued`, `running`) is cancelled.

**Agent mode** — `herd stop --json` returns one record per cancelled trial (empty list if there was nothing live):

```json
{
  "cancelled": [
    {"index": 3, "slurm_job_id": "12345", "previous_status": "running"},
    {"index": 7, "slurm_job_id": "12345", "previous_status": "queued"}
  ]
}
```

## `herd snapshot`

Bundle every read-style command's output (status + sacct + logged metrics + per-trial last-log line + recent failed-trial stderr) into a single JSON document.

```bash
herd snapshot [WORKSPACE] [-n LINES] [--max-failed N]
```

`herd snapshot` is **JSON-only**: it has no human-formatted form. It exists for agent loops where one cheap CLI call per tick beats firing four (`status`, `stats`, `res`, `tail`) and re-stitching the results — and avoids partial-state races between calls when SLURM transitions a trial mid-snapshot.

| Flag | Description |
|------|-------------|
| `-n, --lines` | Max stderr lines to include per failed trial (default: 20) |
| `--max-failed` | Cap on number of failed trials to attach stderr for (default: 20) |

Shape:

```json
{
  "sweep_name": "mnist_sweep",
  "workspace": "/home/you/sweeps/mnist_sweep",
  "totals": {"total": 11, "running": 4, "completed": 5, "failed": 2},
  "trials": [
    {
      "index": 0, "status": "completed", "experiment_name": "lr-0.001_opt-adam",
      "params": {"lr": 0.001, "optimizer": "adam"},
      "slurm_job_id": "12345",
      "slurm_state": "COMPLETED",
      "elapsed": "00:01:30", "elapsed_seconds": 90,
      "max_rss_bytes": 1610612736, "ave_rss_bytes": 858993459,
      "req_mem_bytes": 1610612736,
      "metrics": {"test_acc": 0.978, "test_loss": 0.071},
      "last_log_line": "Test acc: 0.978"
    }
  ],
  "failed_stderr": [
    {
      "index": 5,
      "stderr_path": ".hyperherd/logs/5.err",
      "stderr_lines": ["RuntimeError: CUDA out of memory", "..."],
      "stderr_truncated": false
    }
  ]
}
```

`metrics` is whatever the trial called `log_result()` with — empty dict for trials that haven't logged anything yet (not silently dropped). `last_log_line` is the same one-liner the human `herd status` table shows in its rightmost column. `failed_stderr` is keyed by index in ascending order; an agent that wants to group failures by root cause should fingerprint these stderr blocks.

## `herd monitor`

Run the autonomous monitor daemon. Connects to Discord, runs the boot interview, operates the sweep until it halts. See [Autonomous monitor](monitor.md) for the full picture.

```bash
herd monitor [WORKSPACE] [flags]
```

| Flag | Description |
|------|-------------|
| `--once` | Run exactly one tick and exit (live — calls the model once) |
| `--dry-run` | Assemble the per-tick state and render the prompt without calling the model. For verifying the deterministic path before paying tokens. |
| `--trigger {scheduled,failure,completion,user_message,boot}` | Trigger for `--once` / `--dry-run` (daemon mode picks its own) |
| `--max-ticks N` | Stop after N ticks (safety cap for testing) |

If `WORKSPACE/.hyperherd` doesn't exist, the daemon auto-initializes the manifest first (equivalent to `herd run --dry-run`) so the agent has trial state to read from its first tick.

Requires Python 3.10+ and the `[monitor]` extras (`pip install 'hyperherd[monitor]'`). Discord setup is one-time per server — see [Discord setup](discord-setup.md).

## `herd clean`

Cancel jobs and clean up workspace state.

```bash
herd clean [WORKSPACE] [-l] [-a]
```

| Flag | Description |
|------|-------------|
| *(none)* | Cancel any running jobs but leave the manifest in place |
| `-l, --logs` | Also remove `.hyperherd/logs/` |
| `-a, --all` | Remove the entire `.hyperherd/` state directory |

`herd clean -a` is destructive — manifests, results, and logs are gone after.

## `herd install-skill`

Install the [Claude Code skill](claude-skill.md) for authoring sweep configs.

```bash
herd install-skill [--scope user|project] [-f]
```

Default scope is `user` (writes to `~/.claude/skills/hyperherd-config/SKILL.md`); `project` writes to `./.claude/skills/`. Use `-f` to overwrite an existing install.
