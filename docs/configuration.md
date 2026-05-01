# Sweep config (`hyperherd.yaml`)

The configuration file is YAML. Every field is documented below with its type, whether it's required, and its default. The config is validated with Pydantic at parse time — invalid configs produce clear error messages immediately.

## Top-level fields

| Field         | Type   | Required | Default | Description |
|---------------|--------|----------|---------|-------------|
| `name`        | string | **yes**  | —       | Unique experiment name. Used in the SLURM job name (`hyperherd_<name>`). Must be a valid filename component. |
| `grid`        | string or list | no | *omitted* | Controls which parameters are swept via Cartesian product. See [grid](#grid). |
| `launcher`    | string | **yes**  | —       | Path to the launcher bash script. Relative paths are resolved relative to the config file's directory. See [Launcher Script](launcher.md). |
| `slurm`       | object | no       | see below | SLURM resource requests. See [slurm](#slurm). |
| `hydra`       | object | no       | see below | Static Hydra overrides. See [hydra](#hydra). |
| `watch`       | object | no       | see below | Settings for the `herd watch` notification daemon. See [watch](#watch). |
| `parameters`  | object | **yes**  | —       | Hyperparameter definitions. At least one parameter is required. See [parameters](#parameters). |
| `conditions`  | list   | no       | `[]`    | Conditional rules that filter or modify parameter combinations. See [Conditions](conditions.md). |

The **workspace** is the directory containing `hyperherd.yaml`. HyperHerd stores its state in a `.hyperherd/` subdirectory within the workspace.

## `grid`

Controls how parameter combinations are generated. Three forms:

| Value | Meaning |
|-------|---------|
| `grid: all` | Full Cartesian product of all parameter values. No defaults required. |
| `grid: [lr, weight_decay]` | Cartesian product of the listed parameters only. All other parameters are held at their `default`. |
| *(omitted)* | One-at-a-time: start from defaults, vary each parameter independently. All parameters must have a `default`. |

**Full grid** (`grid: all`) generates every combination. With 3 parameters of 4, 3, and 2 levels, you get 4 × 3 × 2 = 24 trials.

**Partial grid** (`grid: [lr, wd]`) generates a Cartesian product of only the listed parameters while holding everything else at its default. Use it to explore interactions between specific parameters without exploding the trial count.

**One-at-a-time** (no `grid` field) starts from a default combination, then varies each parameter independently. This produces 1 + Σ(levels_i − 1) trials. For the 4/3/2 example: 1 + 3 + 2 + 1 = 7 trials.

```yaml
# Full grid:
grid: all

# Partial grid:
grid: [learning_rate, weight_decay]

# One-at-a-time (omit grid entirely):
# (no grid field)
```

When `grid` is not `"all"`, every parameter that is not in the grid list must have a `default`. When `grid` is omitted, all parameters must have a `default`.

## `slurm`

SLURM resource requests. These map directly to `#SBATCH` directives in the generated batch script.

| Field           | Type         | Required | Default      | Description |
|-----------------|--------------|----------|--------------|-------------|
| `partition`     | string       | no       | `"default"`  | SLURM partition name. |
| `time`          | string       | no       | `"01:00:00"` | Wall-clock time limit in `HH:MM:SS` format. |
| `mem`           | string       | no       | `"8G"`       | Memory per node (e.g. `"16G"`, `"512M"`). |
| `cpus_per_task` | integer      | no       | `1`          | Number of CPU cores per task. |
| `gres`          | string       | no       | *omitted*    | Generic resources (e.g. `"gpu:1"`, `"gpu:a100:2"`). If omitted, the `--gres` line is not included. |
| `max_concurrent`| integer      | no       | *omitted*    | Cap on simultaneously running array tasks. Appended as `%N` to the SLURM array spec (e.g. `--array=0-49%5`). Override per-run with `herd run --max-concurrent N`. |
| `extra_args`    | list[string] | no       | `[]`         | Additional raw `#SBATCH` flags. Each string is placed after `#SBATCH ` verbatim. |

```yaml
slurm:
  partition: gpu
  time: "04:00:00"
  mem: "32G"
  cpus_per_task: 4
  gres: "gpu:1"
  max_concurrent: 8
  extra_args:
    - "--export=ALL"
    - "--exclusive"
```

## `watch`

Settings for the [`herd watch`](commands.md#herd-watch) daemon — a polling loop that posts trial state changes to a webhook (Slack, Discord, ntfy.sh, or any URL that accepts a POST). All fields are optional; the entire `watch:` block can be omitted, in which case the defaults below apply and `herd watch` falls back to a per-workspace ntfy.sh topic generated on first run.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `webhook` | string | *unset* | Webhook URL to POST events to. Unset → daemon prints a generated `https://ntfy.sh/herd-{slug}-{random}` URL on startup. |
| `format` | enum | `raw` | Payload shape: `slack`, `discord`, `ntfy`, or `raw`. |
| `interval_seconds` | int | `60` | How often to poll SLURM. Lower = faster failure alerts, more `sacct` calls. |
| `heartbeat_minutes` | int or null | `5` | Minimum gap between heartbeat digests. Required if `heartbeat` is in `events`. |
| `events` | list | `[failed, done, heartbeat]` | Which events to deliver. Choices: `failed`, `done`, `heartbeat`. |
| `summarize` | bool | `false` | When `true`, shells out to the `claude` CLI on `failed` events for a 1–2 sentence diagnosis built from the SLURM cause + stderr tail. Replaces the raw stderr tail in slack/discord/ntfy bodies; the unabridged tail still appears in `raw` payloads. |

```yaml
watch:
  webhook: https://hooks.slack.com/services/T.../B.../xxxx
  format: slack
  interval_seconds: 60
  heartbeat_minutes: 30
  events: [failed, done, heartbeat]
```

### Events

- **`failed`** — fires once per trial when it transitions into `failed` or `cancelled`. Skipped on the very first poll so a daemon started mid-sweep doesn't replay history. The payload carries a [diagnosis block](#failure-diagnosis) (SLURM cause + stderr tail).
- **`done`** — fires once when every trial is in a terminal state (`completed` / `failed` / `cancelled`). Re-arms if you resubmit a trial.
- **`heartbeat`** — periodic digest with current totals. Suppressed if nothing has changed since the last heartbeat, so a queued sweep stuck behind a partition doesn't spam you.

### Format

`slack`, `discord`, and `ntfy` all send the same one-line text body, just wrapped in the right shape for each service:

| Format | Body | Content-Type |
|--------|------|--------------|
| `slack` | `{"text": "..."}` | `application/json` |
| `discord` | `{"content": "..."}` | `application/json` |
| `ntfy` | the line as plain text | `text/plain` |
| `raw` | the full event JSON (event, trial, totals, sweep, timestamp, summary) | `application/json` |

The line looks like `[mnist_sweep] trial 7 (lr-0.001_opt-sgd) failed — 12/19 done, 4 running, 1 failed`. Pick `raw` if you want to render your own message in a downstream consumer.

### Failure diagnosis

`failed` events automatically carry a diagnosis block built from `sacct` and the trial's stderr log:

| Field | Source | Example |
|-------|--------|---------|
| `cause` | SLURM `State` + `ExitCode` | `TIMEOUT`, `OUT_OF_MEMORY`, `SIGSEGV`, `exit code 1` |
| `slurm_state` | sacct `State` | `TIMEOUT` |
| `exit_code` / `signal` | sacct `ExitCode` | `1` / `0` |
| `reason` | sacct `Reason` | `JobOutOfTime` |
| `job_id` | manifest job ledger | `12345` |
| `stderr_tail` | `.hyperherd/logs/<idx>.err` | last 20 lines / 1500 bytes |
| `stderr_truncated` | bool | `true` when the cap kicked in |

In **slack / discord / ntfy** bodies, the `cause` is appended to the summary line and the stderr tail is included in a triple-backtick code block:

```
[mnist_sweep] trial 7 (lr-0.001_opt-sgd) failed (TIMEOUT) — 12/19 done, 4 running, 1 failed
```
Traceback (most recent call last):
  File "train.py", line 42, in <module>
    main()
slurmstepd: error: JOB 12345 ON node-04 CANCELLED AT ... DUE TO TIME LIMIT
```
```

In **`raw`** payloads, the full block is exposed as `failure: {...}` so a downstream consumer can render it however it wants.

With `summarize: true`, a Claude-generated 1–2 sentence diagnosis replaces the stderr code block in slack/discord/ntfy (the raw tail is still present in `raw` payloads). If `claude` isn't on PATH or the call fails, the daemon silently falls back to the stderr tail.

### Zero-config ntfy fallback

Leave `webhook` unset and the daemon will:

1. Generate `herd-{sweep-slug}-{random}` on first run, persist it to `.hyperherd/watch.json`, and reuse it on every subsequent run.
2. Print the resulting `https://ntfy.sh/...` URL plus subscribe instructions to stderr at startup.
3. Force `format: ntfy` regardless of what's configured (the URL only accepts the ntfy shape).

The topic is unguessable in practice (~48 bits of entropy in the random suffix), but the public ntfy.sh broker is read-by-URL, so anyone with the URL can read your notifications. Use a private webhook for sensitive sweeps.

## `static_overrides`

A list of extra `name=value` tokens appended to every trial's override string. Use them for values that are constant across the sweep but differ from your trainer's defaults (dataset paths, fixed seeds, debug flags). The values are passed through verbatim — your launcher is free to parse them however it wants.

```yaml
static_overrides:
  - "data.path=/scratch/datasets"
  - "trainer.seed=42"
```

> The legacy `hydra: { static_overrides: [...] }` form still parses for back-compat, but new configs should use the top-level field.

### How overrides reach the training command

HyperHerd builds an override string for each trial by combining:

1. `experiment_name=<name>` (built from parameter abbreviations, e.g. `lr-0.001_opt-adam_bs-64`)
2. Each parameter's `name=value`
3. Any `static_overrides`
4. Any condition `set` extras (last → these win)

This string is passed as `$1` to your launcher script. For example:

```
experiment_name=lr-0.001_opt-adam_bs-64 learning_rate=0.001 optimizer=adam batch_size=64 data.path=/scratch/datasets
```

If you're using Hydra this passes through unmodified. If not, your `launch.sh` parses these `name=value` pairs into whatever flags your trainer accepts.

Three environment variables are also exported in the SLURM script:

- `HYPERHERD_WORKSPACE` — absolute path to the workspace directory
- `HYPERHERD_TRIAL_ID` — the array task index (same as `$SLURM_ARRAY_TASK_ID`)
- `HYPERHERD_EXPERIMENT_NAME` — the experiment name (e.g. `lr-0.001_opt-adam_bs-64`)

Use these for output directories, wandb run names, logging paths, and the [`log_result()` API](results.md).

## `parameters`

A YAML mapping where each key is the parameter name and the value is a specification object. Parameter names should match the Hydra config keys you want to override (e.g. `model.learning_rate`, `training.batch_size`). Dotted names work for nested Hydra paths.

Every parameter requires:

- `type` — either `"discrete"` or `"continuous"`
- `abbrev` — *(optional)* a short name used to build `experiment_name` (e.g. `"lr"`, `"opt"`, `"bs"`). Defaults to the parameter name itself. **Required** when the parameter name contains characters outside `[A-Za-z0-9._-]` (e.g. spaces, `/`), since the token ends up as a file-path component.
- `default` — *(optional)* required when the parameter is not in the `grid`

!!! note "Allowed characters in `abbrev`"
    Whether implicit (from the parameter name) or explicit, the abbrev token must match `[A-Za-z0-9._-]+`. Anything else would corrupt the `experiment_name` directory layout or the space-separated `key=value` override string.

### Discrete parameters

| Field     | Type      | Required | Description |
|-----------|-----------|----------|-------------|
| `type`    | string    | **yes**  | Must be `"discrete"`. |
| `abbrev`  | string    | no       | Short name for experiment naming. Defaults to the parameter name; required when the parameter name contains characters outside `[A-Za-z0-9._-]`. |
| `values`  | list[any] | **yes**  | List of values to sweep over. Strings, integers, floats, or booleans. |
| `labels`  | list[string] | no    | Per-value short display tokens used in the experiment name. Must be the same length as `values`, contain unique non-empty strings without `/`. **Required** when any value contains `/` (e.g. file paths). |
| `default` | any       | no*      | Default value. Must be one of `values`. Required when not in grid. |

```yaml
parameters:
  optimizer:
    abbrev: opt
    type: discrete
    values: [adam, sgd, adamw]
    default: adam
  num_layers:
    abbrev: nl
    type: discrete
    values: [2, 4, 8]
    default: 4
  # Path-like values must declare labels so experiment names stay short.
  pretrained:
    abbrev: pre
    type: discrete
    values: ["/scratch/ckpts/resnet50.ckpt", "/scratch/ckpts/vit_base.ckpt"]
    labels: [resnet50, vit_base]
    default: "/scratch/ckpts/resnet50.ckpt"
```

The full value is still passed to Hydra as the override; `labels` only affect the auto-generated `experiment_name`.

### Continuous parameters

| Field     | Type    | Required | Default    | Description |
|-----------|---------|----------|------------|-------------|
| `type`    | string  | **yes**  | —          | Must be `"continuous"`. |
| `abbrev`  | string  | no       | param name | Short name for experiment naming. Defaults to the parameter name; required when the parameter name contains characters outside `[A-Za-z0-9._-]`. |
| `low`     | float   | **yes**  | —          | Lower bound (inclusive). |
| `high`    | float   | **yes**  | —          | Upper bound (inclusive). |
| `scale`   | string  | no       | `"linear"` | `"linear"` for uniform spacing, `"log"` for log-uniform. Log scale requires `low > 0`. |
| `steps`   | integer | no       | `5`        | Number of evenly-spaced points to discretize into. |
| `default` | float   | no*      | —          | Default value. Must be within `[low, high]`. Required when not in grid. |

```yaml
parameters:
  learning_rate:
    abbrev: lr
    type: continuous
    low: 1e-5
    high: 1e-2
    scale: log
    steps: 5
    default: 0.001
```

**Log scale discretization.** With `low: 1e-4, high: 1e-2, scale: log, steps: 3`, values are spaced evenly in log10 space: `[0.0001, 0.001, 0.01]`.

!!! note "Default validation"
    Discrete defaults must be in `values`. Continuous defaults must be within `[low, high]`. Both are validated at parse time.

## Complete example

```yaml
name: resnet_sweep

grid: [learning_rate, weight_decay]

slurm:
  partition: gpu
  time: "08:00:00"
  mem: "32G"
  cpus_per_task: 8
  gres: "gpu:a100:1"
  extra_args:
    - "--export=ALL"

launcher: ./launch.sh

static_overrides:
  - "data.root=/scratch/imagenet"
  - "trainer.max_epochs=90"

parameters:
  learning_rate:
    abbrev: lr
    type: continuous
    low: 1e-4
    high: 1e-1
    scale: log
    steps: 4
    default: 0.001
  optimizer:
    abbrev: opt
    type: discrete
    values: [sgd, adamw]
    default: adamw
  weight_decay:
    abbrev: wd
    type: continuous
    low: 0.0
    high: 0.01
    scale: linear
    steps: 3
    default: 0.001
  batch_size:
    abbrev: bs
    type: discrete
    values: [64, 128, 256]
    default: 128

conditions:
  - name: sgd_no_high_lr
    when:
      optimizer: sgd
    exclude:
      learning_rate: [0.1]
```

`grid: [learning_rate, weight_decay]` creates a 4 × 3 = 12 trial grid over those two parameters; `optimizer` and `batch_size` stay at their defaults (`adamw` and `128`).
