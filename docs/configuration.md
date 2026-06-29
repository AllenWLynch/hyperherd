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
| `discord`     | object | no       | *unset*  | Discord channel for the [`herd monitor`](monitor.md) daemon. See [discord](#discord). |
| `parameters`  | object | **yes**  | —       | Hyperparameter definitions. At least one parameter is required. See [parameters](#parameters). |
| `conditions`  | list   | no       | `[]`    | Conditional rules that filter or modify parameter combinations. See [Conditions](conditions.md). |
| `successive_halving` | object | no | *unset* | Successive-halving pruning parameters for [`herd sh`](commands.md#herd-sh). See [below](#successive-halving-pruning). |

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

## `discord`

Settings for the [`herd monitor`](monitor.md) daemon's Discord channel. Required if you want the monitor to be able to talk to you. The token comes from the `DISCORD_BOT_TOKEN` environment variable — don't put secrets in YAML. Walk through [Discord setup](discord-setup.md) once per server to mint the bot and copy the guild ID.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `guild_id` | string | *unset* | Discord server ID. Required to enable the channel — without it the daemon runs without notifications. |
| `channel_id` | string | *unset* | Pin to a specific existing channel (skips auto-create). |
| `channel_name` | string | *unset* | Override the sweep-derived channel name. Discord lowercases it automatically; non `[a-z0-9-]` chars are stripped. |

```yaml
discord:
  guild_id: "1234567890123456789"
```

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

Four environment variables are also exported in the SLURM script:

- `HYPERHERD_WORKSPACE` — absolute path to the workspace directory
- `HYPERHERD_SWEEP_NAME` — the sweep's `name:` from yaml (shared across all trials)
- `HYPERHERD_TRIAL_ID` — the array task index (same as `$SLURM_ARRAY_TASK_ID`)
- `HYPERHERD_TRIAL_NAME` — the auto-generated per-trial identifier (e.g. `lr-0.001_opt-adam_bs-64`)

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

## Successive-halving pruning

An optional `successive_halving:` block enables [`herd sh`](commands.md#herd-sh) — early-stopping that concentrates compute on the promising trials. As trials reach geometrically-spaced step *rungs*, the better fraction of the field is kept and the rest is pruned.

```yaml
successive_halving:
  metric: val_loss      # the metric the trainer logs via log_result(name, val, step=...)
  direction: min        # "min" (lower is better, e.g. loss) or "max" (e.g. accuracy)
  min_steps: 5          # first rung: earliest step a trial can be pruned
  budget: 50            # total/max steps a trial trains to
  eta: 2                # reduction factor (default 2 → keep the better half each rung)
  mode: sync            # scheduler: "sync" (default) or "asha" — see below
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `metric` | string | **yes** | — | Objective metric name. Must match a streaming `log_result(name, value, step=...)` the trainer emits. |
| `direction` | `min` \| `max` | **yes** | — | Whether lower or higher is better. |
| `min_steps` | int ≥ 1 | **yes** | — | First rung — the earliest step at which a trial can be pruned. |
| `budget` | int ≥ 1 | **yes** | — | Total/maximum steps a trial runs to. Must be ≥ `min_steps`. |
| `eta` | int ≥ 2 | no | `2` | Reduction factor. Rungs are spaced by powers of `eta`; `eta=2` keeps the better half at each rung. |
| `mode` | `sync` \| `asha` | no | `sync` | Scheduler — how the cut is made at each rung (see below). |

**Scheduler (`mode`).** Both keep the top `1/eta` at each rung; they differ in how they handle a partially-arrived field:

- **`sync`** (default) — the conservative bracket. A trial is only pruned/promoted once its standing is certain *regardless of how the trials that haven't reached the rung yet turn out*; undecidable trials are **paused** until enough of the field arrives. Never makes an early-stop mistake, but stalls when trials are launched unevenly (every arrival waits for the stragglers).
- **`asha`** — asynchronous halving ([Li et al. 2020](https://arxiv.org/abs/1810.05934)). At each rung it ranks **only the trials that have arrived** and keeps the top `floor(n/eta)` (all of them while fewer than `eta` have arrived). It never waits and never pauses, so it suits fields that are launched or relaunched over time — at the cost of occasionally stopping a trial that a slower, worse-but-later arrival would have spared. Override per-run with `herd sh --mode asha`.

Rungs are `min_steps × eta^k` for each value `≤ budget` — e.g. `min_steps: 5, budget: 50, eta: 2` gives rungs `[5, 10, 20, 40]`.

The **step units are whatever the trainer logs the metric with.** If you log once per epoch (`log_result("val_loss", v, step=epoch)`), express `min_steps`/`budget` in epochs; if per global training step, use those. Log on a consistent cadence across trials so the field is compared fairly — epoch-aligned logging is the simplest. See the [MNIST example](example.md) for an epoch-aligned setup.

`herd sh` is stateless — run it on a loop, or let the autonomous monitor's `run_sh` tool call it. See [`herd sh`](commands.md#herd-sh) and the [monitor docs](monitor.md#successive-halving) for details.

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
