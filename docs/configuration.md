# HyperWhip Configuration Guide

This document describes how to write a HyperWhip configuration file (`hyperwhip.yaml`) and the companion launcher script (`launch.sh`). HyperWhip uses these two files to generate and submit SLURM job arrays that sweep over hyperparameter combinations.

## Overview

A HyperWhip experiment requires exactly two user-authored files:

1. **`hyperwhip.yaml`** — declares the hyperparameter search space, SLURM resources, conditions, and static Hydra overrides.
2. **`launch.sh`** — a bash script that receives a Hydra override string as its first argument (`$1`) and runs the training command inside whatever environment you need (container, conda, modules, etc.).

Both files live in a **workspace directory**. All HyperWhip commands take the workspace directory as their argument:

```bash
# Scaffold a new experiment:
whip init my_experiment --partition gpu --gres gpu:1

# Validate Hydra config locally:
whip test my_experiment

# Preview what will be submitted (no SLURM interaction):
whip run my_experiment --dry-run

# Submit the job array:
whip run my_experiment

# Check status:
whip status my_experiment

# Tail a specific trial's log:
whip tail my_experiment 3

# Print results TSV:
whip res my_experiment

# Cancel and clean up:
whip clean my_experiment --all
```

---

## Configuration File Reference

The configuration file is YAML. Every field is documented below with its type, whether it is required, and its default value. The config is validated with Pydantic at parse time — invalid configs produce clear error messages immediately.

### Top-level fields

| Field         | Type   | Required | Default | Description |
|---------------|--------|----------|---------|-------------|
| `name`        | string | **yes**  | —       | Unique experiment name. Used in the SLURM job name (`hyperwhip_<name>`). Must be a valid filename component. |
| `grid`        | string or list | no | *omitted* | Controls which parameters are swept via Cartesian product. See [grid](#grid). |
| `launcher`    | string | **yes**  | —       | Path to the launcher bash script. Relative paths are resolved relative to the config file's directory. See [Launcher Script](#launcher-script). |
| `slurm`       | object | no       | see below | SLURM resource requests. See [slurm](#slurm). |
| `hydra`       | object | no       | see below | Static Hydra overrides. See [hydra](#hydra). |
| `parameters`  | object | **yes**  | —       | Hyperparameter definitions. At least one parameter is required. See [parameters](#parameters). |
| `conditions`  | list   | no       | `[]`    | Conditional rules that filter or modify parameter combinations. Also accepts the legacy key `constraints`. See [conditions](#conditions). |

The **workspace** is the directory containing `hyperwhip.yaml`. HyperWhip stores its state in a `.hyperwhip/` subdirectory within the workspace.

---

### `grid`

Controls how parameter combinations are generated. The `grid` field accepts three forms:

| Value | Meaning |
|-------|---------|
| `grid: all` | Full Cartesian product of all parameter values. No defaults needed. |
| `grid: [lr, weight_decay]` | Cartesian product of the listed parameters only. All other parameters are held at their `default` value. |
| *(omitted)* | One-at-a-time: start from defaults, vary each parameter independently. All parameters must have a `default`. |

**Full grid** (`grid: all`) generates every combination. If you have 3 parameters with 4, 3, and 2 levels respectively, you get 4 × 3 × 2 = 24 trials.

**Partial grid** (`grid: [lr, wd]`) generates a Cartesian product of only the listed parameters while holding everything else at its default. Useful when you want to explore interactions between specific parameters without exploding the trial count.

**One-at-a-time** (no `grid` field) starts from a default combination, then varies each parameter independently while holding all others at their default. This produces 1 + Σ(levels_i − 1) trials. For the 4/3/2 example: 1 + 3 + 2 + 1 = 7 trials.

```yaml
# Full grid:
grid: all

# Partial grid (only lr and weight_decay):
grid: [learning_rate, weight_decay]

# One-at-a-time (omit grid entirely):
# (no grid field)
```

When `grid` is not `"all"`, every parameter that is not in the grid list must have a `default` field. When `grid` is omitted, all parameters must have a `default`.

---

### `slurm`

SLURM resource requests. These map directly to `#SBATCH` directives in the generated batch script.

| Field           | Type         | Required | Default      | Description |
|-----------------|--------------|----------|--------------|-------------|
| `partition`     | string       | no       | `"default"`  | SLURM partition name. |
| `time`          | string       | no       | `"01:00:00"` | Wall-clock time limit in `HH:MM:SS` format. |
| `mem`           | string       | no       | `"8G"`       | Memory per node (e.g. `"16G"`, `"512M"`). |
| `cpus_per_task` | integer      | no       | `1`          | Number of CPU cores per task. |
| `gres`          | string       | no       | *omitted*    | Generic resources (e.g. `"gpu:1"`, `"gpu:a100:2"`). If omitted, the `--gres` line is not included. |
| `extra_args`    | list[string] | no       | `[]`         | Additional raw `#SBATCH` flags. Each string is placed after `#SBATCH ` verbatim. |

```yaml
slurm:
  partition: gpu
  time: "04:00:00"
  mem: "32G"
  cpus_per_task: 4
  gres: "gpu:1"
  extra_args:
    - "--export=ALL"
    - "--exclusive"
```

---

### `hydra`

Static Hydra overrides appended to every trial.

| Field              | Type         | Required | Default | Description |
|--------------------|--------------|----------|---------|-------------|
| `static_overrides` | list[string] | no       | `[]`    | Hydra overrides appended to every trial's override string. Use for values that are constant across the sweep but differ from Hydra defaults. |

```yaml
hydra:
  static_overrides:
    - "data.path=/scratch/datasets"
    - "trainer.seed=42"
```

**How overrides reach the training command**: HyperWhip constructs a Hydra override string for each trial by combining:
1. `experiment_name=<name>` (built from parameter abbreviations, e.g. `lr-0.001_opt-adam_bs-64`)
2. Each parameter's `name=value`
3. Any `static_overrides`

This string is passed as `$1` to your launcher script. For example:

```
experiment_name=lr-0.001_opt-adam_bs-64 learning_rate=0.001 optimizer=adam batch_size=64 data.path=/scratch/datasets
```

Additionally, three environment variables are exported in the SLURM script:
- `HYPERWHIP_WORKSPACE` — absolute path to the workspace directory
- `HYPERWHIP_TRIAL_ID` — the array task index (same as `$SLURM_ARRAY_TASK_ID`)
- `HYPERWHIP_EXPERIMENT_NAME` — the experiment name string (e.g. `lr-0.001_opt-adam_bs-64`)

These can be used for output directories, wandb run names, logging, and the `log_result()` API.

---

### `parameters`

A YAML mapping where each key is the parameter name and the value is a specification object. Parameter names should match the Hydra config keys you want to override (e.g. `model.learning_rate`, `training.batch_size`). You can use dotted names for nested Hydra config paths.

Every parameter requires:
- `abbrev` — a short name used to build the `experiment_name` (e.g. `"lr"`, `"opt"`, `"bs"`)
- `type` — either `"discrete"` or `"continuous"`
- `default` — *(optional)* the default value, required when the parameter is not in the `grid`

#### Discrete parameters

| Field     | Type      | Required | Description |
|-----------|-----------|----------|-------------|
| `type`    | string    | **yes**  | Must be `"discrete"`. |
| `abbrev`  | string    | **yes**  | Short name for experiment naming. |
| `values`  | list[any] | **yes**  | List of values to sweep over. Can be strings, integers, floats, or booleans. |
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
```

#### Continuous parameters

| Field     | Type    | Required | Default    | Description |
|-----------|---------|----------|------------|-------------|
| `type`    | string  | **yes**  | —          | Must be `"continuous"`. |
| `abbrev`  | string  | **yes**  | —          | Short name for experiment naming. |
| `low`     | float   | **yes**  | —          | Lower bound (inclusive). |
| `high`    | float   | **yes**  | —          | Upper bound (inclusive). |
| `scale`   | string  | no       | `"linear"` | `"linear"` for uniform spacing, `"log"` for log-uniform spacing. Log scale requires `low > 0`. |
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

**Log scale discretization**: With `low: 1e-4, high: 1e-2, scale: log, steps: 3`, the values are spaced evenly in log10 space: `[0.0001, 0.001, 0.01]`.

*Default values are validated at config parse time: discrete defaults must be in `values`, continuous defaults must be within `[low, high]`.

---

### `conditions`

An ordered list of conditional rules. Conditions are applied as post-filters after all parameter combinations are generated. They are evaluated in order.

The legacy key `constraints` is still accepted as a synonym for `conditions`.

Each condition has:

| Field     | Type   | Required | Description |
|-----------|--------|----------|-------------|
| `name`    | string | no       | Human-readable label (used in error messages). |
| `when`    | object | **yes**  | Condition: a mapping of `parameter_name: matcher`. All listed parameters must match (AND). See [matchers](#when-matchers). |
| `exclude` | object | no*      | Exclusion: `parameter_name: [values]`. Removes matching combinations. |
| `force`   | object | no*      | Override: `parameter_name: value`. Forces the value; duplicates are removed. |
| `set`     | object | no*      | Inject extra Hydra overrides: `hydra.path: value`. Keys are arbitrary Hydra paths (not parameters), and these overrides win over `hydra.static_overrides`. |

*At least one of `exclude`, `force`, or `set` is required per condition.

#### `when` matchers

Each value in the `when` mapping can be one of three forms:

| Form | Example | Meaning |
|------|---------|---------|
| scalar | `optimizer: sgd` | exact equality |
| list   | `optimizer: [sgd, momentum_sgd]` | OR — matches any element |
| operator map | `learning_rate: {gt: 0.01}` | comparison: `eq`, `ne`, `gt`, `ge`, `lt`, `le`, `in`, `not_in` |

Operator maps must contain exactly one operator key. `in` and `not_in` take a list; the others take a scalar. Numeric comparisons (`gt`/`ge`/`lt`/`le`) only match if both sides are numbers.

If a discrete parameter's `values` are themselves lists, use `eq:` to disambiguate from the OR-list form (e.g. `layers: {eq: [3, 5, 7]}`).

#### Examples

```yaml
conditions:
  # Multi-value when: applies to both sgd and momentum_sgd in one rule.
  - name: sgd_family_no_high_lr
    when:
      optimizer: [sgd, momentum_sgd]
      learning_rate: {gt: 0.01}
    exclude:
      learning_rate: [0.05, 0.1]

  # Force a parameter value.
  - name: adamw_fixed_wd
    when:
      optimizer: adamw
    force:
      weight_decay: 0.01

  # Inject non-parameter Hydra overrides conditionally.
  # `set` keys are arbitrary Hydra paths; they're not validated as parameters.
  - name: adamw_warmup
    when:
      optimizer: adamw
    set:
      scheduler.type: cosine
      scheduler.warmup_steps: 1000
```

Constraint references in `when`, `exclude`, and `force` are validated at parse time. `set` keys are intentionally **not** validated against `parameters` — they're meant for non-parameter Hydra paths.

#### Override ordering

The Hydra override string for each trial is built in this order (Hydra applies left-to-right, last wins):

1. `experiment_name=<name>`
2. swept parameter `name=value` pairs
3. `hydra.static_overrides`
4. condition `set` extras (last → these win over `static_overrides`)

This means `set` is the right place for **conditional adjustments** that should override the global baseline, while `static_overrides` is the baseline that always applies.

---

## Launcher Script

The launcher script is a user-provided bash script. HyperWhip does **not** manage your container runtime, environment modules, conda environments, or any other setup.

### Contract

1. HyperWhip calls your launcher as: `bash <launcher_path> "<hydra_overrides>"`
2. The first argument (`$1`) is a space-separated Hydra override string.
3. Your script must invoke the Hydra training command with these overrides.
4. SLURM environment variables (`$SLURM_JOB_ID`, `$SLURM_ARRAY_TASK_ID`, etc.) are available, plus `$HYPERWHIP_WORKSPACE`, `$HYPERWHIP_TRIAL_ID`, and `$HYPERWHIP_EXPERIMENT_NAME`.
5. Exit code 0 means success; nonzero means failure.

### Example: Apptainer/Singularity launcher

```bash
#!/bin/bash
set -euo pipefail

OVERRIDES="$1"
CONTAINER="/path/to/your/container.sif"
BINDS="/scratch:/scratch,/home/$USER:/home/$USER"

apptainer exec --nv --bind "$BINDS" "$CONTAINER" \
    python train.py $OVERRIDES
```

### Example: Conda environment launcher

```bash
#!/bin/bash
set -euo pipefail

OVERRIDES="$1"

source /opt/conda/etc/profile.d/conda.sh
conda activate myenv

python train.py $OVERRIDES
```

### Example: Docker via Enroot/Pyxis launcher

```bash
#!/bin/bash
set -euo pipefail

OVERRIDES="$1"
IMAGE="nvcr.io/nvidia/pytorch:24.01-py3"

srun --container-image="$IMAGE" \
     --container-mounts="/scratch:/scratch" \
     python train.py $OVERRIDES
```

### Idempotency requirements

HyperWhip's `run` command is idempotent — rerunning it resubmits only pending and failed trials with the same array indices and parameters. For this to work, **your Hydra application must also be idempotent**:

- **Checkpoint on a deterministic path**: Use `$HYPERWHIP_EXPERIMENT_NAME` or `$HYPERWHIP_TRIAL_ID` to construct a unique, stable output directory.
- **Resume from checkpoint**: On startup, check if a checkpoint exists and resume.
- **Do not fail on existing output**: Handle pre-existing output directories gracefully.

Example using the experiment name for output paths:

```python
import os
exp_name = os.environ.get("HYPERWHIP_EXPERIMENT_NAME", "default")
output_dir = f"./outputs/{exp_name}"
```

---

## Result Logging

HyperWhip provides a lightweight API for logging per-trial summary metrics from within your training script. This is intended for final/test metrics — use wandb or similar for high-density training logs.

### Logging from training code

```python
from hyperwhip import log_result

# Call at the end of training or evaluation:
log_result("test_accuracy", 0.95)
log_result("final_loss", 0.12)
log_result("epochs_completed", 50)
```

`log_result(name, value)` writes to `.hyperwhip/results/<trial_id>.json` in the workspace. It resolves the write path from the `HYPERWHIP_WORKSPACE` and `HYPERWHIP_TRIAL_ID` environment variables, which are set automatically by `whip run` and `whip test`.

- Can be called multiple times — metrics accumulate in a single JSON file per trial.
- Calling with the same metric name overwrites the previous value.
- Values must be JSON-serializable (numbers, strings, booleans, lists, dicts).

### Viewing results

```bash
whip res my_experiment
```

Prints a TSV table with all trial parameters and logged metrics:

```
trial_id  experiment_name          learning_rate  optimizer  test_acc  test_loss
0         lr-0.01_opt-adam_bs-32   0.01           adam       0.92      0.31
1         lr-0.01_opt-adam_bs-64   0.01           adam       0.87      0.45
2         lr-0.01_opt-sgd_bs-32    0.01           sgd
3         lr-0.01_opt-sgd_bs-64    0.01           sgd
```

Trials without results show empty cells. Pipe to `column -t` for aligned display, or redirect to a file for pandas/Excel.

---

## Validating Before Submission

### Config validation (`whip run --dry-run`)

The `--dry-run` flag on `whip run` validates the HyperWhip config, runs preflight checks (launcher exists, workspace is writable, partition exists), generates the trial manifest, and prints the sbatch script and trial listing — all without submitting to SLURM.

Non-default parameter values are highlighted in the dry-run output so you can verify the sweep at a glance.

### Hydra config validation (`whip test`)

```bash
whip test my_experiment       # test trial 0
whip test my_experiment 5     # test trial 5
```

This runs a single trial locally via your launcher script with `--cfg job` appended to the Hydra overrides. Hydra prints the fully resolved config and exits without running training. This catches:

- Unknown parameter names in the Hydra config
- Type mismatches between overrides and the Hydra schema
- Missing required config fields

Note: `whip test` runs on the login node, so it requires your launcher's environment to be accessible locally (e.g. conda env). If your launcher requires a GPU container, you'll need to test manually or adapt the launcher to handle the `--cfg job` case.

---

## Complete Example

### Directory layout

```
my_project/
  train.py                  # Your Hydra training script
  configs/
    config.yaml             # Your Hydra base config
  my_experiment/
    hyperwhip.yaml           # HyperWhip sweep configuration
    launch.sh                # Launcher script
```

### `my_experiment/hyperwhip.yaml`

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

hydra:
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

In this example, `grid: [learning_rate, weight_decay]` creates a 4 × 3 = 12 trial grid over learning rate and weight decay, while `optimizer` and `batch_size` stay at their defaults (`adamw` and `128`).

### `my_experiment/launch.sh`

```bash
#!/bin/bash
set -euo pipefail

OVERRIDES="$1"
CONTAINER="/shared/containers/pytorch-24.01.sif"

apptainer exec --nv \
    --bind "/scratch:/scratch,/shared:/shared" \
    "$CONTAINER" \
    python train.py $OVERRIDES
```

### `train.py` (excerpt)

```python
from hyperwhip import log_result

# ... training loop ...

# Log summary metrics at the end:
log_result("test_accuracy", test_acc)
log_result("final_loss", final_loss)
```

### Usage

```bash
# Validate Hydra config:
whip test my_experiment

# See what would be submitted:
whip run my_experiment --dry-run

# Submit:
whip run my_experiment

# Check progress:
whip status my_experiment

# Tail trial 5's log:
whip tail my_experiment 5

# Re-run to resubmit any failed trials:
whip run my_experiment

# View results:
whip res my_experiment

# Clean up everything:
whip clean my_experiment --all
```

---

## Workspace Layout

After `whip run`, the workspace directory contains:

```
my_experiment/
  hyperwhip.yaml
  launch.sh
  .hyperwhip/
    manifest.json       # Array of trial objects: {index, params, experiment_name, status}
    job_ids.json        # Records of submitted SLURM jobs
    job.sbatch          # The generated SLURM batch script
    logs/
      0.out, 0.err      # stdout/stderr for array task 0
      1.out, 1.err      # stdout/stderr for array task 1
      ...
    results/
      0.json            # Logged metrics for trial 0 (from log_result())
      1.json
      ...
```

- **manifest.json**: The authoritative mapping of array index to parameter values and experiment names. Do not edit manually.
- **job.sbatch**: The generated script. You can inspect it to verify correctness.
- **logs/**: SLURM captures stdout/stderr here. The `status` command reads the last line of each `.out` file.
- **results/**: Per-trial metric files written by `log_result()`. Read by `whip res`.

---

## Generated SLURM Script

For reference, HyperWhip generates a batch script like this:

```bash
#!/bin/bash
#SBATCH --job-name=hyperwhip_resnet_sweep
#SBATCH --array=0-11
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --output=<workspace>/.hyperwhip/logs/%a.out
#SBATCH --error=<workspace>/.hyperwhip/logs/%a.err
#SBATCH --export=ALL

# Export HyperWhip environment variables
export HYPERWHIP_WORKSPACE="<workspace>"
export HYPERWHIP_TRIAL_ID="$SLURM_ARRAY_TASK_ID"
export HYPERWHIP_EXPERIMENT_NAME=$(python -m hyperwhip resolve-name "<workspace>/.hyperwhip/manifest.json" "$SLURM_ARRAY_TASK_ID")

# Resolve Hydra overrides for this array task (includes experiment_name=...)
OVERRIDES=$(python -m hyperwhip resolve-overrides "<workspace>/.hyperwhip/manifest.json" "$SLURM_ARRAY_TASK_ID" --static "data.root=/scratch/imagenet trainer.max_epochs=90")

# Invoke the user's launcher script
bash "<workspace>/launch.sh" "$OVERRIDES"
```

The `resolve-overrides` and `resolve-name` subcommands are internal HyperWhip utilities. They read `manifest.json` and print the override string or experiment name for a given array task ID. This means `python` and the `hyperwhip` package must be available on the compute node (outside the container).

### Compute nodes without Python

If `python` is not available on the bare compute node, you can read the manifest directly in your launcher with `jq`:

```bash
#!/bin/bash
set -euo pipefail

MANIFEST=".hyperwhip/manifest.json"
TASK_ID="$SLURM_ARRAY_TASK_ID"

# Build overrides from manifest using jq
OVERRIDES=$(jq -r --argjson id "$TASK_ID" '
  .[] | select(.index == $id) | .params | to_entries | map("\(.key)=\(.value)") | join(" ")
' "$MANIFEST")

# Get experiment name
EXPERIMENT_NAME=$(jq -r --argjson id "$TASK_ID" '
  .[] | select(.index == $id) | .experiment_name
' "$MANIFEST")

# Prepend experiment_name and append static overrides
OVERRIDES="experiment_name=$EXPERIMENT_NAME $OVERRIDES data.root=/scratch/imagenet"

CONTAINER="/shared/containers/pytorch-24.01.sif"
apptainer exec --nv --bind "/scratch:/scratch" "$CONTAINER" \
    python train.py $OVERRIDES
```
