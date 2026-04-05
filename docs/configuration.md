# HyperWhip Configuration Guide

This document describes how to write a HyperWhip configuration file (`hyperwhip.yaml`) and the companion launcher script (`launch.sh`). HyperWhip uses these two files to generate and submit SLURM job arrays that sweep over hyperparameter combinations.

## Overview

A HyperWhip experiment requires exactly two user-authored files:

1. **`hyperwhip.yaml`** — declares the hyperparameter search space, SLURM resources, constraints, and how to invoke the training command.
2. **`launch.sh`** — a bash script that receives a Hydra override string as its first argument (`$1`) and runs the training command inside whatever environment you need (container, conda, modules, etc.).

HyperWhip is invoked as:

```bash
# Preview what will be submitted (no SLURM interaction):
hyperwhip launch hyperwhip.yaml --dry-run

# Submit the job array:
hyperwhip launch hyperwhip.yaml

# Check status:
hyperwhip monitor hyperwhip.yaml

# Cancel and clean up:
hyperwhip clean hyperwhip.yaml --all
```

---

## Configuration File Reference

The configuration file is YAML. Every field is documented below with its type, whether it is required, and its default value.

### Top-level fields

| Field       | Type   | Required | Default          | Description |
|-------------|--------|----------|------------------|-------------|
| `name`      | string | **yes**  | —                | Unique experiment name. Used in the SLURM job name (`hyperwhip_<name>`) and as the default workspace directory name. Must be a valid filename component (letters, numbers, underscores, hyphens). |
| `workspace` | string | no       | `./<name>`       | Directory where HyperWhip stores its state (`.hyperwhip/` subdirectory with manifest, logs, generated sbatch script). Relative paths are resolved relative to the config file's directory. |
| `launcher`  | string | **yes**  | —                | Path to the launcher bash script. Relative paths are resolved relative to the config file's directory. See [Launcher Script](#launcher-script) below. |
| `search`    | object | no       | `{mode: "grid"}` | Search strategy configuration. See [search](#search). |
| `slurm`     | object | no       | see below        | SLURM resource requests. See [slurm](#slurm). |
| `hydra`     | object | no       | see below        | Hydra command and static overrides. See [hydra](#hydra). |
| `parameters`| object | **yes**  | —                | Hyperparameter definitions. At least one parameter is required. See [parameters](#parameters). |
| `constraints`| list  | no       | `[]`             | Constraints that filter or modify parameter combinations. See [constraints](#constraints). |

---

### `search`

Controls how parameter combinations are generated.

| Field      | Type   | Required | Default  | Description |
|------------|--------|----------|----------|-------------|
| `mode`     | string | no       | `"grid"` | `"grid"` or `"axes"`. |
| `defaults` | object | **yes** if mode is `"axes"` | — | A mapping of every parameter name to its default value. Required for axes mode, ignored for grid mode. |

**Grid mode** generates the full Cartesian product of all parameter values. If you have 3 parameters with 4, 3, and 2 levels respectively, you get 4 × 3 × 2 = 24 trials.

**Axes mode** (one-at-a-time) starts from a default combination, then varies each parameter independently while holding all others at their default. This produces 1 + Σ(levels_i − 1) trials. For the same 4/3/2 example: 1 + 3 + 2 + 1 = 7 trials.

```yaml
# Grid mode (default):
search:
  mode: grid

# Axes mode:
search:
  mode: axes
  defaults:
    learning_rate: 0.001
    optimizer: adam
    batch_size: 64
```

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

Configuration for the Hydra command that each trial runs.

| Field              | Type         | Required | Default             | Description |
|--------------------|--------------|----------|---------------------|-------------|
| `command`          | string       | no       | `"python train.py"` | The base command to execute. This is informational — the launcher script is responsible for actually running it. It is available for reference in documentation and logs. |
| `config_path`      | string       | no       | *omitted*           | Path to the Hydra config directory. Informational — pass it in your launcher if needed. |
| `static_overrides` | list[string] | no       | `[]`                | Hydra overrides appended to every trial's override string. Use for values that are constant across the sweep but differ from Hydra defaults (e.g. `"data.path=/scratch/datasets"`). |

```yaml
hydra:
  command: "python train.py"
  config_path: "./configs"
  static_overrides:
    - "data.path=/scratch/datasets"
    - "trainer.seed=42"
```

**How overrides reach the training command**: HyperWhip constructs a Hydra override string for each trial by combining the trial's parameter values with any `static_overrides`. This string is passed as `$1` to your launcher script. For example, a trial might receive:

```
learning_rate=0.001 optimizer=adam batch_size=64 data.path=/scratch/datasets trainer.seed=42
```

Your launcher script is responsible for passing this to the Hydra command (see [Launcher Script](#launcher-script)).

---

### `parameters`

A YAML mapping where each key is the parameter name and the value is a specification object. Parameter names should match the Hydra config keys you want to override (e.g. `model.learning_rate`, `training.batch_size`). You can use dotted names for nested Hydra config paths.

Each parameter has a `type` field that determines its other required fields:

#### Discrete parameters

| Field    | Type      | Required | Description |
|----------|-----------|----------|-------------|
| `type`   | string    | **yes**  | Must be `"discrete"`. |
| `values` | list[any] | **yes**  | List of values to sweep over. Can be strings, integers, floats, or booleans. |

```yaml
parameters:
  optimizer:
    type: discrete
    values: [adam, sgd, adamw]
  use_dropout:
    type: discrete
    values: [true, false]
  num_layers:
    type: discrete
    values: [2, 4, 8]
```

#### Continuous parameters

| Field   | Type    | Required | Default    | Description |
|---------|---------|----------|------------|-------------|
| `type`  | string  | **yes**  | —          | Must be `"continuous"`. |
| `low`   | float   | **yes**  | —          | Lower bound (inclusive). |
| `high`  | float   | **yes**  | —          | Upper bound (inclusive). |
| `scale` | string  | no       | `"linear"` | `"linear"` for uniform spacing, `"log"` for log-uniform spacing. Log scale requires `low > 0` and `high > 0`. |
| `steps` | integer | no       | `5`        | Number of evenly-spaced points to discretize into. With `steps: 5` and `low: 0, high: 1`, you get `[0.0, 0.25, 0.5, 0.75, 1.0]`. |

```yaml
parameters:
  learning_rate:
    type: continuous
    low: 1e-5
    high: 1e-2
    scale: log
    steps: 5
  weight_decay:
    type: continuous
    low: 0.0
    high: 0.1
    scale: linear
    steps: 3
```

**Log scale discretization**: With `low: 1e-4, high: 1e-2, scale: log, steps: 3`, the values are spaced evenly in log10 space: `[0.0001, 0.001, 0.01]`.

---

### `constraints`

An ordered list of constraint objects. Constraints are applied as post-filters after all parameter combinations are generated. They are evaluated in order.

Each constraint has:

| Field     | Type   | Required | Description |
|-----------|--------|----------|-------------|
| `name`    | string | no       | Human-readable label for this constraint (used in error messages). |
| `when`    | object | **yes**  | Condition: a mapping of `parameter_name: value`. The constraint activates only for combinations where **all** listed parameters match their specified values. |
| `exclude` | object | no*      | Exclusion rule: a mapping of `parameter_name: [list of values]`. When the `when` condition matches, any combination where the target parameter has one of these values is **removed**. |
| `force`   | object | no*      | Force rule: a mapping of `parameter_name: value`. When the `when` condition matches, the target parameter is **overridden** to the specified value. Resulting duplicates are automatically removed. |

*At least one of `exclude` or `force` is required per constraint.

```yaml
constraints:
  # When optimizer is sgd, remove trials with learning_rate >= 0.01
  - name: sgd_no_high_lr
    when:
      optimizer: sgd
    exclude:
      learning_rate: [0.01, 0.1]

  # When optimizer is adamw, force weight_decay to 0.01
  - name: adamw_fixed_wd
    when:
      optimizer: adamw
    force:
      weight_decay: 0.01

  # Compound condition: when optimizer=sgd AND use_nesterov=true, force momentum
  - name: nesterov_momentum
    when:
      optimizer: sgd
      use_nesterov: true
    force:
      momentum: 0.9
```

**Constraint evaluation order matters**. Constraints are applied sequentially. An earlier `exclude` constraint can remove combinations before a later `force` constraint sees them.

**Deduplication**: After all constraints are applied, duplicate combinations (identical parameter dictionaries) are removed. This commonly happens with `force` constraints that collapse multiple combinations into one.

---

## Launcher Script

The launcher script is a user-provided bash script. HyperWhip does **not** manage your container runtime, environment modules, conda environments, or any other setup. The launcher script is where you handle all of that.

### Contract

1. HyperWhip calls your launcher as: `bash <launcher_path> "<hydra_overrides>"`
2. The first argument (`$1`) is a space-separated Hydra override string, e.g.: `learning_rate=0.001 optimizer=adam batch_size=64 data.path=/scratch/datasets`
3. Your script must invoke the Hydra training command with these overrides.
4. The script runs inside a SLURM job allocation — SLURM environment variables (`$SLURM_JOB_ID`, `$SLURM_ARRAY_TASK_ID`, etc.) are available.
5. Exit code 0 means success; nonzero means failure. SLURM marks the task accordingly.

### Example: Apptainer/Singularity launcher

```bash
#!/bin/bash
# launch.sh — Run training inside an Apptainer container
set -euo pipefail

OVERRIDES="$1"
CONTAINER="/path/to/your/container.sif"

# Bind paths: project directory and data
BINDS="/scratch:/scratch,/home/$USER:/home/$USER"

apptainer exec \
    --nv \
    --bind "$BINDS" \
    "$CONTAINER" \
    python train.py $OVERRIDES
```

### Example: Conda environment launcher

```bash
#!/bin/bash
# launch.sh — Run training in a conda environment
set -euo pipefail

OVERRIDES="$1"

source /opt/conda/etc/profile.d/conda.sh
conda activate myenv

python train.py $OVERRIDES
```

### Example: Docker via Enroot/Pyxis launcher

```bash
#!/bin/bash
# launch.sh — Run training via enroot
set -euo pipefail

OVERRIDES="$1"
IMAGE="nvcr.io/nvidia/pytorch:24.01-py3"

srun --container-image="$IMAGE" \
     --container-mounts="/scratch:/scratch" \
     python train.py $OVERRIDES
```

### Example: Environment modules launcher

```bash
#!/bin/bash
# launch.sh — Run training with environment modules
set -euo pipefail

OVERRIDES="$1"

module purge
module load cuda/12.1
module load python/3.11

source /home/$USER/venvs/training/bin/activate

python train.py $OVERRIDES
```

### Idempotency requirements

HyperWhip's `launch` command is idempotent — rerunning it resubmits only pending and failed trials with the same array indices and parameters. For this to work, **your Hydra application must also be idempotent**:

- **Checkpoint on a deterministic path**: Use the Hydra override values to construct a unique output directory so that the same parameters always write to the same location. For example, configure Hydra's `hydra.run.dir` based on the parameter values or `$SLURM_ARRAY_TASK_ID`.
- **Resume from checkpoint**: On startup, check if a checkpoint exists at the output path and resume from it instead of starting from scratch.
- **Do not fail on existing output**: If the output directory already exists from a previous (possibly failed) run, the training script should handle this gracefully.

Example Hydra config for deterministic output paths:

```yaml
# In your Hydra config (config.yaml):
hydra:
  run:
    dir: ./outputs/${experiment_name}/trial_${slurm_array_task_id}
```

Or use the HyperWhip workspace's array task ID in your training script:

```python
import os
task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
output_dir = f"./outputs/trial_{task_id}"
```

---

## Complete Example

### Directory layout

```
my_project/
  train.py                # Your Hydra training script
  configs/
    config.yaml           # Your Hydra base config
  hyperwhip.yaml          # HyperWhip sweep configuration
  launch.sh               # Launcher script
```

### `hyperwhip.yaml`

```yaml
name: resnet_sweep
workspace: ./experiments/resnet_sweep

search:
  mode: grid

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
  command: "python train.py"
  config_path: "./configs"
  static_overrides:
    - "data.root=/scratch/imagenet"
    - "trainer.max_epochs=90"

parameters:
  model.learning_rate:
    type: continuous
    low: 1e-4
    high: 1e-1
    scale: log
    steps: 4
  model.optimizer:
    type: discrete
    values: [sgd, adamw]
  model.weight_decay:
    type: continuous
    low: 0.0
    high: 0.01
    scale: linear
    steps: 3
  data.batch_size:
    type: discrete
    values: [64, 128, 256]

constraints:
  - name: sgd_no_high_lr
    when:
      model.optimizer: sgd
    exclude:
      model.learning_rate: [0.1]
  - name: adamw_fixed_wd
    when:
      model.optimizer: adamw
    force:
      model.weight_decay: 0.001
```

### `launch.sh`

```bash
#!/bin/bash
set -euo pipefail

OVERRIDES="$1"
CONTAINER="/shared/containers/pytorch-24.01.sif"

apptainer exec \
    --nv \
    --bind "/scratch:/scratch,/shared:/shared" \
    "$CONTAINER" \
    python train.py $OVERRIDES
```

### Usage

```bash
# See what would be submitted:
hyperwhip launch hyperwhip.yaml --dry-run

# Submit:
hyperwhip launch hyperwhip.yaml

# Check progress:
hyperwhip monitor hyperwhip.yaml

# Re-run to resubmit any failed trials:
hyperwhip launch hyperwhip.yaml

# Clean up everything:
hyperwhip clean hyperwhip.yaml --all
```

---

## Workspace Layout

After `hyperwhip launch`, the workspace directory contains:

```
experiments/resnet_sweep/
  .hyperwhip/
    manifest.json       # Array of trial objects: {index, params, status}
    job_ids.json        # Records of submitted SLURM jobs
    job.sbatch          # The generated SLURM batch script
    logs/
      0.out, 0.err      # stdout/stderr for array task 0
      1.out, 1.err      # stdout/stderr for array task 1
      ...
```

- **manifest.json**: The authoritative mapping of array index to parameter values. Do not edit manually.
- **job.sbatch**: The generated script. You can inspect it to verify correctness.
- **logs/**: SLURM captures stdout/stderr here. The `monitor` command reads the last line of each `.out` file.

---

## Generated SLURM Script

For reference, HyperWhip generates a batch script like this:

```bash
#!/bin/bash
#SBATCH --job-name=hyperwhip_resnet_sweep
#SBATCH --array=0-23
#SBATCH --partition=gpu
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --output=<workspace>/.hyperwhip/logs/%a.out
#SBATCH --error=<workspace>/.hyperwhip/logs/%a.err
#SBATCH --export=ALL

# Resolve Hydra overrides for this array task
OVERRIDES=$(python -m hyperwhip resolve-overrides "<workspace>/.hyperwhip/manifest.json" "$SLURM_ARRAY_TASK_ID" --static "data.root=/scratch/imagenet trainer.max_epochs=90")

# Invoke the user's launcher script
bash "/path/to/launch.sh" "$OVERRIDES"
```

The `resolve-overrides` subcommand is an internal HyperWhip utility. It reads `manifest.json`, looks up the entry for `$SLURM_ARRAY_TASK_ID`, and prints the Hydra override string. This means `python` and the `hyperwhip` package must be available on the compute node (outside the container). If your compute nodes don't have Python available outside the container, see the note below.

### Compute nodes without Python

If `python` is not available on the bare compute node, you can work around this by modifying your launcher to read the manifest directly. The manifest is a JSON array where each object has an `index` field matching the array task ID and a `params` object. Example launcher that reads the manifest with `jq`:

```bash
#!/bin/bash
set -euo pipefail

MANIFEST=".hyperwhip/manifest.json"
TASK_ID="$SLURM_ARRAY_TASK_ID"

# Build overrides from manifest using jq
OVERRIDES=$(jq -r --argjson id "$TASK_ID" '
  .[] | select(.index == $id) | .params | to_entries | map("\(.key)=\(.value)") | join(" ")
' "$MANIFEST")

# Append static overrides
OVERRIDES="$OVERRIDES data.root=/scratch/imagenet trainer.max_epochs=90"

CONTAINER="/shared/containers/pytorch-24.01.sif"
apptainer exec --nv --bind "/scratch:/scratch" "$CONTAINER" \
    python train.py $OVERRIDES
```
