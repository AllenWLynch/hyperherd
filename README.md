# HyperWhip

Launch and monitor hyperparameter optimization job arrays on SLURM.

HyperWhip takes a YAML configuration file describing your hyperparameter search space and submits it as a SLURM job array. Each array task runs one parameter combination through your training script via [Hydra](https://hydra.cc/) overrides. A user-provided launcher script handles container setup, environment modules, or any other runtime configuration.

## Installation

### From source (recommended for development)

```bash
git clone <repo-url> && cd hyperwhip
pip install -e .
```

### From the repository directly

```bash
pip install git+<repo-url>
```

### Verify installation

```bash
hyperwhip --help
```

This should print the available subcommands: `launch`, `monitor`, `clean`.

### Dependencies

- Python >= 3.8
- [PyYAML](https://pyyaml.org/) (installed automatically)
- A SLURM cluster with `sbatch`, `sacct`, `squeue`, and `scancel` available on the submission host

## Quick Start

### 1. Write a config file

Create `hyperwhip.yaml`:

```yaml
name: my_sweep
workspace: ./experiments/my_sweep

search:
  mode: grid

slurm:
  partition: gpu
  time: "04:00:00"
  mem: "32G"
  gres: "gpu:1"

launcher: ./launch.sh

hydra:
  command: "python train.py"

parameters:
  learning_rate:
    type: continuous
    low: 1e-4
    high: 1e-2
    scale: log
    steps: 3
  optimizer:
    type: discrete
    values: [adam, sgd]
```

### 2. Write a launcher script

Create `launch.sh`:

```bash
#!/bin/bash
set -euo pipefail
OVERRIDES="$1"

# Example: run inside an Apptainer container
apptainer exec --nv /path/to/container.sif python train.py $OVERRIDES
```

### 3. Launch

```bash
# Preview first:
hyperwhip launch hyperwhip.yaml --dry-run

# Submit:
hyperwhip launch hyperwhip.yaml
```

### 4. Monitor

```bash
hyperwhip monitor hyperwhip.yaml
```

### 5. Resubmit failures

```bash
# Re-running launch only resubmits pending/failed trials:
hyperwhip launch hyperwhip.yaml
```

### 6. Clean up

```bash
hyperwhip clean hyperwhip.yaml --all
```

## Documentation

See [docs/configuration.md](docs/configuration.md) for the full configuration reference, launcher script examples, constraint system, and search mode details.
