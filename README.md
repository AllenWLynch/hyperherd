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
whip --help
```

This should print the available subcommands: `init`, `run`, `test`, `status`, `tail`, `res`, `clean`.

### Dependencies

- Python >= 3.8
- [PyYAML](https://pyyaml.org/) (installed automatically)
- [Pydantic](https://docs.pydantic.dev/) >= 2.0 (installed automatically)
- A SLURM cluster with `sbatch`, `sacct`, `squeue`, and `scancel` available on the submission host

## Quick Start

### 1. Scaffold a new experiment

```bash
whip init my_experiment --partition gpu --gres gpu:1
```

This creates `my_experiment/hyperwhip.yaml` and `my_experiment/launch.sh` with sensible defaults. Edit both files to match your setup:

- **hyperwhip.yaml** — define your parameters, grid mode, SLURM resources, and conditions
- **launch.sh** — set up your container, conda environment, or module loads

### 2. Validate and preview

```bash
# Validate Hydra config for trial 0 (runs locally via launcher with --cfg job):
whip test my_experiment

# Preview the full sweep (no SLURM interaction, runs preflight checks):
whip run my_experiment --dry-run
```

### 3. Launch

```bash
whip run my_experiment
```

### 4. Monitor

```bash
whip status my_experiment
```

### 5. Resubmit failures

```bash
# Re-running only resubmits pending/failed trials:
whip run my_experiment
```

### 6. Collect results

Log metrics from your training script:

```python
from hyperwhip import log_result

log_result("test_accuracy", 0.95)
log_result("final_loss", 0.12)
```

Then print a TSV summary:

```bash
whip res my_experiment
```

### 7. Clean up

```bash
whip clean my_experiment --all
```

## Documentation

See [docs/configuration.md](docs/configuration.md) for the full configuration reference, launcher script examples, constraint system, search mode details, and result logging API.
