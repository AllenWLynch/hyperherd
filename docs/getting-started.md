# Getting started

## Installation

```bash
pip install git+https://github.com/AllenWLynch/hyperherd.git
```

For local development:

```bash
git clone https://github.com/AllenWLynch/hyperherd.git
cd hyperherd
pip install -e .
```

### Install the Claude Code skill

HyperHerd ships with a [Claude Code skill](claude-skill.md) for authoring sweep configs. After `pip install`, run:

```bash
herd install-skill
```

This drops the skill into `~/.claude/skills/hyperherd-config/`. Next time you open Claude Code, you can ask it to write or edit a `hyperherd.yaml` and it will use the skill's checklist + patterns.

### Verify

```bash
herd --help
```

Should print the available subcommands.

### Dependencies

- Python ≥ 3.8
- [PyYAML](https://pyyaml.org/) and [Pydantic](https://docs.pydantic.dev/) ≥ 2.0 (installed automatically)
- A SLURM cluster with `sbatch`, `sacct`, `squeue`, and `scancel` on the submission host

## Your first sweep

### 1. Scaffold a workspace

```bash
herd init my_experiment
```

This creates `my_experiment/hyperherd.yaml` and `my_experiment/launch.sh` with template content. Edit both to match your setup:

- `hyperherd.yaml` — define your parameters, grid mode, SLURM resources, and conditions
- `launch.sh` — set up your container, conda environment, or module loads

If you already have a config or launcher you want to reuse from another experiment, drop them in directly:

```bash
herd init my_experiment --config /path/to/existing.yaml --launcher /path/to/launch.sh
```

| Flag | What it does |
|------|--------------|
| `--config FILE` | Copy `FILE` in as `hyperherd.yaml` instead of generating a template |
| `--launcher FILE` | Copy `FILE` in as `launch.sh` instead of generating a template |
| `-f, --force` | Overwrite existing files in the target directory |

### 2. Validate before submitting

```bash
# Preview the full sweep (no SLURM interaction, runs preflight checks):
herd run my_experiment --dry-run

# Hydra users: validate Hydra config for trial 0 by running locally with --cfg job:
herd test my_experiment
```

`herd test` is the only Hydra-specific subcommand — it appends `--cfg job` to the override string so Hydra prints the resolved config and exits without running training. If your trainer doesn't use Hydra, skip it and use `herd run --dry-run` + `herd local` instead.

### 3. Launch

```bash
herd run my_experiment
```

You'll see the SLURM job array submitted. Most `herd` subcommands work without an argument when run from inside the workspace:

```bash
cd my_experiment
herd status
herd tail 3
```

### 4. Monitor

```bash
herd status        # one-shot status table
herd stats         # sacct accounting (runtime + memory) for every trial
herd tail 3        # last 20 lines of trial 3's stdout
```

### 5. Resubmit failures, or edit the sweep mid-flight

`herd run` is idempotent. Re-running the same workspace only submits trials that are `ready`, `failed`, or `cancelled`:

```bash
herd run        # picks up just the trials that need to run
```

!!! tip "Editing the config while trials are running"
    You can update `hyperherd.yaml` between (or even during) runs. On the next `herd run`, HyperHerd diffs the new manifest against the existing one, **appends new trials**, and asks before dropping any completed/running trials your edits would remove. Already-completed trials keep their results — they aren't re-run.

    Typical workflow: launch a coarse sweep, watch a few results come in, then add another value to a parameter or tighten a `condition` and run again. The new trials get submitted, the existing ones stay. See [reconciliation](workspace.md#re-running-and-reconciliation) for the exact rules.

### 6. Collect results

Log per-trial summary metrics from your training code:

```python
from hyperherd import log_result

log_result("test_accuracy", 0.95)
log_result("final_loss", 0.12)
```

Then print a TSV summary:

```bash
herd res
```

### 7. Clean up

```bash
herd clean --all
```

## Next steps

- The full **[sweep config reference](configuration.md)** covers every field in `hyperherd.yaml`.
- **[Conditions](conditions.md)** are how you filter or modify parameter combinations.
- **[Launcher script](launcher.md)** patterns for Apptainer/Singularity, conda, Docker.
- The **[command reference](commands.md)** documents every subcommand.
