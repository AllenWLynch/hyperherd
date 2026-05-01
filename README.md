# HyperHerd

Launch and monitor hyperparameter optimization job arrays on SLURM.

You write a YAML sweep config and a bash launcher script. HyperHerd generates a SLURM job array, hands each task a string of `name=value` overrides, and tracks state across resubmissions.

[Hydra](https://hydra.cc/) is the recommended trainer harness (its CLI consumes the override format directly), but the launcher is free-form bash — parse the arguments however you want.

📖 **Full documentation: [allenwlynch.github.io/hyperherd](https://allenwlynch.github.io/hyperherd/)**

## Quick start

```bash
# Install
pip install git+https://github.com/AllenWLynch/hyperherd.git

# Install the Claude Code skill for authoring sweep configs
herd install-skill

# Scaffold a workspace
herd init my_experiment

# Edit my_experiment/hyperherd.yaml and my_experiment/launch.sh, then:
herd run my_experiment --dry-run    # preview
herd run my_experiment              # submit
herd status my_experiment           # monitor
```

## Documentation

- [Getting started](https://allenwlynch.github.io/hyperherd/getting-started/) — install, scaffold, run your first sweep
- [Sweep config reference](https://allenwlynch.github.io/hyperherd/configuration/) — every field in `hyperherd.yaml`
- [Conditions](https://allenwlynch.github.io/hyperherd/conditions/) — filter or modify parameter combinations
- [Launcher script](https://allenwlynch.github.io/hyperherd/launcher/) — contract + examples (Apptainer, conda, non-Hydra)
- [Command reference](https://allenwlynch.github.io/hyperherd/commands/) — every `herd` subcommand
- [Claude Code skill](https://allenwlynch.github.io/hyperherd/claude-skill/) — authoring configs by asking Claude

## Requirements

- Python ≥ 3.8
- SLURM cluster with `sbatch`, `sacct`, `squeue`, `scancel` on the submission host
