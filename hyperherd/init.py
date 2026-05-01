"""Scaffold a starter hyperherd.yaml and launch.sh."""

import os
import stat

CONFIG_TEMPLATE = """\
name: {name}

# Grid controls which parameters are swept via Cartesian product.
#   grid: all          — grid over everything
#   grid: [lr, wd]     — grid over lr and wd, hold others at defaults
#   (omit grid)        — one-at-a-time from defaults
grid: all

slurm:
  partition: default
  time: "04:00:00"
  mem: "8G"
  cpus_per_task: 1
  # gres: "gpu:1"          # uncomment for GPU jobs
  # max_concurrent: 8      # cap simultaneously running array tasks

launcher: ./launch.sh

# static_overrides: extra tokens appended to every trial's argument string
# (e.g. dataset paths, fixed Hydra flags). One entry per token; whitespace-
# split when the launcher receives them.
# static_overrides:
#   - "data.path=/scratch/datasets"

# `herd watch` posts trial state changes to a webhook (Slack/Discord/ntfy/
# anything that accepts a JSON or plain-text POST). Run in a nohup/tmux/
# screen session. With no `webhook` set, falls back to a per-workspace
# ntfy.sh topic and prints the subscribe URL on startup.
# watch:
#   webhook: https://hooks.slack.com/services/T.../B.../xxxx
#   format: slack                     # slack | discord | ntfy | raw
#   interval_seconds: 60              # how often to poll SLURM
#   heartbeat_minutes: 5              # min gap between heartbeat digests
#   events: [failed, done, heartbeat] # which events to deliver
#   summarize: true                   # opt-in: `claude -p` paragraph in raw payloads

parameters:
{parameters_block}
"""

LAUNCHER_TEMPLATE = """\
#!/bin/bash
# launch.sh — Launcher script for HyperHerd
# Receives the name=value override string as $1. Modify this script to set up
# your container, conda environment, modules, or any other runtime config.
set -euo pipefail

OVERRIDES="$1"

# --- Option A: Apptainer/Singularity container ---
# CONTAINER="/path/to/your/container.sif"
# apptainer exec --nv --bind "/scratch:/scratch" "$CONTAINER" \\
#     python train.py $OVERRIDES

# --- Option B: Conda environment ---
# source /opt/conda/etc/profile.d/conda.sh
# conda activate myenv
# python train.py $OVERRIDES

# --- Option C: Direct execution ---
python train.py $OVERRIDES
"""


def scaffold(
    directory,
    overwrite=False,
    from_config=None,
    from_launcher=None,
):
    """Generate hyperherd.yaml and launch.sh in the given directory.

    The generated files are templates with sensible defaults; edit the YAML to
    set partition/time/mem/etc. If `from_config` or `from_launcher` is given,
    that file is copied verbatim instead of generating a template.

    Returns (config_path, launcher_path).
    """
    import shutil

    directory = os.path.abspath(directory)
    os.makedirs(directory, exist_ok=True)

    name = os.path.basename(directory) or "experiment"

    config_path = os.path.join(directory, "hyperherd.yaml")
    launcher_path = os.path.join(directory, "launch.sh")

    if not overwrite:
        for path, label in [(config_path, "hyperherd.yaml"), (launcher_path, "launch.sh")]:
            if os.path.exists(path):
                raise FileExistsError(
                    f"{label} already exists at {path}. Use --force to overwrite."
                )

    # Config: copy from source or generate from template
    if from_config:
        src = os.path.abspath(from_config)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"Config source not found: {src}")
        shutil.copy2(src, config_path)
    else:
        config_content = CONFIG_TEMPLATE.format(
            name=name,
            parameters_block=_build_example_parameters(),
        )
        with open(config_path, "w") as f:
            f.write(config_content)

    # Launcher: copy from source or generate from template
    if from_launcher:
        src = os.path.abspath(from_launcher)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"Launcher source not found: {src}")
        shutil.copy2(src, launcher_path)
    else:
        with open(launcher_path, "w") as f:
            f.write(LAUNCHER_TEMPLATE)

    # Ensure launcher is executable
    os.chmod(launcher_path, os.stat(launcher_path).st_mode | stat.S_IEXEC)

    return config_path, launcher_path


def _build_example_parameters():
    """Return an example parameters block with comments."""
    return """\
  # Example continuous parameter (log-scaled):
  learning_rate:
    abbrev: lr        # short name used in experiment_name (e.g. "lr=0.001_opt=adam")
    type: continuous
    low: 1e-5
    high: 1e-2
    scale: log
    steps: 5
    # default: 0.001  # required unless grid: all

  # Example discrete parameter:
  # optimizer:
  #   abbrev: opt
  #   type: discrete
  #   values: [adam, sgd, adamw]
  #   default: adam    # required unless grid: all

# conditions:
#   - name: example_constraint
#     when:
#       optimizer: sgd
#     exclude:
#       learning_rate: [0.01]
"""
