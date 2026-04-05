"""Scaffold a starter hyperwhip.yaml and launch.sh."""

import os
import stat

CONFIG_TEMPLATE = """\
name: {name}

# Grid controls which parameters are swept via Cartesian product.
#   grid: all          — grid over everything
#   grid: [lr, wd]     — grid over lr and wd, hold others at defaults
#   (omit grid)        — one-at-a-time from defaults
{grid_line}

# Default values for each parameter (required unless grid: all)
{defaults_block}
slurm:
  partition: {partition}
  time: "{time}"
  mem: "{mem}"
  cpus_per_task: {cpus}
{gres_line}
launcher: ./launch.sh

hydra:
{static_overrides_block}
parameters:
{parameters_block}
"""

LAUNCHER_TEMPLATE = """\
#!/bin/bash
# launch.sh — Launcher script for HyperWhip
# Receives Hydra overrides as $1. Modify this script to set up your
# container, conda environment, modules, or any other runtime config.
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
    name=None,
    grid="all",
    partition="default",
    time="04:00:00",
    mem="8G",
    cpus=1,
    gres=None,
    overwrite=False,
):
    """Generate hyperwhip.yaml and launch.sh in the given directory.

    Returns (config_path, launcher_path).
    """
    directory = os.path.abspath(directory)
    os.makedirs(directory, exist_ok=True)

    if name is None:
        name = os.path.basename(directory)
        if not name or name == ".":
            name = "experiment"

    config_path = os.path.join(directory, "hyperwhip.yaml")
    launcher_path = os.path.join(directory, "launch.sh")

    if not overwrite:
        for path, label in [(config_path, "hyperwhip.yaml"), (launcher_path, "launch.sh")]:
            if os.path.exists(path):
                raise FileExistsError(
                    f"{label} already exists at {path}. Use --force to overwrite."
                )

    # Build config content
    gres_line = f'  gres: "{gres}"' if gres else "  # gres: \"gpu:1\"  # uncomment for GPU jobs"

    if grid == "all":
        grid_line = "grid: all"
        defaults_block = (
            "# defaults:\n"
            "#   learning_rate: 0.001\n"
            "#   optimizer: adam\n"
        )
    else:
        grid_line = "# grid: all"
        defaults_block = (
            "defaults:\n"
            "  # Set default value for each parameter:\n"
            "  # learning_rate: 0.001\n"
            "  # optimizer: adam\n"
        )

    static_overrides_block = (
        "  # static_overrides:\n"
        "  #   - \"data.path=/scratch/datasets\"\n"
    )

    parameters_block = _build_example_parameters()

    config_content = CONFIG_TEMPLATE.format(
        name=name,
        grid_line=grid_line,
        defaults_block=defaults_block,
        partition=partition,
        time=time,
        mem=mem,
        cpus=cpus,
        gres_line=gres_line,
        static_overrides_block=static_overrides_block,
        parameters_block=parameters_block,
    )

    launcher_content = LAUNCHER_TEMPLATE

    with open(config_path, "w") as f:
        f.write(config_content)

    with open(launcher_path, "w") as f:
        f.write(launcher_content)
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

  # Example discrete parameter:
  # optimizer:
  #   abbrev: opt
  #   type: discrete
  #   values: [adam, sgd, adamw]

# constraints:
#   - name: example_constraint
#     when:
#       optimizer: sgd
#     exclude:
#       learning_rate: [0.01]
"""
