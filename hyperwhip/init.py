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
    from_config=None,
    from_launcher=None,
):
    """Generate hyperwhip.yaml and launch.sh in the given directory.

    If from_config is given, copies that file instead of generating a template.
    If from_launcher is given, copies that file instead of generating a template.

    Returns (config_path, launcher_path).
    """
    import shutil

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

    # Config: copy from source or generate from template
    if from_config:
        src = os.path.abspath(from_config)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"Config source not found: {src}")
        shutil.copy2(src, config_path)
    else:
        gres_line = f'  gres: "{gres}"' if gres else "  # gres: \"gpu:1\"  # uncomment for GPU jobs"
        grid_line = "grid: all" if grid == "all" else "# grid: all"

        static_overrides_block = (
            "  # static_overrides:\n"
            "  #   - \"data.path=/scratch/datasets\"\n"
        )
        parameters_block = _build_example_parameters()

        config_content = CONFIG_TEMPLATE.format(
            name=name,
            grid_line=grid_line,
            partition=partition,
            time=time,
            mem=mem,
            cpus=cpus,
            gres_line=gres_line,
            static_overrides_block=static_overrides_block,
            parameters_block=parameters_block,
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
