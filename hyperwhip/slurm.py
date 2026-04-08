"""SLURM interaction: sbatch submission, status queries, job cancellation."""

import os
import re
import subprocess
from typing import Dict, List, Optional, Tuple

from hyperwhip.config import Config
from hyperwhip import manifest


def generate_sbatch_script(config: Config, indices: List[int]) -> str:
    """Generate a SLURM batch script for the job array."""
    ws = manifest.workspace_path(config.workspace)
    log_dir = manifest.logs_path(config.workspace)

    # Build array spec (e.g. "0-49" or "0,2,5,7-10")
    array_spec = _indices_to_array_spec(indices)

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=hyperwhip_{config.name}",
        f"#SBATCH --array={array_spec}",
        f"#SBATCH --partition={config.slurm.partition}",
        f"#SBATCH --time={config.slurm.time}",
        f"#SBATCH --mem={config.slurm.mem}",
        f"#SBATCH --cpus-per-task={config.slurm.cpus_per_task}",
    ]

    if config.slurm.gres:
        lines.append(f"#SBATCH --gres={config.slurm.gres}")

    lines.extend([
        f"#SBATCH --output={log_dir}/%a.out",
        f"#SBATCH --error={log_dir}/%a.err",
    ])

    for arg in config.slurm.extra_args:
        lines.append(f"#SBATCH {arg}")

    # Build static overrides for the resolve command
    static_flag = ""
    if config.hydra.static_overrides:
        escaped = " ".join(config.hydra.static_overrides)
        static_flag = f' --static "{escaped}"'

    lines.extend([
        "",
        "# Export HyperWhip environment variables",
        f'export HYPERWHIP_WORKSPACE="{config.workspace}"',
        "export HYPERWHIP_TRIAL_ID=\"$SLURM_ARRAY_TASK_ID\"",
        f'export HYPERWHIP_EXPERIMENT_NAME=$(python -m hyperwhip resolve-name '
        f'"{ws}/{manifest.MANIFEST_FILE}" "$SLURM_ARRAY_TASK_ID")',
        "",
        "# Resolve Hydra overrides for this array task (includes experiment_name=...)",
        f'OVERRIDES=$(python -m hyperwhip resolve-overrides "{ws}/{manifest.MANIFEST_FILE}" '
        f'"$SLURM_ARRAY_TASK_ID"{static_flag})',
        "",
        "# Invoke the user's launcher script",
        f'bash "{config.launcher}" "$OVERRIDES"',
    ])

    return "\n".join(lines) + "\n"


def _indices_to_array_spec(indices: List[int]) -> str:
    """Convert a list of indices to a compact SLURM array spec.

    Examples:
        [0, 1, 2, 3] -> "0-3"
        [0, 1, 3, 5, 6, 7] -> "0-1,3,5-7"
    """
    if not indices:
        raise ValueError("No indices to submit")

    indices = sorted(set(indices))
    ranges = []
    start = indices[0]
    end = indices[0]

    for i in indices[1:]:
        if i == end + 1:
            end = i
        else:
            ranges.append((start, end))
            start = i
            end = i
    ranges.append((start, end))

    parts = []
    for s, e in ranges:
        if s == e:
            parts.append(str(s))
        else:
            parts.append(f"{s}-{e}")

    return ",".join(parts)


def submit_job(config: Config, script_content: str, dry_run: bool = False) -> Optional[str]:
    """Submit a job array via sbatch. Returns the SLURM job ID, or None for dry run."""
    sbatch_file = manifest.sbatch_path(config.workspace)

    with open(sbatch_file, "w") as f:
        f.write(script_content)
    os.chmod(sbatch_file, 0o755)

    if dry_run:
        return None

    result = subprocess.run(
        ["sbatch", sbatch_file],
        capture_output=True,
        text=True,
        cwd=config.workspace,
    )

    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed:\n{result.stderr}")

    # Parse job ID from "Submitted batch job 12345"
    match = re.search(r"Submitted batch job (\d+)", result.stdout)
    if not match:
        raise RuntimeError(f"Could not parse job ID from sbatch output: {result.stdout}")

    return match.group(1)


def query_job_status(job_ids: List[str]) -> Dict[Tuple[str, int], str]:
    """Query SLURM for the status of job array tasks.

    Returns a dict of (job_id, array_index) -> status string.
    Status values: PENDING, RUNNING, COMPLETED, FAILED, CANCELLED, TIMEOUT, etc.
    """
    if not job_ids:
        return {}

    job_spec = ",".join(job_ids)

    result = subprocess.run(
        [
            "sacct",
            "-j", job_spec,
            "--format=JobID,State,Elapsed",
            "--noheader",
            "--parsable2",
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        # sacct might not be available or jobs might have aged out
        # Fall back to squeue for running/pending jobs
        return _query_squeue(job_ids)

    statuses = {}
    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        job_id_str = parts[0]
        state = parts[1].split()[0] if parts[1] else "UNKNOWN"  # strip trailing text

        # Parse "12345_0" format (individual job array tasks)
        match = re.match(r"(\d+)_(\d+)$", job_id_str)
        if match:
            jid = match.group(1)
            array_idx = int(match.group(2))
            statuses[(jid, array_idx)] = state
            continue

        # Parse "12345_[0-10]" compact range format (e.g. cancelled before running)
        match = re.match(r"(\d+)_\[(.+)\]$", job_id_str)
        if match:
            jid = match.group(1)
            for idx in _parse_array_range(match.group(2)):
                statuses[(jid, idx)] = state

    return statuses


def _parse_array_range(spec: str) -> List[int]:
    """Parse a SLURM array range spec like '0-10' or '0-3,5,7-9' into indices."""
    indices = []
    for part in spec.split(","):
        if "-" in part:
            start, end = part.split("-", 1)
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(part))
    return indices


def _query_squeue(job_ids: List[str]) -> Dict[Tuple[str, int], str]:
    """Fallback: query squeue for running/pending jobs."""
    job_spec = ",".join(job_ids)
    result = subprocess.run(
        [
            "squeue",
            "-j", job_spec,
            "--format=%i %t",
            "--noheader",
        ],
        capture_output=True,
        text=True,
    )

    statuses = {}
    if result.returncode != 0:
        return statuses

    state_map = {
        "PD": "PENDING",
        "R": "RUNNING",
        "CG": "COMPLETING",
        "CD": "COMPLETED",
        "F": "FAILED",
        "CA": "CANCELLED",
        "TO": "TIMEOUT",
    }

    for line in result.stdout.strip().split("\n"):
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        job_id_str = parts[0]
        state_code = parts[1]

        match = re.match(r"(\d+)_(\d+)", job_id_str)
        if match:
            jid = match.group(1)
            array_idx = int(match.group(2))
            statuses[(jid, array_idx)] = state_map.get(state_code, state_code)

    return statuses


def cancel_jobs(job_ids: List[str]) -> None:
    """Cancel SLURM jobs via scancel."""
    if not job_ids:
        return
    for jid in job_ids:
        subprocess.run(["scancel", jid], capture_output=True, text=True)


def get_log_tail(base: str, index: int, lines: int = 1) -> str:
    """Read the last N lines from a task's stdout log."""
    log_file = os.path.join(manifest.logs_path(base), f"{index}.out")
    if not os.path.isfile(log_file):
        return ""
    try:
        with open(log_file, "r") as f:
            all_lines = f.readlines()
            tail = all_lines[-lines:] if all_lines else []
            return "".join(tail).strip()
    except (OSError, UnicodeDecodeError):
        return "<unreadable>"
