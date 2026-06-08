"""Deterministic command handlers, transport-agnostic.

These are the "no agent in the loop" actions a user can invoke directly
from chat — slash commands in Discord, and (eventually) Slack equivalents.
Each function takes a workspace path plus the parsed parameters, runs the
underlying `herd` operation, and returns plain text suitable for posting
back to the channel.

No transport-specific code lives here; chat platforms handle their own
registration UI (slash command tree, etc.) and call into these handlers.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional


_RUNNABLE = [sys.executable, "-m", "hyperherd.cli"]
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[mK]")


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


# --- /status --------------------------------------------------------------

# How interesting is each status, smaller = more important. Used to
# sort the per-trial table so when Discord truncates a long status post
# the user loses the boring tail (completeds, readys) instead of the
# things they actually care about.
_STATUS_PRIORITY = {
    "running":   0,
    "queued":    1,
    "submitted": 2,
    "paused":    3,
    "failed":    4,
    "pruned":    5,
    "cancelled": 6,
    "completed": 7,
    "ready":     8,
}

# Statuses considered "active" by /running — i.e. work that hasn't
# settled into a terminal state.
_ACTIVE_STATUSES = ("running", "queued", "submitted")


def trial_sort_key(trial: dict) -> tuple:
    """Sort by status-priority then index — active trials first, then
    problems, then completed, then never-ran. Index breaks ties so the
    order is deterministic within each status bucket."""
    status = (trial.get("status") or "").lower()
    return (_STATUS_PRIORITY.get(status, 99), trial.get("index", 0))


def cmd_status(workspace: Path) -> str:
    """Show sweep totals + per-trial table. Backed by `herd snapshot`."""
    try:
        proc = subprocess.run(
            _RUNNABLE + ["snapshot", str(workspace)],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as e:
        return f"`herd snapshot` failed (exit {e.returncode}):\n{e.stderr or e.stdout}"

    try:
        snap = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return f"Unparseable snapshot output:\n{proc.stdout[:500]}"

    return _format_status(snap)


def cmd_running(workspace: Path) -> str:
    """Show only active trials (running + queued + submitted). Use
    when /status is too long to fit in a Discord message."""
    try:
        proc = subprocess.run(
            _RUNNABLE + ["snapshot", str(workspace)],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as e:
        return f"`herd snapshot` failed (exit {e.returncode}):\n{e.stderr or e.stdout}"

    try:
        snap = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return f"Unparseable snapshot output:\n{proc.stdout[:500]}"

    return _format_status(snap, only_active=True)


# Cap the trial table at this many rows so a sweep with hundreds of
# completed trials doesn't overflow Discord's 2000-char message limit
# (which would truncate the END of the message — exactly the spot
# we'd otherwise rely on for the "everything is fine" footer).
_STATUS_TRIAL_CAP = 35


def _format_status(snap: dict, only_active: bool = False) -> str:
    """Mobile-friendly status: totals header + per-trial `name: status`.

    Each line is just the experiment name plus its current state — the
    one piece of info a user reading on a phone needs from /status.
    Wider details (elapsed time, params, etc.) live behind /stats and
    the live dashboard."""
    name = snap.get("sweep_name", "?")
    totals = snap.get("totals") or {}
    trials = snap.get("trials") or []

    order = ["ready", "submitted", "queued", "running", "paused",
             "completed", "failed", "pruned", "cancelled"]
    counts = ", ".join(
        f"{totals[k]} {k}" for k in order if totals.get(k)
    ) or "(no trials yet)"
    counts += f" ({totals.get('total', len(trials))} total)"

    title = f"{name} — {counts}"
    if only_active:
        trials = [t for t in trials if (t.get("status") or "").lower() in _ACTIVE_STATUSES]
        title += "\n(active trials only — `/status` for the full list)"

    if not trials:
        if only_active:
            return f"{title}\n\n_No active trials right now._"
        return title

    trials = sorted(trials, key=trial_sort_key)
    truncated = 0
    if not only_active and len(trials) > _STATUS_TRIAL_CAP:
        truncated = len(trials) - _STATUS_TRIAL_CAP
        trials = trials[:_STATUS_TRIAL_CAP]

    idx_w = max((len(str(t.get("index", 0))) for t in trials), default=1) + 1
    aligned = _align_names([t.get("experiment_name") or "-" for t in trials])

    lines = []
    for t, nm_col in zip(trials, aligned):
        idx_col = f"#{t.get('index', '?')}".ljust(idx_w)
        status = (t.get("status") or "?")
        lines.append(f"  {idx_col}  {nm_col}  {status}")

    body = f"{title}\n\n" + "\n".join(lines)
    if truncated:
        body += (
            f"\n... and {truncated} more "
            f"(use `/running` or `herd status` for the full list)"
        )
    return body


# --- /stop ----------------------------------------------------------------

def cmd_stop(workspace: Path, index: int) -> str:
    """Cancel a single trial. Backed by `herd stop -i <index>`."""
    try:
        proc = subprocess.run(
            _RUNNABLE + ["stop", "-i", str(index), str(workspace)],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as e:
        return f"`herd stop -i {index}` failed: {_strip_ansi(e.stderr or e.stdout)}"
    out = _strip_ansi(proc.stdout or "").strip() or "(no output)"
    return f"Stopped trial {index}.\n{out}"


def cmd_stop_all(workspace: Path) -> str:
    """Cancel every running/queued trial. Backed by `herd stop --all`."""
    try:
        proc = subprocess.run(
            _RUNNABLE + ["stop", "--all", str(workspace)],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as e:
        return f"`herd stop --all` failed: {_strip_ansi(e.stderr or e.stdout)}"
    out = _strip_ansi(proc.stdout or "").strip() or "(no output)"
    return f"Stopped all live trials.\n{out}"


# --- /metrics -------------------------------------------------------------

def _format_metric_value(v: float) -> str:
    if abs(v) > 0 and (abs(v) < 1e-3 or abs(v) >= 1e6):
        return f"{v:.4g}"
    return f"{v:.6g}"


def _align_names(names: list) -> list:
    """Split each underscore-separated experiment name into segments and
    return versions where every segment position is padded to the maximum
    width seen across all names. Segments are joined with two spaces so
    they read as distinct columns in a monospace font.

    Trailing spaces on each row are stripped so the last column stays
    flush with the next field (status, value, etc.)."""
    rows = [n.split("_") for n in names]
    if not rows:
        return []
    n_cols = max(len(r) for r in rows)
    padded = [r + [""] * (n_cols - len(r)) for r in rows]
    widths = [max(len(r[i]) for r in padded) for i in range(n_cols)]
    return [
        "  ".join(seg.ljust(widths[i]) for i, seg in enumerate(row)).rstrip()
        for row in padded
    ]


def cmd_metrics(
    workspace: Path,
    metric: Optional[str] = None,
    smooth: int = 0,
) -> str:
    """Per-trial value of one metric, one line per trial.

    With no `metric`: list the available metric names so the user can
    pick one. With a metric: print `#<idx> <name>: <value>` for every
    non-ready trial that logged it. Optional `smooth` averages the
    last N points instead of taking the most recent."""
    if smooth < 0 or smooth > 1000:
        return f"`smooth` must be 0..1000 (got {smooth})."
    try:
        from hyperherd import manifest
        from hyperherd.logging import list_metric_streams, load_metric_stream
    except Exception as e:
        return f"Couldn't import: {e}"

    try:
        trials = manifest.load_manifest(str(workspace))
    except Exception as e:
        return f"Couldn't load manifest: {e}"

    non_ready = [t for t in trials if t.get("status") != "ready"]

    if metric is None or not metric.strip():
        names: set = set()
        for t in non_ready:
            for n in list_metric_streams(str(workspace), t["index"]):
                names.add(n)
        if not names:
            return "No metrics logged yet."
        listing = "\n".join(f"  • `{n}`" for n in sorted(names))
        return (
            f"Available metrics ({len(names)}):\n{listing}\n\n"
            f"Use `/metrics <name>` to see each trial's value."
        )

    metric = metric.strip()
    rows = []
    for trial in non_ready:
        idx = trial["index"]
        if metric not in list_metric_streams(str(workspace), idx):
            continue
        stream = load_metric_stream(str(workspace), idx, metric)
        numeric = [
            p["value"] for p in stream
            if isinstance(p.get("value"), (int, float))
        ]
        if not numeric:
            continue
        if smooth > 0 and len(numeric) > 1:
            tail = numeric[-smooth:]
            v = sum(tail) / len(tail)
        else:
            v = numeric[-1]
        rows.append((
            idx,
            (trial.get("experiment_name") or ""),
            float(v),
        ))

    if not rows:
        return f"No trial has logged `{metric}` yet."

    # Sort: loss-like metrics ascending (low = good), others descending.
    lc = metric.lower()
    reverse = "loss" not in lc and "error" not in lc
    rows.sort(key=lambda r: r[2], reverse=reverse)
    direction = "low → high" if not reverse else "high → low"

    idx_w = max(len(str(r[0])) for r in rows) + 1  # +1 for '#'
    val_w = max(len(_format_metric_value(r[2])) for r in rows)
    aligned = _align_names([r[1] for r in rows])

    suffix = f"  (mean of last {smooth})" if smooth > 0 else ""
    n = len(rows)
    header = f"{metric} — {n} trial{'s' if n != 1 else ''}, {direction}{suffix}"
    out = [header]
    for (idx, _, v), nm_col in zip(rows, aligned):
        idx_col = f"#{idx}".ljust(idx_w)
        val_col = _format_metric_value(v).rjust(val_w)
        out.append(f"  {idx_col}  {val_col}  {nm_col}")
    return "\n".join(out)


# --- /prune ---------------------------------------------------------------

def cmd_prune(workspace: Path, index: int, reason: str = "user-pruned via /prune") -> str:
    """Prune one trial: scancel via `herd stop` + stamp manifest as
    `pruned`. Distinct from `/stop` (cancel, resubmittable) and `/pause`
    (resumable) — pruned trials are NOT resubmitted by future `herd run`
    calls."""
    proc = subprocess.run(
        _RUNNABLE + ["stop", "-i", str(index), str(workspace)],
        capture_output=True, text=True,
    )
    # Don't fail if stop returned nonzero — the trial might already
    # be terminal in SLURM, in which case the "pruned" stamp still
    # accomplishes the goal (no resubmit on next `herd run`).
    stop_out = _strip_ansi(proc.stdout or "").strip()
    try:
        from hyperherd import manifest
        manifest.update_trial_status(str(workspace), index, "pruned")
    except Exception as e:
        return (
            f"Cancelled trial {index} via SLURM, but couldn't stamp "
            f"manifest as pruned: {e}\n{stop_out}"
        )
    return (
        f"Pruned trial {index}. Reason: {reason}\n"
        f"(Won't be resubmitted by future `herd run` calls.)\n{stop_out}"
    )


# --- /pause ---------------------------------------------------------------

def cmd_pause(workspace: Path, index: int) -> str:
    """Pause one trial cooperatively: write a `pause` signal and stamp the
    manifest `paused`. The running trial self-terminates at its next
    `log_result(step=...)` call (checkpointing first if it checkpoints) —
    no scancel. Distinct from `/stop` (cancel) and `/prune` (terminal):
    a paused trial is resumable, by `herd sh` (if it's later judged
    top-half) or `herd run -i <index>`."""
    try:
        from hyperherd import manifest
        from hyperherd.logging import write_prune_signal
    except Exception as e:
        return f"Couldn't import hyperherd: {e}"

    trials = manifest.load_manifest(str(workspace))
    trial = next((t for t in trials if t["index"] == index), None)
    if trial is None:
        return f"No trial with index {index}."
    status = trial.get("status", "unknown")
    try:
        write_prune_signal(str(workspace), index, "pause")
        manifest.update_trial_status(str(workspace), index, "paused")
    except Exception as e:
        return f"Couldn't pause trial {index}: {e}"
    return (
        f"Paused trial {index} (was {status}). It stops at its next logged "
        f"step and is resumable (`herd sh`, or `herd run -i {index}`)."
    )


# --- /run -----------------------------------------------------------------

def cmd_run(workspace: Path, index: int) -> str:
    """Submit one trial. Backed by `herd run -i <index>`."""
    proc = subprocess.run(
        _RUNNABLE + ["run", "-i", str(index), str(workspace)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return f"`herd run -i {index}` failed: {_strip_ansi(proc.stderr or proc.stdout)}"
    out = _strip_ansi(proc.stdout or "").strip() or "(no output)"
    return f"Submitted trial {index}.\n{out}"


def cmd_run_all(workspace: Path) -> str:
    """Submit every ready trial. Backed by `herd run`."""
    proc = subprocess.run(
        _RUNNABLE + ["run", str(workspace)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return f"`herd run` failed: {_strip_ansi(proc.stderr or proc.stdout)}"
    out = _strip_ansi(proc.stdout or "").strip() or "(no output)"
    return f"Submitted all ready trials.\n{out}"


# --- /plan ----------------------------------------------------------------

def cmd_plan(workspace: Path) -> str:
    """Show the agent's MONITOR_PLAN.md contents."""
    path = Path(workspace) / ".hyperherd" / "MONITOR_PLAN.md"
    if not path.is_file():
        return ("No plan yet — the agent writes one on its first tick. "
                "If the daemon just started, give it a moment.")
    try:
        content = path.read_text()
    except OSError as e:
        return f"Couldn't read plan: {e}"
    return content.rstrip() or "(plan is empty)"


# --- /info ----------------------------------------------------------------

def cmd_info(
    workspace: Path,
    *,
    ticks: int = 0,
    total_cost_usd: float = 0.0,
    started_at_iso: Optional[str] = None,
) -> str:
    """Daemon-wide status: workspace, sweep, phase, uptime, costs.

    `ticks`, `total_cost_usd`, `started_at_iso` come from the live daemon
    via its `info_handler` callback. Without those (e.g. running this
    function standalone), only file-derived fields are populated.
    """
    from datetime import datetime, timezone

    lines = []
    lines.append(f"Workspace: {workspace}")

    # Sweep name from hyperherd.yaml.
    try:
        from hyperherd.config import load_config
        config = load_config(str(workspace))
        lines.append(f"Sweep: {config.name}")
    except Exception:
        pass

    # Phase from the agent's plan.
    plan_path = Path(workspace) / ".hyperherd" / "MONITOR_PLAN.md"
    phase = "(unknown — no plan yet)"
    if plan_path.is_file():
        try:
            for line in plan_path.read_text().splitlines():
                stripped = line.strip().lstrip("-").strip()
                if stripped.lower().startswith("phase:"):
                    phase = stripped.split(":", 1)[1].strip()
                    break
        except OSError:
            pass
    lines.append(f"Phase: {phase}")
    lines.append("")

    # Daemon-runtime fields.
    if started_at_iso:
        try:
            started = datetime.fromisoformat(started_at_iso)
            now = datetime.now(timezone.utc)
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            uptime_s = int((now - started).total_seconds())
            lines.append(f"Daemon uptime: {_format_duration(uptime_s)}")
        except Exception:
            pass
    lines.append(f"Ticks completed: {ticks}")
    lines.append(f"Total cost: ${total_cost_usd:.4f}")

    # Next tick info from .hyperherd/next-tick.json.
    next_path = Path(workspace) / ".hyperherd" / "next-tick.json"
    if next_path.is_file():
        try:
            data = json.loads(next_path.read_text())
            if data.get("halted"):
                lines.append(f"Halted: {data.get('reason', '?')}")
            elif "scheduled_at" in data and "delay_seconds" in data:
                scheduled = datetime.fromisoformat(data["scheduled_at"])
                if scheduled.tzinfo is None:
                    scheduled = scheduled.replace(tzinfo=timezone.utc)
                fire_at = scheduled.timestamp() + int(data["delay_seconds"])
                remaining = int(fire_at - datetime.now(timezone.utc).timestamp())
                if remaining > 0:
                    lines.append(f"Next scheduled tick in: {_format_duration(remaining)}")
                else:
                    lines.append("Next scheduled tick: due now")
        except (OSError, json.JSONDecodeError, ValueError):
            pass

    return "\n".join(lines)


def _format_duration(seconds: int) -> str:
    """Human-readable duration, e.g. 90 -> '1m 30s', 3700 -> '1h 1m'."""
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{m}m {s}s"
    h, rem = divmod(seconds, 3600)
    m, _ = divmod(rem, 60)
    return f"{h}h {m}m"


# --- /tail ----------------------------------------------------------------

def cmd_tail(
    workspace: Path,
    index: int,
    lines: int = 20,
    stream: str = "both",
) -> str:
    """Last N lines of trial <index>'s logs. `stream` selects which
    file(s) to read: "stderr", "stdout", or "both" (default — labeled
    sections, useful when frameworks split training prints between the
    two streams)."""
    if lines <= 0 or lines > 1000:
        return f"`lines` must be between 1 and 1000 (got {lines})."
    if stream not in ("stderr", "stdout", "both"):
        return f"`stream` must be 'stderr', 'stdout', or 'both' (got {stream!r})."

    logs_dir = Path(workspace) / ".hyperherd" / "logs"
    targets = []
    if stream in ("stderr", "both"):
        targets.append(("stderr", logs_dir / f"{index}.err"))
    if stream in ("stdout", "both"):
        targets.append(("stdout", logs_dir / f"{index}.out"))

    sections = []
    any_present = False
    for label, path in targets:
        if not path.is_file():
            sections.append(f"=== {label} === (no file at `{path}`)")
            continue
        any_present = True
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except OSError as e:
            sections.append(f"=== {label} === (couldn't read: {e})")
            continue
        tail = "\n".join(content.splitlines()[-lines:])
        if not tail.strip():
            sections.append(f"=== {label} === (empty)")
        else:
            sections.append(f"=== {label} (last {lines} lines) ===\n{tail}")

    if not any_present:
        return (
            f"No log files for trial {index} — it may not have started yet."
        )
    return "\n\n".join(sections)


# --- /stats ---------------------------------------------------------------

def cmd_stats(workspace: Path) -> str:
    """Per-trial timing/memory stats. Backed by `herd stats`."""
    proc = subprocess.run(
        _RUNNABLE + ["stats", str(workspace)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return f"`herd stats` failed: {proc.stderr or proc.stdout}"
    return _strip_ansi(proc.stdout).rstrip() or "(no stats yet)"


# --- /params --------------------------------------------------------------

def cmd_params(workspace: Path) -> str:
    """Show the parameter grid the sweep will run over. Reads
    `hyperherd.yaml` directly and regenerates the combinations so the
    output is sweep-shape rather than current-state."""
    try:
        from hyperherd.config import load_config
        from hyperherd.search import generate_combinations
        from hyperherd.constraints import apply_constraints
    except Exception as e:
        return f"Couldn't load config: {e}"

    try:
        config = load_config(str(workspace))
    except Exception as e:
        return f"Couldn't read hyperherd.yaml: {e}"

    try:
        combos = apply_constraints(generate_combinations(config), config.conditions)
    except Exception as e:
        return f"Couldn't generate combinations: {e}"

    lines = [f"{config.name} — {len(combos)} trial(s)", ""]

    lines.append("Parameters:")
    for pname, pspec in config.parameters.items():
        ptype = getattr(pspec, "type", "?")
        if ptype == "discrete":
            vals = getattr(pspec, "values", [])
            default = getattr(pspec, "default", None)
            lines.append(
                f"  {pname} (discrete): {vals}"
                + (f" [default {default}]" if default is not None else "")
            )
        elif ptype == "continuous":
            low = getattr(pspec, "low", None)
            high = getattr(pspec, "high", None)
            scale = getattr(pspec, "scale", "linear")
            steps = getattr(pspec, "steps", None)
            lines.append(
                f"  {pname} (continuous, {scale}): "
                f"{low}..{high} in {steps} step(s)"
            )
        else:
            lines.append(f"  {pname} ({ptype})")
    lines.append("")

    grid = config.grid
    if grid is not None:
        lines.append(f"Grid: {grid}")
        lines.append("")

    if not combos:
        lines.append("(no combinations after applying conditions)")
        return "\n".join(lines)

    lines.append("Trials:")
    for i, trial in enumerate(combos[:50]):
        # Trials are dataclass-like with .params and .extras dicts.
        params = getattr(trial, "params", trial)
        extras = getattr(trial, "extras", {}) or {}
        kv = " ".join(f"{k}={v}" for k, v in params.items())
        if extras:
            kv += "  +(" + " ".join(f"{k}={v}" for k, v in extras.items()) + ")"
        lines.append(f"  {i:3d}  {kv}")
    if len(combos) > 50:
        lines.append(f"  ... ({len(combos) - 50} more)")

    return "\n".join(lines)


# --- /results -------------------------------------------------------------

def build_results_table(workspace: Path) -> Optional[dict]:
    """Build the `herd res` summary as both a column-aligned text table and
    a TSV blob, plus the bare list of (header, rows) for callers that want
    to do their own formatting.

    Returns None when the workspace has no trials. Otherwise:
        {
          "header": [str, ...],
          "rows":   [[str, ...], ...],
          "table_text": "...",   # column-aligned, suitable for code block
          "tsv":        "...",   # tab-separated, suitable for file upload
          "n_trials":  int,
          "n_metrics": int,
        }
    """
    from hyperherd.logging import load_all_results
    from hyperherd import manifest as manifest_mod

    trials = manifest_mod.load_manifest(str(workspace))
    if not trials:
        return None
    results = load_all_results(str(workspace))

    # Derive parameter column order from the first trial's `params` dict
    # (insertion order matches the YAML's declaration order — that's how
    # the manifest is written), then accumulate any extra keys later trials
    # introduce. We avoid going through `load_config` so this still works
    # if the YAML develops a transient validation error mid-sweep.
    param_names: list = []
    seen_params: set = set()
    for trial in trials:
        for p in (trial.get("params") or {}):
            if p not in seen_params:
                param_names.append(p)
                seen_params.add(p)

    metric_names: list = []
    seen: set = set()
    for trial_results in results.values():
        for k in trial_results:
            if k not in seen:
                metric_names.append(k)
                seen.add(k)

    header = ["idx", "experiment_name"] + param_names + metric_names

    def _fmt(v):
        if v == "" or v is None:
            return ""
        if isinstance(v, float):
            return f"{v:.6g}"
        return str(v)

    rows: list = []
    for trial in trials:
        idx = trial["index"]
        params = trial.get("params", {})
        trial_results = results.get(idx, {})
        row = [str(idx), trial.get("experiment_name") or ""]
        for p in param_names:
            row.append(_fmt(params.get(p, "")))
        for m in metric_names:
            row.append(_fmt(trial_results.get(m, "")))
        rows.append(row)

    widths = [len(h) for h in header]
    for row in rows:
        for i, cell in enumerate(row):
            if len(cell) > widths[i]:
                widths[i] = len(cell)

    def _line(cells):
        return "  ".join(c.ljust(widths[i]) for i, c in enumerate(cells)).rstrip()

    table_text = "\n".join([_line(header)] + [_line(r) for r in rows])
    tsv = "\n".join(["\t".join(header)] + ["\t".join(r) for r in rows]) + "\n"

    return {
        "header": header,
        "rows": rows,
        "table_text": table_text,
        "tsv": tsv,
        "n_trials": len(trials),
        "n_metrics": len(metric_names),
    }


def build_steps_blob(workspace: Path) -> Optional[dict]:
    """Build a gzipped TSV of every per-step metric record across the
    workspace's trials, in long form (trial_id,step,metric,timestamp,value).

    Returns None when no trial has streamed any metric. Otherwise:
        {
          "gz_bytes":      bytes,    # gzip-compressed UTF-8 TSV
          "n_rows":        int,
          "n_trials":      int,
          "n_metrics":     int,
          "uncompressed":  int,      # bytes
        }

    The TSV uses tabs (matching `results.tsv` from build_results_table).
    Discord caps a single attachment at 10 MB on a non-boosted server,
    50 MB at boost tier 2; the caller decides whether to upload based on
    `len(gz_bytes)`.
    """
    import gzip
    from hyperherd import manifest as manifest_mod
    from hyperherd.logging import collect_step_rows

    trials = manifest_mod.load_manifest(str(workspace))
    if not trials:
        return None

    rows = collect_step_rows(str(workspace), trials)
    if not rows:
        return None

    metrics_seen: set = set()
    trials_seen: set = set()
    body_lines: list = []
    for tid, step, metric, ts_iso, value in rows:
        trials_seen.add(tid)
        metrics_seen.add(metric)
        if isinstance(value, float):
            v = f"{value:.6g}"
        else:
            v = str(value)
        body_lines.append("\t".join([str(tid), str(step), metric, ts_iso, v]))

    header = "\t".join(["trial_id", "step", "metric", "timestamp", "value"])
    text = header + "\n" + "\n".join(body_lines) + "\n"
    raw = text.encode("utf-8")
    gz = gzip.compress(raw, compresslevel=6)
    return {
        "gz_bytes": gz,
        "n_rows": len(rows),
        "n_trials": len(trials_seen),
        "n_metrics": len(metrics_seen),
        "uncompressed": len(raw),
    }


# --- /help ----------------------------------------------------------------

def cmd_help() -> str:
    """List of available slash commands."""
    return (
        "**HerdDog commands**\n"
        "`/status` — refresh and move the live dashboard embed to the bottom of the chat\n"
        "`/running` — active trials only (running + queued + submitted)\n"
        "`/stats` — timing and memory stats per trial\n"
        "`/params` — sweep config: parameters, grid shape, all trial combos\n"
        "`/metrics` — list logged metric names; `/metrics <name>` shows each trial's value\n"
        "`/plot <metric> [trials] [smooth]` — PNG plot of a metric across trials (e.g. `/plot train/loss 0-3`)\n"
        "`/results [with_steps]` — upload the per-trial parameters + final metrics table as a TSV (and `steps.tsv.gz` with the full per-step history unless `with_steps=false`)\n"
        "`/info` — daemon metadata: workspace, phase, uptime, ticks, cost\n"
        "`/plan` — show the agent's `MONITOR_PLAN.md`\n"
        "`/run <index>` — submit (or resubmit) one trial\n"
        "`/run_all` — submit every ready trial\n"
        "`/stop <index>` — cancel one trial (resubmitted on next `herd run`)\n"
        "`/stop_all` — cancel every live trial\n"
        "`/pause <index>` — pause one trial (graceful, resumable; SH-aware)\n"
        "`/prune <index> [reason]` — algorithmic kill, NOT resubmitted on `herd run`\n"
        "`/tail <index> [lines]` — last N lines of a trial's logs (default 20)\n"
        "`/shutdown` — stop the monitor daemon entirely\n"
        "`/help` — this list\n"
        "\n"
        "For anything else (cadence changes, remediation policy, questions), "
        "`@<botname>` me — I'll wake the agent."
    )
