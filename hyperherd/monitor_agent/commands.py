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
import subprocess
import sys
from pathlib import Path
from typing import Optional


_RUNNABLE = [sys.executable, "-m", "hyperherd.cli"]


# --- /status --------------------------------------------------------------

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


def _format_status(snap: dict) -> str:
    name = snap.get("sweep_name", "?")
    totals = snap.get("totals") or {}
    trials = snap.get("trials") or []

    order = ["ready", "submitted", "queued", "running",
             "completed", "failed", "pruned", "cancelled"]
    counts = ", ".join(
        f"{totals[k]} {k}" for k in order if totals.get(k)
    ) or "(no trials yet)"
    counts += f" ({totals.get('total', len(trials))} total)"

    if not trials:
        return f"{name} — {counts}"

    rows = [("idx", "status", "elapsed", "name")]
    for t in trials:
        rows.append((
            str(t.get("index", "?")),
            (t.get("status") or "?")[:10],
            (t.get("elapsed") or "-")[:10],
            (t.get("experiment_name") or "-")[:40],
        ))

    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    lines = [
        " ".join(c.ljust(w) for c, w in zip(rows[0], widths)),
        " ".join("-" * w for w in widths),
    ]
    for r in rows[1:]:
        lines.append(" ".join(c.ljust(w) for c, w in zip(r, widths)))

    return f"{name} — {counts}\n\n" + "\n".join(lines)


# --- /stop ----------------------------------------------------------------

def cmd_stop(workspace: Path, index: int) -> str:
    """Cancel a single trial. Backed by `herd stop -i <index>`."""
    try:
        proc = subprocess.run(
            _RUNNABLE + ["stop", "-i", str(index), str(workspace)],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as e:
        return f"`herd stop -i {index}` failed: {e.stderr or e.stdout}"
    out = (proc.stdout or "").strip() or "(no output)"
    return f"Stopped trial {index}.\n{out}"


def cmd_stop_all(workspace: Path) -> str:
    """Cancel every running/queued trial. Backed by `herd stop --all`."""
    try:
        proc = subprocess.run(
            _RUNNABLE + ["stop", "--all", str(workspace)],
            capture_output=True, text=True, check=True,
        )
    except subprocess.CalledProcessError as e:
        return f"`herd stop --all` failed: {e.stderr or e.stdout}"
    out = (proc.stdout or "").strip() or "(no output)"
    return f"Stopped all live trials.\n{out}"


# --- /metrics -------------------------------------------------------------

def cmd_metrics(workspace: Path, smooth: int = 0) -> str:
    """Cross-trial metric summary: for each non-ready trial and each
    logged metric, the most recent value (or mean of last `smooth`
    points if smooth>0). Renders as a vertical list — Discord's
    pinned-message side panel can't render wide tables."""
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

    rows = []
    metric_names = set()
    for trial in trials:
        status = trial.get("status", "?")
        if status == "ready":
            continue
        idx = trial["index"]
        metrics_for_trial: dict = {}
        for name in list_metric_streams(str(workspace), idx):
            stream = load_metric_stream(str(workspace), idx, name)
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
            metrics_for_trial[name] = v
            metric_names.add(name)
        rows.append({
            "idx": idx,
            "status": status,
            "name": trial.get("experiment_name", ""),
            "metrics": metrics_for_trial,
        })

    if not rows:
        return "No trials yet (or none have been submitted)."
    if not metric_names:
        return f"{len(rows)} trial(s); none have logged streamed metrics."

    sorted_metrics = sorted(metric_names)
    suffix = (
        " (smoothed: mean of last "
        f"{smooth} points)" if smooth > 0 else " (last logged value)"
    )
    out = [f"Metric summary{suffix}:", ""]
    for r in rows:
        idx, status = r["idx"], r["status"]
        nm = r["name"][:32]
        out.append(f"  #{idx} ({status}) {nm}")
        if not r["metrics"]:
            out.append(f"      (no streamed metrics)")
            continue
        for m in sorted_metrics:
            v = r["metrics"].get(m)
            if v is None:
                continue
            # Format scientifically for very small/large, fixed otherwise.
            if abs(v) > 0 and (abs(v) < 1e-3 or abs(v) >= 1e6):
                out.append(f"      {m}: {v:.4g}")
            else:
                out.append(f"      {m}: {v:.6g}")
    return "\n".join(out)


# --- /prune ---------------------------------------------------------------

def cmd_prune(workspace: Path, index: int, reason: str = "user-pruned via /prune") -> str:
    """Prune one trial: scancel via `herd stop` + stamp manifest as
    `pruned`. Distinct from `/cancel` — pruned trials are NOT
    resubmitted by future `herd run` calls."""
    proc = subprocess.run(
        _RUNNABLE + ["stop", "-i", str(index), str(workspace)],
        capture_output=True, text=True,
    )
    # Don't fail if stop returned nonzero — the trial might already
    # be terminal in SLURM, in which case the "pruned" stamp still
    # accomplishes the goal (no resubmit on next `herd run`).
    stop_out = (proc.stdout or "").strip()
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


# --- /run -----------------------------------------------------------------

def cmd_run(workspace: Path, index: int) -> str:
    """Submit one trial. Backed by `herd run -i <index>`."""
    proc = subprocess.run(
        _RUNNABLE + ["run", "-i", str(index), str(workspace)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return f"`herd run -i {index}` failed: {proc.stderr or proc.stdout}"
    out = (proc.stdout or "").strip() or "(no output)"
    return f"Submitted trial {index}.\n{out}"


def cmd_run_all(workspace: Path) -> str:
    """Submit every ready trial. Backed by `herd run`."""
    proc = subprocess.run(
        _RUNNABLE + ["run", str(workspace)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return f"`herd run` failed: {proc.stderr or proc.stdout}"
    out = (proc.stdout or "").strip() or "(no output)"
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
    # Strip ANSI color codes from terminal output — Discord doesn't render them.
    import re as _re
    cleaned = _re.sub(r"\x1b\[[0-9;]*m", "", proc.stdout)
    return cleaned.rstrip() or "(no stats yet)"


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


# --- /help ----------------------------------------------------------------

def cmd_help() -> str:
    """List of available slash commands."""
    return (
        "**HerdDog commands**\n"
        "`/status` — current sweep totals + per-trial table\n"
        "`/stats` — timing and memory stats per trial\n"
        "`/params` — sweep config: parameters, grid shape, all trial combos\n"
        "`/metrics [smooth]` — cross-trial metric summary; optional smoothing window (mean of last N points)\n"
        "`/info` — daemon metadata: workspace, phase, uptime, ticks, cost\n"
        "`/plan` — show the agent's `MONITOR_PLAN.md`\n"
        "`/run <index>` — submit (or resubmit) one trial\n"
        "`/run_all` — submit every ready trial\n"
        "`/cancel <index>` — cancel one trial (will be resubmitted on next `herd run`)\n"
        "`/cancel_all` — cancel every live trial\n"
        "`/prune <index> [reason]` — algorithmic kill, NOT resubmitted on `herd run`\n"
        "`/tail <index> [lines]` — last N lines of a trial's logs (default 20)\n"
        "`/stop` — stop the monitor daemon entirely\n"
        "`/help` — this list\n"
        "\n"
        "For anything else (cadence changes, remediation policy, questions), "
        "`@<botname>` me — I'll wake the agent."
    )
