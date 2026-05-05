"""Render small PNG plots of streamed trial metrics.

Used by:
- The `/plot` slash command (user-driven).
- The `post_plot` agent tool (agent-driven).
- The SLURM event poll's auto-plot on completion/failure.

matplotlib is lazy-imported because (a) startup is ~1s, and (b) the
`hyperherd` core package shouldn't pull it in for non-monitor users.
The plot APIs raise `PlotUnavailable` if matplotlib isn't installed,
and callers degrade gracefully by skipping the plot.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional

log = logging.getLogger(__name__)


class PlotUnavailable(RuntimeError):
    """matplotlib not installed, or a metric has no points to plot."""


def _read_plan_metric(workspace: Path) -> Optional[str]:
    """Return the success metric name from MONITOR_PLAN.md, or None.

    The setup interview writes a line like:
        - Success metric: val/loss, min
    or:
        - Success metric: none
    We extract the name part and return it, or None if unset/none."""
    plan_path = workspace / ".hyperherd" / "MONITOR_PLAN.md"
    if not plan_path.is_file():
        return None
    try:
        for line in plan_path.read_text().splitlines():
            s = line.strip().lstrip("-").strip()
            if s.lower().startswith("success metric:"):
                value = s.split(":", 1)[1].strip()
                if not value or value.lower() == "none":
                    return None
                # Strip optional direction suffix: "val/loss, min" → "val/loss"
                name = value.split(",")[0].strip()
                return name or None
    except OSError:
        pass
    return None


def pick_auto_plot_metric(
    workspace: Path, trial_index: int,
) -> Optional[str]:
    """Choose the best metric to auto-plot for one trial.

    Priority:
    1. The success metric from MONITOR_PLAN.md (set during setup interview).
    2. First alphabetically from whatever the trial actually logged.

    Returns None if the trial has no streamed metrics at all."""
    from hyperherd.logging import list_metric_streams
    streams = list_metric_streams(str(workspace), trial_index)
    if not streams:
        return None
    plan_metric = _read_plan_metric(workspace)
    if plan_metric and plan_metric in set(streams):
        return plan_metric
    return sorted(streams)[0]


def render_metric_plot(
    workspace: Path,
    metric: str,
    *,
    trial_indices: Optional[Iterable[int]] = None,
    smooth: int = 0,
    out_path: Optional[Path] = None,
) -> Path:
    """Render `metric` for the given trials (or all non-ready trials)
    as a PNG. Returns the path. Caller is responsible for cleanup if
    `out_path` is None — we use a NamedTemporaryFile that survives the
    function (since Discord upload happens after).

    Raises `PlotUnavailable` if matplotlib is missing OR no trials have
    points for this metric."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless — no DISPLAY needed
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise PlotUnavailable(
            "matplotlib not installed — `pip install hyperherd[monitor]` "
            "(or `pip install matplotlib`) to enable plotting."
        ) from e

    from hyperherd import manifest
    from hyperherd.logging import load_metric_stream

    trials = manifest.load_manifest(str(workspace))
    if trial_indices is not None:
        wanted = set(int(i) for i in trial_indices)
        trials = [t for t in trials if t["index"] in wanted]
    else:
        # Skip never-submitted trials — their streams are guaranteed empty.
        trials = [t for t in trials if t.get("status") != "ready"]

    series: List[Tuple[int, str, List[float], List[float]]] = []
    for t in trials:
        idx = t["index"]
        stream = load_metric_stream(str(workspace), idx, metric)
        xs, ys = [], []
        for p in stream:
            v = p.get("value")
            if not isinstance(v, (int, float)):
                continue
            step = p.get("step", len(ys))
            xs.append(step)
            ys.append(float(v))
        if not ys:
            continue
        if smooth > 1 and len(ys) > smooth:
            ys = _rolling_mean(ys, smooth)
            xs = xs[len(xs) - len(ys):]
        name = (t.get("experiment_name") or f"trial-{idx}")[:30]
        series.append((idx, name, xs, ys))

    if not series:
        raise PlotUnavailable(
            f"No trial has points for metric `{metric}`."
        )

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=110)
    for idx, name, xs, ys in series:
        ax.plot(xs, ys, label=f"#{idx} {name}", linewidth=1.4)
    ax.set_xlabel("step")
    ax.set_ylabel(metric)
    ax.grid(True, alpha=0.25)
    title = metric
    if smooth > 1:
        title = f"{metric} (smoothed, window={smooth})"
    ax.set_title(title)
    if len(series) <= 12:
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()

    if out_path is None:
        tmp = tempfile.NamedTemporaryFile(
            prefix="hyperherd-plot-", suffix=".png", delete=False,
        )
        tmp.close()
        out_path = Path(tmp.name)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def available_metrics(
    workspace: Path,
    trial_indices: Optional[Iterable[int]] = None,
) -> List[str]:
    """Union of metric names across the requested trials (or all trials
    if None). Sorted for determinism."""
    from hyperherd import manifest
    from hyperherd.logging import list_metric_streams

    trials = manifest.load_manifest(str(workspace))
    if trial_indices is not None:
        wanted = set(int(i) for i in trial_indices)
        trials = [t for t in trials if t["index"] in wanted]
    out: set = set()
    for t in trials:
        for name in list_metric_streams(str(workspace), t["index"]):
            out.add(name)
    return sorted(out)


def _rolling_mean(ys: List[float], window: int) -> List[float]:
    """Trailing rolling mean — drops the first `window-1` points."""
    if window <= 1 or len(ys) < window:
        return ys
    out = []
    s = sum(ys[:window])
    out.append(s / window)
    for i in range(window, len(ys)):
        s += ys[i] - ys[i - window]
        out.append(s / window)
    return out
