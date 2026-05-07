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

import colorsys
import logging
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

log = logging.getLogger(__name__)

# Tufte-ish ink colors (kept inline so we don't depend on the
# tufte-plots skill — that's a personal-tooling repo, not a runtime dep).
_PRIMARY_INK = "#222222"
_SECONDARY_INK = "#bbbbbb"


def _husl_palette(n: int) -> List[Tuple[float, float, float]]:
    """n perceptually-spaced colors via even hue rotation in HLS.

    Approximates seaborn's `color_palette('husl', n)` without the
    seaborn dep — even hue spacing at fixed lightness/saturation gives
    a CVD-passable wheel of distinguishable hues for many trials."""
    if n <= 0:
        return []
    return [
        colorsys.hls_to_rgb(i / n, 0.55, 0.65)
        for i in range(n)
    ]


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
    2. Heuristic ranking over the names the trial actually logged (see
       `_rank_metric_name`).
    3. Alphabetical fallback.

    Returns None if the trial has no streamed metrics at all."""
    from hyperherd.logging import list_metric_streams
    streams = list_metric_streams(str(workspace), trial_index)
    if not streams:
        return None
    plan_metric = _read_plan_metric(workspace)
    if plan_metric and plan_metric in set(streams):
        return plan_metric
    return min(streams, key=lambda n: (_rank_metric_name(n), n))


# Lower rank = more preferred. Tuple ordering: split → kind → "main".
# The key reads as a search tree: pick on split first, then loss-vs-score,
# then promote anything with "main" in the name.
_SPLIT_RANK = {"test": 0, "val": 1, "train": 3}  # nothing/other = 2
_KIND_RANK = {"loss": 0, "score": 1}             # nothing/other = 2


def _rank_metric_name(name: str) -> tuple:
    """Score a metric name lower = better-to-plot.

    Heuristic: prefer test > val > (uncategorized) > train; within that,
    prefer 'loss' > 'score' > other; then promote names containing
    'main' (e.g. `main_metric`, `val/main_loss`)."""
    lc = name.lower()

    if "test" in lc:
        split = _SPLIT_RANK["test"]
    elif "val" in lc:
        split = _SPLIT_RANK["val"]
    elif "train" in lc:
        split = _SPLIT_RANK["train"]
    else:
        split = 2

    if "loss" in lc:
        kind = _KIND_RANK["loss"]
    elif "score" in lc:
        kind = _KIND_RANK["score"]
    else:
        kind = 2

    main_bonus = 0 if "main" in lc else 1
    return (split, kind, main_bonus)


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

    n = len(series)
    palette = _husl_palette(n)
    # Many trials → thinner, more transparent lines so overlap stays
    # readable; few trials → opaque and slightly thicker.
    if n <= 6:
        lw, alpha = 1.2, 0.95
    elif n <= 16:
        lw, alpha = 1.0, 0.85
    else:
        lw, alpha = 0.8, 0.7

    fig, ax = plt.subplots(figsize=(6.5, 4.0), dpi=140)
    for (idx, name, xs, ys), color in zip(series, palette):
        ax.plot(xs, ys, label=f"#{idx} {name}",
                linewidth=lw, alpha=alpha, color=color,
                solid_capstyle="round")

    # Tufte: minimum non-data ink. No grid, no top/right spines, no
    # axis title — the metric name lives on the y-axis label.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(_PRIMARY_INK)
    ax.spines["bottom"].set_color(_PRIMARY_INK)
    ax.spines["left"].set_linewidth(0.6)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="both", colors=_PRIMARY_INK, width=0.6,
                   labelsize=8, length=3)
    ax.set_xlabel("step", fontsize=9, color=_PRIMARY_INK)
    ylabel = metric if smooth <= 1 else f"{metric} (smoothed, w={smooth})"
    ax.set_ylabel(ylabel, fontsize=9, color=_PRIMARY_INK)

    # Range frame: bound spines to the data extent and use only
    # min/mid/max ticks. Defensively skip when the extent collapses.
    all_xs = [x for _, _, xs, _ in series for x in xs]
    all_ys = [y for _, _, _, ys in series for y in ys]
    if all_xs and all_ys:
        xmin, xmax = min(all_xs), max(all_xs)
        ymin, ymax = min(all_ys), max(all_ys)
        if xmax > xmin:
            ax.spines["bottom"].set_bounds(xmin, xmax)
            ax.set_xticks([xmin, (xmin + xmax) / 2, xmax])
        if ymax > ymin:
            ax.spines["left"].set_bounds(ymin, ymax)
            ax.set_yticks([ymin, (ymin + ymax) / 2, ymax])

    # Legend: off-plot only when it'll fit. Past ~16 trials, drop it
    # entirely — the eye can't disambiguate that many lines from a
    # legend anyway and the husl wheel + line shape carries the
    # comparison.
    if n <= 16:
        long_labels = any(len(line.get_label()) > 14 for line in ax.get_lines())
        if long_labels:
            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5),
                      ncol=1, frameon=False, fontsize=7)
        else:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18),
                      ncol=min(n, 4), frameon=False, fontsize=7)
    fig.tight_layout()

    if out_path is None:
        tmp = tempfile.NamedTemporaryFile(
            prefix="hyperherd-plot-", suffix=".png", delete=False,
        )
        tmp.close()
        out_path = Path(tmp.name)
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
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
