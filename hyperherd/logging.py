"""Runtime helpers for HyperHerd trials — called from within a launched job.

Two complementary helpers live here:

  * `log_result(name, value)` — write a metric for the current trial. Results
    accumulate in `.hyperherd/results/<trial_id>.json` and are surfaced by
    `herd res`.
  * `parse_overrides(arg_string=None)` — turn the launcher's `$1` argument
    string ("lr=0.001 optimizer=adam ...") into a dict of Python values.
    Useful as a lightweight argparse replacement when the trainer doesn't
    use Hydra (and the launcher delegates to a Python program).

Both rely on `HYPERHERD_WORKSPACE` / `HYPERHERD_TRIAL_ID` env vars (logging)
and `sys.argv` (parsing), which the sbatch script and `herd test` both set
up automatically.
"""

import json
import os
import sys
from typing import Any, Dict, Optional

from hyperherd.manifest import MANIFEST_FILE, WORKSPACE_DIR

RESULTS_DIR = "results"

# Name of the per-trial signal file (lives at results/<trial_id>/_signal).
# Written by `herd sh` when a trial is pruned/paused; read in-trial by
# `log_result(step=...)` so a cooperative trainer self-terminates at its next
# logging point. Contents are the action word: "prune" or "pause".
SIGNAL_FILE = "_signal"

# Valid signal actions.
_SIGNAL_ACTIONS = ("prune", "pause")


class TrialPruned(Exception):
    """Raised inside a trial when the pruner has asked it to stop.

    `log_result(..., step=...)` raises this when `herd sh` has written a prune/
    pause signal for the current trial — letting a cooperative trainer unwind
    its loop, checkpoint, and exit cleanly instead of being force-killed.

    `action` is "prune" (terminal; won't be resumed by SH) or "pause"
    (resumable; SH may resubmit it later). Trainers can catch this to exit 0
    gracefully; the trial's terminal manifest status is already owned by
    `herd sh`, so uncaught propagation is also fine.
    """

    def __init__(self, action: str = "prune", index: Optional[int] = None):
        self.action = action
        self.index = index
        who = f"trial {index}" if index is not None else "this trial"
        super().__init__(
            f"{who} was asked to {action} by successive-halving pruning"
        )


def signal_path(workspace: str, trial_id) -> str:
    """Path to a trial's prune/pause signal file."""
    return os.path.join(
        workspace, WORKSPACE_DIR, RESULTS_DIR, str(trial_id), SIGNAL_FILE
    )


def write_prune_signal(workspace: str, trial_id, action: str) -> None:
    """Write a prune/pause signal for a trial (called by `herd sh`)."""
    if action not in _SIGNAL_ACTIONS:
        raise ValueError(f"signal action must be one of {_SIGNAL_ACTIONS}, got {action!r}")
    path = signal_path(workspace, trial_id)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write(action)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_prune_signal(workspace: str, trial_id) -> Optional[str]:
    """Read a trial's prune/pause signal, or None if none is set."""
    path = signal_path(workspace, trial_id)
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        action = f.read().strip()
    return action if action in _SIGNAL_ACTIONS else None


def clear_prune_signal(workspace: str, trial_id) -> None:
    """Remove a trial's prune/pause signal (e.g. before resuming it)."""
    path = signal_path(workspace, trial_id)
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def assert_writable() -> None:
    """Verify the trainer is running in a usable HyperHerd trial context.

    Raises RuntimeError if `HYPERHERD_WORKSPACE` / `HYPERHERD_TRIAL_ID`
    are unset, the workspace directory isn't visible to this process
    (typical container bind-mount failure), or the workspace's manifest
    is missing. Call once at trainer startup to fail-fast instead of
    discovering empty result files at the end of a long run.

    This is the same check `log_result(strict=True)` runs per call,
    exposed so trainers can do it once up-front.
    """
    _check_writable()


def _check_writable() -> None:
    """Raise RuntimeError if the current process can't actually write a
    metric to the host's workspace.

    Catches the silent-failure mode where a container starts with
    `HYPERHERD_WORKSPACE` set but the workspace path isn't bind-mounted
    in: `os.makedirs(exist_ok=True)` then creates the directory inside
    the container's ephemeral filesystem and writes succeed but land
    nowhere durable. We detect this by requiring `.hyperherd/manifest.json`
    to be visible — the host writes that file before launch, so its
    presence proves the mount made it through.
    """
    workspace = os.environ.get("HYPERHERD_WORKSPACE")
    if not workspace:
        raise RuntimeError(
            "HYPERHERD_WORKSPACE is not set. log_result(strict=True) must "
            "be called from within a HyperHerd trial."
        )
    if os.environ.get("HYPERHERD_TRIAL_ID") is None:
        raise RuntimeError(
            "HYPERHERD_TRIAL_ID is not set. log_result(strict=True) must "
            "be called from within a HyperHerd trial."
        )
    if not os.path.isdir(workspace):
        raise RuntimeError(
            f"HYPERHERD_WORKSPACE={workspace!r} is set but the directory is "
            f"not visible to this process. If you're running inside a "
            f"container, the workspace path is likely not bind-mounted "
            f"(apptainer: add `--bind {workspace}:{workspace}`)."
        )
    manifest_path = os.path.join(workspace, WORKSPACE_DIR, MANIFEST_FILE)
    if not os.path.isfile(manifest_path):
        raise RuntimeError(
            f"Workspace {workspace!r} is visible but {WORKSPACE_DIR}/{MANIFEST_FILE} "
            f"is missing. If you're running inside a container, only part of "
            f"the workspace tree is mounted — bind the whole workspace, not "
            f"just a subdirectory. Otherwise this trial was launched without "
            f"a populated manifest, which shouldn't happen via `herd run` / "
            f"`herd test`."
        )


def _results_dir() -> str:
    workspace = os.environ.get("HYPERHERD_WORKSPACE")
    if not workspace:
        raise RuntimeError(
            "HYPERHERD_WORKSPACE not set. "
            "log_result() must be called from within a HyperHerd trial "
            "(launched via 'mush launch' or 'mush test')."
        )
    return os.path.join(workspace, WORKSPACE_DIR, RESULTS_DIR)


def _results_path() -> str:
    trial_id = os.environ.get("HYPERHERD_TRIAL_ID")
    if trial_id is None:
        raise RuntimeError(
            "HYPERHERD_TRIAL_ID not set. "
            "log_result() must be called from within a HyperHerd trial."
        )
    return os.path.join(_results_dir(), f"{trial_id}.json")


def log_result(
    name: str,
    value,
    step: Optional[int] = None,
    *,
    strict: bool = True,
) -> None:
    """Log a named metric for the current trial.

    Two modes:

    * **Final-summary mode** (no `step`): writes to
      `.hyperherd/results/<trial_id>.json` as a flat `{name: value}` dict.
      Calling twice with the same name overwrites. This is the original
      behavior, preserved for backward compatibility.
    * **Streaming mode** (`step` given): appends `{step, value}` to
      `.hyperherd/results/<trial_id>/stream/<name>.jsonl`. The autonomous
      monitor reads these streams to make decisions (warnings, pruning).

    The two modes are not mutually exclusive — call streaming
    `log_result("val_loss", v, step=s)` during training, then the bare
    `log_result("test_acc", final)` at the end. The flat JSON keeps
    `herd res`'s TSV view clean while the per-metric stream files give
    the agent detail.

    Args:
        name: Metric name (e.g. "val_loss", "test_accuracy").
        value: Metric value (must be JSON-serializable).
        step: Optional step counter. If given, switches to streaming mode.
        strict: Default True. Verifies the host workspace is actually
            visible to this process before writing — catches the case
            where a container sets HYPERHERD_WORKSPACE but the path
            isn't bind-mounted (the write would otherwise succeed
            silently into the container's ephemeral FS). Set to False
            only if you have a specific reason to skip the mount check
            (e.g. an isolated test that genuinely runs outside a trial).
    """
    if strict:
        _check_writable()
    if step is None:
        _log_result_final(name, value)
    else:
        _log_result_stream(name, value, step)
        # After recording the step, honor any pending prune/pause signal so a
        # cooperative trainer stops at this logging point. Only in streaming
        # mode — the bare final-summary call at the end of training must never
        # raise. Absent signal (the common case) is a cheap stat and a no-op.
        _raise_if_signalled()


def _raise_if_signalled() -> None:
    """Raise `TrialPruned` if `herd sh` has signalled the current trial.

    Reads the same `HYPERHERD_WORKSPACE` / `HYPERHERD_TRIAL_ID` env the logging
    path uses, so it only ever fires for *this* trial. Missing env or missing
    file → no-op (e.g. running outside a trial, or no pruning configured).
    """
    workspace = os.environ.get("HYPERHERD_WORKSPACE")
    trial_id = os.environ.get("HYPERHERD_TRIAL_ID")
    if not workspace or trial_id is None:
        return
    action = read_prune_signal(workspace, trial_id)
    if action is not None:
        raise TrialPruned(action, index=int(trial_id) if str(trial_id).isdigit() else None)


def _log_result_final(name: str, value) -> None:
    path = _results_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.isfile(path):
        with open(path, "r") as f:
            data = json.load(f)
    else:
        data = {}

    data[name] = value

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _log_result_stream(name: str, value, step: int) -> None:
    """Append one (step, value, ts) record to the per-metric stream file.
    `ts` is the Unix timestamp at append time — lets the agent ask
    'val_loss over the last 5 minutes' against the file.

    Slashes in `name` are honored as nested subdirectories under
    ``stream/`` (so Lightning-style metric names like ``train/loss``
    land at ``stream/train/loss.jsonl``). Names that try to escape
    ``stream/`` (absolute paths, ``..`` components) are rejected.
    """
    import time
    trial_id = os.environ.get("HYPERHERD_TRIAL_ID")
    if trial_id is None:
        raise RuntimeError(
            "HYPERHERD_TRIAL_ID not set. "
            "log_result() must be called from within a HyperHerd trial."
        )
    stream_dir = os.path.join(_results_dir(), str(trial_id), "stream")
    if os.path.isabs(name) or ".." in name.replace("\\", "/").split("/"):
        raise ValueError(
            f"Invalid metric name {name!r}: must be relative and must not "
            f"contain '..' components."
        )
    stream_path = os.path.join(stream_dir, f"{name}.jsonl")
    os.makedirs(os.path.dirname(stream_path), exist_ok=True)
    with open(stream_path, "a") as f:
        f.write(json.dumps({
            "step": int(step),
            "value": value,
            "ts": time.time(),
        }) + "\n")


def load_metric_stream(workspace: str, trial_id: int, name: str) -> list:
    """Read the per-step stream for one metric of one trial.

    Returns a list of `{"step": int, "value": ...}` dicts in append order
    (which is usually but not guaranteed to be step order). Empty list if
    the stream doesn't exist.
    """
    path = os.path.join(
        workspace, WORKSPACE_DIR, RESULTS_DIR,
        str(trial_id), "stream", f"{name}.jsonl",
    )
    if not os.path.isfile(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def list_metric_streams(workspace: str, trial_id: int) -> list:
    """List the metric names that have a stream file for this trial.

    Recurses into ``stream/``, so nested names like ``train/loss``
    (written by ``log_result("train/loss", ...)``) appear in the result
    with the slash preserved. Always returns POSIX-style separators
    regardless of host OS.
    """
    stream_dir = os.path.join(
        workspace, WORKSPACE_DIR, RESULTS_DIR, str(trial_id), "stream",
    )
    if not os.path.isdir(stream_dir):
        return []
    names = []
    for root, _dirs, files in os.walk(stream_dir):
        for f in files:
            if not f.endswith(".jsonl"):
                continue
            full = os.path.join(root, f)
            rel = os.path.relpath(full, stream_dir)
            names.append(rel[:-len(".jsonl")].replace(os.sep, "/"))
    return sorted(names)


def load_trial_results(workspace: str, trial_id: int) -> dict:
    """Load results for a specific trial. Returns empty dict if no results."""
    path = os.path.join(workspace, WORKSPACE_DIR, RESULTS_DIR, f"{trial_id}.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def collect_step_rows(workspace: str, trials: list) -> list:
    """Walk every trial's per-metric stream files and produce long-form rows
    (trial_id, step, metric, ts_iso, value) suitable for CSV/TSV emission.

    When the same (trial, metric, step) was logged more than once — e.g.
    a trainer re-emitted a metric after a restart — keep only the record
    with the largest `ts`.

    Rows are sorted (trial_id, metric, step). `ts_iso` is UTC ISO 8601
    with seconds precision; missing timestamps render as the empty string.
    """
    from datetime import datetime, timezone

    rows: list = []
    for trial in trials:
        idx = trial["index"]
        for metric in list_metric_streams(workspace, idx):
            by_step: dict = {}
            for rec in load_metric_stream(workspace, idx, metric):
                step = rec.get("step")
                if step is None:
                    continue
                ts = rec.get("ts", 0)
                if step not in by_step or ts >= by_step[step].get("ts", 0):
                    by_step[step] = rec
            for step in sorted(by_step):
                rec = by_step[step]
                ts = rec.get("ts")
                if isinstance(ts, (int, float)):
                    ts_iso = datetime.fromtimestamp(
                        ts, tz=timezone.utc,
                    ).isoformat(timespec="seconds").replace("+00:00", "Z")
                else:
                    ts_iso = ""
                rows.append((idx, int(step), metric, ts_iso, rec.get("value")))
    rows.sort(key=lambda r: (r[0], r[2], r[1]))
    return rows


def load_all_results(workspace: str) -> dict:
    """Load results for all trials. Returns {trial_id: {metric: value, ...}}."""
    results_dir = os.path.join(workspace, WORKSPACE_DIR, RESULTS_DIR)
    if not os.path.isdir(results_dir):
        return {}
    all_results = {}
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".json"):
            trial_id = int(fname.replace(".json", ""))
            with open(os.path.join(results_dir, fname), "r") as f:
                all_results[trial_id] = json.load(f)
    return all_results


def _coerce_token(token: str) -> Any:
    """Convert a single override value-token to int/float/bool/None when possible.

    Mirrors the format used by `manifest._format_override_value` so a
    round-trip (Python value → override string → `parse_overrides`) lands on
    the original type for the common cases: ints stay ints, floats stay
    floats, bools/None survive. Anything that doesn't parse cleanly is
    returned as-is, so string-valued params still come through unmodified.
    """
    if token == "null" or token == "None":
        return None
    if token == "true":
        return True
    if token == "false":
        return False
    # Try int when there's no decimal/exponent — keeps "10" as int(10),
    # "1.0" as float(1.0), and "1e-3" as float(0.001).
    if token and token.lstrip("+-").isdigit():
        try:
            return int(token)
        except ValueError:
            pass
    try:
        return float(token)
    except ValueError:
        return token


def parse_overrides(arg_string: Optional[str] = None) -> Dict[str, Any]:
    """Parse a HyperHerd override string into a dict of {name: value}.

    The launcher script receives overrides as `$1`, a whitespace-separated
    string of `name=value` tokens (e.g.
    `"experiment_name=lr-0.001_opt-adam lr=0.001 optimizer=adam"`).

    Pass that string in, or omit `arg_string` to read `sys.argv[1]`. Values
    are coerced back to int/float/bool/None when they parse cleanly,
    otherwise left as strings.

    Hydra-style key prefixes (`+foo`, `++foo`, `~foo`) are left in the
    returned key as-is — non-Hydra trainers won't be emitting them, and
    Hydra trainers use the override string directly anyway.

    Example use from a non-Hydra trainer's `launch.sh`:

        # launch.sh
        python train.py "$1"

        # train.py
        from hyperherd import parse_overrides, log_result
        params = parse_overrides()
        lr = params["lr"]
        ...
        log_result("test_accuracy", acc)
    """
    if arg_string is None:
        if len(sys.argv) < 2:
            raise RuntimeError(
                "parse_overrides() called with no argument and sys.argv[1] "
                "is missing. Make sure your launcher forwards the override "
                "string, e.g. `python train.py \"$1\"`."
            )
        arg_string = sys.argv[1]

    result: Dict[str, Any] = {}
    for token in arg_string.split():
        if "=" not in token:
            # Plain flags (e.g. `--cfg` `job` from `herd test --cfg-job`) are
            # not name=value pairs; ignore rather than raise so the same
            # parser can be used for cfg-job invocations too.
            continue
        key, _, value = token.partition("=")
        if not key:
            continue
        result[key] = _coerce_token(value)
    return result
