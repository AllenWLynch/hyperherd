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

from hyperherd.manifest import WORKSPACE_DIR

RESULTS_DIR = "results"


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


def log_result(name: str, value, step: Optional[int] = None) -> None:
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
    """
    if step is None:
        _log_result_final(name, value)
    else:
        _log_result_stream(name, value, step)


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
    'val_loss over the last 5 minutes' against the file."""
    import time
    trial_id = os.environ.get("HYPERHERD_TRIAL_ID")
    if trial_id is None:
        raise RuntimeError(
            "HYPERHERD_TRIAL_ID not set. "
            "log_result() must be called from within a HyperHerd trial."
        )
    stream_dir = os.path.join(_results_dir(), str(trial_id), "stream")
    os.makedirs(stream_dir, exist_ok=True)
    stream_path = os.path.join(stream_dir, f"{name}.jsonl")
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
    """List the metric names that have a stream file for this trial."""
    stream_dir = os.path.join(
        workspace, WORKSPACE_DIR, RESULTS_DIR, str(trial_id), "stream",
    )
    if not os.path.isdir(stream_dir):
        return []
    return sorted(
        f[:-len(".jsonl")] for f in os.listdir(stream_dir)
        if f.endswith(".jsonl")
    )


def load_trial_results(workspace: str, trial_id: int) -> dict:
    """Load results for a specific trial. Returns empty dict if no results."""
    path = os.path.join(workspace, WORKSPACE_DIR, RESULTS_DIR, f"{trial_id}.json")
    if not os.path.isfile(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


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
