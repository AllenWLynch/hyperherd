"""The agent's complete action surface — typed in-process MCP tools.

The Claude Agent SDK's `@tool` decorator passes a single dict argument
matching the declared schema and expects a `{"content": [{"type": "text",
"text": ...}], ...}` return — a deviation from "normal" Python signatures
that I learned the hard way during smoke testing. Each tool here unpacks
the args dict and returns a wrapped text response via `_text_response`.

Tools either call existing HyperHerd machinery directly (msg, manifest
reads) or shell out to the `herd` CLI with `--json`. Every wrapper
records what happened in `.hyperherd/agent_log.jsonl` for the audit
trail.

`schedule_next` and `halt` don't take action against the workspace; they
communicate the agent's intent for the *next* tick by writing to
`.hyperherd/next-tick.json`. The daemon's outer loop reads that file
after `query()` returns.

Tools register under a single `hyperherd` MCP server; the agent calls them
as `mcp__hyperherd__<name>`.
"""

import asyncio
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# `claude_agent_sdk` is an optional dependency — only needed when the
# daemon actually runs. Importing it lazily keeps `pip install hyperherd`
# without `[monitor]` extras working for the v1 path.
try:
    from claude_agent_sdk import tool  # type: ignore
except ImportError:  # pragma: no cover
    def tool(name, description, schema):  # noqa: D401 - SDK shim
        """No-op decorator used when claude-agent-sdk isn't installed."""
        def _wrap(fn):
            fn._tool_name = name
            fn._tool_description = description
            fn._tool_schema = schema
            return fn
        return _wrap


# Process-wide context. Set once by `tick.run_tick()` before the agent
# starts — the SDK runs each tool function as a callback so we can't pass
# extra arguments through the tool signature.
_CTX: Dict[str, Any] = {}


def set_context(
    workspace: Path,
    sweep_name: str,
    last_state_json: str,
    channel=None,
) -> None:
    """Bind the per-tick context. Called by `tick.run_tick` before
    `query()`. The agent's tools read this dict instead of taking the
    workspace as an arg (the SDK's tool schemas are flat — no closures)."""
    _CTX.clear()
    _CTX["workspace"] = Path(workspace)
    _CTX["sweep_name"] = sweep_name
    _CTX["last_state_json"] = last_state_json
    _CTX["audit_log_path"] = Path(workspace) / ".hyperherd" / "agent_log.jsonl"
    _CTX["next_tick_path"] = Path(workspace) / ".hyperherd" / "next-tick.json"
    _CTX["plan_path"] = Path(workspace) / ".hyperherd" / "MONITOR_PLAN.md"
    _CTX["user_prompt_path"] = Path(workspace) / "PROMPT.md"
    _CTX["user_prompt_ack_path"] = Path(workspace) / ".hyperherd" / "user-prompt.sha256"
    _CTX["channel"] = channel


# --- response helper -------------------------------------------------------

def _text_response(value: Any, *, is_error: bool = False) -> Dict[str, Any]:
    """Wrap a value into the SDK's expected return shape:
    `{"content": [{"type": "text", "text": "..."}]}`. Strings are kept
    as-is; everything else is JSON-encoded so the agent can parse it."""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, default=str)
        except (TypeError, ValueError):
            text = str(value)
    out: Dict[str, Any] = {"content": [{"type": "text", "text": text}]}
    if is_error:
        out["is_error"] = True
    return out


# --- audit log -------------------------------------------------------------

def _audit(event: str, **fields: Any) -> None:
    """Append a JSON line to `.hyperherd/agent_log.jsonl`. Best-effort —
    audit logging never raises into the agent."""
    path = _CTX.get("audit_log_path")
    if path is None:
        return
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "event": event,
        **fields,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass


# --- tools -----------------------------------------------------------------

@tool(
    "read_state",
    "Return the full per-tick state dict — totals, every trial's status and "
    "metrics, newly-failed with stderr, newly-completed, inbox, chat_history. "
    "The summary fields are already in the per-tick user message; call this "
    "ONLY when you need data the summary omits, like the per-trial table for "
    "live-phase decisions or to re-check after a long tool chain.",
    {},
)
async def read_state(args: Dict[str, Any]) -> Dict[str, Any]:
    raw = _CTX.get("last_state_json", "{}")
    return _text_response(json.loads(raw))


@tool(
    "read_plan",
    "Return MONITOR_PLAN.md contents. Skip this at tick start — the plan is "
    "already in the user message. Use it after `write_plan` if you need to "
    "verify what was saved.",
    {},
)
async def read_plan(args: Dict[str, Any]) -> Dict[str, Any]:
    path = _CTX.get("plan_path")
    if path is None or not path.is_file():
        return _text_response("")
    try:
        return _text_response(path.read_text())
    except OSError as e:
        return _text_response(f"failed to read plan: {e}", is_error=True)


@tool(
    "read_user_prompt",
    "Return the workspace-root PROMPT.md contents and its sha256. Use this when "
    "the per-tick state says PROMPT.md is `new` or `changed` and you need to "
    "fold its instructions into MONITOR_PLAN.md. Skip when status is `unchanged` "
    "— the file hasn't moved since you last acknowledged it. Returns "
    "{exists, sha256, text}.",
    {},
)
async def read_user_prompt(args: Dict[str, Any]) -> Dict[str, Any]:
    import hashlib
    path = _CTX.get("user_prompt_path")
    if path is None or not path.is_file():
        return _text_response({"exists": False, "sha256": None, "text": None})
    try:
        text = path.read_text()
    except OSError as e:
        return _text_response(f"failed to read PROMPT.md: {e}", is_error=True)
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return _text_response({"exists": True, "sha256": sha, "text": text})


@tool(
    "mark_user_prompt_read",
    "Record that you have read PROMPT.md at the given sha256 and folded its "
    "instructions into MONITOR_PLAN.md. Future ticks will then see "
    "user_prompt.status=`unchanged` until the file is edited again. Pass the "
    "sha256 from `read_user_prompt` (or from state.user_prompt.sha256).",
    {"sha256": str},
)
async def mark_user_prompt_read(args: Dict[str, Any]) -> Dict[str, Any]:
    sha = (args.get("sha256") or "").strip()
    if not sha:
        return _text_response("sha256 is required", is_error=True)
    path = _CTX["user_prompt_ack_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(sha + "\n")
    _audit("mark_user_prompt_read", sha256=sha)
    return _text_response({"acknowledged_sha256": sha, "path": str(path)})


@tool(
    "write_plan",
    "Replace MONITOR_PLAN.md with the given Markdown. Use after every tick that "
    "advances Phase, increments Quiet ticks, or adds a trial to Warned indices.",
    {"plan": str},
)
async def write_plan(args: Dict[str, Any]) -> Dict[str, Any]:
    plan = args["plan"]
    path = _CTX["plan_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plan)
    _audit("write_plan", bytes=len(plan))
    return _text_response({"path": str(path), "bytes": len(plan)})


@tool(
    "bump_mem",
    "Increase slurm.mem in hyperherd.yaml by `percent` percent (e.g. 50 → +50%). "
    "Returns {old, new}. Cap one bump per failure class per sweep — the agent "
    "tracks this in MONITOR_PLAN.md, not here.",
    {"percent": int},
)
async def bump_mem(args: Dict[str, Any]) -> Dict[str, Any]:
    return _text_response(await _bump_yaml_resource("mem", int(args["percent"])))


@tool(
    "bump_time",
    "Increase slurm.time by `percent` percent. Same shape and same cap as bump_mem.",
    {"percent": int},
)
async def bump_time(args: Dict[str, Any]) -> Dict[str, Any]:
    return _text_response(await _bump_yaml_resource("time", int(args["percent"])))


@tool(
    "run_indices",
    "Submit (or resubmit with force=true) the given trial indices. Returns "
    "{slurm_job_id, submitted_indices}. Use force=true after a YAML bump to "
    "re-run failed trials.",
    {"indices": list, "force": bool},
)
async def run_indices(args: Dict[str, Any]) -> Dict[str, Any]:
    indices = args.get("indices") or []
    force = bool(args.get("force", False))
    if not indices:
        return _text_response({"slurm_job_id": None, "submitted_indices": []})
    spec = ",".join(str(int(i)) for i in sorted(indices))
    cmd = ["herd", "run", "--json", "-i", spec, str(_CTX["workspace"])]
    if force:
        cmd.append("--force")
    return _text_response(await _run_herd_json(
        cmd, audit_event="run_indices",
        audit_fields={"indices": indices, "force": force},
    ))


@tool(
    "stop_index",
    "Cancel one running/queued trial. Returns {cancelled: [{index, slurm_job_id, "
    "previous_status}]} (empty list if the trial wasn't live).",
    {"index": int},
)
async def stop_index(args: Dict[str, Any]) -> Dict[str, Any]:
    index = int(args["index"])
    cmd = ["herd", "stop", "--json", str(_CTX["workspace"]), str(index)]
    return _text_response(await _run_herd_json(
        cmd, audit_event="stop_index", audit_fields={"index": index},
    ))


@tool(
    "stop_all",
    "Cancel every running/queued trial in the workspace. Returns the same "
    "shape as stop_index, with one entry per cancelled trial.",
    {},
)
async def stop_all(args: Dict[str, Any]) -> Dict[str, Any]:
    cmd = ["herd", "stop", "--json", "--all", str(_CTX["workspace"])]
    return _text_response(await _run_herd_json(
        cmd, audit_event="stop_all", audit_fields={},
    ))


@tool(
    "prune_index",
    "Algorithmically kill one trial — distinct from `stop_index`, which is "
    "for user-driven cancellations. The trial is scancel'd in SLURM AND "
    "marked `pruned` in the manifest, so `herd run` will NOT resubmit it on "
    "subsequent runs. Use this when the agent decides a trial is hopeless "
    "(NaN/inf, sustained divergence) — see the Pruning policy section in "
    "the skill. Reason is recorded in the audit log and shown to the user.",
    {"index": int, "reason": str},
)
async def prune_index(args: Dict[str, Any]) -> Dict[str, Any]:
    index = int(args["index"])
    reason = str(args["reason"])
    workspace = str(_CTX["workspace"])

    # Cancel via the existing herd-stop path so SLURM actually kills the job.
    stop_cmd = ["herd", "stop", "--json", workspace, str(index)]
    stop_result = await _run_herd_json(
        stop_cmd, audit_event="prune_stop",
        audit_fields={"index": index, "reason": reason},
    )

    # Then overwrite the resulting `cancelled` status with `pruned`.
    # _sync_slurm_status protects pruned trials from being flipped back
    # to cancelled when SLURM reports CANCELLED for the same job.
    try:
        from hyperherd import manifest
        manifest.update_trial_status(workspace, index, "pruned")
    except Exception as e:
        _audit("prune_status_update_failed", index=index, error=str(e))
        return _text_response(
            {"pruned": False, "error": f"manifest update failed: {e}"},
            is_error=True,
        )

    _audit("prune", index=index, reason=reason, stop_result=stop_result)
    return _text_response({
        "pruned": True, "index": index, "reason": reason,
    })


@tool(
    "summarize_metrics",
    "Summarize the optimization state in one call: for each non-ready "
    "trial and each logged metric, the most recent value (or the mean "
    "of the last `smooth` points if smooth>0). Useful when the user "
    "asks 'how is the sweep doing?' or you need a cross-trial view. "
    "Cheaper than calling list_metrics+compute_metric per trial. "
    "Returns {smooth, metric_names: [...], trials: [{index, name, "
    "status, metrics: {<metric>: <value>}}]}.",
    {"smooth": int},
)
async def summarize_metrics(args: Dict[str, Any]) -> Dict[str, Any]:
    smooth = int(args.get("smooth") or 0)
    workspace = str(_CTX["workspace"])
    from hyperherd import manifest as _manifest
    from hyperherd.logging import list_metric_streams, load_metric_stream

    trials = _manifest.load_manifest(workspace)
    out_trials = []
    metric_names = set()
    for trial in trials:
        status = trial.get("status", "?")
        if status == "ready":
            continue
        idx = trial["index"]
        metrics_for_trial: Dict[str, float] = {}
        for name in list_metric_streams(workspace, idx):
            stream = load_metric_stream(workspace, idx, name)
            numeric = [p["value"] for p in stream
                       if isinstance(p.get("value"), (int, float))]
            if not numeric:
                continue
            if smooth > 0 and len(numeric) > 1:
                tail = numeric[-smooth:]
                metrics_for_trial[name] = sum(tail) / len(tail)
            else:
                metrics_for_trial[name] = numeric[-1]
            metric_names.add(name)
        out_trials.append({
            "index": idx,
            "name": trial.get("experiment_name", ""),
            "status": status,
            "metrics": metrics_for_trial,
        })

    return _text_response({
        "smooth": smooth,
        "metric_names": sorted(metric_names),
        "trials": out_trials,
    })


@tool(
    "list_metrics",
    "List every metric name recorded for a trial, with `n` (logged "
    "points), `step_first`, `step_last`, and `last` (most recent "
    "value). Returns `{index, metrics: [...]}`. Use this for discovery "
    "— call once per sweep to learn what names the trainer is using "
    "(`val/loss` vs `val_loss`, `train/acc` vs `train_acc`, etc.) "
    "before reaching for `compute_metric`. Cache the result in the "
    "plan (e.g. `Available metrics: [val/loss, train/loss, ...]`) so "
    "future ticks don't re-discover.",
    {"index": int},
)
async def list_metrics(args: Dict[str, Any]) -> Dict[str, Any]:
    from hyperherd.logging import list_metric_streams, load_metric_stream
    index = int(args["index"])
    workspace = str(_CTX["workspace"])
    names = list_metric_streams(workspace, index)
    if not names:
        return _text_response({"index": index, "metrics": []})
    out = []
    for name in names:
        stream = load_metric_stream(workspace, index, name)
        if not stream:
            out.append({"name": name, "n": 0})
            continue
        first = stream[0]
        last = stream[-1]
        out.append({
            "name": name,
            "n": len(stream),
            "step_first": first.get("step"),
            "step_last": last.get("step"),
            "last": last.get("value"),
        })
    return _text_response({"index": index, "metrics": out})


@tool(
    "validate_config",
    "Run `herd test --cfg-job <workspace> <index>` to preflight a "
    "trial's resolved config without spending SLURM time. Hydra-"
    "specific: appends `--cfg job` to the override string so a Hydra "
    "trainer prints the resolved config and exits. Catches missing "
    "required fields, type mismatches, unknown parameter names, and "
    "launcher-level errors (missing container, bad conda env). Use as "
    "a canary preflight when the user said `yes` to the Hydra interview "
    "question. Returns {valid, returncode, stdout_tail, stderr_tail}.",
    {"index": int},
)
async def validate_config(args: Dict[str, Any]) -> Dict[str, Any]:
    index = int(args["index"])
    workspace = str(_CTX["workspace"])
    proc = await asyncio.create_subprocess_exec(
        "herd", "test", "--cfg-job", workspace, str(index),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    out = (stdout or b"").decode("utf-8", errors="replace")
    err = (stderr or b"").decode("utf-8", errors="replace")
    ok = proc.returncode == 0
    _audit("validate_config", index=index, valid=ok, returncode=proc.returncode)
    return _text_response({
        "valid": ok,
        "returncode": proc.returncode,
        "stdout_tail": out[-2000:],
        "stderr_tail": err[-2000:],
    })


@tool(
    "tail_log",
    "Read the last N lines of a trial's logs from disk. `stream` "
    "selects which file(s): 'both' (default — reads .out and .err with "
    "labeled sections; the right choice when verifying a canary because "
    "frameworks split training output between stdout and stderr "
    "inconsistently), 'stderr' (just .err — Python tracebacks, SLURM "
    "notices), or 'stdout' (just .out — print() output, framework "
    "progress bars). Default lines=40, max 1000. Returns plain text — "
    "pattern-match for training evidence (loss values, step/iteration/"
    "epoch counters, framework startup messages) or for stack traces.",
    {"index": int, "lines": int, "stream": str},
)
async def tail_log(args: Dict[str, Any]) -> Dict[str, Any]:
    from hyperherd.monitor_agent import commands as _cmd
    index = int(args["index"])
    lines = int(args.get("lines") or 40)
    stream = str(args.get("stream") or "both")
    text = _cmd.cmd_tail(Path(_CTX["workspace"]), index, lines, stream=stream)
    _audit("tail_log", index=index, lines=lines, stream=stream)
    return _text_response(text)


@tool(
    "compute_metric",
    "Aggregate a logged metric over a trial's stream. Reads "
    "`.hyperherd/results/<index>/stream/<metric>.jsonl` (written by "
    "`log_result(name, value, step=...)` in the trial code). Each metric "
    "lives in its own file — pass the metric name to query just that one. "
    "Returns {n, last, mean, median, stddev, min, max, has_nan_or_inf, "
    "recent} where `recent` is the last few raw values (for trend "
    "inspection). Optional windowing args narrow the result: `last_n` "
    "(last N entries by file order), `step_min`/`step_max` (closed "
    "interval on the step counter), `since_seconds` (entries logged "
    "within the last N seconds). Multiple windows compose (AND). Returns "
    "{n: 0} (no error) if the stream doesn't exist or no entries match — "
    "that just means there's no data to act on yet.",
    {
        "index": int, "metric": str,
        "last_n": int, "step_min": int, "step_max": int,
        "since_seconds": int,
    },
)
async def compute_metric(args: Dict[str, Any]) -> Dict[str, Any]:
    index = int(args["index"])
    metric = str(args["metric"])
    last_n = args.get("last_n")
    step_min = args.get("step_min")
    step_max = args.get("step_max")
    since_seconds = args.get("since_seconds")

    from hyperherd.logging import load_metric_stream

    stream = load_metric_stream(str(_CTX["workspace"]), index, metric)
    if not stream:
        return _text_response({"n": 0, "metric": metric, "index": index})

    # Window filters (compose AND).
    filtered = stream
    if step_min is not None:
        filtered = [p for p in filtered if p.get("step", 0) >= int(step_min)]
    if step_max is not None:
        filtered = [p for p in filtered if p.get("step", 0) <= int(step_max)]
    if since_seconds is not None:
        import time
        cutoff = time.time() - int(since_seconds)
        # Skip entries without a timestamp (legacy streams from before
        # ts was added) — they can't satisfy a since filter either way.
        filtered = [p for p in filtered if p.get("ts", 0) >= cutoff]
    if last_n is not None:
        filtered = filtered[-int(last_n):]

    if not filtered:
        return _text_response({
            "n": 0, "metric": metric, "index": index,
            "n_total": len(stream),
            "note": "no entries match the window",
        })

    values = [
        p["value"] for p in filtered
        if isinstance(p.get("value"), (int, float))
    ]
    if not values:
        return _text_response({
            "n": 0, "metric": metric, "index": index,
            "note": "entries exist but no numeric values",
        })

    import statistics, math
    has_nan = any(math.isnan(v) or math.isinf(v) for v in values
                  if isinstance(v, float))
    finite = [v for v in values
              if not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))]

    out: Dict[str, Any] = {
        "n": len(values),
        "n_total": len(stream),
        "last": values[-1],
        "has_nan_or_inf": has_nan,
        # Trail of the last few raw values so the agent can verify
        # "growing gap" / "decreasing" / "plateau" claims without
        # fetching the full history.
        "recent": values[-min(8, len(values)):],
    }
    if filtered:
        first = filtered[0]
        last = filtered[-1]
        if "step" in first:
            out["step_first"] = first["step"]
        if "step" in last:
            out["step_last"] = last["step"]
    if finite:
        out["mean"] = statistics.fmean(finite)
        out["median"] = statistics.median(finite)
        out["min"] = min(finite)
        out["max"] = max(finite)
        if len(finite) > 1:
            out["stddev"] = statistics.stdev(finite)
    return _text_response(out)


@tool(
    "msg",
    "Post a conversational message — replies to the user, alerts, "
    "questions, or anything else that's part of the back-and-forth. "
    "Recorded in chat history so future ticks remember it. For the "
    "obligatory per-tick heartbeat summary, use `tick_summary` instead. "
    "The Discord bot's display name is already attached to every post, "
    "so don't prefix the text with the bot name.",
    {"text": str},
)
async def msg(args: Dict[str, Any]) -> Dict[str, Any]:
    text = args["text"]
    body = _agent_prefix(text)
    channel = _CTX.get("channel")
    if channel is None:
        # No chat surface configured — message has nowhere to go. Audit
        # it so it's still in the JSONL log, return a soft failure to
        # the agent rather than crashing the tick.
        _audit("msg_skipped_no_channel", text=text[:200])
        return _text_response(
            {"posted": False, "reason": "no chat channel configured"},
            is_error=True,
        )
    try:
        await channel.post(body)
        _audit("msg", text=text[:200], via=channel.name)
        record_chat_entry(
            Path(_CTX["workspace"]),
            role="agent", text=text, via=channel.name, author="agent",
        )
        return _text_response({"posted": True, "via": channel.name})
    except Exception as e:
        _audit("msg_failed", error=str(e), via=channel.name)
        return _text_response(
            {"posted": False, "error": str(e), "via": channel.name},
            is_error=True,
        )


@tool(
    "tick_summary",
    "Post the obligatory per-tick heartbeat summary. Same routing as "
    "`msg`, but NOT recorded in chat history — heartbeats would otherwise "
    "drown out actual conversation. Use this for the once-per-tick "
    "'tick clean — ... Next tick in X' message and nothing else.",
    {"text": str},
)
async def tick_summary(args: Dict[str, Any]) -> Dict[str, Any]:
    text = args["text"]
    body = _agent_prefix(text)
    channel = _CTX.get("channel")
    if channel is None:
        _audit("tick_summary_skipped_no_channel", text=text[:200])
        return _text_response(
            {"posted": False, "reason": "no chat channel configured"},
            is_error=True,
        )
    try:
        await channel.post(body)
        _audit("tick_summary", text=text[:200], via=channel.name)
        return _text_response({"posted": True, "via": channel.name})
    except Exception as e:
        _audit("tick_summary_failed", error=str(e), via=channel.name)
        return _text_response(
            {"posted": False, "error": str(e), "via": channel.name},
            is_error=True,
        )


@tool(
    "post_plot",
    "Render a metric across one or more trials as a PNG and upload it "
    "to the chat channel. Use this when a plot would communicate "
    "diagnostic info more clearly than text — divergence, ranking, "
    "training dynamics. `metric` is the same name the trial logs (e.g. "
    "`train/loss`, `val/acc`). `trial_indices` defaults to all "
    "non-ready trials. `smooth` is a rolling-mean window (0 = off). "
    "If `caption` is set, it's posted alongside the file as a chat "
    "message — keep it short.",
    {
        "metric": str,
        "trial_indices": Optional[List[int]],
        "caption": Optional[str],
        "smooth": Optional[int],
    },
)
async def post_plot(args: Dict[str, Any]) -> Dict[str, Any]:
    metric = str(args["metric"])
    trial_indices = args.get("trial_indices")
    caption = args.get("caption")
    smooth = int(args.get("smooth") or 0)
    channel = _CTX.get("channel")
    if channel is None:
        _audit("post_plot_skipped_no_channel", metric=metric)
        return _text_response(
            {"posted": False, "reason": "no chat channel configured"},
            is_error=True,
        )
    from hyperherd.monitor_agent import plots

    workspace = _CTX["workspace"]

    def _render():
        return plots.render_metric_plot(
            workspace, metric,
            trial_indices=trial_indices, smooth=smooth,
        )

    try:
        png_path = await asyncio.get_running_loop().run_in_executor(
            None, _render,
        )
    except plots.PlotUnavailable as e:
        _audit("post_plot_failed_render", metric=metric, error=str(e))
        return _text_response(
            {"posted": False, "error": str(e)}, is_error=True,
        )
    except Exception as e:
        _audit("post_plot_failed_render", metric=metric, error=str(e))
        return _text_response(
            {"posted": False, "error": f"{type(e).__name__}: {e}"},
            is_error=True,
        )

    body = _agent_prefix(caption) if caption else None
    try:
        await channel.post_file(png_path, body=body)
        _audit("post_plot", metric=metric,
               trial_indices=list(trial_indices) if trial_indices else None,
               caption=(caption or "")[:200])
        return _text_response({"posted": True, "metric": metric})
    except Exception as e:
        _audit("post_plot_failed_upload", metric=metric, error=str(e))
        return _text_response(
            {"posted": False, "error": str(e)}, is_error=True,
        )
    finally:
        try:
            png_path.unlink()
        except OSError:
            pass


@tool(
    "post_results_table",
    "Upload trial parameters + logged metrics (the same data `herd res` "
    "shows) to the chat channel as a TSV file, so the user can download "
    "it for plotting or analysis. When the rendered table is small "
    "enough, an inline code-block preview is included in the message "
    "body alongside the file. By default also attaches `steps.tsv.gz` — "
    "a gzipped long-form TSV of every per-step metric record (the same "
    "data `herd res --steps` produces). Pass `with_steps=False` to "
    "suppress that attachment, e.g. when the user only wants the "
    "summary. Use this at sweep end — call it before "
    "`halt('sweep complete')`. Optional `caption` is posted as the "
    "message body above the file.",
    {"caption": Optional[str], "with_steps": Optional[bool]},
)
async def post_results_table(args: Dict[str, Any]) -> Dict[str, Any]:
    caption = args.get("caption")
    with_steps = args.get("with_steps")
    if with_steps is None:
        with_steps = True
    channel = _CTX.get("channel")
    if channel is None:
        _audit("post_results_table_skipped_no_channel")
        return _text_response(
            {"posted": False, "reason": "no chat channel configured"},
            is_error=True,
        )

    workspace = _CTX["workspace"]
    try:
        from hyperherd.monitor_agent.commands import (
            build_results_table, build_steps_blob,
        )
        built = build_results_table(workspace)
        steps = build_steps_blob(workspace) if with_steps else None
    except Exception as e:
        _audit("post_results_table_failed_load", error=str(e))
        return _text_response(
            {"posted": False, "error": f"{type(e).__name__}: {e}"},
            is_error=True,
        )

    if built is None:
        return _text_response(
            {"posted": False, "reason": "no trials in manifest"},
            is_error=True,
        )

    # Always upload the TSV — users want a downloadable file for plotting
    # / analysis, not just an inline table that's a pain to copy out of
    # Discord. When the rendered code-block table also fits in the
    # message body, include it inline as a preview alongside the upload.
    body_prefix = (_agent_prefix(caption) + "\n") if caption else ""
    code_block = f"```\n{built['table_text']}\n```"
    inline = body_prefix + code_block
    INLINE_LIMIT = 1900
    body = inline if len(inline) <= INLINE_LIMIT else (body_prefix.rstrip() or None)

    # Discord caps a single attachment at 10 MB on a non-boosted server.
    # Skip the steps blob if it's too big (the agent can suggest the user
    # pull it via `herd res --steps` on the host instead).
    DISCORD_FILE_LIMIT = 10 * 1024 * 1024
    steps_dropped_reason: Optional[str] = None
    if steps is not None and len(steps["gz_bytes"]) > DISCORD_FILE_LIMIT:
        steps_dropped_reason = (
            f"steps.tsv.gz is {len(steps['gz_bytes']) / 1e6:.1f} MB — "
            f"over Discord's 10 MB attachment cap; pull it via "
            f"`herd res --steps` on the host."
        )
        steps = None

    import tempfile
    tsv_path = Path(tempfile.mkstemp(suffix=".tsv", prefix="results-")[1])
    steps_path: Optional[Path] = None
    try:
        tsv_path.write_text(built["tsv"])
        paths: list = [tsv_path]
        if steps is not None:
            steps_path = Path(tempfile.mkstemp(suffix=".tsv.gz", prefix="steps-")[1])
            steps_path.write_bytes(steps["gz_bytes"])
            paths.append(steps_path)

        try:
            await channel.post_file(paths if len(paths) > 1 else paths[0], body=body)
            audit_fields = {
                "trials": built["n_trials"],
                "metrics": built["n_metrics"],
                "inline_preview": (body == inline),
                "steps_attached": steps is not None,
            }
            if steps is not None:
                audit_fields.update({
                    "steps_rows": steps["n_rows"],
                    "steps_gz_bytes": len(steps["gz_bytes"]),
                })
            if steps_dropped_reason:
                audit_fields["steps_dropped"] = steps_dropped_reason
            _audit("post_results_table", **audit_fields)
            response: Dict[str, Any] = {
                "posted": True,
                "trials": built["n_trials"], "metrics": built["n_metrics"],
                "inline_preview": (body == inline),
                "steps_attached": steps is not None,
            }
            if steps is not None:
                response["steps_rows"] = steps["n_rows"]
            if steps_dropped_reason:
                response["steps_dropped"] = steps_dropped_reason
            return _text_response(response)
        except Exception as e:
            _audit("post_results_table_failed_upload", error=str(e))
            return _text_response(
                {"posted": False, "error": str(e)}, is_error=True,
            )
    finally:
        for p in (tsv_path, steps_path):
            if p is None:
                continue
            try:
                p.unlink()
            except OSError:
                pass


@tool(
    "msg_thread",
    "Post a message into a per-trial discussion thread (creating the "
    "thread if it doesn't exist yet). Use this for trial-specific "
    "follow-ups: failure diagnosis, remediation plan, comparison with "
    "siblings. Keeps the main channel scannable when many trials are "
    "in flight.",
    {"trial_index": int, "text": str},
)
async def msg_thread(args: Dict[str, Any]) -> Dict[str, Any]:
    idx = int(args["trial_index"])
    text = str(args["text"])
    body = _agent_prefix(text)
    channel = _CTX.get("channel")
    if channel is None:
        _audit("msg_thread_skipped_no_channel", trial_index=idx)
        return _text_response(
            {"posted": False, "reason": "no chat channel configured"},
            is_error=True,
        )
    try:
        await channel.post_to_trial_thread(
            idx, body,
            thread_seed_text=f"Trial #{idx} — discussion",
        )
        _audit("msg_thread", trial_index=idx, text=text[:200],
               via=channel.name)
        record_chat_entry(
            Path(_CTX["workspace"]),
            role="agent", text=f"[trial-{idx} thread] {text}",
            via=channel.name, author="agent",
        )
        return _text_response({"posted": True, "trial_index": idx})
    except Exception as e:
        _audit("msg_thread_failed", trial_index=idx, error=str(e))
        return _text_response(
            {"posted": False, "error": str(e)}, is_error=True,
        )


_AGENT_EMOJI = "🐕"


def _agent_prefix(text: str) -> str:
    """Prepend the agent emoji to outbound posts so the user can spot
    agent voice against the daemon-side ▶️/✅/⚠️/🐾/🛑 stream. Idempotent
    if the text already starts with the emoji."""
    stripped = text.lstrip()
    if stripped.startswith(_AGENT_EMOJI):
        return text
    return f"{_AGENT_EMOJI} {text}"


# --- chat history ----------------------------------------------------------
# Rolling buffer of recent messages on both sides so the agent can stitch
# its own questions to the user's replies across ticks. Both sides write
# here: this `msg` tool when the agent posts, and `state._drain_inbox`
# when a user reply lands.

CHAT_HISTORY_FILENAME = "chat-history.jsonl"
CHAT_HISTORY_KEEP = 6  # last N entries total, mixed roles — "the last few"


def record_chat_entry(
    workspace: Path,
    *,
    role: str,        # "agent" | "user"
    text: str,
    via: str,         # "discord" | "webhook" | etc.
    author: str = "",
    timestamp: Optional[str] = None,
) -> None:
    """Append a chat entry to chat-history.jsonl, trimmed to last N."""
    path = Path(workspace) / ".hyperherd" / CHAT_HISTORY_FILENAME
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        existing: list[str] = []
        if path.is_file():
            existing = [ln for ln in path.read_text().splitlines() if ln.strip()]
        if timestamp is None:
            timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        line = json.dumps({
            "timestamp": timestamp,
            "role": role,
            "author": author,
            "via": via,
            "text": text,
        })
        kept = (existing + [line])[-CHAT_HISTORY_KEEP:]
        path.write_text("\n".join(kept) + "\n")
    except OSError as e:
        # Non-fatal — the post itself already succeeded.
        _audit("record_chat_entry_failed", error=str(e))


@tool(
    "schedule_next",
    "Set how many seconds until the next scheduled tick. Required: every tick "
    "MUST call this exactly once before returning. Clamped to [60, 3600]. The "
    "daemon's scheduler reads this immediately after the agent's turn ends.",
    {"delay_seconds": int},
)
async def schedule_next(args: Dict[str, Any]) -> Dict[str, Any]:
    delay = max(60, min(3600, int(args["delay_seconds"])))
    path = _CTX["next_tick_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "delay_seconds": delay,
        "scheduled_at": datetime.now(timezone.utc).isoformat(),
    }))
    _audit("schedule_next", delay_seconds=delay)
    return _text_response({"delay_seconds": delay})


@tool(
    "halt",
    "Stop the agent loop entirely. Use for unrecoverable conditions: recurring "
    "code-bug exceptions, sweep complete, user replied 'pause'. The daemon "
    "stops scheduling new ticks once this is called.",
    {"reason": str},
)
async def halt(args: Dict[str, Any]) -> Dict[str, Any]:
    reason = str(args["reason"])
    path = _CTX["next_tick_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "halted": True,
        "reason": reason,
        "halted_at": datetime.now(timezone.utc).isoformat(),
    }))
    _audit("halt", reason=reason)
    return _text_response({"halted": True, "reason": reason})


# --- helpers ----------------------------------------------------------------

async def _run_herd_json(
    cmd: List[str],
    *, audit_event: str, audit_fields: Dict[str, Any],
) -> Dict[str, Any]:
    """Subprocess `herd ... --json`, parse stdout, surface stderr to audit log
    on non-zero exit. Async so concurrent tool calls don't block."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate()
    if proc.returncode != 0:
        _audit(f"{audit_event}_failed", returncode=proc.returncode,
               stderr=err.decode("utf-8", errors="replace")[:500],
               **audit_fields)
        return {"error": err.decode("utf-8", errors="replace")[:500],
                "returncode": proc.returncode}
    try:
        result = json.loads(out)
    except json.JSONDecodeError as e:
        _audit(f"{audit_event}_parse_error", stdout=out.decode("utf-8", errors="replace")[:500])
        return {"error": f"failed to parse herd JSON: {e}"}
    _audit(audit_event, **audit_fields, result=result)
    return result


async def _bump_yaml_resource(field: str, percent: int) -> Dict[str, str]:
    """Edit hyperherd.yaml's slurm.<field>, preserving comments. The field
    is a string with units (e.g. '8G' or '01:00:00'); we parse, scale,
    serialize back."""
    from ruamel.yaml import YAML  # imported lazily — only here when bump fires
    yaml = YAML()
    yaml.preserve_quotes = True

    config_path = _CTX["workspace"] / "hyperherd.yaml"
    with open(config_path) as f:
        data = yaml.load(f)

    slurm = data.get("slurm") or {}
    old = str(slurm.get(field, ""))
    if not old:
        return {"error": f"slurm.{field} is unset; cannot bump"}

    if field == "mem":
        new = _scale_mem(old, percent)
    elif field == "time":
        new = _scale_time(old, percent)
    else:  # pragma: no cover
        return {"error": f"unsupported field {field}"}

    slurm[field] = new
    data["slurm"] = slurm
    with open(config_path, "w") as f:
        yaml.dump(data, f)

    _audit(f"bump_{field}", percent=percent, old=old, new=new)
    return {"old": old, "new": new}


_MEM_UNITS = {"K": 1, "M": 1024, "G": 1024 ** 2, "T": 1024 ** 3}


def _scale_mem(value: str, percent: int) -> str:
    """e.g. ('8G', 50) -> '12G'. Returns same unit suffix."""
    s = value.strip()
    suffix = s[-1].upper() if s and s[-1].upper() in _MEM_UNITS else ""
    num_str = s[:-1] if suffix else s
    n = float(num_str) * (1 + percent / 100.0)
    if suffix in ("G", "T"):
        return f"{n:.2f}{suffix}".rstrip("0").rstrip(".") + (suffix if "." not in f"{n:.2f}" else "")
    return f"{int(round(n))}{suffix}"


def _scale_time(value: str, percent: int) -> str:
    """e.g. ('01:00:00', 50) -> '01:30:00'. Accepts HH:MM:SS or D-HH:MM:SS."""
    s = value.strip()
    days = 0
    if "-" in s:
        d, _, s = s.partition("-")
        days = int(d)
    parts = [int(p) for p in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts
    total = days * 86400 + h * 3600 + m * 60 + sec
    scaled = int(total * (1 + percent / 100.0))
    new_days, rem = divmod(scaled, 86400)
    new_h, rem = divmod(rem, 3600)
    new_m, new_s = divmod(rem, 60)
    if new_days:
        return f"{new_days}-{new_h:02d}:{new_m:02d}:{new_s:02d}"
    return f"{new_h:02d}:{new_m:02d}:{new_s:02d}"


# --- registry ---------------------------------------------------------------

ALL = [
    read_state, read_plan, write_plan,
    read_user_prompt, mark_user_prompt_read,
    bump_mem, bump_time,
    run_indices, stop_index, stop_all, prune_index,
    validate_config, tail_log, list_metrics, compute_metric,
    summarize_metrics,
    msg, tick_summary, post_results_table, schedule_next, halt,
]
"""All in-process tools, in the order they're registered with the SDK's
`create_sdk_mcp_server(name='hyperherd', tools=ALL)`."""
