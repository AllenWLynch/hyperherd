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
    "msg",
    "Post a conversational message — replies to the user, alerts, "
    "questions, or anything else that's part of the back-and-forth. "
    "Recorded in chat history so future ticks remember it. For the "
    "obligatory per-tick heartbeat summary, use `tick_summary` instead. "
    "Voice rule: prefix the body with 'Herd dog:' so the user can spot "
    "agent messages.",
    {"text": str},
)
async def msg(args: Dict[str, Any]) -> Dict[str, Any]:
    text = args["text"]
    channel = _CTX.get("channel")
    if channel is not None:
        try:
            await channel.post(text)
            _audit("msg", text=text[:200], via=channel.name)
            record_chat_entry(
                Path(_CTX["workspace"]),
                role="agent", text=text, via=channel.name, author="Herd dog",
            )
            return _text_response({"posted": True, "via": channel.name})
        except Exception as e:
            _audit("msg_failed", error=str(e), via=channel.name)
            return _text_response(
                {"posted": False, "error": str(e), "via": channel.name},
                is_error=True,
            )

    from hyperherd import watch
    from hyperherd.config import load_config
    config = load_config(str(_CTX["workspace"]))
    webhook = config.watch.webhook
    fmt = config.watch.format
    if not webhook:
        webhook, _ = watch.resolve_default_webhook(config.workspace, config.name)
        fmt = "ntfy"

    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(
            None, lambda: watch.post_message(webhook, fmt, text, config.name)
        )
    except OSError as e:
        _audit("msg_failed", error=str(e))
        return _text_response({"posted": False, "error": str(e)}, is_error=True)
    _audit("msg", text=text[:200], via="webhook")
    record_chat_entry(
        Path(_CTX["workspace"]),
        role="agent", text=text, via="webhook", author="Herd dog",
    )
    return _text_response({"posted": True, "via": "webhook", "webhook": webhook})


@tool(
    "tick_summary",
    "Post the obligatory per-tick heartbeat summary. Same routing as "
    "`msg`, but NOT recorded in chat history — heartbeats would otherwise "
    "drown out actual conversation. Use this for the once-per-tick "
    "'Herd dog: tick clean — ... Next tick in X' message and nothing else.",
    {"text": str},
)
async def tick_summary(args: Dict[str, Any]) -> Dict[str, Any]:
    text = args["text"]
    channel = _CTX.get("channel")
    if channel is not None:
        try:
            await channel.post(text)
            _audit("tick_summary", text=text[:200], via=channel.name)
            return _text_response({"posted": True, "via": channel.name})
        except Exception as e:
            _audit("tick_summary_failed", error=str(e), via=channel.name)
            return _text_response(
                {"posted": False, "error": str(e), "via": channel.name},
                is_error=True,
            )

    from hyperherd import watch
    from hyperherd.config import load_config
    config = load_config(str(_CTX["workspace"]))
    webhook = config.watch.webhook
    fmt = config.watch.format
    if not webhook:
        webhook, _ = watch.resolve_default_webhook(config.workspace, config.name)
        fmt = "ntfy"

    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(
            None, lambda: watch.post_message(webhook, fmt, text, config.name)
        )
    except OSError as e:
        _audit("tick_summary_failed", error=str(e))
        return _text_response({"posted": False, "error": str(e)}, is_error=True)
    _audit("tick_summary", text=text[:200], via="webhook")
    return _text_response({"posted": True, "via": "webhook", "webhook": webhook})


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
    bump_mem, bump_time,
    run_indices, stop_index, stop_all,
    msg, tick_summary, schedule_next, halt,
]
"""All in-process tools, in the order they're registered with the SDK's
`create_sdk_mcp_server(name='hyperherd', tools=ALL)`."""
