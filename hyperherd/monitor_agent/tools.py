"""The agent's complete action surface — exactly 11 typed tools.

Each function is a thin wrapper that either calls existing HyperHerd
machinery directly (msg, manifest reads) or shells out to the `herd` CLI
with `--json` so we don't duplicate the launch/stop logic. Every wrapper
records what happened in `.hyperherd/agent_log.jsonl` for the audit trail.

`schedule_next` and `halt` are special — they don't take action against the
workspace; they communicate the agent's intent for the *next* tick by
writing to `.hyperherd/next-tick.json`. The daemon's outer loop reads that
file after `agent.run()` returns.

The SDK's `@tool` decorator wraps these into in-process MCP tools; we
register them under a single `hyperherd` MCP server and the agent calls them
as `mcp__hyperherd__<name>`.
"""

import asyncio
import json
import os
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
    `agent.run()`. The agent's tools read this dict instead of taking the
    workspace as an arg (the SDK's tool schemas are flat — no closures)."""
    _CTX.clear()
    _CTX["workspace"] = Path(workspace)
    _CTX["sweep_name"] = sweep_name
    _CTX["last_state_json"] = last_state_json
    _CTX["audit_log_path"] = Path(workspace) / ".hyperherd" / "agent_log.jsonl"
    _CTX["next_tick_path"] = Path(workspace) / ".hyperherd" / "next-tick.json"
    _CTX["plan_path"] = Path(workspace) / ".hyperherd" / "MONITOR_PLAN.md"
    _CTX["channel"] = channel


# --- audit log --------------------------------------------------------------

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


# --- tools ------------------------------------------------------------------

@tool(
    "read_state",
    "Return the per-tick state snapshot: totals, every trial's status and metrics, "
    "newly-failed trials with stderr tails, newly-completed trials, and any user "
    "messages received via Discord since the last tick. Call this first.",
    {},
)
async def read_state() -> Dict[str, Any]:
    raw = _CTX.get("last_state_json", "{}")
    return json.loads(raw)


@tool(
    "read_plan",
    "Return the contents of MONITOR_PLAN.md (the agent's cross-tick notepad). "
    "Empty string if the plan doesn't exist yet (first tick after `herd monitor-v2 init`).",
    {},
)
async def read_plan() -> str:
    path = _CTX.get("plan_path")
    if path is None or not path.is_file():
        return ""
    try:
        return path.read_text()
    except OSError:
        return ""


@tool(
    "write_plan",
    "Replace MONITOR_PLAN.md with the given Markdown. Use after every tick that "
    "advances Phase, increments Quiet ticks, or adds a trial to Warned indices.",
    {"plan": str},
)
async def write_plan(plan: str) -> Dict[str, str]:
    path = _CTX["plan_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(plan)
    _audit("write_plan", bytes=len(plan))
    return {"path": str(path), "bytes": len(plan)}


@tool(
    "bump_mem",
    "Increase slurm.mem in hyperherd.yaml by `percent` percent (e.g. 50 → +50%). "
    "Returns {old, new}. Cap one bump per failure class per sweep — the agent "
    "tracks this in MONITOR_PLAN.md, not here.",
    {"percent": int},
)
async def bump_mem(percent: int) -> Dict[str, str]:
    return await _bump_yaml_resource("mem", percent)


@tool(
    "bump_time",
    "Increase slurm.time by `percent` percent. Same shape and same cap as bump_mem.",
    {"percent": int},
)
async def bump_time(percent: int) -> Dict[str, str]:
    return await _bump_yaml_resource("time", percent)


@tool(
    "run_indices",
    "Submit (or resubmit with force=true) the given trial indices. Returns "
    "{slurm_job_id, submitted_indices}. Use force=true after a YAML bump to "
    "re-run failed trials.",
    {"indices": list, "force": bool},
)
async def run_indices(indices: List[int], force: bool = False) -> Dict[str, Any]:
    if not indices:
        return {"slurm_job_id": None, "submitted_indices": []}
    spec = ",".join(str(i) for i in sorted(indices))
    cmd = ["herd", "run", "--json", "-i", spec, str(_CTX["workspace"])]
    if force:
        cmd.append("--force")
    return await _run_herd_json(cmd, audit_event="run_indices",
                                 audit_fields={"indices": indices, "force": force})


@tool(
    "stop_index",
    "Cancel one running/queued trial. Returns {cancelled: [{index, slurm_job_id, "
    "previous_status}]} (empty list if the trial wasn't live).",
    {"index": int},
)
async def stop_index(index: int) -> Dict[str, Any]:
    cmd = ["herd", "stop", "--json", str(_CTX["workspace"]), str(index)]
    return await _run_herd_json(cmd, audit_event="stop_index",
                                 audit_fields={"index": index})


@tool(
    "stop_all",
    "Cancel every running/queued trial in the workspace. Returns the same "
    "shape as stop_index, with one entry per cancelled trial.",
    {},
)
async def stop_all() -> Dict[str, Any]:
    cmd = ["herd", "stop", "--json", "--all", str(_CTX["workspace"])]
    return await _run_herd_json(cmd, audit_event="stop_all", audit_fields={})


@tool(
    "msg",
    "Post a notification to the user. Routes through the configured chat "
    "channel (Discord) if available, otherwise the legacy webhook. Voice "
    "rule: prefix the body with 'Herd dog:' so the user can spot agent "
    "messages.",
    {"text": str},
)
async def msg(text: str) -> Dict[str, Any]:
    # Channel takes priority over webhook when both are configured —
    # the user explicitly opted into a channel by setting it up.
    channel = _CTX.get("channel")
    if channel is not None:
        try:
            await channel.post(text)
            _audit("msg", text=text[:200], via=channel.name)
            return {"posted": True, "via": channel.name}
        except Exception as e:
            _audit("msg_failed", error=str(e), via=channel.name)
            return {"posted": False, "error": str(e), "via": channel.name}

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
        return {"posted": False, "error": str(e)}
    _audit("msg", text=text[:200], via="webhook")
    return {"posted": True, "via": "webhook", "webhook": webhook}


@tool(
    "schedule_next",
    "Set how many seconds until the next scheduled tick. Required: every tick "
    "MUST call this exactly once before returning. Clamped to [60, 3600]. The "
    "daemon's scheduler reads this immediately after the agent's turn ends.",
    {"delay_seconds": int},
)
async def schedule_next(delay_seconds: int) -> Dict[str, int]:
    delay = max(60, min(3600, int(delay_seconds)))
    path = _CTX["next_tick_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "delay_seconds": delay,
        "scheduled_at": datetime.now(timezone.utc).isoformat(),
    }))
    _audit("schedule_next", delay_seconds=delay)
    return {"delay_seconds": delay}


@tool(
    "halt",
    "Stop the agent loop entirely. Use for unrecoverable conditions: recurring "
    "code-bug exceptions, sweep complete, user replied 'pause'. The daemon "
    "stops scheduling new ticks once this is called.",
    {"reason": str},
)
async def halt(reason: str) -> Dict[str, str]:
    path = _CTX["next_tick_path"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "halted": True,
        "reason": reason,
        "halted_at": datetime.now(timezone.utc).isoformat(),
    }))
    _audit("halt", reason=reason)
    return {"halted": True, "reason": reason}


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
    msg, schedule_next, halt,
]
"""All in-process tools, in the order they're registered with the SDK's
`create_sdk_mcp_server(name='hyperherd', tools=ALL)`."""
