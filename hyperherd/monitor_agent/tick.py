"""One tick = build state, hand it to the agent, return.

Two entrypoints:

- `run_tick(workspace, trigger)` — the live path: assembles state, configures
  the SDK, calls `query()`, streams tool calls into the audit log, returns
  the next-tick delay (or halted=True) for the daemon to act on.
- `dry_run(workspace, trigger)` — assembles state and renders the prompt
  *without* calling Anthropic. Useful for verifying the deterministic part
  works before paying for tokens.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

_log = logging.getLogger(__name__)


_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Z_][A-Z0-9_]*)\}")


def _resolve_env(env: Dict[str, str]) -> Dict[str, str]:
    """Expand `${VAR}` references in values from the daemon's environment.
    Missing vars become empty strings — the subprocess MCP can decide
    whether that's fatal."""
    out = {}
    for k, v in env.items():
        out[k] = _ENV_VAR_PATTERN.sub(
            lambda m: os.environ.get(m.group(1), ""), v
        )
    return out

from hyperherd.monitor_agent import state as state_mod
from hyperherd.monitor_agent import tools as tools_mod
from hyperherd.monitor_agent import prompt


@dataclass
class TickResult:
    """What the daemon needs after a tick: how long until the next one, or
    whether the agent halted, plus accounting for visibility."""
    next_delay_seconds: Optional[int]
    halted: bool
    halt_reason: Optional[str]
    cost_usd: float
    turns: int


def dry_run(workspace: Path, trigger: state_mod.TickTrigger = "scheduled") -> dict:
    """Assemble state + render the prompt. No SDK call, no API spend.

    Returns a dict the caller can pretty-print to show the user what the
    agent would see on this tick.
    """
    workspace = Path(workspace).resolve()
    s = state_mod.compute(workspace, trigger)
    return {
        "trigger": trigger,
        "workspace": str(workspace),
        "system_prompt_chars": len(prompt.system_prompt()),
        "state": s.to_dict(),
        "user_message": prompt.render_state(s),
    }


async def run_tick(
    workspace: Path,
    trigger: state_mod.TickTrigger = "scheduled",
    *, max_turns: int = 16,
    channel=None,
) -> TickResult:
    """Live tick: assemble → configure SDK → run → return.

    Importing the SDK lazily so a missing `claude-agent-sdk` install only
    bites users on the live path, not on dry runs / unit tests.
    """
    # Auth: the SDK shells out to the `claude` CLI, which accepts either an
    # ANTHROPIC_API_KEY (pay-per-token API console billing) or the OAuth
    # credentials from `claude /login` (subscription billing). We don't
    # require the env var here — if neither is present, the CLI itself will
    # surface a clear auth error.

    try:
        from claude_agent_sdk import (  # type: ignore
            query, ClaudeAgentOptions, create_sdk_mcp_server,
        )
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "claude-agent-sdk not installed. Run "
            "`pip install hyperherd[monitor]` to install the monitor extras."
        ) from e

    workspace = Path(workspace).resolve()
    s = state_mod.compute(workspace, trigger)

    _log.info(
        "Tick state assembled: trigger=%s inbox=%d newly_failed=%d "
        "newly_completed=%d newly_pruned=%d totals=%s",
        trigger, len(s.inbox), len(s.newly_failed),
        len(s.newly_completed), len(s.newly_pruned), s.totals,
    )

    # Bind tool context before the agent starts — tools read this dict
    # since their @tool schemas can't carry workspace as an argument.
    tools_mod.set_context(
        workspace=workspace,
        sweep_name=s.sweep_name,
        last_state_json=json.dumps(s.to_dict()),
        channel=channel,
    )

    # Reset the next-tick file so we can detect whether the agent called
    # schedule_next() — if not, the daemon falls back to a sensible default.
    next_tick_path = workspace / ".hyperherd" / "next-tick.json"
    if next_tick_path.is_file():
        next_tick_path.unlink()

    in_process = create_sdk_mcp_server(name="hyperherd", tools=tools_mod.ALL)
    mcp_servers = {"hyperherd": in_process}

    # User-configured external MCP servers (wandb, mlflow, etc.) loaded
    # from `mcp_servers:` in hyperherd.yaml. They get passed through to
    # the SDK as subprocess MCPs; tools appear to the agent under
    # `mcp__<name>__*`. Allowed-tools includes a wildcard for each so
    # the agent can use any tool the server exposes without us
    # enumerating them up front.
    extra_allowed: list = []
    try:
        from hyperherd.config import load_config
        config = load_config(str(workspace))
        for server in config.mcp_servers:
            mcp_servers[server.name] = {
                "command": server.command,
                "args": list(server.args),
                "env": _resolve_env(server.env),
            }
            extra_allowed.append(f"mcp__{server.name}__*")
    except Exception as e:
        # A bad mcp_servers block shouldn't kill the tick — log and run
        # with just the built-in server.
        _log.warning("Could not load external MCP servers: %s", e)

    options = ClaudeAgentOptions(
        system_prompt=prompt.system_prompt(),
        max_turns=max_turns,
        mcp_servers=mcp_servers,
        allowed_tools=[
            "mcp__hyperherd__read_state",
            "mcp__hyperherd__read_plan",
            "mcp__hyperherd__write_plan",
            "mcp__hyperherd__bump_mem",
            "mcp__hyperherd__bump_time",
            "mcp__hyperherd__run_indices",
            "mcp__hyperherd__stop_index",
            "mcp__hyperherd__stop_all",
            "mcp__hyperherd__prune_index",
            "mcp__hyperherd__validate_config",
            "mcp__hyperherd__tail_log",
            "mcp__hyperherd__list_metrics",
            "mcp__hyperherd__compute_metric",
            "mcp__hyperherd__summarize_metrics",
            "mcp__hyperherd__msg",
            "mcp__hyperherd__tick_summary",
            "mcp__hyperherd__schedule_next",
            "mcp__hyperherd__halt",
            *extra_allowed,
        ],
        permission_mode="acceptEdits",
    )

    user_msg = prompt.render_state(s)

    cost_usd = 0.0
    turns = 0
    last_error_text: Optional[str] = None
    try:
        async for message in query(prompt=user_msg, options=options):
            # The SDK yields typed message objects. We use duck-typing because
            # the exact class names vary across SDK minor versions and the only
            # thing we rely on is the `type`/role/cost fields.
            cost_usd += float(getattr(message, "total_cost_usd", 0.0) or 0.0)
            if getattr(message, "role", None) == "assistant":
                turns += 1
            # ResultMessage carries is_error/result. The SDK's transport layer
            # raises a generic "Command failed with exit code 1" when the CLI
            # exits non-zero (e.g. billing errors), without surfacing the
            # actual reason — capture it here so we can re-raise with context.
            if getattr(message, "is_error", False):
                last_error_text = getattr(message, "result", None) or str(message)
    except Exception as e:
        if last_error_text:
            raise RuntimeError(f"Agent SDK call failed: {last_error_text}") from e
        raise

    # Read the next-tick file the schedule_next/halt tool wrote.
    return _resolve_outcome(next_tick_path, cost_usd=cost_usd, turns=turns)


def _resolve_outcome(next_tick_path: Path, *, cost_usd: float, turns: int) -> TickResult:
    if not next_tick_path.is_file():
        # Agent forgot to call schedule_next — pick a defensive default.
        return TickResult(
            next_delay_seconds=1800,    # 30 min, the steady-state cadence
            halted=False, halt_reason=None,
            cost_usd=cost_usd, turns=turns,
        )
    try:
        data = json.loads(next_tick_path.read_text())
    except (OSError, json.JSONDecodeError):
        return TickResult(next_delay_seconds=1800, halted=False, halt_reason=None,
                          cost_usd=cost_usd, turns=turns)
    if data.get("halted"):
        # Coerce reason to str — older buggy tool calls could leave a
        # nested dict here, and downstream formatting expects a string.
        reason = data.get("reason")
        if reason is not None and not isinstance(reason, str):
            reason = str(reason)
        return TickResult(next_delay_seconds=None, halted=True,
                          halt_reason=reason,
                          cost_usd=cost_usd, turns=turns)
    return TickResult(
        next_delay_seconds=int(data.get("delay_seconds", 1800)),
        halted=False, halt_reason=None,
        cost_usd=cost_usd, turns=turns,
    )
