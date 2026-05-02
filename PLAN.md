# Agent-SDK Monitor: Architecture & Implementation Plan

**Branch:** `agent-sdk-monitor`. Status: design only — no code yet.

## What this replaces

The current `herd monitor` runs an interactive Claude Code session in the user's terminal, self-pacing via dynamic `/loop` + `ScheduleWakeup`. It works, but:

- Can't accept external events (failures from `herd watch`, replies from Slack/Discord) — the parked session has no entrypoint.
- Inherits Claude Code's full general-purpose toolset and permission machinery, which is broader than this loop needs and a constant source of friction (off-list bash commands, jq quoting, `Unhandled node type: string`, `--slurpfile` blocks).
- Requires the user to keep a tmux session alive on the login node.

The new design replaces it with a **Python daemon** built on the **Claude Agent SDK**. The daemon owns the loop, exposes ~10 typed tools as the agent's full action surface, and converts three external sources (scheduled tick, SLURM failures, Discord replies) into agent invocations.

## Goals

1. **Event-driven.** A failure detected by SLURM polling, or a reply from the user in Discord, can trigger an agent tick within seconds — not at the next scheduled cadence.
2. **Bounded blast radius.** The agent's action surface is exactly the tools we register — no Bash, no Edit, no Read. Permission concerns evaporate.
3. **wandb-aware.** Agent can fetch live training metrics (val_acc, val_loss, NaN/inf detection) from wandb via MCP, without us hand-rolling the SDK calls.
4. **Headless.** Daemon runs as a service (systemd / nohup / tmux). No interactive Claude Code session required.
5. **Migrate, don't replace.** The current `herd monitor` stays as the easy on-ramp. The daemon mode is opt-in via a new entrypoint.

## Non-goals (out of scope for v1)

- Multi-sweep coordination from one daemon (one daemon per workspace, like watch).
- Slack support (Discord first; Slack is a parallel adapter we add later if there's demand).
- A web UI / dashboard.
- Replacing `herd watch`'s webhook-posting machinery (the daemon subsumes its polling, but the post-to-webhook path is reused as-is).

---

## Architecture

```
                ┌────────────────────────────────────────────────────┐
                │                    daemon                          │
                │                                                    │
   SLURM ──────►│  poll loop      ─┐                                 │
                │                  ├──► event queue ──► tick worker ─┼──► Claude Agent SDK ──► Anthropic API
   Discord ────►│  websocket gw   ─┘         ▲                       │           │
                │                            │                       │           │
                │  scheduler ────────────────┘                       │       tool calls
                │                                                    │           │
                └────────────────────────────────────────────────────┘           │
                                                                                  ▼
                                                    ┌────────────────────────────────┐
                                                    │ in-process tools (10)          │
                                                    │  + wandb-mcp-server (subproc)  │
                                                    └────────────────────────────────┘
                                                                  │
                                          herd run/stop/msg, edit YAML, write plan
                                                                  ▼
                                                            workspace state
```

Three event sources, one tick worker, one bounded tool surface.

## Module layout

```
hyperherd/
  monitor_agent/
    __init__.py
    daemon.py        # main(): assembles the components, runs the event loop
    events.py        # event types + queue
    sources/
      slurm_poll.py  # extracted from existing watch.py — emits FailureEvent / CompletionEvent
      discord.py     # discord bot using discord.py — emits UserMessageEvent
      schedule.py    # next-tick timer — emits ScheduledTickEvent
    tick.py          # one tick = build state → agent.run() → done (pure function)
    state.py         # deterministic state assembler (no LLM)
    tools.py         # the 10 typed tool functions
    prompt.py        # system prompt = adapted skill markdown; user-message renderer
    mcp.py           # MCP server registration (wandb-mcp-server config, optional custom servers)
```

`herd watch` keeps existing — the daemon's `slurm_poll.py` is an internal extraction, not a replacement of the user-facing CLI command.

---

## Tool surface (the entire blast radius)

Every action the agent can take. Each tool is an `async` Python function decorated with `@tool` from the SDK. Docstrings become tool descriptions visible to the model.

```python
@tool("read_state", "Return the current TickState snapshot — totals, per-trial status, recent failures, inbox.", {})
async def read_state() -> dict: ...

@tool("read_plan", "Return the current MONITOR_PLAN.md contents (empty string on first tick).", {})
async def read_plan() -> str: ...

@tool("write_plan", "Replace MONITOR_PLAN.md with the given Markdown.", {"plan": str})
async def write_plan(plan: str) -> None: ...

@tool("bump_mem", "Increase slurm.mem in hyperherd.yaml by `percent`%. Capped at one bump per failure class per sweep.", {"percent": int})
async def bump_mem(percent: int) -> dict: ...

@tool("bump_time", "Increase slurm.time by `percent`%. Same cap as bump_mem.", {"percent": int})
async def bump_time(percent: int) -> dict: ...

@tool("run_indices", "Submit (or resubmit with force=True) the given trial indices.", {"indices": list[int], "force": bool})
async def run_indices(indices: list[int], force: bool = False) -> dict: ...

@tool("stop_index", "Cancel one running/queued trial by index.", {"index": int})
async def stop_index(index: int) -> dict: ...

@tool("stop_all", "Cancel every running/queued trial in the workspace.", {})
async def stop_all() -> dict: ...

@tool("msg", "Post a notification to the configured webhook (Discord/ntfy/Slack). Sweep prefix added automatically.", {"text": str})
async def msg(text: str) -> None: ...

@tool("schedule_next", "Set the next scheduled tick delay in seconds. Required: every tick must call this exactly once before returning.", {"delay_seconds": int})
async def schedule_next(delay_seconds: int) -> None: ...

@tool("halt", "Stop the loop entirely (recurring code-bug exceptions, sweep complete, user said 'pause').", {"reason": str})
async def halt(reason: str) -> None: ...
```

Plus tools loaded via MCP (see "wandb integration" below):

- `mcp__wandb__query_wandb_tool` — run metadata + metric queries
- `mcp__wandb__get_run_history_tool` — per-step time-series (sampled)
- `mcp__wandb__scan_history` — exact NaN/inf detection (custom thin wrapper if the official sampling bites)

The agent's `allowed_tools` list is exactly the 11 in-process tools + the MCP-prefixed wandb tools. Anything else is silently unavailable.

---

## State assembler (`state.py`)

The deterministic part. Runs in Python before the agent is invoked. One function:

```python
@dataclass
class TickState:
    sweep_name: str
    workspace: Path
    plan: str                          # MONITOR_PLAN.md contents (or "" if first tick)
    totals: dict[str, int]             # status → count
    trials: list[TrialView]            # per-trial: index, status, params, slurm_state,
                                       # elapsed_seconds, max_rss_bytes, metrics, last_log_line,
                                       # wandb_run_id (if discoverable from experiment_name)
    newly_failed: list[FailureView]    # diff vs prev snapshot — incl. stderr_tail
    newly_completed: list[int]
    inbox: list[InboundMessage]        # Discord replies received since last tick
    trigger: Literal["scheduled", "failure", "completion", "user_message", "boot"]


def compute(workspace: Path, trigger: str) -> TickState:
    cur  = json.loads(subprocess.run(["herd", "snapshot"], ...).stdout)
    prev = _read_prev_snapshot(workspace)
    _rotate_snapshot(workspace, cur)
    return TickState(
        sweep_name=cur["sweep_name"],
        workspace=workspace,
        plan=_read_plan(workspace),
        totals=cur["totals"],
        trials=[_view(t, workspace) for t in cur["trials"]],
        newly_failed=_diff_failed(prev, cur, with_stderr=True),
        newly_completed=_diff_completed(prev, cur),
        inbox=_drain_inbox(workspace),  # reads .hyperherd/inbox.jsonl, truncates after
        trigger=trigger,
    )
```

The agent never calls `herd snapshot`, `cat`, `jq`, or any other plumbing — it gets the dict handed to it via `read_state()`.

---

## Discord integration (the receive path)

**Library:** `discord.py` (mature, ~14k★, supports gateway events natively).

**Setup:**

1. User creates a Discord application + bot in the Discord developer portal (one-time, ~5 min).
2. Bot is invited to the user's server with `Read Messages` and `Send Messages` permissions in one channel.
3. Bot token + channel ID go into `hyperherd.yaml`:

```yaml
monitor:
  discord:
    bot_token_env: DISCORD_BOT_TOKEN   # name of env var holding the token
    channel_id: 123456789012345678
    listen_to_user_id: 987654321098765432  # only react to messages from this user
```

**Daemon component (`sources/discord.py`):**

```python
async def run(config, event_q):
    intents = discord.Intents.default()
    intents.message_content = True
    bot = discord.Client(intents=intents)

    @bot.event
    async def on_message(msg):
        if msg.channel.id != config.channel_id: return
        if msg.author.id != config.listen_to_user_id: return
        await event_q.put(UserMessageEvent(text=msg.content, author=msg.author.name))

    await bot.start(os.environ[config.bot_token_env])
```

The daemon's outbound posts (the `msg` tool) ALSO go via Discord — same channel, same bot. Replaces the current ntfy/Slack webhook path for users on this code path. Ntfy/Slack remain available for users who don't want the bot setup.

**No public IP required.** `discord.py` opens an outbound websocket to Discord's gateway, so the bot works on a cluster login node behind any firewall. This is the key advantage over Slack (which can do the same via Socket Mode but is more annoying to set up).

---

## wandb integration

**Strategy:** install the official `wandb-mcp-server` ([github.com/wandb/wandb-mcp-server](https://github.com/wandb/wandb-mcp-server)) as a subprocess MCP server, register it with the agent, expose its tools under the `mcp__wandb__` prefix.

```python
# mcp.py
def build_wandb_mcp_config():
    return {
        "wandb": {
            "command": "uvx",
            "args": ["--from", "git+https://github.com/wandb/wandb-mcp-server",
                     "wandb_mcp_server"],
            "env": {"WANDB_API_KEY": os.environ["WANDB_API_KEY"]},
        },
    }
```

The agent's `allowed_tools` includes `mcp__wandb__query_wandb_tool` and `mcp__wandb__get_run_history_tool`.

**Wandb run identification.** The skill needs to know how to map a HyperHerd trial index to a wandb run. Two options:

1. **Trainer responsibility.** User's trainer initializes wandb with `name=os.environ["HYPERHERD_EXPERIMENT_NAME"]`. The agent searches wandb for runs named matching the experiment_name.
2. **Trainer reports run ID.** User's trainer calls `hyperherd.log_result("wandb_run_id", wandb.run.id)`. The agent reads the per-trial `metrics["wandb_run_id"]` from snapshot and queries that run directly.

Option 2 is more reliable. Document both; recommend (2) in the skill.

**Sampling caveat.** `wandb-mcp-server`'s `get_run_history_tool` uses sampled history (~500 points by default). For "did this go NaN at any step" questions, sampled data may miss the NaN step. **v1.5 enhancement (deferred):** ship a thin custom MCP server `hyperherd-wandb-mcp` that wraps `wandb.Api().run().scan_history(keys=[...])` for exact reads, ~150 LOC.

---

## Tick lifecycle (`tick.py`)

```python
SKILL_MD = (Path(__file__).parent / "prompt" / "monitor.md").read_text()

async def run_tick(workspace: Path, trigger: str) -> TickResult:
    state = state.compute(workspace, trigger)
    user_msg = prompt.render_state(state)

    options = ClaudeAgentOptions(
        system_prompt=SKILL_MD,
        max_turns=8,
        allowed_tools=[
            "read_state", "read_plan", "write_plan",
            "bump_mem", "bump_time",
            "run_indices", "stop_index", "stop_all",
            "msg", "schedule_next", "halt",
            "mcp__wandb__query_wandb_tool",
            "mcp__wandb__get_run_history_tool",
        ],
        mcp_servers={
            **build_wandb_mcp_config(),
            "hyperherd": create_sdk_mcp_server(name="hyperherd", tools=tools.ALL),
        },
        permission_mode="acceptEdits",  # tools we registered are pre-approved
    )

    cost_usd = 0.0
    next_delay = None
    halted = False
    async for message in query(prompt=user_msg, options=options):
        if isinstance(message, AssistantMessage):
            log_event(workspace, "assistant", message)
        elif isinstance(message, ToolUseMessage):
            log_event(workspace, "tool_use", message)  # for the audit log
            if message.tool_name == "schedule_next":
                next_delay = message.args["delay_seconds"]
            elif message.tool_name == "halt":
                halted = True
        elif isinstance(message, ResultMessage):
            cost_usd = message.total_cost_usd

    return TickResult(next_delay_seconds=next_delay, halted=halted, cost_usd=cost_usd)
```

**Per-tick cost target:** ≤$0.05 with prompt caching active (~5k-token system prompt + ~1-2k state message + ~2k of agent decision = mostly cache-hit territory after the first tick).

The daemon catches `next_delay is None` (agent forgot to call `schedule_next`) and falls back to a cadence-table default.

---

## Daemon (`daemon.py`)

```python
async def main(workspace: Path):
    config = load_monitor_config(workspace)
    event_q: asyncio.Queue = asyncio.Queue()

    # Three event sources fan in.
    sources = [
        sources.slurm_poll.run(workspace, event_q),     # poll sacct, emit FailureEvent/CompletionEvent
        sources.discord.run(config, event_q),           # gateway websocket, emit UserMessageEvent
        sources.schedule.run(workspace, event_q),       # next-tick timer
    ]
    asyncio.gather(*sources, return_exceptions=True)

    while True:
        event = await event_q.get()
        if isinstance(event, ShutdownEvent):
            break
        result = await tick.run_tick(workspace, trigger=event.trigger)
        if result.halted:
            log_info("Loop halted by agent.")
            break
        # The schedule source picks up next_delay_seconds via the workspace state file.
        # Failure/UserMessage events bypass the schedule entirely — the next event in
        # the queue gets processed immediately.
```

**Coalescing.** If three failures arrive in quick succession, we don't want three separate ticks. The event queue collapses adjacent events of the same type within a 30-second window into one tick whose `state.newly_failed` includes all three.

**Cost control.** Hard cap on ticks per hour (default 30). If exceeded, the daemon waits — the user gets a Discord post saying it's rate-limited.

---

## Skill markdown adaptation (`prompt/monitor.md`)

Port today's `hyperherd-monitor` skill with these edits:

**Keep:**
- Voice / "Herd dog:" prefix rule
- Phased rollout state machine (`Phase: not-started/canary-pending/...`)
- Failure triage table (host OOM / CUDA OOM / TIMEOUT / NODE_FAIL / recurring exception)
- NaN/inf-only kill rule, "no early stopping pretense"
- Cadence selection table
- Always-post-on-tick rule with next-tick countdown
- "Plan already exists is normal — read it, don't re-interview"

**Strip:**
- Approved tooling section (no Bash, no allowlist — moot)
- "Keep bash commands simple" (no Bash)
- jq recipes (no jq — `read_state()` returns a Python dict)
- Snapshot rotation recipe (state assembler does it)
- "Don't reach for python -c" (no shell)
- Anything mentioning `herd snapshot --json | jq` style invocations

**Add:**
- The 11 tool functions with one-line examples each
- "Discord inbox" section: how to interpret `state.inbox` (user replies — they may pause, change cadence, ask questions, change the plan)
- "wandb metric source" section: how to call `mcp__wandb__get_run_history_tool` for live val_acc, NaN/inf checks
- "Cost discipline" section: prefer `read_plan()` over re-asking the user, cache wandb queries by run_id+step, don't fetch metrics for every trial every tick

Final length target: ~3-4k tokens, half the current one.

---

## Phases of work (incremental milestones)

Each phase ends with something runnable and shippable.

### Phase 0 — Project setup (½ day)
- Add `claude-agent-sdk`, `discord.py`, `wandb` to `pyproject.toml` extras
- New `monitor_agent/` package skeleton
- `herd monitor-v2 [WORKSPACE]` CLI entrypoint, hidden behind `--experimental` flag

### Phase 1 — In-process tools, no MCP (1.5 days)
- Implement all 11 tools as plain async functions calling existing `herd` machinery (most of them already exist as Python functions in `hyperherd.cli`/`hyperherd.manifest`; just async-wrap)
- Implement `state.py` with deterministic snapshot/diff
- Single-tick mode: `herd monitor-v2 --once` runs one tick and exits
- Skill markdown ported and trimmed
- Manual test against the mnist example: agent reads state, posts a status `msg`, calls `schedule_next`, exits

### Phase 2 — Schedule loop (1 day)
- `sources/schedule.py` — timer that emits `ScheduledTickEvent`
- `daemon.py` event loop with one event source
- Persistence: next-delay survives daemon restart via `.hyperherd/next-tick.json`
- `herd monitor-v2 --daemon` runs forever, ticks on schedule

### Phase 3 — SLURM event source (1 day)
- Extract polling logic from `watch.py` into `sources/slurm_poll.py`
- Emit `FailureEvent` / `CompletionEvent` when sacct state transitions detected
- Coalesce events within a 30s window
- Ticks now fire within ~1 minute of a real failure (poll interval)

### Phase 4 — Discord receiver (1.5 days)
- `sources/discord.py` with `discord.py`
- Outbound Discord posts replace ntfy for users on this code path
- One-page docs section: Discord bot setup walkthrough
- End-to-end test: type a message in Discord, daemon ticks within 1s, agent processes the inbox

### Phase 5 — wandb MCP (1 day)
- `mcp.py` registers `wandb-mcp-server` as a subprocess
- Skill section explaining how to use the wandb tools
- Test: agent fetches val_acc for a running trial via `mcp__wandb__get_run_history_tool`

### Phase 6 — Hardening (2 days)
- Audit log of every tool call → `.hyperherd/agent_log.jsonl`
- Cost tracking → daily budget cap, Discord post when budget exhausted
- Reconnection logic for Discord gateway disconnects
- Systemd unit file + docs page
- Rate-limiting (≤30 ticks/hour default, configurable)

### Phase 7 — Migration & docs (1 day)
- `docs/monitor.md` updated with v2 setup
- `docs/dev/agent-sdk-architecture.md` (this doc, post-mortem-cleaned)
- The current `herd monitor` (Claude Code mode) stays as `herd monitor` — the new path is `herd monitor-v2` until v2 is stable, then they swap.

**Total estimate:** ~9 working days for a single dev. Can compress to ~6 if Phase 4 (Discord) and Phase 5 (wandb) overlap.

---

## Migration & coexistence with v1

- **`herd monitor`** keeps working as today (Claude Code in tmux + dynamic `/loop`). It's the easy on-ramp — no API key, no Discord setup.
- **`herd monitor-v2`** (the daemon) is opt-in. Requires `ANTHROPIC_API_KEY` env var and (for Discord) a bot token.
- **Same workspace, same `MONITOR_PLAN.md`.** A user can run v1 today, switch to v2 next week, and the agent picks up where it left off.
- **Same `herd watch` interop.** v2's `slurm_poll.py` is internal; users can still run `herd watch` separately if they want the per-trial webhook posts. (v2 will avoid double-posting by detecting watch's state file.)
- **Eventual consolidation.** Once v2 has been live for ~2 months and the API-key cost is well-characterized, fold v1 into v2 — v1 becomes a thin shim that spawns v2 with sensible defaults.

---

## Open questions / risks

1. **Cost.** Per-tick cost depends on prompt-caching effectiveness. We need a real measurement on a typical sweep (50 trials, 4-hour runs, ~30 ticks total) before committing to defaults. Plausible range $1–5 per sweep at current Sonnet pricing.
2. **wandb run-to-trial mapping.** Option 2 (`log_result("wandb_run_id", ...)`) is more robust but requires the user's trainer to do it. Document both paths; recommend the explicit one.
3. **Discord reply latency.** `discord.py`'s gateway connection can drop on flaky cluster networks. Plan exponential-backoff reconnect with a Discord post on prolonged disconnect.
4. **Token rotation.** What happens when `ANTHROPIC_API_KEY` is rotated mid-run? Daemon needs SIGHUP-equivalent to re-read env, or graceful crash + restart.
5. **MCP server lifecycle.** The wandb MCP runs as a subprocess; it can crash. Daemon needs to restart it on failure.
6. **Concurrent sweeps.** Each workspace gets its own daemon. If a user runs three concurrent sweeps, three daemons. We'll ship simple — no multi-tenancy in v1.
7. **The "set up a Discord bot" UX.** Two manual steps in a developer portal. Mitigation: a thorough docs page with screenshots, and a `herd discord-setup` helper that prints the exact invite URL once the user has a bot token.
8. **Sampling caveat for NaN detection.** May force us to ship our own thin wandb MCP earlier than Phase 5.

---

## What we know vs. what we'd verify before coding

**Confirmed:**
- `claude-agent-sdk` API surface (from claude-code-guide research)
- `wandb-mcp-server` exists and is vendor-maintained
- `discord.py` works on cluster login nodes (outbound websocket only)
- Most of the in-process tool implementations already exist as Python functions in `hyperherd.cli` / `hyperherd.manifest`

**To verify before Phase 1:**
- Exact `query()` vs `ClaudeSDKClient` choice for our tick shape (claude-code-guide leans `query()` per-tick)
- Prompt-caching hit-rate visibility in `ResultMessage` (so we can measure)
- Whether `ToolUseMessage` exposes async tool errors clean enough for our audit log

**To verify before Phase 4:**
- Discord developer portal still allows free-tier bots with message-content intent (it does as of 2026, but worth a one-line confirmation)

**To verify before Phase 5:**
- Whether `wandb-mcp-server`'s sampled history is good enough for our prompts in practice, or whether we need the custom `scan_history` wrapper from day one
