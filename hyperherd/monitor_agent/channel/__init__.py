"""Transport-agnostic chat channel for the monitor daemon.

The daemon wants two things from whatever chat platform it's connected to:

1. **Outbound**: send a body of text. (`post`)
2. **Inbound**: receive user messages and route them into the per-tick inbox.

This module defines a `MessageChannel` Protocol that hides the concrete
transport (Discord today, Slack tomorrow), plus an `InboundEvent` dataclass
matching the shape `state.InboundMessage` expects.

Concrete implementations live in sibling modules (e.g. `discord_channel.py`).
A `build_channel(config, sweep_name)` factory picks one based on the
workspace config; if no channel is configured, it returns None and the
daemon runs without a chat surface (using the legacy webhook path for
outbound only).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import (
    AsyncContextManager, Awaitable, Callable, Optional, Protocol,
    runtime_checkable,
)

log = logging.getLogger(__name__)


@dataclass
class InboundEvent:
    """A user message received via a channel. The shape mirrors
    `state.InboundMessage` so the inbox-writer can serialize it directly."""

    timestamp: str       # ISO-8601 UTC
    source: str          # "discord", "slack", ...
    author: str          # display name or handle
    text: str


InboundHandler = Callable[[InboundEvent], Awaitable[None]]
StopHandler = Callable[[], None]
InfoHandler = Callable[[], dict]


@runtime_checkable
class MessageChannel(Protocol):
    """The contract the daemon depends on. Each transport implements this."""

    name: str

    async def start(self) -> None:
        """Connect, do any provisioning (e.g. find-or-create the target
        channel), and begin listening. Returns once ready to send/receive."""

    async def stop(self) -> None:
        """Disconnect cleanly. Idempotent."""

    async def post(self, body: str) -> None:
        """Send a message body to the configured outbound destination."""

    def set_inbound_handler(self, handler: InboundHandler) -> None:
        """Register a callback fired for each user message."""

    def set_stop_handler(self, handler: StopHandler) -> None:
        """Register a callback the channel can invoke to ask the daemon
        to shut down (e.g. from a `/stop` slash command). Optional —
        transports without a way to receive 'stop' commands can ignore."""

    def set_info_handler(self, handler: InfoHandler) -> None:
        """Register a callback the channel can invoke to fetch live
        daemon stats (tick count, total cost, uptime). The dict should
        match the kwargs of `commands.cmd_info`."""

    def thinking(self) -> AsyncContextManager:
        """Async context manager that signals 'agent is working' to the
        user (Discord typing indicator, Slack 'Bot is typing...', etc.).
        Default no-op for transports without an analogue."""


# --- inbox writer ----------------------------------------------------------

INBOX_FILENAME = "inbox.jsonl"


def make_inbox_writer(
    workspace: Path,
    *,
    on_write: Optional[Callable[[], None]] = None,
) -> InboundHandler:
    """Return an `InboundHandler` that appends each event as one JSONL line
    to `<workspace>/.hyperherd/inbox.jsonl`. The state assembler drains and
    truncates this file on every tick.

    `on_write` is an optional sync callback invoked after a successful
    append — the daemon uses it to set its wake-up event so a user message
    can interrupt the inter-tick sleep and trigger an immediate tick.
    """
    inbox_path = workspace / ".hyperherd" / INBOX_FILENAME
    inbox_path.parent.mkdir(parents=True, exist_ok=True)

    async def write(event: InboundEvent) -> None:
        line = json.dumps({
            "timestamp": event.timestamp,
            "source": event.source,
            "author": event.author,
            "text": event.text,
        })
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, _append_line, inbox_path, line)
        except OSError as e:
            log.warning("Failed to write inbox event from %s: %s",
                        event.author, e)
            return
        log.info(
            "Inbox: wrote message from %s (text head: %r)",
            event.author, event.text[:80],
        )
        if on_write is not None:
            try:
                on_write()
            except Exception as e:
                log.warning("inbox on_write callback raised: %s", e)

    return write


def _append_line(path: Path, line: str) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# --- factory ---------------------------------------------------------------

def build_channel(
    workspace_config,
    sweep_name: str,
    workspace: Path,
    *,
    force_token_conflict: bool = False,
) -> Optional[MessageChannel]:
    """Inspect the workspace config and return a configured channel, or
    None if no channel is set up (in which case the daemon falls back to
    the legacy webhook path for outbound and has no inbound surface).

    Discord requires `DISCORD_BOT_TOKEN` in the environment plus a
    `discord:` section in `hyperherd.yaml` with `guild_id`. If the env var
    is missing we treat it as 'not configured' and log a hint.

    `force_token_conflict=True` bypasses the same-token preflight — use
    when a previous daemon was killed uncleanly and its stale heartbeat
    hasn't aged out yet.
    """
    discord_cfg = getattr(workspace_config, "discord", None)
    if discord_cfg is None or getattr(discord_cfg, "guild_id", None) is None:
        return None

    import os
    token = os.environ.get("DISCORD_BOT_TOKEN")
    if not token:
        log.warning(
            "discord.guild_id is set in hyperherd.yaml but "
            "DISCORD_BOT_TOKEN is not in the environment — running "
            "without a Discord channel. Set the token to enable it."
        )
        return None

    from hyperherd.monitor_agent.channel.discord_channel import DiscordChannel
    return DiscordChannel(
        token=token,
        guild_id=int(discord_cfg.guild_id),
        sweep_name=sweep_name,
        workspace=workspace,
        channel_id=(
            int(discord_cfg.channel_id) if discord_cfg.channel_id else None
        ),
        channel_name=discord_cfg.channel_name,
        dashboard_refresh_seconds=int(discord_cfg.dashboard_refresh_seconds),
        force_token_conflict=force_token_conflict,
    )
