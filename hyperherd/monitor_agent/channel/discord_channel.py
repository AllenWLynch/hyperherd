"""Discord transport for the monitor daemon.

Connects to Discord as a bot, finds or creates a text channel for the
sweep inside the configured guild, listens for messages there, and posts
the agent's outbound `msg` calls into the same channel.

Setup steps the user does once per Discord server:

1. Create an application + bot in the Discord Developer Portal.
2. Enable the **MESSAGE CONTENT** privileged gateway intent on the bot.
3. Generate an invite URL with scopes `bot` and the permissions
   `View Channels`, `Send Messages`, `Read Message History`,
   `Manage Channels` (the last is needed for auto-creation).
4. Invite the bot to their server.
5. Copy the bot token → `DISCORD_BOT_TOKEN` env var.
6. Right-click the server name in Discord (with Developer Mode on) →
   Copy Server ID → put it in `hyperherd.yaml` under
   `discord.guild_id`.

Restart the daemon. It will create a channel named after the sweep
(e.g. `mnist-sweep`) on first run, then reuse it on subsequent runs.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Optional

from hyperherd.monitor_agent.channel import (
    InboundEvent, InboundHandler, MessageChannel,
)

log = logging.getLogger(__name__)


def sweep_to_channel_name(sweep_name: str) -> str:
    """Discord text-channel names: lowercase, max 100 chars, only letters,
    digits, hyphens, and underscores. Map underscores in sweep names to
    hyphens for readability and strip anything else."""
    s = sweep_name.lower().strip()
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"[^a-z0-9\-]", "", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return (s or "hyperherd")[:100]


class DiscordChannel(MessageChannel):
    """`MessageChannel` over Discord's gateway via discord.py.

    The bot connects to the configured guild on `start()`, finds an
    existing text channel matching the sweep-derived name (or the
    explicit `channel_name`/`channel_id` overrides), creates one if
    none exists, and uses it for both inbound and outbound traffic.
    """

    name = "discord"

    def __init__(
        self,
        *,
        token: str,
        guild_id: int,
        sweep_name: str,
        channel_id: Optional[int] = None,
        channel_name: Optional[str] = None,
    ):
        self._token = token
        self._guild_id = guild_id
        self._sweep_name = sweep_name
        self._explicit_channel_id = channel_id
        self._explicit_channel_name = channel_name
        self._client = None  # type: ignore[assignment]
        self._client_task: Optional[asyncio.Task] = None
        self._channel = None
        self._on_inbound: Optional[InboundHandler] = None
        self._ready = asyncio.Event()

    def set_inbound_handler(self, handler: InboundHandler) -> None:
        self._on_inbound = handler

    async def start(self) -> None:
        try:
            import discord
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "discord.py not installed. Install the monitor extras: "
                "`pip install hyperherd[monitor]`."
            ) from e

        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        self._client = discord.Client(intents=intents)

        @self._client.event
        async def on_ready():  # noqa: ARG001 — discord.py-required signature
            log.info("Discord connected as %s", self._client.user)
            try:
                await self._resolve_or_create_channel()
                self._ready.set()
            except Exception as e:
                log.error("Failed to resolve Discord channel: %s", e)
                # Surface the failure by closing the client so the gather
                # in the daemon raises rather than hanging on _ready.
                await self._client.close()

        @self._client.event
        async def on_message(message):
            if self._client is None:
                return
            if message.author.id == self._client.user.id:
                return
            if self._channel is None or message.channel.id != self._channel.id:
                return
            if self._on_inbound is None:
                return
            event = InboundEvent(
                timestamp=message.created_at.isoformat(),
                source="discord",
                author=str(message.author),
                text=message.content or "",
            )
            try:
                await self._on_inbound(event)
            except Exception as e:
                log.warning("Inbound handler raised: %s", e)

        # Run discord.py's connect-and-poll loop as a background task. We
        # race the ready-event against the client task so that connection
        # failures (bad token, network) and post-connect failures (missing
        # permissions inside on_ready, which closes the client) both
        # surface here as exceptions instead of hanging on the wait.
        self._client_task = asyncio.create_task(
            self._client.start(self._token), name="discord-client"
        )
        ready_task = asyncio.create_task(self._ready.wait(), name="discord-ready")
        done, _ = await asyncio.wait(
            {ready_task, self._client_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if ready_task not in done:
            ready_task.cancel()
            exc = self._client_task.exception() if self._client_task.done() else None
            raise RuntimeError(
                f"Discord client exited before becoming ready: {exc}"
            )

    async def stop(self) -> None:
        if self._client is not None and not self._client.is_closed():
            await self._client.close()
        if self._client_task is not None:
            try:
                await self._client_task
            except (asyncio.CancelledError, Exception) as e:
                log.debug("Discord client task ended: %s", e)
        self._client = None
        self._channel = None

    async def post(self, body: str) -> None:
        if self._channel is None:
            log.warning("post() called before channel is ready; dropping message.")
            return
        try:
            await self._channel.send(body)
        except Exception as e:
            log.warning("Failed to post to Discord: %s", e)

    async def _resolve_or_create_channel(self) -> None:
        """Find the target channel inside the configured guild, creating
        it if necessary. Caches the resulting channel object on `self`."""
        if self._client is None:
            return
        guild = self._client.get_guild(self._guild_id)
        if guild is None:
            try:
                guild = await self._client.fetch_guild(self._guild_id)
            except Exception as e:
                raise RuntimeError(
                    f"Bot can't see guild {self._guild_id} — verify it has "
                    f"been invited to the server. ({e})"
                )

        # Explicit channel_id takes precedence over name-based resolution.
        if self._explicit_channel_id is not None:
            ch = guild.get_channel(self._explicit_channel_id)
            if ch is None:
                ch = await self._client.fetch_channel(self._explicit_channel_id)
            self._channel = ch
            log.info("Using configured Discord channel: %s", ch)
            return

        target_name = (
            self._explicit_channel_name
            or sweep_to_channel_name(self._sweep_name)
        )

        for ch in guild.text_channels:
            if ch.name == target_name:
                self._channel = ch
                log.info("Reusing existing Discord channel #%s", target_name)
                return

        # Channel didn't exist — create it. Requires Manage Channels.
        try:
            self._channel = await guild.create_text_channel(
                target_name,
                topic=f"HyperHerd sweep: {self._sweep_name}",
            )
            log.info("Created Discord channel #%s", target_name)
        except Exception as e:
            raise RuntimeError(
                f"Failed to create channel #{target_name}. The bot likely "
                f"lacks 'Manage Channels' permission. ({e})"
            )
