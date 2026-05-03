"""Discord transport for the monitor daemon.

Connects to Discord as a bot, finds or creates a text channel for the
sweep inside the configured guild, and uses it as both the agent's
notification surface and the inbox for user replies.

Three message paths in the channel:

1. **Slash commands** (`/status`, `/stop`, `/tail`, `/help`, …) — handled
   locally via `monitor_agent.commands` without invoking the agent.
   Discord's UI provides typed parameter prompts.
2. **Mentions / replies** (`@HerdDog ...` or replying to a bot message)
   — stripped of the mention and routed to the daemon's inbox handler,
   which wakes the loop so the agent can respond on the next tick.
3. **Plain channel messages** — ignored. The channel is shared; people
   can chat without summoning the agent.

Setup steps the user does once per Discord server:

1. Create an application + bot in the Discord Developer Portal.
2. Enable the **MESSAGE CONTENT** privileged gateway intent on the bot.
3. Generate an invite URL with scopes `bot` + `applications.commands`
   and the permissions `View Channels`, `Send Messages`,
   `Read Message History`, `Manage Channels`.
4. Invite the bot to their server.
5. Copy the bot token → `DISCORD_BOT_TOKEN` env var.
6. Right-click the server name (Developer Mode on) → Copy Server ID →
   `discord.guild_id` in `hyperherd.yaml`.

Restart the daemon. It registers slash commands per-guild on first
connect (instant; global sync would take an hour to propagate).
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Optional

from hyperherd.monitor_agent import commands as cmd_mod
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


def strip_mention(text: str, bot_user_id: int) -> str:
    """Remove leading @-mentions of the bot from a message body. Discord
    serializes them as `<@USERID>` or `<@!USERID>`."""
    pattern = re.compile(rf"<@!?{bot_user_id}>")
    return pattern.sub("", text).strip()


def _strip_name_prefix(text: str, bot_name: str) -> Optional[str]:
    """Detect a plain-text address like `@HerdDog ...`, `HerdDog: ...`,
    or `HerdDog, ...` (case-insensitive) and return the body without the
    prefix. Returns None if no prefix is present.

    This catches users who type the name by hand instead of picking the
    bot from Discord's autocomplete dropdown — without it, the message
    would not register as a mention and would be silently dropped.
    """
    if not bot_name:
        return None
    s = text.lstrip()
    name_lower = bot_name.lower()
    s_lower = s.lower()

    for prefix in (f"@{name_lower}", name_lower):
        if s_lower.startswith(prefix):
            after = s[len(prefix):]
            # Require a separator so we don't match "HerdDoggy" as an
            # address to "HerdDog".
            if not after or after[0] in " :,\t\n":
                return after.lstrip(" :,\t\n").rstrip()
    return None


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
        workspace: Path,
        channel_id: Optional[int] = None,
        channel_name: Optional[str] = None,
    ):
        self._token = token
        self._guild_id = guild_id
        self._sweep_name = sweep_name
        self._workspace = Path(workspace)
        self._explicit_channel_id = channel_id
        self._explicit_channel_name = channel_name
        self._client = None  # type: ignore[assignment]
        self._tree = None
        self._client_task: Optional[asyncio.Task] = None
        self._channel = None
        self._on_inbound: Optional[InboundHandler] = None
        self._ready = asyncio.Event()

    def set_inbound_handler(self, handler: InboundHandler) -> None:
        self._on_inbound = handler

    async def start(self) -> None:
        try:
            import discord
            from discord import app_commands
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "discord.py not installed. Install the monitor extras: "
                "`pip install hyperherd[monitor]`."
            ) from e

        intents = discord.Intents.default()
        intents.message_content = True
        intents.guilds = True
        self._client = discord.Client(intents=intents)
        self._tree = app_commands.CommandTree(self._client)
        self._register_slash_commands(app_commands)

        @self._client.event
        async def on_ready():  # noqa: ARG001 — discord.py-required signature
            log.info("Discord connected as %s", self._client.user)
            try:
                await self._resolve_or_create_channel()
                # Per-guild sync is instant; global sync takes ~1h.
                await self._tree.sync(guild=discord.Object(id=self._guild_id))
                log.info("Slash commands synced to guild %s", self._guild_id)
                self._ready.set()
            except Exception as e:
                log.error("Discord setup failed: %s", e)
                await self._client.close()

        @self._client.event
        async def on_message(message):
            await self._handle_inbound_message(message)

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

    # --- inbound: only mentions/replies reach the agent ------------------

    async def _handle_inbound_message(self, message) -> None:
        if self._client is None or self._channel is None:
            return
        if message.author.id == self._client.user.id:
            return
        if message.channel.id != self._channel.id:
            return

        bot_user = self._client.user
        raw = message.content or ""

        # Three ways to address the bot, in priority order:
        #   1. Real Discord mention (autocomplete-resolved <@USERID>).
        #   2. A reply to one of the bot's messages.
        #   3. Plain-text "@<botname>" / "<botname>:" / "<botname>," prefix —
        #      a fallback for users who type the name manually instead of
        #      picking it from the autocomplete dropdown. Without this,
        #      "@HerdDog please pause" would silently drop on the floor.
        is_mention = bot_user in message.mentions
        is_reply_to_bot = (
            message.reference is not None
            and message.reference.resolved is not None
            and getattr(message.reference.resolved.author, "id", None)
                == bot_user.id
        )
        text_after_prefix = _strip_name_prefix(raw, bot_user.name)
        is_text_address = text_after_prefix is not None

        if not (is_mention or is_reply_to_bot or is_text_address):
            # Plain channel chatter — ignored on purpose.
            return

        if self._on_inbound is None:
            return

        if is_text_address:
            cleaned = text_after_prefix
        else:
            cleaned = strip_mention(raw, bot_user.id)
            # Real mentions can also coexist with text-prefix forms; strip
            # both so we never forward "@HerdDog HerdDog: please pause".
            stripped_again = _strip_name_prefix(cleaned, bot_user.name)
            if stripped_again is not None:
                cleaned = stripped_again

        if not cleaned.strip():
            return

        event = InboundEvent(
            timestamp=message.created_at.isoformat(),
            source="discord",
            author=str(message.author),
            text=cleaned,
        )
        try:
            await self._on_inbound(event)
        except Exception as e:
            log.warning("Inbound handler raised: %s", e)

    # --- slash commands --------------------------------------------------

    def _register_slash_commands(self, app_commands) -> None:
        """Wire `commands.py` handlers into Discord's CommandTree. The
        decorators run synchronously here at start(); the actual handlers
        are awaited when Discord delivers an Interaction."""
        import discord
        guild = discord.Object(id=self._guild_id)

        @self._tree.command(
            name="status",
            description="Show sweep totals and per-trial table",
            guild=guild,
        )
        async def status_cmd(interaction: discord.Interaction) -> None:
            await interaction.response.defer(thinking=True)
            text = await asyncio.get_running_loop().run_in_executor(
                None, cmd_mod.cmd_status, self._workspace,
            )
            await interaction.followup.send(_codeblock(text))

        @self._tree.command(
            name="stop", description="Cancel a single trial", guild=guild,
        )
        @app_commands.describe(index="Trial index to stop")
        async def stop_cmd(interaction: discord.Interaction, index: int) -> None:
            await interaction.response.defer(thinking=True)
            text = await asyncio.get_running_loop().run_in_executor(
                None, cmd_mod.cmd_stop, self._workspace, index,
            )
            await interaction.followup.send(text)

        @self._tree.command(
            name="stop_all", description="Cancel every live trial", guild=guild,
        )
        async def stop_all_cmd(interaction: discord.Interaction) -> None:
            await interaction.response.defer(thinking=True)
            text = await asyncio.get_running_loop().run_in_executor(
                None, cmd_mod.cmd_stop_all, self._workspace,
            )
            await interaction.followup.send(text)

        @self._tree.command(
            name="tail",
            description="Last N lines of a trial's stderr log",
            guild=guild,
        )
        @app_commands.describe(
            index="Trial index",
            lines="How many lines (default 20, max 1000)",
        )
        async def tail_cmd(
            interaction: discord.Interaction,
            index: int,
            lines: int = 20,
        ) -> None:
            await interaction.response.defer(thinking=True)
            text = await asyncio.get_running_loop().run_in_executor(
                None, cmd_mod.cmd_tail, self._workspace, index, lines,
            )
            await interaction.followup.send(_codeblock(text))

        @self._tree.command(
            name="help", description="List of HerdDog commands", guild=guild,
        )
        async def help_cmd(interaction: discord.Interaction) -> None:
            await interaction.response.send_message(cmd_mod.cmd_help())

    # --- channel resolution ----------------------------------------------

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


def _codeblock(text: str) -> str:
    """Wrap text in a Discord triple-backtick code block, truncating if
    it would exceed Discord's 2000-char message limit."""
    MAX_BODY = 1900  # leave room for fences + "(truncated)" line
    if len(text) > MAX_BODY:
        text = text[:MAX_BODY] + "\n... (truncated)"
    return f"```\n{text}\n```"
