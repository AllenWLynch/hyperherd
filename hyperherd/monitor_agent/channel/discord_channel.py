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
import contextlib
import logging
import re
from pathlib import Path
from typing import AsyncContextManager, Optional

from hyperherd.monitor_agent import commands as cmd_mod
from hyperherd.monitor_agent.channel import (
    InboundEvent, InboundHandler, InfoHandler, MessageChannel, StopHandler,
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
        dashboard_refresh_seconds: int = 60,
    ):
        self._token = token
        self._guild_id = guild_id
        self._sweep_name = sweep_name
        self._workspace = Path(workspace)
        self._explicit_channel_id = channel_id
        self._explicit_channel_name = channel_name
        self._dashboard_refresh = int(dashboard_refresh_seconds)
        self._client = None  # type: ignore[assignment]
        self._tree = None
        self._client_task: Optional[asyncio.Task] = None
        self._dashboard_task: Optional[asyncio.Task] = None
        self._dashboard_msg = None
        self._channel = None
        self._on_inbound: Optional[InboundHandler] = None
        self._on_stop: Optional[StopHandler] = None
        self._on_info: Optional[InfoHandler] = None
        self._ready = asyncio.Event()

    def set_inbound_handler(self, handler: InboundHandler) -> None:
        self._on_inbound = handler

    def set_stop_handler(self, handler: StopHandler) -> None:
        self._on_stop = handler

    def set_info_handler(self, handler: InfoHandler) -> None:
        self._on_info = handler

    def thinking(self) -> AsyncContextManager:
        """Return an async context manager that shows Discord's typing
        indicator for the duration. Falls back to a no-op if the channel
        isn't connected yet."""
        if self._channel is None:
            return _noop_async_cm()
        # discord.py's `Messageable.typing()` returns an async CM that
        # auto-renews while the body runs.
        return self._channel.typing()

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

        # Channel is connected and ready. Start the live-dashboard task —
        # one self-editing message that takes the place of the user
        # repeatedly typing /status. Disabled with refresh_seconds=0.
        if self._dashboard_refresh > 0:
            # Unpin any stale dashboards from prior daemon runs first —
            # otherwise pinned messages accumulate one-per-restart and
            # the user has to manually clean them up.
            await self._unpin_stale_dashboards()
            self._dashboard_task = asyncio.create_task(
                self._dashboard_loop(), name="discord-dashboard",
            )

    async def stop(self) -> None:
        if self._dashboard_task is not None:
            self._dashboard_task.cancel()
            try:
                await self._dashboard_task
            except (asyncio.CancelledError, Exception):
                pass
            self._dashboard_task = None
        if self._client is not None and not self._client.is_closed():
            await self._client.close()
        if self._client_task is not None:
            try:
                await self._client_task
            except (asyncio.CancelledError, Exception) as e:
                log.debug("Discord client task ended: %s", e)
        self._client = None
        self._channel = None
        self._dashboard_msg = None

    async def post(self, body: str) -> None:
        if self._channel is None:
            log.warning("post() called before channel is ready; dropping message.")
            return
        try:
            await self._channel.send(body)
        except Exception as e:
            log.warning("Failed to post to Discord: %s", e)

    # --- live dashboard --------------------------------------------------

    async def _unpin_stale_dashboards(self) -> None:
        """Sweep the channel's pinned messages, unpin any that look like
        a leftover dashboard from a previous daemon run. Match by
        author=self + content prefix `📊 ` so we don't touch unrelated
        pins (user-pinned conversation, etc.). Best-effort."""
        if self._channel is None or self._client is None:
            return
        try:
            pins = await self._channel.pins()
        except Exception as e:
            log.debug("Couldn't fetch channel pins: %s", e)
            return
        for msg in pins:
            try:
                if msg.author.id != self._client.user.id:
                    continue
                if not (msg.content or "").startswith("📊"):
                    continue
                await msg.unpin(reason="HyperHerd: replacing stale dashboard")
                log.info("Unpinned stale dashboard message %s", msg.id)
            except Exception as e:
                log.debug("Couldn't unpin %s: %s", getattr(msg, "id", "?"), e)

    def _build_dashboard_view(self):
        """A discord.ui.View carrying the Refresh button. timeout=None
        keeps it clickable for the daemon's whole lifetime (the view's
        in-memory state lives on this DiscordChannel instance, so the
        button stops working if the daemon restarts — the next dashboard
        post will install a fresh one)."""
        import discord

        class _DashboardView(discord.ui.View):
            def __init__(self_v, channel: "DiscordChannel") -> None:
                super().__init__(timeout=None)
                self_v._channel = channel

            @discord.ui.button(
                label="Refresh",
                emoji="🔄",
                style=discord.ButtonStyle.secondary,
                custom_id="hyperherd_dashboard_refresh",
            )
            async def refresh(  # noqa: ARG002 — discord.py signature
                self_v,
                interaction: "discord.Interaction",
                button: "discord.ui.Button",
            ) -> None:
                try:
                    content = self_v._channel._build_dashboard_content()
                except Exception as e:
                    await interaction.response.send_message(
                        f"⚠️ Refresh failed: {type(e).__name__}: {e}",
                        ephemeral=True,
                    )
                    return
                await interaction.response.edit_message(
                    content=content, view=self_v
                )

        return _DashboardView(self)

    async def _dashboard_loop(self) -> None:
        """Maintain a single self-editing 'live status' message in the
        channel so users don't have to keep typing `/status`. Best-effort
        on every front: a single failure to edit / pin / build never
        kills the loop, since the rest of the daemon is more important."""
        # Initial post + best-effort pin.
        view = self._build_dashboard_view()
        try:
            self._dashboard_msg = await self._channel.send(
                "📊 _loading dashboard…_", view=view,
            )
        except Exception as e:
            log.warning("Could not post initial dashboard: %s", e)
            return
        try:
            await self._dashboard_msg.pin(reason="HyperHerd live dashboard")
        except Exception as e:
            # Most likely Forbidden — bot lacks Manage Messages. The
            # dashboard still works unpinned; users can pin manually.
            log.info(
                "Couldn't pin the dashboard (%s); it'll still update in "
                "place. Add the 'Manage Messages' permission to the bot "
                "if you want it pinned automatically.",
                type(e).__name__,
            )

        while True:
            try:
                await asyncio.sleep(self._dashboard_refresh)
            except asyncio.CancelledError:
                return
            try:
                content = self._build_dashboard_content()
            except Exception as e:
                log.warning("Dashboard content build failed: %s", e)
                continue
            try:
                await self._dashboard_msg.edit(content=content, view=view)
            except Exception as e:
                # Could be NotFound (user deleted the message) or any
                # other transient glitch. Try to recreate; if THAT
                # fails, just keep going and try again next cycle.
                log.info("Dashboard edit failed (%s); reposting.",
                         type(e).__name__)
                try:
                    view = self._build_dashboard_view()
                    self._dashboard_msg = await self._channel.send(
                        content, view=view,
                    )
                    try:
                        await self._dashboard_msg.pin(
                            reason="HyperHerd live dashboard"
                        )
                    except Exception:
                        pass
                except Exception as e2:
                    log.warning("Dashboard repost also failed: %s", e2)

    def _build_dashboard_content(self) -> str:
        """Vertical-layout dashboard, optimized for Discord's pinned-
        message side panel (skinny, can't render wide tables). Each
        trial is one line with an emoji + index + experiment name.

        Reads `.hyperherd/last-snapshot.json` directly rather than
        going through `cmd_status` (which formats a wide table)."""
        import json as _json
        from datetime import datetime, timezone

        snap_path = self._workspace / ".hyperherd" / "last-snapshot.json"
        if not snap_path.is_file():
            return f"📊 **{self._sweep_name}** · _no snapshot yet_"
        try:
            snap = _json.loads(snap_path.read_text())
        except (OSError, _json.JSONDecodeError):
            return f"📊 **{self._sweep_name}** · _snapshot unreadable_"

        totals = snap.get("totals") or {}
        trials = snap.get("trials") or []
        info_kwargs = self._on_info() if self._on_info is not None else {}
        ticks = info_kwargs.get("ticks", 0)
        cost = info_kwargs.get("total_cost_usd", 0.0)
        started_iso = info_kwargs.get("started_at_iso", "")

        # Phase from the agent's plan, if any.
        plan_path = self._workspace / ".hyperherd" / "MONITOR_PLAN.md"
        phase = "?"
        if plan_path.is_file():
            try:
                for line in plan_path.read_text().splitlines():
                    s = line.strip().lstrip("-").strip()
                    if s.lower().startswith("phase:"):
                        phase = s.split(":", 1)[1].strip()
                        break
            except OSError:
                pass

        now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        order = ["ready", "submitted", "queued", "running",
                 "completed", "failed", "pruned", "cancelled"]
        # Colored-circle scheme for at-a-glance scanning in Discord's
        # pinned panel — single-codepoint emojis (no variation selectors)
        # render consistently across web/desktop/mobile clients.
        emoji = {
            "ready": "⚪", "submitted": "🔵", "queued": "🟡",
            "running": "🟢", "completed": "✅", "failed": "🔴",
            "pruned": "🟣", "cancelled": "⚫",
        }
        health_emoji = {
            "running": "🟢", "degraded": "🟡",
            "halted": "🔴", "stopping": "⚫",
            "passive": "⚪",
        }
        total = totals.get("total", len(trials))

        lines = []
        lines.append(f"📊 **{self._sweep_name}** · {now}")
        lines.append("")
        lines.append(f"**Status** ({total} trials)")
        for k in order:
            v = totals.get(k, 0)
            if v:
                lines.append(f"{emoji.get(k, '·')} {v} {k}")

        lines.append("")
        lines.append("**Daemon**")
        health = info_kwargs.get("health", "running")
        consec = info_kwargs.get("consecutive_failures", 0)
        health_label = health
        if health == "degraded" and consec:
            health_label = f"degraded ({consec} fail)"
        lines.append(f"{health_emoji.get(health, '·')} `{health_label}`")
        lines.append(f"phase · `{phase}`")
        # In passive mode the agent never runs, so ticks/cost are always 0.
        # Hide them to avoid implying the daemon is doing more than it is.
        is_passive = (health == "passive")
        if not is_passive:
            lines.append(f"ticks · {ticks}")
            lines.append(f"cost · ${cost:.4f}")
        next_tick_iso = info_kwargs.get("next_tick_at_iso")
        if next_tick_iso:
            try:
                nt = datetime.fromisoformat(next_tick_iso)
                if nt.tzinfo is None:
                    nt = nt.replace(tzinfo=timezone.utc)
                remaining = int((nt - datetime.now(timezone.utc)).total_seconds())
                label = "next snapshot" if is_passive else "next tick"
                if remaining > 0:
                    lines.append(f"{label} · in {_format_uptime(remaining)}")
                else:
                    lines.append(f"{label} · _waking_")
            except Exception:
                pass
        if started_iso:
            try:
                started_dt = datetime.fromisoformat(started_iso)
                if started_dt.tzinfo is None:
                    started_dt = started_dt.replace(tzinfo=timezone.utc)
                up_secs = int(
                    (datetime.now(timezone.utc) - started_dt).total_seconds()
                )
                lines.append(f"uptime · {_format_uptime(up_secs)}")
            except Exception:
                pass

        if trials:
            lines.append("")
            lines.append("**Trials**")
            # Active-first so the dashboard's CAP loses boring trials
            # (completed/ready) when there are many, not the running
            # ones the user is actually watching.
            sorted_trials = sorted(trials, key=cmd_mod.trial_sort_key)
            CAP = 25  # keep room for the rest of the message body
            for t in sorted_trials[:CAP]:
                idx = t.get("index", "?")
                status = t.get("status", "?")
                ic = emoji.get(status, "·")
                name = (t.get("experiment_name") or "?")[:24]
                tail = ""
                if status == "running":
                    el = t.get("elapsed") or ""
                    tail = f" · {el}" if el else ""
                elif status == "completed":
                    last = (t.get("last_log_line") or "").strip()[:24]
                    tail = f" · {last}" if last else ""
                elif status == "failed":
                    last = (t.get("last_log_line") or "").strip()[:24]
                    tail = f" · {last}" if last else ""
                lines.append(f"{ic} #{idx} `{name}`{tail}")
            if len(sorted_trials) > CAP:
                lines.append(
                    f"_… and {len(sorted_trials) - CAP} more "
                    f"(use `/running` or `/status`)_"
                )

        body = "\n".join(lines)
        if len(body) > 1990:
            body = body[:1980] + "\n_(truncated)_"
        return body

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

        # Try every name we can think of: the bot's global username and
        # any per-guild nickname. Either may be what the user typed.
        candidate_names = [bot_user.name]
        if self._channel is not None:
            try:
                me_in_guild = self._channel.guild.me
                if me_in_guild and me_in_guild.display_name:
                    candidate_names.append(me_in_guild.display_name)
            except Exception:
                pass

        # Three ways to address the bot, in priority order:
        #   1. Real Discord mention (autocomplete-resolved <@USERID>).
        #   2. A reply to one of the bot's messages.
        #   3. Plain-text "@<botname>" / "<botname>:" / "<botname>," prefix.
        is_mention = bot_user in message.mentions
        is_reply_to_bot = (
            message.reference is not None
            and message.reference.resolved is not None
            and getattr(message.reference.resolved.author, "id", None)
                == bot_user.id
        )
        text_after_prefix = None
        for name in candidate_names:
            text_after_prefix = _strip_name_prefix(raw, name)
            if text_after_prefix is not None:
                break
        is_text_address = text_after_prefix is not None

        if not (is_mention or is_reply_to_bot or is_text_address):
            # Diagnostic: if a message arrives with empty content from a
            # human, that's almost always the MESSAGE CONTENT INTENT not
            # being enabled in the Developer Portal — Discord ships an
            # empty content for non-resolved-mention messages without it.
            if not raw and not is_mention:
                log.warning(
                    "Got a message with empty content from %s — likely "
                    "MESSAGE CONTENT INTENT is disabled in the Discord "
                    "Developer Portal. Enable it under Bot → Privileged "
                    "Gateway Intents and restart the daemon.",
                    message.author,
                )
            else:
                log.debug(
                    "Ignoring channel message from %s: not addressing the "
                    "bot. Names tried: %s. Raw content head: %r",
                    message.author, candidate_names, raw[:80],
                )
            return

        log.info(
            "Inbound from %s (mention=%s reply=%s text_prefix=%s)",
            message.author, is_mention, is_reply_to_bot, is_text_address,
        )

        # Immediate ack so the user knows we saw their message — even if
        # the agent is mid-tick and won't respond properly for a few
        # seconds. The 👀 reaction is the conventional "I see you, working
        # on it" signal in chat platforms.
        try:
            await message.add_reaction("👀")
        except Exception as e:
            log.debug("Couldn't add reaction (missing permission?): %s", e)

        if self._on_inbound is None:
            return

        if is_text_address:
            cleaned = text_after_prefix
        else:
            cleaned = strip_mention(raw, bot_user.id)
            # Real mentions can also coexist with text-prefix forms; strip
            # both so we never forward "@HerdDog HerdDog: please pause".
            for name in candidate_names:
                stripped_again = _strip_name_prefix(cleaned, name)
                if stripped_again is not None:
                    cleaned = stripped_again
                    break

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

    async def _in_bound_channel(self, interaction) -> bool:
        """True if the interaction was triggered in this daemon's
        sweep channel. On False, sends an ephemeral redirect message —
        caller should early-return without doing more work.

        Slash commands are guild-scoped (Discord doesn't support
        channel scoping), so without this gate `/status` typed in any
        channel of the guild would respond there but report data from
        whichever sweep this daemon happens to monitor — confusing
        when the user has multiple sweeps or just types in the wrong
        channel by accident.
        """
        if self._channel is None:
            return True  # not yet connected; let it through
        if interaction.channel_id == self._channel.id:
            return True
        await interaction.response.send_message(
            f"This command is bound to {self._channel.mention} "
            f"(sweep `{self._sweep_name}`) — please run it there.",
            ephemeral=True,
        )
        return False

    def _register_slash_commands(self, app_commands) -> None:
        """Wire `commands.py` handlers into Discord's CommandTree. The
        decorators run synchronously here at start(); the actual handlers
        are awaited when Discord delivers an Interaction."""
        import discord
        guild = discord.Object(id=self._guild_id)
        ws = self._workspace
        in_bound_channel = self._in_bound_channel  # capture for closures

        @self._tree.command(
            name="status",
            description="Sweep totals + per-trial table",
            guild=guild,
        )
        async def status_cmd(interaction: discord.Interaction) -> None:
            if not await in_bound_channel(interaction):
                return
            await interaction.response.defer(thinking=True)
            text = await asyncio.get_running_loop().run_in_executor(
                None, cmd_mod.cmd_status, ws,
            )
            await interaction.followup.send(_codeblock(text))

        @self._tree.command(
            name="running",
            description="Active trials only (running, queued, submitted)",
            guild=guild,
        )
        async def running_cmd(interaction: discord.Interaction) -> None:
            if not await in_bound_channel(interaction):
                return
            await interaction.response.defer(thinking=True)
            text = await asyncio.get_running_loop().run_in_executor(
                None, cmd_mod.cmd_running, ws,
            )
            await interaction.followup.send(_codeblock(text))

        @self._tree.command(
            name="stats",
            description="Per-trial timing and memory stats",
            guild=guild,
        )
        async def stats_cmd(interaction: discord.Interaction) -> None:
            if not await in_bound_channel(interaction):
                return
            await interaction.response.defer(thinking=True)
            text = await asyncio.get_running_loop().run_in_executor(
                None, cmd_mod.cmd_stats, ws,
            )
            await interaction.followup.send(_codeblock(text))

        @self._tree.command(
            name="params",
            description="Parameter grid: sweep config and all trial combos",
            guild=guild,
        )
        async def params_cmd(interaction: discord.Interaction) -> None:
            if not await in_bound_channel(interaction):
                return
            await interaction.response.defer(thinking=True)
            text = await asyncio.get_running_loop().run_in_executor(
                None, cmd_mod.cmd_params, ws,
            )
            await interaction.followup.send(_codeblock(text))

        @self._tree.command(
            name="run", description="Submit (or resubmit) one trial", guild=guild,
        )
        @app_commands.describe(index="Trial index to run")
        async def run_cmd(
            interaction: discord.Interaction, index: int,
        ) -> None:
            if not await in_bound_channel(interaction):
                return
            await interaction.response.defer(thinking=True)
            text = await asyncio.get_running_loop().run_in_executor(
                None, cmd_mod.cmd_run, ws, index,
            )
            await interaction.followup.send(text)

        @self._tree.command(
            name="run_all",
            description="Submit every ready trial",
            guild=guild,
        )
        async def run_all_cmd(interaction: discord.Interaction) -> None:
            if not await in_bound_channel(interaction):
                return
            await interaction.response.defer(thinking=True)
            text = await asyncio.get_running_loop().run_in_executor(
                None, cmd_mod.cmd_run_all, ws,
            )
            await interaction.followup.send(text)

        @self._tree.command(
            name="plan",
            description="Show the agent's MONITOR_PLAN.md",
            guild=guild,
        )
        async def plan_cmd(interaction: discord.Interaction) -> None:
            if not await in_bound_channel(interaction):
                return
            await interaction.response.defer(thinking=True)
            text = await asyncio.get_running_loop().run_in_executor(
                None, cmd_mod.cmd_plan, ws,
            )
            await interaction.followup.send(_codeblock(text))

        @self._tree.command(
            name="info",
            description="Daemon metadata: phase, uptime, tick count, total cost",
            guild=guild,
        )
        async def info_cmd(interaction: discord.Interaction) -> None:
            if not await in_bound_channel(interaction):
                return
            await interaction.response.defer(thinking=True)
            stats = self._on_info() if self._on_info is not None else {}

            def _build():
                return cmd_mod.cmd_info(ws, **stats)

            text = await asyncio.get_running_loop().run_in_executor(None, _build)
            await interaction.followup.send(_codeblock(text))

        @self._tree.command(
            name="cancel", description="Cancel one trial", guild=guild,
        )
        @app_commands.describe(index="Trial index to cancel")
        async def cancel_cmd(
            interaction: discord.Interaction, index: int,
        ) -> None:
            if not await in_bound_channel(interaction):
                return
            await interaction.response.defer(thinking=True)
            text = await asyncio.get_running_loop().run_in_executor(
                None, cmd_mod.cmd_stop, ws, index,
            )
            await interaction.followup.send(text)

        @self._tree.command(
            name="cancel_all",
            description="Cancel every live trial",
            guild=guild,
        )
        async def cancel_all_cmd(interaction: discord.Interaction) -> None:
            if not await in_bound_channel(interaction):
                return
            await interaction.response.defer(thinking=True)
            text = await asyncio.get_running_loop().run_in_executor(
                None, cmd_mod.cmd_stop_all, ws,
            )
            await interaction.followup.send(text)

        @self._tree.command(
            name="prune",
            description="Algorithmic kill — NOT resubmitted by `herd run`",
            guild=guild,
        )
        @app_commands.describe(
            index="Trial index to prune",
            reason="Reason for pruning (optional, recorded in audit log)",
        )
        async def prune_cmd(
            interaction: discord.Interaction,
            index: int,
            reason: str = "user-pruned via /prune",
        ) -> None:
            if not await in_bound_channel(interaction):
                return
            await interaction.response.defer(thinking=True)
            text = await asyncio.get_running_loop().run_in_executor(
                None, cmd_mod.cmd_prune, ws, index, reason,
            )
            await interaction.followup.send(text)

        @self._tree.command(
            name="metrics",
            description="Cross-trial metric summary; optional smoothing window",
            guild=guild,
        )
        @app_commands.describe(
            smooth="If > 0, mean of last N points instead of last value",
        )
        async def metrics_cmd(
            interaction: discord.Interaction,
            smooth: int = 0,
        ) -> None:
            if not await in_bound_channel(interaction):
                return
            await interaction.response.defer(thinking=True)
            text = await asyncio.get_running_loop().run_in_executor(
                None, cmd_mod.cmd_metrics, ws, smooth,
            )
            await interaction.followup.send(_codeblock(text))

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
            if not await in_bound_channel(interaction):
                return
            await interaction.response.defer(thinking=True)
            text = await asyncio.get_running_loop().run_in_executor(
                None, cmd_mod.cmd_tail, ws, index, lines,
            )
            await interaction.followup.send(_codeblock(text))

        @self._tree.command(
            name="stop",
            description="Stop the monitor daemon entirely",
            guild=guild,
        )
        async def stop_cmd(interaction: discord.Interaction) -> None:
            if not await in_bound_channel(interaction):
                return
            if self._on_stop is None:
                await interaction.response.send_message(
                    "Not connected to a daemon — nothing to stop.",
                )
                return
            await interaction.response.send_message(
                "Stopping daemon. Final summary will follow.",
            )
            try:
                self._on_stop()
            except Exception as e:
                log.warning("Stop handler raised: %s", e)

        @self._tree.command(
            name="help", description="List of HerdDog commands", guild=guild,
        )
        async def help_cmd(interaction: discord.Interaction) -> None:
            if not await in_bound_channel(interaction):
                return
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


@contextlib.asynccontextmanager
async def _noop_async_cm():
    yield


def _format_uptime(seconds: int) -> str:
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        m, s = divmod(seconds, 60)
        return f"{m}m {s}s"
    h, rem = divmod(seconds, 3600)
    m, _ = divmod(rem, 60)
    return f"{h}h {m}m"
