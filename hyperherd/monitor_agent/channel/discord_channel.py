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

One bot token per daemon: Discord enforces a single gateway connection
per token, so two daemons sharing a token will keep kicking each other
off. If you're running multiple sweeps in parallel, create one bot per
sweep workspace and put each token in its own `.env`. The startup
preflight (`_check_for_token_conflicts`) detects the same-token case
by reading per-channel heartbeat markers (written by `_heartbeat_loop`
into the channel's topic field) and refuses to start with an actionable
error. On clean shutdown the marker is cleared so a quick restart
isn't blocked. `--force-discord` bypasses the check after an unclean
kill.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import AsyncContextManager, Optional

from hyperherd.monitor_agent import commands as cmd_mod
from hyperherd.monitor_agent.channel import (
    InboundEvent, InboundHandler, InfoHandler, MessageChannel, StopHandler,
)

log = logging.getLogger(__name__)

# A single-token-per-bot signal stored in the channel's `topic` field.
# Each running daemon refreshes the timestamp inside its own channel's
# topic at HEARTBEAT_INTERVAL. Other daemons starting up later read
# every text channel's topic in the guild — if any *other* channel has
# a fresh heartbeat from this same bot identity, that's a same-token
# conflict and we refuse to start.
#
# Discord rate-limits channel.edit() at 2 changes per 10 minutes, so
# the heartbeat interval has to be ≥ 5 min. We use 6 min to leave
# margin for retries.
_HEARTBEAT_PREFIX = "[hyperherd-heartbeat: "
_HEARTBEAT_SUFFIX = "]"
_HEARTBEAT_INTERVAL = 360  # seconds — must satisfy Discord topic rate limit
_HEARTBEAT_RE = re.compile(
    re.escape(_HEARTBEAT_PREFIX) + r"([^\]]+)" + re.escape(_HEARTBEAT_SUFFIX)
)

# Cooldown between manual dashboard-refresh-button presses. Too fast and
# we'd be hammering sacct; too slow and the button feels broken. 20s
# matches the typical sacct cache window.
_MANUAL_REFRESH_COOLDOWN_S = 20


def _parse_heartbeat_topic(topic: str) -> Optional[datetime]:
    """Return the embedded heartbeat datetime if the topic carries a
    parseable HyperHerd marker, else None. Tolerates surrounding text
    so users can keep a human-readable channel description alongside."""
    if not topic:
        return None
    m = _HEARTBEAT_RE.search(topic)
    if not m:
        return None
    try:
        dt = datetime.fromisoformat(m.group(1).strip())
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _strip_heartbeat_marker(topic: str) -> str:
    """Remove any HyperHerd heartbeat marker from a topic string,
    leaving the user-facing description intact."""
    if not topic:
        return ""
    return _HEARTBEAT_RE.sub("", topic).rstrip()


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
        force_token_conflict: bool = False,
    ):
        self._token = token
        self._guild_id = guild_id
        self._sweep_name = sweep_name
        self._workspace = Path(workspace)
        self._explicit_channel_id = channel_id
        self._explicit_channel_name = channel_name
        self._dashboard_refresh = int(dashboard_refresh_seconds)
        self._force_token_conflict = bool(force_token_conflict)
        self._client = None  # type: ignore[assignment]
        self._tree = None
        self._client_task: Optional[asyncio.Task] = None
        self._dashboard_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._dashboard_msg = None
        self._channel = None
        # Monotonic timestamp of the last manual dashboard refresh, used
        # to rate-limit the Refresh button so a user spamming it can't
        # turn into a sacct-pounding loop.
        self._last_manual_refresh: float = 0.0
        # Per-trial Discord Thread objects, populated lazily on the
        # first failure post for that trial. Cleared on daemon
        # restart — recovering threads from message history is
        # possible but the agent's first thread post will just
        # synthesize a new parent message.
        self._threads_by_trial: dict = {}
        self._on_inbound: Optional[InboundHandler] = None
        self._on_stop: Optional[StopHandler] = None
        self._on_info: Optional[InfoHandler] = None
        self._ready = asyncio.Event()
        self._setup_error: Optional[BaseException] = None

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
                await self._check_for_token_conflicts()
                # Per-guild sync is instant; global sync takes ~1h.
                await self._tree.sync(guild=discord.Object(id=self._guild_id))
                log.info("Slash commands synced to guild %s", self._guild_id)
                self._ready.set()
            except Exception as e:
                log.error("Discord setup failed: %s", e)
                # Stash so start() can surface the real cause to the
                # daemon supervisor, instead of just "client exited".
                self._setup_error = e
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
            if self._setup_error is not None:
                raise self._setup_error
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

        # Heartbeat marker in our channel's topic — read by other
        # daemons' `_check_for_token_conflicts` to detect a same-token
        # conflict on their startup.
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(), name="discord-heartbeat",
        )

    async def stop(self) -> None:
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except (asyncio.CancelledError, Exception):
                pass
            self._heartbeat_task = None
        # Best-effort: erase our heartbeat marker so a quick restart
        # (or another daemon under the same token) doesn't trip the
        # conflict check on a freshly-stopped daemon.
        await self._clear_heartbeat_topic()
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

    async def _clear_heartbeat_topic(self) -> None:
        """Strip the heartbeat marker from our channel's topic on
        clean shutdown. Best-effort — failures degrade to "user must
        wait out the heartbeat staleness window before restarting"."""
        if self._channel is None:
            return
        try:
            current = getattr(self._channel, "topic", None) or ""
            stripped = _strip_heartbeat_marker(current)
            if stripped == current:
                return
            await self._channel.edit(
                topic=stripped or None,
                reason="HyperHerd heartbeat clear (clean shutdown)",
            )
        except Exception as e:
            log.debug("Couldn't clear heartbeat topic on shutdown: %s", e)

    async def post(self, body: str) -> None:
        if self._channel is None:
            log.warning("post() called before channel is ready; dropping message.")
            return
        try:
            await self._channel.send(body)
        except Exception as e:
            log.warning("Failed to post to Discord: %s", e)

    async def post_file(
        self, path, *, body: Optional[str] = None,
    ) -> None:
        """Upload one or more files to the channel. `path` is either a
        single Path or a list of Paths (multi-attachment in one message,
        which Discord renders as a stacked carousel). `body` is the
        optional accompanying message text rendered above the files."""
        if self._channel is None:
            log.warning(
                "post_file() called before channel is ready; dropping %s",
                path,
            )
            return
        try:
            import discord
        except ImportError:  # pragma: no cover
            return
        paths = path if isinstance(path, list) else [path]
        try:
            files = [discord.File(str(p)) for p in paths]
            if len(files) == 1:
                await self._channel.send(content=body, file=files[0])
            else:
                await self._channel.send(content=body, files=files)
        except Exception as e:
            log.warning("Failed to upload %s to Discord: %s", paths, e)

    async def post_to_trial_thread(
        self,
        trial_index: int,
        body: Optional[str] = None,
        *,
        file_path: Optional[Path] = None,
        thread_seed_text: Optional[str] = None,
    ) -> None:
        """Post into the per-trial thread, creating it on first call.

        Discord's `Message.create_thread()` is the only way to attach a
        thread to a *specific* message — a useful UX detail because it
        means clicking the failure post opens the discussion. If we
        don't have a stored anchor message yet (cold start, daemon
        restart), we post `thread_seed_text` to the channel and use
        that as the anchor."""
        if self._channel is None:
            log.warning(
                "post_to_trial_thread(%s) called before channel is "
                "ready; dropping.", trial_index,
            )
            return
        try:
            import discord
        except ImportError:  # pragma: no cover
            return

        thread = self._threads_by_trial.get(trial_index)
        if thread is None:
            seed = (
                thread_seed_text
                or f"Trial #{trial_index} — discussion thread"
            )
            try:
                anchor = await self._channel.send(seed)
                thread = await anchor.create_thread(
                    name=f"trial-{trial_index}",
                    auto_archive_duration=1440,  # 24h
                )
                self._threads_by_trial[trial_index] = thread
            except Exception as e:
                # Thread creation can fail if the bot lacks Manage
                # Threads. Fall back to plain channel post so the
                # message isn't lost.
                log.info(
                    "Couldn't create thread for trial %s (%s); falling "
                    "back to channel.", trial_index, type(e).__name__,
                )
                if file_path is not None:
                    await self.post_file(file_path, body=body)
                elif body is not None:
                    await self.post(body)
                return

        try:
            if file_path is not None:
                f = discord.File(str(file_path))
                await thread.send(content=body, file=f)
            elif body is not None:
                await thread.send(body)
        except Exception as e:
            log.warning(
                "Failed to post into trial-%s thread: %s",
                trial_index, e,
            )

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
                # Match either the old plain-text format (content starts
                # with 📊) or the new embed format (embed title starts
                # with 📊 and content is empty/loading).
                is_embed_dash = bool(
                    msg.embeds
                    and (msg.embeds[0].title or "").startswith("📊")
                )
                is_text_dash = (msg.content or "").startswith("📊")
                if not (is_embed_dash or is_text_dash):
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
                await self_v._channel._handle_refresh_click(
                    interaction, self_v
                )

        return _DashboardView(self)

    async def _handle_refresh_click(self, interaction, view) -> None:
        """Refresh-button click: re-poll SLURM (sacct/squeue) for a
        fresh snapshot, then re-render the dashboard against it.

        Rate-limited by `_MANUAL_REFRESH_COOLDOWN_S` — within the
        cooldown window we just re-render the existing snapshot
        (cheap) and tell the user how long to wait. Without the
        cooldown a user spamming the button could turn into a tight
        sacct loop on the controller.

        Defers the interaction first because `refresh_snapshot()`
        shells out to `herd snapshot`, which can take 1–2s — past
        Discord's 3s response deadline if sacct is slow.
        """
        import time
        from hyperherd.monitor_agent import state as state_mod

        now = time.monotonic()
        elapsed = now - self._last_manual_refresh
        cooled = elapsed >= _MANUAL_REFRESH_COOLDOWN_S
        await interaction.response.defer()
        if cooled:
            try:
                await asyncio.get_running_loop().run_in_executor(
                    None, state_mod.refresh_snapshot, self._workspace,
                )
            except Exception as e:
                # Refresh failed — fall back to whatever's on disk so
                # the user still gets a working dashboard.
                log.warning("Manual snapshot refresh failed: %s", e)
            else:
                self._last_manual_refresh = now
        try:
            embed = self._build_dashboard_embed()
        except Exception as e:
            await interaction.followup.send(
                f"⚠️ Refresh failed: {type(e).__name__}: {e}",
                ephemeral=True,
            )
            return
        await interaction.edit_original_response(
            content="", embed=embed, view=view,
        )
        if not cooled:
            wait = max(1, int(_MANUAL_REFRESH_COOLDOWN_S - elapsed))
            await interaction.followup.send(
                f"⏳ Re-polling SLURM is on cooldown — try again in "
                f"{wait}s. (Showing the cached snapshot.)",
                ephemeral=True,
            )

    async def _dashboard_loop(self) -> None:
        """Maintain a single self-editing 'live status' message in the
        channel so users don't have to keep typing `/status`. Best-effort
        on every front: a single failure to edit / pin / build never
        kills the loop, since the rest of the daemon is more important."""
        import discord as _discord

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
                embed = self._build_dashboard_embed()
            except Exception as e:
                log.warning("Dashboard content build failed: %s", e)
                continue
            try:
                await self._dashboard_msg.edit(content="", embed=embed, view=view)
            except Exception as e:
                # Could be NotFound (user deleted the message) or any
                # other transient glitch. Try to recreate; if THAT
                # fails, just keep going and try again next cycle.
                log.info("Dashboard edit failed (%s); reposting.",
                         type(e).__name__)
                try:
                    view = self._build_dashboard_view()
                    self._dashboard_msg = await self._channel.send(
                        embed=embed, view=view,
                    )
                    try:
                        await self._dashboard_msg.pin(
                            reason="HyperHerd live dashboard"
                        )
                    except Exception:
                        pass
                except Exception as e2:
                    log.warning("Dashboard repost also failed: %s", e2)

    def _build_dashboard_embed(self):
        """Build a `discord.Embed` dashboard, optimized for Discord's
        pinned-message panel. Uses a colored sidebar to signal sweep
        health at a glance, structured fields for status/daemon/trials."""
        import json as _json
        from datetime import datetime, timezone
        import discord as _discord

        snap_path = self._workspace / ".hyperherd" / "last-snapshot.json"
        if not snap_path.is_file():
            embed = _discord.Embed(
                title=f"📊 {self._sweep_name}",
                description="_No snapshot yet — waiting for first poll._",
                color=0x95a5a6,
            )
            return embed
        try:
            snap = _json.loads(snap_path.read_text())
        except (OSError, _json.JSONDecodeError):
            embed = _discord.Embed(
                title=f"📊 {self._sweep_name}",
                description="_Snapshot unreadable — check disk._",
                color=0xe74c3c,
            )
            return embed

        totals = snap.get("totals") or {}
        trials = snap.get("trials") or []
        info_kwargs = self._on_info() if self._on_info is not None else {}
        ticks = info_kwargs.get("ticks", 0)
        cost = info_kwargs.get("total_cost_usd", 0.0)
        started_iso = info_kwargs.get("started_at_iso", "")
        health = info_kwargs.get("health", "running")
        consec = info_kwargs.get("consecutive_failures", 0)
        is_passive = (health == "passive")

        # Sidebar color reflects sweep health.
        health_color = {
            "running": 0x2ecc71,   # green
            "degraded": 0xf1c40f,  # yellow
            "halted": 0xe74c3c,    # red
            "stopping": 0x34495e,  # dark
            "passive": 0x95a5a6,   # grey
        }.get(health, 0x2ecc71)

        # If any trials are failed, pull color toward warning.
        if totals.get("failed", 0) and health not in ("halted", "degraded"):
            health_color = 0xf39c12  # amber

        now = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
        embed = _discord.Embed(
            title=f"📊 {self._sweep_name}",
            color=health_color,
        )
        embed.set_footer(text=f"updated {now}")

        # --- Status field ---
        emoji = {
            "ready": "⚪", "submitted": "🔵", "queued": "🟡",
            "running": "🟢", "completed": "✅", "failed": "🔴",
            "pruned": "🟣", "cancelled": "⚫",
        }
        order = ["running", "queued", "submitted", "failed",
                 "completed", "pruned", "cancelled", "ready"]
        total = totals.get("total", len(trials))
        status_lines = []
        for k in order:
            v = totals.get(k, 0)
            if v:
                status_lines.append(f"{emoji.get(k, '·')} {v} {k}")
        embed.add_field(
            name=f"Status ({total} trials)",
            value="\n".join(status_lines) if status_lines else "_none_",
            inline=True,
        )

        # --- Daemon field ---
        health_emoji = {
            "running": "🟢", "degraded": "🟡",
            "halted": "🔴", "stopping": "⚫",
            "passive": "⚪",
        }
        health_label = health
        if health == "degraded" and consec:
            health_label = f"degraded ({consec} fail)"

        # Phase from agent plan.
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

        daemon_lines = [
            f"{health_emoji.get(health, '·')} `{health_label}`",
            f"phase · `{phase}`",
        ]
        if not is_passive:
            daemon_lines.append(f"ticks · {ticks}")
            daemon_lines.append(f"cost · ${cost:.4f}")
        next_tick_iso = info_kwargs.get("next_tick_at_iso")
        if next_tick_iso:
            try:
                nt = datetime.fromisoformat(next_tick_iso)
                if nt.tzinfo is None:
                    nt = nt.replace(tzinfo=timezone.utc)
                remaining = int(
                    (nt - datetime.now(timezone.utc)).total_seconds()
                )
                label = "next snapshot" if is_passive else "next tick"
                if remaining > 0:
                    daemon_lines.append(
                        f"{label} · in {_format_uptime(remaining)}"
                    )
                else:
                    daemon_lines.append(f"{label} · _waking_")
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
                daemon_lines.append(f"uptime · {_format_uptime(up_secs)}")
            except Exception:
                pass
        embed.add_field(
            name="Daemon",
            value="\n".join(daemon_lines),
            inline=True,
        )

        # --- Trials field ---
        if trials:
            sorted_trials = sorted(trials, key=cmd_mod.trial_sort_key)
            CAP = 20  # embed field value ≤ 1024 chars; each line ~30 chars
            trial_lines = []
            for t in sorted_trials[:CAP]:
                idx = t.get("index", "?")
                status = t.get("status", "?")
                ic = emoji.get(status, "·")
                name = (t.get("experiment_name") or "?")[:20]
                tail = ""
                if status == "running":
                    el = t.get("elapsed") or ""
                    tail = f" · {el}" if el else ""
                elif status in ("completed", "failed"):
                    last = (t.get("last_log_line") or "").strip()[:20]
                    tail = f" · {last}" if last else ""
                trial_lines.append(f"{ic} #{idx} `{name}`{tail}")
            if len(sorted_trials) > CAP:
                trial_lines.append(
                    f"_… +{len(sorted_trials) - CAP} more_"
                )
            embed.add_field(
                name="Trials",
                value="\n".join(trial_lines),
                inline=False,
            )

        return embed

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
            description="Refresh and move the live dashboard embed to the bottom of the chat",
            guild=guild,
        )
        async def status_cmd(interaction: discord.Interaction) -> None:
            if not await in_bound_channel(interaction):
                return
            await interaction.response.defer(ephemeral=True, thinking=True)
            from hyperherd.monitor_agent import state as state_mod
            try:
                await asyncio.get_running_loop().run_in_executor(
                    None, state_mod.refresh_snapshot, ws,
                )
            except Exception as e:
                log.warning("/status snapshot refresh failed: %s", e)
            try:
                embed = self._build_dashboard_embed()
            except Exception as e:
                await interaction.followup.send(
                    f"⚠️ Dashboard build failed: {e}", ephemeral=True,
                )
                return
            old_msg = self._dashboard_msg
            if old_msg is not None:
                try:
                    await old_msg.delete()
                except Exception:
                    pass
            view = self._build_dashboard_view()
            try:
                new_msg = await self._channel.send(embed=embed, view=view)
                self._dashboard_msg = new_msg
                try:
                    await new_msg.pin(reason="HyperHerd live dashboard")
                except Exception:
                    pass
            except Exception as e:
                await interaction.followup.send(
                    f"⚠️ Failed to repost dashboard: {e}", ephemeral=True,
                )
                return
            await interaction.followup.send("✅ Dashboard moved to bottom.", ephemeral=True)

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
            description="List logged metric names, or show one metric's value per trial",
            guild=guild,
        )
        @app_commands.describe(
            metric="Metric name to look up (omit to list all available metrics)",
            smooth="If > 0, mean of last N points instead of last value",
        )
        async def metrics_cmd(
            interaction: discord.Interaction,
            metric: str = "",
            smooth: int = 0,
        ) -> None:
            if not await in_bound_channel(interaction):
                return
            await interaction.response.defer(thinking=True)
            text = await asyncio.get_running_loop().run_in_executor(
                None, cmd_mod.cmd_metrics, ws,
                metric or None, smooth,
            )
            await interaction.followup.send(_codeblock(text))

        @self._tree.command(
            name="plot",
            description="Plot a metric across one or more trials as a PNG",
            guild=guild,
        )
        @app_commands.describe(
            metric="Metric name (e.g. train/loss). /metrics shows what's logged.",
            trials="Comma- or range-separated indices, e.g. '0,2,5' or '0-3' (default: all)",
            smooth="Rolling-mean window (default 0 = no smoothing)",
        )
        async def plot_cmd(
            interaction: discord.Interaction,
            metric: str,
            trials: str = "",
            smooth: int = 0,
        ) -> None:
            if not await in_bound_channel(interaction):
                return
            await interaction.response.defer(thinking=True)
            try:
                idxs = _parse_index_spec(trials) if trials.strip() else None
            except ValueError as e:
                await interaction.followup.send(
                    f"Bad trials spec {trials!r}: {e}", ephemeral=True,
                )
                return
            await self._render_and_post_plot(
                interaction, metric=metric, trial_indices=idxs,
                smooth=smooth,
            )

        @self._tree.command(
            name="results",
            description="Upload the results table (and per-step history) as files",
            guild=guild,
        )
        @app_commands.describe(
            with_steps=(
                "Also attach steps.tsv.gz — the long-form per-step "
                "history (default true). Set false for summary only."
            ),
        )
        async def results_cmd(
            interaction: discord.Interaction, with_steps: bool = True,
        ) -> None:
            if not await in_bound_channel(interaction):
                return
            await interaction.response.defer(thinking=True)
            try:
                built = await asyncio.get_running_loop().run_in_executor(
                    None, cmd_mod.build_results_table, self._workspace,
                )
            except Exception as e:
                await interaction.followup.send(
                    f"⚠️ Couldn't build results table: {type(e).__name__}: {e}",
                    ephemeral=True,
                )
                return
            if built is None:
                await interaction.followup.send(
                    "No trials in manifest yet — nothing to summarize.",
                    ephemeral=True,
                )
                return

            steps = None
            if with_steps:
                try:
                    steps = await asyncio.get_running_loop().run_in_executor(
                        None, cmd_mod.build_steps_blob, self._workspace,
                    )
                except Exception as e:
                    log.warning("Couldn't build steps blob: %s", e)

            DISCORD_FILE_LIMIT = 10 * 1024 * 1024
            steps_oversize = (
                steps is not None
                and len(steps["gz_bytes"]) > DISCORD_FILE_LIMIT
            )
            if steps_oversize:
                steps = None

            code_block = f"```\n{built['table_text']}\n```"
            INLINE_LIMIT = 1900
            content_parts: list = []
            if len(code_block) <= INLINE_LIMIT:
                content_parts.append(code_block)
            if steps_oversize:
                content_parts.append(
                    "(per-step history was over Discord's 10 MB cap — "
                    "pull it via `herd res --steps` on the host)"
                )
            content = "\n".join(content_parts) if content_parts else None

            import tempfile
            tsv_path = Path(tempfile.mkstemp(suffix=".tsv", prefix="results-")[1])
            steps_path: Optional[Path] = None
            try:
                tsv_path.write_text(built["tsv"])
                files = [discord.File(str(tsv_path), filename="results.tsv")]
                if steps is not None:
                    steps_path = Path(tempfile.mkstemp(
                        suffix=".tsv.gz", prefix="steps-",
                    )[1])
                    steps_path.write_bytes(steps["gz_bytes"])
                    files.append(discord.File(
                        str(steps_path), filename="steps.tsv.gz",
                    ))
                try:
                    if len(files) == 1:
                        await interaction.followup.send(content=content, file=files[0])
                    else:
                        await interaction.followup.send(content=content, files=files)
                except Exception as e:
                    await interaction.followup.send(
                        f"⚠️ Results upload failed: {e}", ephemeral=True,
                    )
            finally:
                for p in (tsv_path, steps_path):
                    if p is None:
                        continue
                    try:
                        p.unlink()
                    except OSError:
                        pass

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

    async def _check_for_token_conflicts(self) -> None:
        """Refuse to start if another daemon is already running with the
        same DISCORD_BOT_TOKEN.

        Discord allows only one gateway connection per bot token, so two
        daemons sharing a token boot each other off in a flap loop.
        Detect the case by reading every text channel's `topic` field
        and looking for a HyperHerd heartbeat marker (set by
        `_heartbeat_loop`) in any channel *other* than ours that's
        within `_HEARTBEAT_INTERVAL × 3` of now. Topic is already
        cached on the channel object — no extra API call.
        """
        if self._client is None or self._channel is None:
            return
        guild = self._channel.guild
        if guild is None:
            return
        cutoff = datetime.now(timezone.utc) - timedelta(
            seconds=_HEARTBEAT_INTERVAL * 3
        )

        conflicts: list[str] = []
        for ch in guild.text_channels:
            if ch.id == self._channel.id:
                continue
            ts = _parse_heartbeat_topic(getattr(ch, "topic", None) or "")
            if ts is None or ts < cutoff:
                continue
            conflicts.append(f"#{ch.name}")

        if not conflicts:
            return
        others = ", ".join(conflicts)
        if self._force_token_conflict:
            log.warning(
                "Skipping same-token conflict check (--force-discord set): "
                "fresh heartbeat present in %s. If the other daemon is still "
                "running, both will keep kicking each other off the gateway.",
                others,
            )
            return
        raise RuntimeError(
            f"Another HyperHerd daemon appears to be running with this "
            f"DISCORD_BOT_TOKEN — fresh heartbeat found in {others}. "
            f"Discord allows only one gateway connection per bot "
            f"token, so two daemons sharing a token will keep "
            f"kicking each other off.\n"
            f"\n"
            f"If you just stopped that daemon, either:\n"
            f"  1. Wait ~{(_HEARTBEAT_INTERVAL * 3) // 60} minutes for "
            f"the stale heartbeat to time out, or\n"
            f"  2. Pass `--force-discord` to bypass this check now.\n"
            f"\n"
            f"For the long-term fix, create a second bot in the Discord "
            f"Developer Portal "
            f"(https://discord.com/developers/applications), invite it "
            f"to your guild, and set DISCORD_BOT_TOKEN=<new_token> in "
            f"this workspace's .env."
        )

    async def _heartbeat_loop(self) -> None:
        """Refresh this daemon's heartbeat marker in our channel's
        topic on a fixed cadence so other daemons starting later can
        detect a same-token conflict via `_check_for_token_conflicts`.

        Best-effort: failures (rate-limit, missing Manage Channels,
        transient errors) are logged but never crash the daemon —
        losing the heartbeat just degrades gracefully to no protection
        for the next restart, which is no worse than not having the
        feature at all."""
        # Fire once immediately so a same-token daemon starting a few
        # seconds after us still sees a marker on its conflict scan.
        try:
            await self._update_heartbeat_topic()
        except Exception as e:
            log.debug("Initial heartbeat write failed: %s", e)
        while True:
            try:
                await asyncio.sleep(_HEARTBEAT_INTERVAL)
            except asyncio.CancelledError:
                return
            try:
                await self._update_heartbeat_topic()
            except Exception as e:
                log.debug("Heartbeat write failed: %s", e)

    async def _update_heartbeat_topic(self) -> None:
        """Stamp the current UTC timestamp into our channel's topic,
        preserving any user-facing description."""
        if self._channel is None:
            return
        now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        marker = f"{_HEARTBEAT_PREFIX}{now_iso}{_HEARTBEAT_SUFFIX}"
        current = getattr(self._channel, "topic", None) or ""
        base = _strip_heartbeat_marker(current)
        if not base:
            base = f"HyperHerd sweep: {self._sweep_name}"
        new_topic = f"{base} {marker}"
        if new_topic == current:
            return
        await self._channel.edit(topic=new_topic, reason="HyperHerd heartbeat")


    async def _render_and_post_plot(
        self,
        interaction,
        *,
        metric: str,
        trial_indices: Optional[list] = None,
        smooth: int = 0,
    ) -> None:
        """Render a metric plot in the executor (matplotlib is sync) and
        post it as a follow-up file. Errors -> ephemeral so they don't
        clutter the channel."""
        from hyperherd.monitor_agent import plots

        def _render():
            return plots.render_metric_plot(
                self._workspace, metric,
                trial_indices=trial_indices, smooth=smooth,
            )

        try:
            png_path = await asyncio.get_running_loop().run_in_executor(
                None, _render,
            )
        except plots.PlotUnavailable as e:
            await interaction.followup.send(str(e), ephemeral=True)
            return
        except Exception as e:
            await interaction.followup.send(
                f"⚠️ Plot failed: {type(e).__name__}: {e}", ephemeral=True,
            )
            return

        try:
            import discord
            f = discord.File(str(png_path))
            caption = f"`{metric}`"
            if trial_indices:
                caption += f" — trials {sorted(trial_indices)}"
            if smooth > 1:
                caption += f" (smoothed, window={smooth})"
            await interaction.followup.send(content=caption, file=f)
        except Exception as e:
            await interaction.followup.send(
                f"⚠️ Plot upload failed: {e}", ephemeral=True,
            )
        finally:
            try:
                Path(png_path).unlink()
            except OSError:
                pass


def _parse_index_spec(spec: str) -> list:
    """Parse '0,2,5' or '0-3' or '0-2,5,7-9' into a sorted unique list."""
    out: set = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


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
