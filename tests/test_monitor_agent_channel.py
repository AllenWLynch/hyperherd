"""Tests for the transport-agnostic channel layer.

These tests don't touch the network or discord.py — they exercise:
- The inbox writer round-trip
- Sweep-name → Discord channel-name normalization
- The daemon's early-wake on inbound events (via a fake channel)
"""

import asyncio
import json
import shutil
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from hyperherd.monitor_agent import daemon as daemon_mod
from hyperherd.monitor_agent.channel import (
    InboundEvent, MessageChannel, make_inbox_writer,
)
from hyperherd.monitor_agent.channel.discord_channel import (
    _strip_name_prefix, sweep_to_channel_name,
)
from hyperherd.monitor_agent.tick import TickResult


class TestStripNamePrefix(unittest.TestCase):
    """Plain-text address detection — for users who type @HerdDog instead
    of picking the bot from Discord's autocomplete (which would otherwise
    drop on the floor since Discord didn't resolve a real mention)."""

    def test_at_prefix_with_space(self):
        out = _strip_name_prefix("@HerdDog please pause", "HerdDog")
        self.assertEqual(out, "please pause")

    def test_at_prefix_case_insensitive(self):
        out = _strip_name_prefix("@herddog status?", "HerdDog")
        self.assertEqual(out, "status?")

    def test_bare_name_with_colon(self):
        out = _strip_name_prefix("HerdDog: bump mem to 16G", "HerdDog")
        self.assertEqual(out, "bump mem to 16G")

    def test_bare_name_with_comma(self):
        out = _strip_name_prefix("HerdDog, what's idx 3 doing?", "HerdDog")
        self.assertEqual(out, "what's idx 3 doing?")

    def test_leading_whitespace_tolerated(self):
        out = _strip_name_prefix("   @HerdDog hi", "HerdDog")
        self.assertEqual(out, "hi")

    def test_no_prefix_returns_none(self):
        self.assertIsNone(_strip_name_prefix("hi everyone", "HerdDog"))

    def test_substring_match_rejected(self):
        """`HerdDoggy` shouldn't be treated as addressing `HerdDog`."""
        self.assertIsNone(_strip_name_prefix("HerdDoggy how's it going", "HerdDog"))

    def test_at_prefix_alone_returns_empty(self):
        """Just `@HerdDog` with nothing after — caller should drop."""
        out = _strip_name_prefix("@HerdDog", "HerdDog")
        self.assertEqual(out, "")


class TestSweepToChannelName(unittest.TestCase):
    """Discord text channels: lowercase, [a-z0-9-], max 100 chars."""

    def test_underscores_become_hyphens(self):
        self.assertEqual(sweep_to_channel_name("mnist_sweep"), "mnist-sweep")

    def test_lowercased(self):
        self.assertEqual(sweep_to_channel_name("MNIST_Sweep"), "mnist-sweep")

    def test_strips_punctuation(self):
        self.assertEqual(sweep_to_channel_name("foo/bar.baz!"), "foobarbaz")

    def test_collapses_repeated_separators(self):
        self.assertEqual(sweep_to_channel_name("a__b  c"), "a-b-c")

    def test_caps_at_100_chars(self):
        out = sweep_to_channel_name("x" * 200)
        self.assertEqual(len(out), 100)

    def test_empty_falls_back(self):
        self.assertEqual(sweep_to_channel_name("!!!"), "hyperherd")


class TestHeartbeatTopic(unittest.TestCase):
    """Channel-topic heartbeat parser/stripper. Pure functions over
    strings — easy to verify in isolation."""

    def test_parse_returns_none_on_empty(self):
        from hyperherd.monitor_agent.channel.discord_channel import (
            _parse_heartbeat_topic,
        )
        self.assertIsNone(_parse_heartbeat_topic(""))
        self.assertIsNone(_parse_heartbeat_topic(None or ""))
        self.assertIsNone(_parse_heartbeat_topic("just a description"))

    def test_parse_returns_datetime(self):
        from hyperherd.monitor_agent.channel.discord_channel import (
            _parse_heartbeat_topic,
        )
        topic = "Sweep foo [hyperherd-heartbeat: 2026-05-05T12:34:56Z]"
        dt = _parse_heartbeat_topic(topic)
        self.assertIsNotNone(dt)
        self.assertEqual(dt.year, 2026)
        self.assertEqual(dt.month, 5)
        self.assertIsNotNone(dt.tzinfo)

    def test_parse_tolerates_naive_iso(self):
        from datetime import timezone
        from hyperherd.monitor_agent.channel.discord_channel import (
            _parse_heartbeat_topic,
        )
        dt = _parse_heartbeat_topic("[hyperherd-heartbeat: 2026-05-05T12:34:56]")
        self.assertEqual(dt.tzinfo, timezone.utc)

    def test_parse_returns_none_on_garbage(self):
        from hyperherd.monitor_agent.channel.discord_channel import (
            _parse_heartbeat_topic,
        )
        self.assertIsNone(_parse_heartbeat_topic("[hyperherd-heartbeat: nope]"))

    def test_strip_leaves_user_text(self):
        from hyperherd.monitor_agent.channel.discord_channel import (
            _strip_heartbeat_marker,
        )
        out = _strip_heartbeat_marker(
            "Sweep foo [hyperherd-heartbeat: 2026-05-05T12:34:56Z]"
        )
        self.assertEqual(out, "Sweep foo")

    def test_strip_idempotent(self):
        from hyperherd.monitor_agent.channel.discord_channel import (
            _strip_heartbeat_marker,
        )
        s = "Sweep foo"
        self.assertEqual(_strip_heartbeat_marker(s), s)


class TestTokenConflictDetection(unittest.TestCase):
    """Same-token preflight: refuse to start if any *other* channel in
    the guild carries a fresh HyperHerd heartbeat marker in its topic.
    Discord allows only one gateway connection per token."""

    def _build(self, force=False):
        from hyperherd.monitor_agent.channel.discord_channel import (
            DiscordChannel,
        )
        return DiscordChannel(
            token="x", guild_id=1, sweep_name="s",
            workspace=Path("/tmp"),
            dashboard_refresh_seconds=60,
            force_token_conflict=force,
        )

    def _wire(self, ch, *, our_channel_id, channels):
        """Attach fake client/channel/guild objects to a DiscordChannel."""
        class _Ch:
            def __init__(self, cid, name, topic):
                self.id = cid
                self.name = name
                self.topic = topic

        class _U:
            id = 100  # our bot user id
        class _Cl:
            user = _U()
        ch._client = _Cl()

        text_channels = []
        our_ch = None
        for spec in channels:
            cobj = _Ch(spec["id"], spec.get("name", f"c{spec['id']}"),
                       spec.get("topic"))
            text_channels.append(cobj)
            if cobj.id == our_channel_id:
                our_ch = cobj

        class _G:
            pass
        guild = _G()
        guild.text_channels = text_channels
        our_ch.guild = guild  # type: ignore[attr-defined]
        ch._channel = our_ch

    def _fresh_marker(self, offset_seconds=-30):
        from datetime import datetime, timedelta, timezone
        ts = (datetime.now(timezone.utc)
              + timedelta(seconds=offset_seconds)).strftime(
                  "%Y-%m-%dT%H:%M:%SZ")
        return f"[hyperherd-heartbeat: {ts}]"

    def test_no_topics_no_conflict(self):
        ch = self._build()
        self._wire(ch, our_channel_id=1, channels=[
            {"id": 1, "name": "ours", "topic": None},
            {"id": 2, "name": "theirs", "topic": "just a description"},
        ])
        asyncio.run(ch._check_for_token_conflicts())  # must not raise

    def test_recent_heartbeat_in_other_channel_raises(self):
        ch = self._build()
        topic = f"Their sweep {self._fresh_marker(-30)}"
        self._wire(ch, our_channel_id=1, channels=[
            {"id": 1, "name": "ours", "topic": None},
            {"id": 2, "name": "theirs", "topic": topic},
        ])
        with self.assertRaises(RuntimeError) as cx:
            asyncio.run(ch._check_for_token_conflicts())
        self.assertIn("DISCORD_BOT_TOKEN", str(cx.exception))
        self.assertIn("theirs", str(cx.exception))
        self.assertIn("--force-discord", str(cx.exception))

    def test_stale_heartbeat_does_not_raise(self):
        ch = self._build()
        # Heartbeat from 2 hours ago — well past 3× interval.
        old_marker = self._fresh_marker(-7200)
        self._wire(ch, our_channel_id=1, channels=[
            {"id": 1, "name": "ours", "topic": None},
            {"id": 2, "name": "theirs", "topic": f"old {old_marker}"},
        ])
        asyncio.run(ch._check_for_token_conflicts())  # too old, no raise

    def test_heartbeat_in_our_channel_ignored(self):
        ch = self._build()
        marker = self._fresh_marker(-10)
        self._wire(ch, our_channel_id=1, channels=[
            {"id": 1, "name": "ours", "topic": f"ours {marker}"},
            {"id": 2, "name": "theirs", "topic": None},
        ])
        asyncio.run(ch._check_for_token_conflicts())  # our own, no raise

    def test_force_flag_bypasses_conflict(self):
        ch = self._build(force=True)
        topic = f"Their sweep {self._fresh_marker(-30)}"
        self._wire(ch, our_channel_id=1, channels=[
            {"id": 1, "name": "ours", "topic": None},
            {"id": 2, "name": "theirs", "topic": topic},
        ])
        asyncio.run(ch._check_for_token_conflicts())  # force=True, no raise


class TestRefreshButtonCooldown(unittest.TestCase):
    """The dashboard's refresh button must trigger a real SLURM re-poll
    (not just a re-render of the cached snapshot) — but rate-limited
    so a user spamming clicks can't hammer sacct."""

    def _build(self):
        from hyperherd.monitor_agent.channel.discord_channel import (
            DiscordChannel,
        )
        return DiscordChannel(
            token="x", guild_id=1, sweep_name="s",
            workspace=Path("/tmp"),
            dashboard_refresh_seconds=60,
        )

    def _fake_interaction(self):
        """Minimal stand-in for discord.Interaction."""
        from unittest.mock import AsyncMock, MagicMock
        inter = MagicMock()
        inter.response = MagicMock()
        inter.response.defer = AsyncMock()
        inter.followup = MagicMock()
        inter.followup.send = AsyncMock()
        inter.edit_original_response = AsyncMock()
        return inter

    def test_first_click_triggers_snapshot_refresh(self):
        from unittest.mock import patch, MagicMock
        ch = self._build()
        ch._build_dashboard_embed = lambda: MagicMock()
        inter = self._fake_interaction()
        with patch(
            "hyperherd.monitor_agent.state.refresh_snapshot",
        ) as refresh:
            asyncio.run(ch._handle_refresh_click(inter, view=object()))
        refresh.assert_called_once_with(Path("/tmp"))
        inter.edit_original_response.assert_awaited_once()
        # No cooldown notice on the first click.
        inter.followup.send.assert_not_called()

    def test_second_click_within_cooldown_skips_sacct(self):
        from unittest.mock import patch, MagicMock
        ch = self._build()
        ch._build_dashboard_embed = lambda: MagicMock()
        inter1 = self._fake_interaction()
        inter2 = self._fake_interaction()
        with patch(
            "hyperherd.monitor_agent.state.refresh_snapshot",
        ) as refresh:
            asyncio.run(ch._handle_refresh_click(inter1, view=object()))
            asyncio.run(ch._handle_refresh_click(inter2, view=object()))
        # First click polled; second was within 20s, no second poll.
        self.assertEqual(refresh.call_count, 1)
        # Second click still re-renders so the user sees the latest.
        inter2.edit_original_response.assert_awaited_once()
        # And gets an ephemeral cooldown notice.
        inter2.followup.send.assert_awaited_once()

    def test_failed_refresh_falls_back_to_cached_snapshot(self):
        from unittest.mock import patch, MagicMock
        ch = self._build()
        ch._build_dashboard_embed = lambda: MagicMock()
        inter = self._fake_interaction()
        with patch(
            "hyperherd.monitor_agent.state.refresh_snapshot",
            side_effect=RuntimeError("sacct timeout"),
        ):
            asyncio.run(ch._handle_refresh_click(inter, view=object()))
        # Refresh failed, but we still showed the cached dashboard.
        inter.edit_original_response.assert_awaited_once()
        # Cooldown is NOT updated on failure, so the next click can retry.
        self.assertEqual(ch._last_manual_refresh, 0.0)


class TestInboxWriter(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.workspace = Path(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_appends_jsonl_and_fires_callback(self):
        called = []
        writer = make_inbox_writer(self.workspace, on_write=lambda: called.append(1))

        async def go():
            await writer(InboundEvent(
                timestamp="2026-05-03T00:00:00",
                source="discord", author="alice", text="pause please",
            ))
            await writer(InboundEvent(
                timestamp="2026-05-03T00:00:01",
                source="discord", author="alice", text="actually go",
            ))

        asyncio.run(go())

        path = self.workspace / ".hyperherd" / "inbox.jsonl"
        self.assertTrue(path.is_file())
        lines = path.read_text().splitlines()
        self.assertEqual(len(lines), 2)
        first = json.loads(lines[0])
        self.assertEqual(first["text"], "pause please")
        self.assertEqual(first["source"], "discord")
        self.assertEqual(called, [1, 1])


@dataclass
class _FakeChannel:
    """Minimal `MessageChannel` that records calls and lets tests trigger
    inbound events on demand."""
    name: str = "fake"
    _started: bool = False
    _stopped: bool = False
    _posts: Optional[List[str]] = None
    _handler = None
    _stop_handler = None
    _info_handler = None

    def __post_init__(self):
        self._posts = []

    async def start(self) -> None:
        self._started = True

    async def stop(self) -> None:
        self._stopped = True

    async def post(self, body: str) -> None:
        self._posts.append(body)

    def set_inbound_handler(self, handler):
        self._handler = handler

    def set_stop_handler(self, handler):
        self._stop_handler = handler

    def set_info_handler(self, handler):
        self._info_handler = handler

    def thinking(self):
        import contextlib

        @contextlib.asynccontextmanager
        async def _cm():
            yield
        return _cm()

    async def inject(self, event: InboundEvent) -> None:
        """Test-only hook: simulate an incoming user message."""
        await self._handler(event)


class TestDaemonInboxWake(unittest.TestCase):
    """The daemon should fire an immediate `user_message` tick when an
    inbound event arrives during the inter-tick sleep."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.workspace = Path(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_inbound_message_wakes_daemon(self):
        triggers = []
        channel = _FakeChannel()

        # First tick returns a long delay so the daemon would normally
        # sleep — but we'll inject an inbound event during that sleep,
        # which should wake it for an immediate user_message tick.
        results = iter([
            TickResult(next_delay_seconds=600, halted=False,
                       halt_reason=None, cost_usd=0.01, turns=1),
            TickResult(next_delay_seconds=None, halted=True,
                       halt_reason="user said pause",
                       cost_usd=0.01, turns=1),
        ])

        async def fake_run_tick(workspace, trigger, channel=None):
            triggers.append(trigger)
            # Right after the first tick returns, simulate a user reply.
            if trigger == "boot":
                # Inject the inbound event after a short delay so the
                # daemon has time to enter its sleep.
                async def inject_later():
                    await asyncio.sleep(0.05)
                    await channel.inject(InboundEvent(
                        timestamp="2026-05-03T00:00:00",
                        source="discord", author="alice", text="pause",
                    ))
                asyncio.create_task(inject_later())
            return next(results)

        async def go():
            return await daemon_mod.run_daemon(
                self.workspace,
                run_tick=fake_run_tick,
                channel=channel,
                post_final=False,
                enable_slurm_poll=False,
            )

        out = asyncio.run(go())

        # The boot tick fires first; the inbound wakeup should produce
        # a second tick with trigger=user_message; that one halts.
        self.assertEqual(triggers, ["boot", "user_message"])
        self.assertTrue(out.halted)
        self.assertEqual(out.halt_reason, "user said pause")
        # Channel lifecycle: started before the loop, stopped after.
        self.assertTrue(channel._started)
        self.assertTrue(channel._stopped)


class TestDashboardEmbed(unittest.TestCase):
    """_build_dashboard_embed returns a discord.Embed-like object with
    correct fields. We stub discord so the test doesn't need discord.py."""

    def _build(self):
        from hyperherd.monitor_agent.channel.discord_channel import DiscordChannel
        return DiscordChannel(
            token="x", guild_id=1, sweep_name="my-sweep",
            workspace=Path("/tmp"),
            dashboard_refresh_seconds=60,
        )

    def _fake_snap(self, path, totals, trials):
        import json
        snap = {"totals": totals, "trials": trials}
        path.write_text(json.dumps(snap))

    def _fake_discord(self):
        """Return a module stub that records Embed construction."""
        from unittest.mock import MagicMock
        discord_stub = MagicMock()
        # Embed needs to be a real class so we can inspect add_field calls.
        class FakeEmbed:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.fields = []
            def add_field(self, *, name, value, inline):
                self.fields.append({"name": name, "value": value})
            def set_footer(self, **kwargs):
                pass
        discord_stub.Embed = FakeEmbed
        return discord_stub

    def test_no_snapshot_returns_loading_embed(self):
        import sys
        from unittest.mock import patch, MagicMock
        ch = self._build()
        # No snapshot file in /tmp/.hyperherd — expect graceful fallback.
        discord_stub = self._fake_discord()
        with patch.dict(sys.modules, {"discord": discord_stub}):
            embed = ch._build_dashboard_embed()
        self.assertIn("no snapshot", embed.kwargs.get("description", "").lower())

    def test_embed_has_status_daemon_trials_fields(self):
        import sys, tempfile, json
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            snap_dir = ws / ".hyperherd"
            snap_dir.mkdir()
            snap = {
                "totals": {"total": 3, "running": 1, "completed": 2},
                "trials": [
                    {"index": 0, "status": "completed",
                     "experiment_name": "exp-a", "last_log_line": ""},
                    {"index": 1, "status": "running",
                     "experiment_name": "exp-b", "elapsed": "2m"},
                    {"index": 2, "status": "completed",
                     "experiment_name": "exp-c", "last_log_line": ""},
                ],
            }
            (snap_dir / "last-snapshot.json").write_text(json.dumps(snap))
            from hyperherd.monitor_agent.channel.discord_channel import (
                DiscordChannel,
            )
            ch = DiscordChannel(
                token="x", guild_id=1, sweep_name="my-sweep",
                workspace=ws, dashboard_refresh_seconds=60,
            )
            discord_stub = self._fake_discord()
            with patch.dict(sys.modules, {"discord": discord_stub}):
                embed = ch._build_dashboard_embed()
        field_names = [f["name"] for f in embed.fields]
        # Should have Status, Daemon, and Trials fields.
        self.assertTrue(any("Status" in n for n in field_names), field_names)
        self.assertTrue(any("Daemon" in n for n in field_names), field_names)
        self.assertTrue(any("Trials" in n for n in field_names), field_names)

    def test_embed_color_red_when_halted(self):
        import sys, tempfile, json
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as tmp:
            ws = Path(tmp)
            snap_dir = ws / ".hyperherd"
            snap_dir.mkdir()
            (snap_dir / "last-snapshot.json").write_text(
                json.dumps({"totals": {}, "trials": []})
            )
            from hyperherd.monitor_agent.channel.discord_channel import (
                DiscordChannel,
            )
            ch = DiscordChannel(
                token="x", guild_id=1, sweep_name="s",
                workspace=ws, dashboard_refresh_seconds=60,
            )
            ch._on_info = lambda: {"health": "halted"}
            discord_stub = self._fake_discord()
            with patch.dict(sys.modules, {"discord": discord_stub}):
                embed = ch._build_dashboard_embed()
        self.assertEqual(embed.kwargs.get("color"), 0xe74c3c)


class TestPickAutoPlotMetric(unittest.TestCase):
    """pick_auto_plot_metric reads the success metric from MONITOR_PLAN.md
    and matches it against the trial's logged streams."""

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        self.ws = Path(self.tmp)
        (self.ws / ".hyperherd").mkdir()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp)

    def _write_plan(self, line):
        (self.ws / ".hyperherd" / "MONITOR_PLAN.md").write_text(line + "\n")

    def _pick(self, streams):
        from unittest.mock import patch
        from hyperherd.monitor_agent.plots import pick_auto_plot_metric
        with patch(
            "hyperherd.logging.list_metric_streams",
            return_value=streams,
        ):
            return pick_auto_plot_metric(self.ws, trial_index=0)

    def test_uses_plan_metric_when_present(self):
        self._write_plan("- Success metric: val/loss, min")
        result = self._pick(["train/loss", "val/loss", "lr"])
        self.assertEqual(result, "val/loss")

    def test_plan_metric_with_maximize(self):
        self._write_plan("- Success metric: val_accuracy, max")
        result = self._pick(["val_accuracy", "train_loss"])
        self.assertEqual(result, "val_accuracy")

    def test_plan_metric_not_in_streams_falls_back_alphabetically(self):
        """If the plan names a metric the trial never logged, don't crash —
        fall back to sorted(streams)[0]."""
        self._write_plan("- Success metric: val/loss, min")
        result = self._pick(["train_loss", "lr"])
        self.assertEqual(result, "lr")

    def test_plan_metric_none_falls_back_alphabetically(self):
        self._write_plan("- Success metric: none")
        result = self._pick(["train/loss", "val/loss"])
        self.assertEqual(result, "train/loss")

    def test_no_plan_file_falls_back_alphabetically(self):
        result = self._pick(["zz_metric", "aa_metric"])
        self.assertEqual(result, "aa_metric")

    def test_no_streams_returns_none(self):
        self._write_plan("- Success metric: val/loss, min")
        result = self._pick([])
        self.assertIsNone(result)


class TestAutoPlot(unittest.IsolatedAsyncioTestCase):
    """SlurmPoll._auto_plot: best-effort plot-and-post on completion/failure."""

    async def _make_poller(self, channel=None):
        from hyperherd.monitor_agent.event_source.slurm import SlurmPoll
        return SlurmPoll(
            workspace="/tmp/ws",
            interval_seconds=0.01,
            channel=channel,
        )

    async def test_no_channel_returns_immediately(self):
        """No channel → _auto_plot should not call pick_auto_plot_metric."""
        from unittest.mock import patch
        poller = await self._make_poller(channel=None)
        with patch(
            "hyperherd.monitor_agent.plots.pick_auto_plot_metric"
        ) as pick:
            await poller._auto_plot(0, seed_text="⚠️ trial 0 failed")
        pick.assert_not_called()

    async def test_no_metric_streams_skips_plot(self):
        """If pick_auto_plot_metric returns None, no render is attempted."""
        from unittest.mock import AsyncMock, MagicMock, patch
        channel = MagicMock()
        channel.post_to_trial_thread = AsyncMock()
        poller = await self._make_poller(channel=channel)
        with patch(
            "hyperherd.monitor_agent.plots.pick_auto_plot_metric",
            return_value=None,
        ):
            await poller._auto_plot(0, seed_text="⚠️ trial 0 failed")
        channel.post_to_trial_thread.assert_not_called()

    async def test_successful_auto_plot_posts_to_thread(self):
        """Happy path: pick metric → render → post_to_trial_thread, cleanup."""
        import tempfile
        from pathlib import Path
        from unittest.mock import AsyncMock, MagicMock, patch

        channel = MagicMock()
        channel.post_to_trial_thread = AsyncMock()
        poller = await self._make_poller(channel=channel)

        # Create a real temp file so the unlink() call works.
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        png_path = Path(tmp.name)

        with (
            patch(
                "hyperherd.monitor_agent.plots.pick_auto_plot_metric",
                return_value="train/loss",
            ),
            patch(
                "hyperherd.monitor_agent.plots.render_metric_plot",
                return_value=png_path,
            ),
        ):
            await poller._auto_plot(3, seed_text="✅ trial 3 done")

        channel.post_to_trial_thread.assert_awaited_once()
        call_kwargs = channel.post_to_trial_thread.call_args
        self.assertEqual(call_kwargs.args[0], 3)
        self.assertEqual(call_kwargs.kwargs.get("thread_seed_text"), "✅ trial 3 done")
        # Temp file should be cleaned up after posting.
        self.assertFalse(png_path.exists())

    async def test_plot_unavailable_doesnt_crash(self):
        """PlotUnavailable (matplotlib not installed / no points) is swallowed."""
        from unittest.mock import AsyncMock, MagicMock, patch
        from hyperherd.monitor_agent.plots import PlotUnavailable

        channel = MagicMock()
        channel.post_to_trial_thread = AsyncMock()
        poller = await self._make_poller(channel=channel)
        with (
            patch(
                "hyperherd.monitor_agent.plots.pick_auto_plot_metric",
                return_value="val/loss",
            ),
            patch(
                "hyperherd.monitor_agent.plots.render_metric_plot",
                side_effect=PlotUnavailable("no data"),
            ),
        ):
            await poller._auto_plot(1, seed_text="seed")  # must not raise
        channel.post_to_trial_thread.assert_not_called()


if __name__ == "__main__":
    unittest.main()
