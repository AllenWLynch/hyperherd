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
from hyperherd.monitor_agent.channel.discord_channel import sweep_to_channel_name
from hyperherd.monitor_agent.tick import TickResult


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


if __name__ == "__main__":
    unittest.main()
