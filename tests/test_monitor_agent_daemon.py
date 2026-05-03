"""Tests for the Phase-2 daemon loop.

The loop wraps `run_tick`, which we mock here so the tests don't need
ANTHROPIC_API_KEY, the SDK, a workspace, or network access.
"""

import asyncio
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hyperherd.monitor_agent import daemon as daemon_mod
from hyperherd.monitor_agent.tick import TickResult


def _make_run_tick(results, calls):
    """Async fn that returns the canned TickResults in order, recording
    each call's trigger into the `calls` list."""
    it = iter(results)

    async def fake_run_tick(workspace, trigger, **_):
        calls.append(trigger)
        return next(it)

    return fake_run_tick


class TestDaemonLoop(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.workspace = Path(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _run(self, fake, **kwargs):
        kwargs.setdefault("post_final", False)
        # Tests don't want a real SLURM poller subprocessing against tmp dirs.
        kwargs.setdefault("enable_slurm_poll", False)
        return asyncio.run(
            daemon_mod.run_daemon(self.workspace, run_tick=fake, **kwargs)
        )

    def test_halts_on_first_tick(self):
        calls = []
        fake = _make_run_tick(
            [TickResult(next_delay_seconds=None, halted=True,
                        halt_reason="sweep complete",
                        cost_usd=0.05, turns=2)],
            calls,
        )
        out = self._run(fake)

        self.assertEqual(out.ticks, 1)
        self.assertTrue(out.halted)
        self.assertEqual(out.halt_reason, "sweep complete")
        self.assertFalse(out.stopped_by_signal)
        self.assertEqual(calls, ["boot"])
        self.assertAlmostEqual(out.total_cost_usd, 0.05, places=6)

    def test_multiple_ticks_then_halt(self):
        calls = []
        fake = _make_run_tick(
            [
                TickResult(0.001, halted=False, halt_reason=None,
                           cost_usd=0.10, turns=3),
                TickResult(0.001, halted=False, halt_reason=None,
                           cost_usd=0.20, turns=4),
                TickResult(None, halted=True, halt_reason="done",
                           cost_usd=0.05, turns=2),
            ],
            calls,
        )
        out = self._run(fake)

        self.assertEqual(out.ticks, 3)
        self.assertTrue(out.halted)
        self.assertEqual(out.halt_reason, "done")
        # First call uses boot trigger; subsequent ones are scheduled.
        self.assertEqual(calls, ["boot", "scheduled", "scheduled"])
        self.assertAlmostEqual(out.total_cost_usd, 0.35, places=6)

    def test_max_ticks_cap(self):
        """When the agent never halts, --max-ticks bounds the run."""
        calls = []
        fake = _make_run_tick(
            [TickResult(0.001, halted=False, halt_reason=None,
                        cost_usd=0.01, turns=1) for _ in range(10)],
            calls,
        )
        out = self._run(fake, max_ticks=3)

        self.assertEqual(out.ticks, 3)
        self.assertFalse(out.halted)
        self.assertFalse(out.stopped_by_signal)
        self.assertEqual(calls, ["boot", "scheduled", "scheduled"])

    def test_none_delay_does_not_crash(self):
        """If the agent forgets schedule_next (delay=None), the loop must
        not raise. We bound with max_ticks=1 so we never sleep the fallback."""
        calls = []
        fake = _make_run_tick(
            [TickResult(next_delay_seconds=None, halted=False,
                        halt_reason=None, cost_usd=0.0, turns=0)],
            calls,
        )
        out = self._run(fake, max_ticks=1)
        self.assertEqual(out.ticks, 1)
        self.assertFalse(out.halted)

    def test_final_message_fires_on_halt(self):
        """The daemon must post a 'stopped' notification on exit so the user
        always knows when API calls have ceased."""
        calls = []
        fake = _make_run_tick(
            [TickResult(next_delay_seconds=None, halted=True,
                        halt_reason="sweep complete",
                        cost_usd=0.05, turns=2)],
            calls,
        )
        with mock.patch.object(daemon_mod, "_post_final_message") as posted:
            posted.return_value = asyncio.sleep(0)  # awaitable no-op
            asyncio.run(daemon_mod.run_daemon(
                self.workspace, run_tick=fake, post_final=True,
                enable_slurm_poll=False,
            ))
        posted.assert_called_once()
        kwargs = posted.call_args.kwargs
        self.assertTrue(kwargs["halted"])
        self.assertEqual(kwargs["halt_reason"], "sweep complete")
        self.assertEqual(kwargs["ticks"], 1)


class TestDaemonSlurmEventWake(unittest.TestCase):
    """Verify the queue-driven path: a SLURM event during the inter-tick
    sleep should wake the daemon and run the next tick with the
    appropriate trigger."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.workspace = Path(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_failure_event_wakes_with_failure_trigger(self):
        from hyperherd.monitor_agent.event_source import WakeEvent

        triggers = []
        results = iter([
            TickResult(next_delay_seconds=600, halted=False,
                       halt_reason=None, cost_usd=0.01, turns=1),
            TickResult(next_delay_seconds=None, halted=True,
                       halt_reason="recurring exception",
                       cost_usd=0.01, turns=1),
        ])

        # We poke the daemon's internal queue from the first tick. We
        # need a way to access it from the fake; use a closure on a list
        # that the run_daemon writes into via a hook.
        injected_queue = {}

        async def fake_run_tick(workspace, trigger, channel=None):
            triggers.append(trigger)
            if trigger == "boot":
                # Find the daemon's event queue (only one in flight) and
                # push a failure event during a short delay so the daemon
                # is in its inter-tick sleep.
                async def push_later():
                    await asyncio.sleep(0.05)
                    queue = injected_queue.get("q")
                    if queue is not None:
                        await queue.put(WakeEvent(trigger="failure"))
                asyncio.create_task(push_later())
            return next(results)

        # Monkey-patch `asyncio.Queue.put_nowait` is overkill — instead,
        # we wrap run_daemon's _wait_next_event to capture the queue
        # arg. Simpler: peek at the asyncio.Queue created inside
        # run_daemon by patching daemon_mod.asyncio.Queue.
        original_queue_cls = daemon_mod.asyncio.Queue

        def queue_factory(*args, **kwargs):
            q = original_queue_cls(*args, **kwargs)
            injected_queue["q"] = q
            return q

        with mock.patch.object(daemon_mod.asyncio, "Queue", queue_factory):
            out = asyncio.run(daemon_mod.run_daemon(
                self.workspace, run_tick=fake_run_tick,
                enable_slurm_poll=False, post_final=False,
            ))

        self.assertEqual(triggers, ["boot", "failure"])
        self.assertTrue(out.halted)


class TestDaemonInboxRequeue(unittest.TestCase):
    """If a user message lands on disk after state.compute drained the
    inbox (i.e. during the tick itself), the wake event was drained but
    the data is in a fresh inbox.jsonl. The daemon must detect this and
    fire an immediate user_message tick instead of sleeping the full
    scheduled delay."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.workspace = Path(self.tmp)
        (self.workspace / ".hyperherd").mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_post_tick_inbox_content_triggers_wake(self):
        triggers = []
        results = iter([
            TickResult(next_delay_seconds=600, halted=False,
                       halt_reason=None, cost_usd=0.01, turns=1),
            TickResult(next_delay_seconds=None, halted=True,
                       halt_reason="acknowledged",
                       cost_usd=0.01, turns=1),
        ])

        async def fake_run_tick(workspace, trigger, **_):
            triggers.append(trigger)
            if trigger == "boot":
                # Simulate a writer landing inbox content AFTER state.compute
                # has already drained — write to inbox.jsonl directly.
                inbox = self.workspace / ".hyperherd" / "inbox.jsonl"
                inbox.write_text(
                    '{"timestamp":"t1","source":"discord",'
                    '"author":"alice","text":"hi"}\n'
                )
            return next(results)

        out = asyncio.run(daemon_mod.run_daemon(
            self.workspace, run_tick=fake_run_tick,
            enable_slurm_poll=False, post_final=False,
        ))

        # Without the re-queue fix, triggers would be ["boot"] and the
        # daemon would sleep the full 600s before checking again. With
        # the fix, the daemon detects post-tick inbox content and fires
        # an immediate user_message tick.
        self.assertEqual(triggers, ["boot", "user_message"])
        self.assertTrue(out.halted)


class TestHeartbeatBuilder(unittest.TestCase):
    """The heartbeat helper formats a one-line digest from
    .hyperherd/last-snapshot.json plus runtime tick count."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.workspace = Path(self.tmp)
        (self.workspace / ".hyperherd").mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_returns_none_without_snapshot(self):
        text = daemon_mod._build_heartbeat_text(self.workspace, {"ticks": 0})
        self.assertIsNone(text)

    def test_renders_totals_and_ticks(self):
        import json as _json
        snap = self.workspace / ".hyperherd" / "last-snapshot.json"
        snap.write_text(_json.dumps({
            "totals": {"total": 3, "running": 1, "completed": 2}
        }))
        text = daemon_mod._build_heartbeat_text(self.workspace, {"ticks": 4})
        self.assertIsNotNone(text)
        self.assertTrue(text.startswith("🐾"))
        self.assertIn("1 running", text)
        self.assertIn("2 completed", text)
        self.assertIn("4 agent tick", text)


if __name__ == "__main__":
    unittest.main()
