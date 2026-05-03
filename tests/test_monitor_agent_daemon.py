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

    async def fake_run_tick(workspace, trigger):
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
            ))
        posted.assert_called_once()
        kwargs = posted.call_args.kwargs
        self.assertTrue(kwargs["halted"])
        self.assertEqual(kwargs["halt_reason"], "sweep complete")
        self.assertEqual(kwargs["ticks"], 1)


if __name__ == "__main__":
    unittest.main()
