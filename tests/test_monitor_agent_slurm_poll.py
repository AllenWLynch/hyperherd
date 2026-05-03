"""Tests for the SLURM polling event source.

The poller shells out to `herd snapshot`; we patch that to return canned
JSON so the tests don't need a live SLURM cluster (or even a workspace).
"""

import asyncio
import json
import unittest
from unittest import mock

from hyperherd.monitor_agent.event_source import WakeEvent
from hyperherd.monitor_agent.event_source.slurm import SlurmPoll


def _fake_snapshot(failed_idxs=(), completed_idxs=()):
    trials = []
    for i in failed_idxs:
        trials.append({"index": i, "status": "failed"})
    for i in completed_idxs:
        trials.append({"index": i, "status": "completed"})
    return {"trials": trials}


class TestSlurmPoll(unittest.IsolatedAsyncioTestCase):
    """One iteration of the polling loop, exercised by injecting a sequence
    of snapshots via mock and pumping the queue."""

    async def asyncSetUp(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.poller = SlurmPoll(workspace="/tmp/anywhere", interval_seconds=0.01)

    async def _pump(self, snapshots, max_steps):
        """Patch _snapshot to yield each canned snapshot in sequence,
        then run the poller until it has consumed `max_steps` of them."""
        snap_iter = iter(snapshots)

        async def fake_snapshot():
            try:
                return next(snap_iter)
            except StopIteration:
                # Stop iterating by cancelling the task.
                raise asyncio.CancelledError()

        with mock.patch.object(self.poller, "_snapshot", side_effect=fake_snapshot):
            task = asyncio.create_task(self.poller.run(self.queue))
            # Give the loop time to consume snapshots — the interval is
            # 0.01s, so a few hundredths of a second covers many ticks.
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    async def _drain_queue(self):
        events = []
        while not self.queue.empty():
            events.append(self.queue.get_nowait())
        return events

    async def test_no_events_emitted_for_baseline(self):
        """Trials already terminal at startup must not fire events —
        otherwise restarting the daemon would emit a spurious flood."""
        snapshots = [_fake_snapshot(failed_idxs=[0, 1], completed_idxs=[2])]
        await self._pump(snapshots, max_steps=1)
        events = await self._drain_queue()
        self.assertEqual(events, [])

    async def test_new_failure_emits_failure_event(self):
        snapshots = [
            _fake_snapshot(failed_idxs=[]),                # baseline
            _fake_snapshot(failed_idxs=[3]),               # idx 3 fails
        ]
        await self._pump(snapshots, max_steps=2)
        events = await self._drain_queue()
        self.assertIn(WakeEvent(trigger="failure"), events)

    async def test_new_completion_emits_completion_event(self):
        snapshots = [
            _fake_snapshot(completed_idxs=[]),
            _fake_snapshot(completed_idxs=[4]),
        ]
        await self._pump(snapshots, max_steps=2)
        events = await self._drain_queue()
        self.assertIn(WakeEvent(trigger="completion"), events)

    async def test_simultaneous_transitions_emit_both(self):
        snapshots = [
            _fake_snapshot(),
            _fake_snapshot(failed_idxs=[0], completed_idxs=[1]),
        ]
        await self._pump(snapshots, max_steps=2)
        events = await self._drain_queue()
        triggers = {e.trigger for e in events}
        self.assertEqual(triggers, {"failure", "completion"})

    async def test_transient_snapshot_error_doesnt_kill_loop(self):
        """If sacct hiccups one tick, the next snapshot should still
        produce events — the poller swallows transient errors."""
        snapshots = [
            _fake_snapshot(),                     # baseline
            RuntimeError("sacct timeout"),        # transient failure
            _fake_snapshot(failed_idxs=[5]),      # recovers, should fire
        ]
        snap_iter = iter(snapshots)

        async def fake_snapshot():
            try:
                v = next(snap_iter)
            except StopIteration:
                raise asyncio.CancelledError()
            if isinstance(v, Exception):
                raise v
            return v

        with mock.patch.object(self.poller, "_snapshot", side_effect=fake_snapshot):
            task = asyncio.create_task(self.poller.run(self.queue))
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        events = await self._drain_queue()
        self.assertIn(WakeEvent(trigger="failure"), events)


class TestSlurmPollChannelPosts(unittest.IsolatedAsyncioTestCase):
    """The poller posts emoji-prefixed notifications to the channel for
    each transition, in addition to queuing wake events for the daemon."""

    async def asyncSetUp(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self.posts: list = []

        class FakeChannel:
            name = "fake"

            async def post(_self, body):
                self.posts.append(body)

        self.poller = SlurmPoll(
            workspace="/tmp/anywhere",
            interval_seconds=0.01,
            channel=FakeChannel(),
        )

    async def _pump(self, snapshots):
        snap_iter = iter(snapshots)

        async def fake_snapshot():
            try:
                return next(snap_iter)
            except StopIteration:
                raise asyncio.CancelledError()

        with mock.patch.object(self.poller, "_snapshot", side_effect=fake_snapshot):
            task = asyncio.create_task(self.poller.run(self.queue))
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    async def test_transitions_post_with_emoji(self):
        running_snap = {"trials": [{"index": 0, "status": "running"}]}
        complete_snap = {"trials": [
            {"index": 0, "status": "completed"},
            {"index": 1, "status": "failed"},
        ]}
        await self._pump([
            {"trials": []},        # baseline empty
            running_snap,          # idx 0 starts
            complete_snap,         # idx 0 done, idx 1 fails
        ])

        joined = "\n".join(self.posts)
        self.assertIn("▶️ Trial 0 started running", joined)
        self.assertIn("✅ Trial 0 completed", joined)
        self.assertIn("⚠️ Trial 1 failed", joined)

    async def test_baseline_does_not_post(self):
        # Trials already in any state at startup must NOT trigger posts.
        await self._pump([
            {"trials": [
                {"index": 0, "status": "running"},
                {"index": 1, "status": "completed"},
            ]},
        ])
        self.assertEqual(self.posts, [])


if __name__ == "__main__":
    unittest.main()
