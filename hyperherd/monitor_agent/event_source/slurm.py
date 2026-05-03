"""SLURM polling event source.

Periodically asks `herd snapshot` for the sweep state, diffs trial-status
sets against the previous poll, and:

1. Pushes `WakeEvent("failure")` / `WakeEvent("completion")` into the
   daemon's queue so the agent ticks promptly on consequential events.
2. Posts a one-line notification into the chat channel for each
   transition (trial started running, trial completed, trial failed).
   These are non-agent posts — emoji-prefixed, no model spend, fire the
   moment SLURM reports the transition.

The post + tick split lets the user see "trial 3 started running" the
instant SLURM picks it up, and then a few seconds later (when the
agent's tick fires), the agent's policy decision. Two different
signals, two different voices.

Why polling instead of `sacct --json` directly: `herd snapshot` already
encapsulates all the manifest-aware logic (slurm job IDs to trial
indices, status mapping, etc.); reusing it keeps this module thin and
ensures the poller and the per-tick state assembler agree on what a
trial's status is.
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Set

from hyperherd.monitor_agent.event_source import WakeEvent

log = logging.getLogger(__name__)


DEFAULT_POLL_INTERVAL_SECONDS = 60


class SlurmPoll:
    """Poll the workspace for trial state transitions on a fixed interval.

    Stateful — tracks the set of indices it has already seen in each
    state, so on startup it doesn't fire spurious events for trials that
    were already in flight (or terminal) before the daemon launched.
    """

    def __init__(
        self,
        workspace: Path,
        *,
        interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
        channel=None,
    ):
        self._workspace = Path(workspace)
        self._interval = float(interval_seconds)
        self._channel = channel
        self._seen_running: Set[int] = set()
        self._seen_failed: Set[int] = set()
        self._seen_completed: Set[int] = set()

    async def run(self, queue: asyncio.Queue) -> None:
        """Run forever (or until cancelled). Push WakeEvents into `queue`
        and emoji-prefixed posts into the channel as transitions occur."""

        # Initial baseline: capture every trial's current bucket so we
        # don't emit a flood of events for state that already existed
        # before the daemon launched.
        try:
            snap = await self._snapshot()
            self._seen_running = self._set_for(snap, "running")
            self._seen_failed = self._set_for(snap, "failed")
            self._seen_completed = self._set_for(snap, "completed")
            log.info(
                "SlurmPoll baseline: %d running, %d failed, %d completed",
                len(self._seen_running),
                len(self._seen_failed),
                len(self._seen_completed),
            )
        except Exception as e:
            log.warning("SlurmPoll baseline snapshot failed: %s", e)

        while True:
            try:
                await asyncio.sleep(self._interval)
            except asyncio.CancelledError:
                return

            try:
                snap = await self._snapshot()
            except asyncio.CancelledError:
                return
            except Exception as e:
                log.warning("SlurmPoll snapshot error: %s", e)
                continue

            running = self._set_for(snap, "running")
            failed = self._set_for(snap, "failed")
            completed = self._set_for(snap, "completed")

            new_running = running - self._seen_running
            new_failed = failed - self._seen_failed
            new_completed = completed - self._seen_completed

            for idx in sorted(new_running):
                await self._post(f"▶️ Trial {idx} started running.")

            if new_failed:
                log.info("SlurmPoll: new failed indices %s", sorted(new_failed))
                for idx in sorted(new_failed):
                    await self._post(f"⚠️ Trial {idx} failed.")
                await queue.put(WakeEvent(trigger="failure"))

            if new_completed:
                log.info("SlurmPoll: new completed indices %s",
                         sorted(new_completed))
                for idx in sorted(new_completed):
                    await self._post(f"✅ Trial {idx} completed.")
                await queue.put(WakeEvent(trigger="completion"))

            self._seen_running = running
            self._seen_failed = failed
            self._seen_completed = completed

    async def _post(self, text: str) -> None:
        """Best-effort post to the chat channel. Logged on failure but
        never propagates — a channel hiccup shouldn't kill the poller."""
        if self._channel is None:
            return
        try:
            await self._channel.post(text)
        except Exception as e:
            log.warning("SlurmPoll channel.post failed: %s", e)

    async def _snapshot(self) -> dict:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "-m", "hyperherd.cli", "snapshot", str(self._workspace),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"`herd snapshot` exit {proc.returncode}: {err.decode().strip()}"
            )
        return json.loads(out)

    @staticmethod
    def _set_for(snap: dict, status: str) -> Set[int]:
        return {
            t["index"] for t in snap.get("trials", [])
            if t.get("status") == status
        }
