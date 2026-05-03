"""Daemon mode: schedule + event-driven loop on top of run_tick().

The agent picks the next-tick delay via `schedule_next`; the daemon
sleeps that long, races the timer against any event the sources push
into the wake queue, and runs another tick. SIGINT/SIGTERM trigger a
clean exit at the next iteration boundary.

Event sources running alongside the loop:

- The `MessageChannel`'s inbox writer pushes `WakeEvent("user_message")`
  when a user @-mentions the bot or replies to one of its posts.
- `SlurmPoll` (Phase 3) pushes `WakeEvent("failure")` /
  `WakeEvent("completion")` on trial-state transitions.

Final-stop notification: when the loop exits (halt or signal), the
daemon posts one "stopped" message through the channel (or webhook
fallback) so the user sees the daemon is no longer running.
"""

import asyncio
import contextlib
import logging
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@contextlib.asynccontextmanager
async def _noop_thinking_cm():
    yield


async def _heartbeat_loop(
    channel,
    workspace: Path,
    runtime_stats: dict,
    interval_seconds: int,
) -> None:
    """Periodic 'still alive' digest to the channel. Non-agent — no
    model spend, posts the totals + tick count from the most recent
    snapshot. Suppressed when the snapshot is missing (e.g. before the
    first tick has run)."""
    while True:
        try:
            await asyncio.sleep(interval_seconds)
        except asyncio.CancelledError:
            return

        text = _build_heartbeat_text(workspace, runtime_stats)
        if text is None:
            continue
        try:
            await channel.post(text)
        except Exception as e:
            log.warning("Heartbeat post failed: %s", e)


def _build_heartbeat_text(workspace: Path, runtime_stats: dict) -> Optional[str]:
    """Format the heartbeat from .hyperherd/last-snapshot.json + runtime
    stats. Returns None if there's no snapshot yet."""
    import json
    snap_path = workspace / ".hyperherd" / "last-snapshot.json"
    if not snap_path.is_file():
        return None
    try:
        snap = json.loads(snap_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None

    totals = snap.get("totals") or {}
    order = ["ready", "submitted", "queued", "running",
             "completed", "failed", "cancelled"]
    parts = [f"{totals[k]} {k}" for k in order if totals.get(k)]
    digest = ", ".join(parts) or "no trials yet"
    ticks = runtime_stats.get("ticks", 0)
    return f"🐾 {digest} · {ticks} agent tick(s) so far."

from hyperherd.monitor_agent import tick as tick_mod
from hyperherd.monitor_agent.channel import (
    MessageChannel, build_channel, make_inbox_writer,
)
from hyperherd.monitor_agent.event_source import WakeEvent
from hyperherd.monitor_agent.event_source.slurm import SlurmPoll

log = logging.getLogger(__name__)


@dataclass
class DaemonResult:
    ticks: int
    total_cost_usd: float
    halted: bool
    halt_reason: Optional[str]
    stopped_by_signal: bool


async def run_daemon(
    workspace: Path,
    *,
    max_ticks: Optional[int] = None,
    run_tick=None,                # injectable for tests
    channel: Optional[MessageChannel] = None,  # if None, built from config
    enable_slurm_poll: bool = True,
    slurm_poll_interval: Optional[float] = None,
    heartbeat_seconds: int = 300,
    post_final: bool = True,
) -> DaemonResult:
    """Run ticks in a loop until the agent halts or a signal arrives.

    First tick fires immediately with `trigger="boot"`. Subsequent ticks
    wait for `result.next_delay_seconds` or until an event source pushes
    a wake-up onto the queue.
    """
    workspace = Path(workspace).resolve()
    if run_tick is None:
        run_tick = tick_mod.run_tick

    # Build the chat channel from workspace config if the caller didn't
    # inject one. None = no channel; the daemon runs without inbox/post.
    if channel is None:
        channel = _build_channel_from_config(workspace)

    shutdown = asyncio.Event()
    event_q: asyncio.Queue = asyncio.Queue()

    # Mutable runtime stats — the channel's /info slash command reads
    # these via an info_handler callback. We update the dict in-place
    # each tick so the callback sees fresh values.
    from datetime import datetime, timezone
    runtime_stats = {
        "ticks": 0,
        "total_cost_usd": 0.0,
        "started_at_iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    def _on_signal(signum):
        log.info("Received signal %s, shutting down after current tick.", signum)
        shutdown.set()

    def _on_inbox_write():
        # Inbox writer fires after each user message lands; turn that into
        # a queue event so the loop wakes early.
        try:
            event_q.put_nowait(WakeEvent(trigger="user_message"))
        except asyncio.QueueFull:
            pass

    loop = asyncio.get_running_loop()
    installed_handlers = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _on_signal, sig)
            installed_handlers.append(sig)
        except (NotImplementedError, RuntimeError):
            pass

    # Wire inbound: events written to inbox.jsonl, then pushed onto the
    # event queue.
    if channel is not None:
        writer = make_inbox_writer(workspace, on_write=_on_inbox_write)
        channel.set_inbound_handler(writer)
        # Let /stop trigger the same shutdown path SIGINT/SIGTERM use.
        channel.set_stop_handler(lambda: shutdown.set())
        # /info pulls live runtime stats via this callback.
        channel.set_info_handler(lambda: dict(runtime_stats))
        try:
            await channel.start()
            log.info("Channel '%s' connected.", channel.name)
        except Exception as e:
            log.error("Failed to start channel '%s': %s — continuing without it.",
                      channel.name, e)
            channel = None

    # Start the SLURM event source. It's purely additive — even if the
    # poller fails, the daemon still runs scheduled ticks and reacts to
    # user messages.
    slurm_task: Optional[asyncio.Task] = None
    if enable_slurm_poll:
        kwargs = {}
        if slurm_poll_interval is not None:
            kwargs["interval_seconds"] = slurm_poll_interval
        poller = SlurmPoll(workspace, channel=channel, **kwargs)
        slurm_task = asyncio.create_task(
            poller.run(event_q), name="slurm-poll",
        )

    # Start the periodic heartbeat. Posts a one-line totals digest to
    # the channel every few minutes so the user always knows the daemon
    # is alive, even when the agent's tick cadence is long.
    heartbeat_task: Optional[asyncio.Task] = None
    if channel is not None and heartbeat_seconds > 0:
        heartbeat_task = asyncio.create_task(
            _heartbeat_loop(
                channel, workspace, runtime_stats, heartbeat_seconds,
            ),
            name="heartbeat",
        )

    ticks = 0
    total_cost = 0.0
    trigger = "boot"
    halted = False
    halt_reason: Optional[str] = None

    try:
        while not shutdown.is_set():
            log.info("Tick %d starting (trigger=%s)", ticks + 1, trigger)
            # Show the platform's "thinking" indicator (Discord typing dots,
            # etc.) for the duration of the tick. No-op if no channel.
            thinking_cm = (
                channel.thinking() if channel is not None
                else _noop_thinking_cm()
            )
            async with thinking_cm:
                result = await run_tick(workspace, trigger=trigger, channel=channel)
            ticks += 1
            total_cost += result.cost_usd
            runtime_stats["ticks"] = ticks
            runtime_stats["total_cost_usd"] = total_cost
            log.info(
                "Tick %d done. cost=$%.4f turns=%d halted=%s next_delay=%s",
                ticks, result.cost_usd, result.turns, result.halted,
                result.next_delay_seconds,
            )

            if result.halted:
                halted = True
                halt_reason = result.halt_reason
                break

            if max_ticks is not None and ticks >= max_ticks:
                log.info("Reached max-ticks cap (%d), exiting.", max_ticks)
                break

            # The tick's state.compute already absorbed every transition
            # known up to now — drain the queue so we don't fire a
            # redundant tick for events the agent has already seen.
            _drain(event_q)

            # BUT: if a user message landed AFTER state.compute renamed
            # the inbox aside (i.e. during the tick itself), the wake
            # event was already drained, but the message is durable on
            # disk in a fresh inbox.jsonl. Re-queue a wake so we read it
            # promptly instead of waiting for the scheduled timeout.
            inbox_path = workspace / ".hyperherd" / "inbox.jsonl"
            try:
                if inbox_path.is_file() and inbox_path.stat().st_size > 0:
                    log.info("Inbox has post-tick content; queuing immediate wake.")
                    event_q.put_nowait(WakeEvent(trigger="user_message"))
            except OSError:
                pass

            delay = result.next_delay_seconds or 1800
            log.info("Sleeping up to %ds until next tick.", delay)

            outcome = await _wait_next_event(event_q, shutdown, timeout=delay)
            if outcome == "shutdown":
                break
            elif outcome == "timeout":
                trigger = "scheduled"
            else:
                # outcome is the trigger string from a WakeEvent
                trigger = outcome

    finally:
        for sig in installed_handlers:
            try:
                loop.remove_signal_handler(sig)
            except (NotImplementedError, RuntimeError):
                pass
        if slurm_task is not None:
            slurm_task.cancel()
            try:
                await slurm_task
            except (asyncio.CancelledError, Exception):
                pass
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            try:
                await heartbeat_task
            except (asyncio.CancelledError, Exception):
                pass

    stopped_by_signal = shutdown.is_set() and not halted

    if post_final:
        await _post_final_message(
            workspace,
            channel=channel,
            ticks=ticks,
            total_cost_usd=total_cost,
            halted=halted,
            halt_reason=halt_reason,
            stopped_by_signal=stopped_by_signal,
        )

    if channel is not None:
        try:
            await channel.stop()
        except Exception as e:
            log.warning("Channel stop raised: %s", e)

    return DaemonResult(
        ticks=ticks,
        total_cost_usd=total_cost,
        halted=halted,
        halt_reason=halt_reason,
        stopped_by_signal=stopped_by_signal,
    )


# --- queue helpers --------------------------------------------------------

def _drain(queue: asyncio.Queue) -> None:
    """Empty the queue without blocking. The caller has just run a tick
    that absorbed all known state, so any queued events are stale."""
    while not queue.empty():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            return


async def _wait_next_event(
    queue: asyncio.Queue,
    shutdown: asyncio.Event,
    *,
    timeout: float,
) -> str:
    """Block until shutdown, an event arrives, or timeout. Returns:
    - 'shutdown' if shutdown fired
    - 'timeout' if the timer elapsed
    - the event's trigger string otherwise
    """
    shutdown_task = asyncio.create_task(shutdown.wait(), name="wait-shutdown")
    queue_task = asyncio.create_task(queue.get(), name="wait-event")
    try:
        done, pending = await asyncio.wait(
            {shutdown_task, queue_task},
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        for t in (shutdown_task, queue_task):
            if not t.done():
                t.cancel()
        for t in (shutdown_task, queue_task):
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

    if shutdown_task in done:
        return "shutdown"
    if queue_task in done:
        event = queue_task.result()
        return event.trigger
    return "timeout"


def _build_channel_from_config(workspace: Path) -> Optional[MessageChannel]:
    """Load the workspace config and ask the channel factory whether one
    can be built. Returns None on any error so the daemon still runs."""
    try:
        from hyperherd.config import load_config
        config = load_config(str(workspace))
        return build_channel(config, sweep_name=config.name, workspace=workspace)
    except Exception as e:
        log.warning("Could not build channel from config: %s", e)
        return None


async def _post_final_message(
    workspace: Path,
    *,
    channel: Optional[MessageChannel],
    ticks: int,
    total_cost_usd: float,
    halted: bool,
    halt_reason: Optional[str],
    stopped_by_signal: bool,
) -> None:
    """Post a 'daemon stopped' notification through the channel.
    Best-effort: failures are logged but don't propagate."""
    if stopped_by_signal:
        reason_text = "stopped by signal"
    elif halted:
        reason_text = f"halted — {halt_reason or 'no reason given'}"
    else:
        reason_text = "stopped (max-ticks reached)"

    body = (
        f"🛑 Daemon {reason_text}. "
        f"Ran {ticks} tick(s), ${total_cost_usd:.4f} total. "
        f"Won't post again unless you restart it."
    )

    if channel is None:
        log.info("No channel configured; skipping final-stop notification.")
        return
    try:
        await channel.post(body)
        log.info("Posted daemon-stopped notification via channel.")
    except Exception as e:
        log.warning("Failed to post daemon-stopped notification: %s", e)
