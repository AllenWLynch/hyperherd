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
    # Order presented to the user. `pruned` is distinct from `failed`
    # (algorithmic kill vs SLURM error) and `cancelled` (user kill);
    # surfacing all three keeps the running+terminal sum visibly equal
    # to the trial total.
    order = ["ready", "submitted", "queued", "running",
             "completed", "failed", "pruned", "cancelled"]
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
    agent_enabled: bool = True,
    passive_refresh_seconds: int = 60,
    force_token_conflict: bool = False,
) -> DaemonResult:
    """Run ticks in a loop until the agent halts or a signal arrives.

    First tick fires immediately with `trigger="boot"`. Subsequent ticks
    wait for `result.next_delay_seconds` or until an event source pushes
    a wake-up onto the queue.

    With ``agent_enabled=False`` the agent loop is skipped entirely — no
    model calls, no token spend. The daemon still runs the SLURM poll,
    heartbeat, and Discord channel (so slash commands keep working) and
    refreshes ``last-snapshot.json`` every ``passive_refresh_seconds``
    so the live dashboard stays current. Use this for very long sweeps
    where you want the chat surface but don't want the agent burning
    tokens at every tick.
    """
    workspace = Path(workspace).resolve()
    if run_tick is None:
        run_tick = tick_mod.run_tick

    # Build the chat channel from workspace config if the caller didn't
    # inject one. None = no channel; the daemon runs without inbox/post.
    if channel is None:
        channel = _build_channel_from_config(
            workspace, force_token_conflict=force_token_conflict,
        )

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
        # Health is one of: "running" (green), "degraded" (yellow — at least
        # one consecutive tick failure but under the halt threshold),
        # "halted" (red), "stopping" (a /stop or signal is mid-shutdown),
        # "passive" (white — agent loop disabled, daemon just keeps the
        # chat surface and dashboard alive without spending tokens).
        # The dashboard reads this to render a one-char status emoji.
        "health": "passive" if not agent_enabled else "running",
        "consecutive_failures": 0,
        "agent_enabled": agent_enabled,
    }

    def _on_signal(signum):
        log.info("Received signal %s, shutting down after current tick.", signum)
        runtime_stats["health"] = "stopping"
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
        def _stop_via_channel():
            runtime_stats["health"] = "stopping"
            shutdown.set()
        channel.set_stop_handler(_stop_via_channel)
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
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 5
    FAILURE_COOLDOWN_SECONDS = 60

    if not agent_enabled:
        log.info(
            "Passive mode: agent loop disabled. Refreshing snapshot every %ds.",
            passive_refresh_seconds,
        )
        if channel is not None:
            try:
                await channel.post(
                    "⚪ Monitor running in **passive mode** — slash commands, "
                    "dashboard, and event posts are live, but the agent loop "
                    "is disabled (no token spend). Restart without `--no-agent` "
                    "to re-enable autonomous monitoring."
                )
            except Exception as e:
                log.warning("Could not post passive-mode banner: %s", e)

    try:
        # ── Passive mode ────────────────────────────────────────────────
        # Skip the agent entirely. Periodically refresh last-snapshot.json
        # so the dashboard / heartbeat / SlurmPoll baselines have current
        # data; the channel keeps slash commands working on its own.
        while not shutdown.is_set() and not agent_enabled:
            try:
                from hyperherd.monitor_agent import state as _state
                _state.refresh_snapshot(workspace)
            except Exception as e:
                log.warning("Passive snapshot refresh failed: %s", e)

            from datetime import datetime as _dt, timezone as _tz, timedelta as _td
            runtime_stats["next_tick_at_iso"] = (
                _dt.now(_tz.utc) + _td(seconds=passive_refresh_seconds)
            ).isoformat(timespec="seconds")
            try:
                await asyncio.wait_for(
                    shutdown.wait(), timeout=passive_refresh_seconds,
                )
            except asyncio.TimeoutError:
                pass
            runtime_stats.pop("next_tick_at_iso", None)
            # Drain non-agent wake events so the queue doesn't grow
            # unbounded (in passive mode nothing consumes them).
            _drain(event_q)

        while not shutdown.is_set():
            log.info("Tick %d starting (trigger=%s)", ticks + 1, trigger)
            # Show the platform's "thinking" indicator (Discord typing dots,
            # etc.) for the duration of the tick. No-op if no channel.
            thinking_cm = (
                channel.thinking() if channel is not None
                else _noop_thinking_cm()
            )
            try:
                async with thinking_cm:
                    result = await run_tick(
                        workspace, trigger=trigger, channel=channel,
                    )
                consecutive_failures = 0
                runtime_stats["consecutive_failures"] = 0
                runtime_stats["health"] = "running"
            except Exception as e:
                # Tick blew up (SDK error, max_turns exceeded, network blip,
                # etc.). Don't kill the daemon — log, alert the user,
                # cooldown briefly, retry. Halt only after several
                # consecutive failures, since that suggests a real problem
                # we can't recover from automatically.
                consecutive_failures += 1
                runtime_stats["consecutive_failures"] = consecutive_failures
                runtime_stats["health"] = "degraded"
                log.exception(
                    "Tick %d raised (consecutive failures: %d): %s",
                    ticks + 1, consecutive_failures, e,
                )
                if channel is not None:
                    try:
                        await channel.post(
                            f"⚠️ Tick failed: {type(e).__name__}: {e}. "
                            f"Retrying in {FAILURE_COOLDOWN_SECONDS}s "
                            f"(consecutive failures: {consecutive_failures}/"
                            f"{MAX_CONSECUTIVE_FAILURES})."
                        )
                    except Exception:
                        pass
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    halted = True
                    halt_reason = (
                        f"halting after {MAX_CONSECUTIVE_FAILURES} "
                        f"consecutive tick failures — check daemon log"
                    )
                    runtime_stats["health"] = "halted"
                    break
                # Cooldown — but interruptible by shutdown signal.
                try:
                    await asyncio.wait_for(
                        shutdown.wait(),
                        timeout=FAILURE_COOLDOWN_SECONDS,
                    )
                    break  # shutdown fired during cooldown
                except asyncio.TimeoutError:
                    pass
                trigger = "scheduled"
                continue
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

            from datetime import datetime as _dt, timezone as _tz, timedelta as _td
            runtime_stats["next_tick_at_iso"] = (
                _dt.now(_tz.utc) + _td(seconds=delay)
            ).isoformat(timespec="seconds")
            outcome = await _wait_next_event(event_q, shutdown, timeout=delay)
            runtime_stats.pop("next_tick_at_iso", None)
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


def _build_channel_from_config(
    workspace: Path,
    *,
    force_token_conflict: bool = False,
) -> Optional[MessageChannel]:
    """Load the workspace config and ask the channel factory whether one
    can be built. Returns None on any error so the daemon still runs."""
    try:
        from hyperherd.config import load_config
        config = load_config(str(workspace))
        return build_channel(
            config, sweep_name=config.name, workspace=workspace,
            force_token_conflict=force_token_conflict,
        )
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
