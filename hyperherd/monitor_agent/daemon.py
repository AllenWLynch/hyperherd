"""Daemon mode: schedule-driven loop on top of run_tick().

The agent picks the next-tick delay via its `schedule_next` tool; the
daemon sleeps that long and runs another tick. SIGINT/SIGTERM trigger
clean exit at the next iteration boundary.

If a `MessageChannel` is configured (Discord today), the daemon also:

- Starts/stops it alongside the loop.
- Routes inbound user messages into `.hyperherd/inbox.jsonl` and wakes
  the loop early so the next tick fires with `trigger="user_message"`.
- Routes the agent's outbound `msg` calls through the channel
  (`tools.msg` reads it via `_CTX["channel"]`).
- Posts the final-stop notification through the channel as well, so the
  user gets it in the same place as everything else.

Phase 3 will add the SLURM-event source to the same wake-up mechanism.
"""

import asyncio
import logging
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hyperherd.monitor_agent import tick as tick_mod
from hyperherd.monitor_agent.channel import (
    MessageChannel, build_channel, make_inbox_writer,
)

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
    post_final: bool = True,
) -> DaemonResult:
    """Run ticks in a loop until the agent halts or a signal arrives.

    First tick fires immediately with `trigger="boot"`. Subsequent ticks
    wait for `result.next_delay_seconds`, interruptible by SIGINT/SIGTERM
    or (when configured) an inbound user message.
    """
    workspace = Path(workspace).resolve()
    if run_tick is None:
        run_tick = tick_mod.run_tick

    # Build the chat channel from workspace config if the caller didn't
    # inject one. None = no channel; the daemon runs without inbox/post.
    if channel is None:
        channel = _build_channel_from_config(workspace)

    shutdown = asyncio.Event()
    inbox_signal = asyncio.Event()

    def _on_signal(signum):
        log.info("Received signal %s, shutting down after current tick.", signum)
        shutdown.set()

    def _on_inbox_write():
        inbox_signal.set()

    loop = asyncio.get_running_loop()
    installed_handlers = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _on_signal, sig)
            installed_handlers.append(sig)
        except (NotImplementedError, RuntimeError):
            pass

    # Wire inbound: events written to inbox.jsonl, then wake the loop.
    if channel is not None:
        writer = make_inbox_writer(workspace, on_write=_on_inbox_write)
        channel.set_inbound_handler(writer)
        try:
            await channel.start()
            log.info("Channel '%s' connected.", channel.name)
        except Exception as e:
            log.error("Failed to start channel '%s': %s — continuing without it.",
                      channel.name, e)
            channel = None

    ticks = 0
    total_cost = 0.0
    trigger = "boot"
    halted = False
    halt_reason: Optional[str] = None

    try:
        while not shutdown.is_set():
            log.info("Tick %d starting (trigger=%s)", ticks + 1, trigger)
            result = await run_tick(workspace, trigger=trigger, channel=channel)
            ticks += 1
            total_cost += result.cost_usd
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

            delay = result.next_delay_seconds or 1800
            log.info("Sleeping up to %ds until next tick.", delay)

            # Race three signals: timeout (normal scheduled tick), shutdown
            # (signal received), inbox_signal (user message). Whichever
            # fires first dictates the next iteration.
            inbox_signal.clear()
            woke_for = await _wait_first(
                shutdown, inbox_signal, timeout=delay,
            )
            if woke_for == "shutdown":
                break
            elif woke_for == "inbox":
                trigger = "user_message"
            else:  # timeout
                trigger = "scheduled"

    finally:
        for sig in installed_handlers:
            try:
                loop.remove_signal_handler(sig)
            except (NotImplementedError, RuntimeError):
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


async def _wait_first(
    shutdown: asyncio.Event,
    inbox: asyncio.Event,
    *,
    timeout: float,
) -> str:
    """Block until shutdown, inbox, or timeout fires. Returns 'shutdown',
    'inbox', or 'timeout'."""
    shutdown_task = asyncio.create_task(shutdown.wait(), name="wait-shutdown")
    inbox_task = asyncio.create_task(inbox.wait(), name="wait-inbox")
    try:
        done, pending = await asyncio.wait(
            {shutdown_task, inbox_task},
            timeout=timeout,
            return_when=asyncio.FIRST_COMPLETED,
        )
    finally:
        for t in (shutdown_task, inbox_task):
            if not t.done():
                t.cancel()
        # Allow cancellations to propagate so we don't leak warnings.
        for t in (shutdown_task, inbox_task):
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

    if shutdown_task in done:
        return "shutdown"
    if inbox_task in done:
        return "inbox"
    return "timeout"


def _build_channel_from_config(workspace: Path) -> Optional[MessageChannel]:
    """Load the workspace config and ask the channel factory whether one
    can be built. Returns None on any error so the daemon still runs."""
    try:
        from hyperherd.config import load_config
        config = load_config(str(workspace))
        return build_channel(config, sweep_name=config.name)
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
    """Post a 'daemon stopped' notification. Routes through the channel
    if configured, else falls back to the watch webhook. Best-effort."""
    if stopped_by_signal:
        reason_text = "stopped by signal"
    elif halted:
        reason_text = f"halted — {halt_reason or 'no reason given'}"
    else:
        reason_text = "stopped (max-ticks reached)"

    body = (
        f"Herd dog: daemon {reason_text}. "
        f"Ran {ticks} tick(s), ${total_cost_usd:.4f} total. "
        f"Won't post again unless you restart it."
    )

    if channel is not None:
        try:
            await channel.post(body)
            log.info("Posted daemon-stopped notification via channel.")
            return
        except Exception as e:
            log.warning("Channel post failed; falling back to webhook: %s", e)

    try:
        from hyperherd import watch
        from hyperherd.config import load_config

        config = load_config(str(workspace))
        webhook = config.watch.webhook
        fmt = config.watch.format
        if not webhook:
            webhook, _ = watch.resolve_default_webhook(config.workspace, config.name)
            fmt = "ntfy"

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, lambda: watch.post_message(webhook, fmt, body, config.name)
        )
        log.info("Posted daemon-stopped notification via webhook.")
    except Exception as e:
        log.warning("Failed to post daemon-stopped notification: %s", e)
