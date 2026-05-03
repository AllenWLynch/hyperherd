"""Daemon mode: schedule-driven loop on top of run_tick().

Phase-2 implementation per PLAN.md. The agent picks the next-tick delay via
its `schedule_next` tool; the daemon sleeps that long and runs another tick.
SIGINT/SIGTERM trigger a clean exit at the next iteration boundary.

Phase 3/4 will add SLURM-event and Discord event sources; this loop will
become a fan-in over an asyncio.Queue at that point. For now, one source.
"""

import asyncio
import logging
import signal
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hyperherd.monitor_agent import tick as tick_mod

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
    run_tick=None,        # injectable for tests
    post_final: bool = True,
) -> DaemonResult:
    """Run ticks in a loop until the agent halts or a signal arrives.

    First tick fires immediately with `trigger="boot"`. Each subsequent tick
    waits for `result.next_delay_seconds`, interruptible by SIGINT/SIGTERM
    (which exit cleanly after the in-flight tick).
    """
    workspace = Path(workspace).resolve()
    if run_tick is None:
        run_tick = tick_mod.run_tick

    shutdown = asyncio.Event()

    def _on_signal(signum):
        log.info("Received signal %s, shutting down after current tick.", signum)
        shutdown.set()

    loop = asyncio.get_running_loop()
    installed_handlers = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _on_signal, sig)
            installed_handlers.append(sig)
        except (NotImplementedError, RuntimeError):
            # Windows / non-main-thread event loops can't install handlers.
            # The cluster targets neither, but tests may run in threads.
            pass

    ticks = 0
    total_cost = 0.0
    trigger = "boot"
    halted = False
    halt_reason: Optional[str] = None

    try:
        while not shutdown.is_set():
            log.info("Tick %d starting (trigger=%s)", ticks + 1, trigger)
            result = await run_tick(workspace, trigger=trigger)
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
            log.info("Sleeping %ds until next tick.", delay)
            try:
                await asyncio.wait_for(shutdown.wait(), timeout=delay)
                # Exited the wait before timeout — shutdown was set.
                break
            except asyncio.TimeoutError:
                pass

            trigger = "scheduled"
    finally:
        for sig in installed_handlers:
            try:
                loop.remove_signal_handler(sig)
            except (NotImplementedError, RuntimeError):
                pass

    stopped_by_signal = shutdown.is_set() and not halted

    # Daemon-side guarantee: always post a final notification on exit. The
    # agent's skill already tells it to post a summary before halting, but
    # if it forgets (or the model errors out), the user still hears that
    # the daemon is no longer running. A double-post is preferable to
    # silent shutdown.
    if post_final:
        await _post_final_message(
            workspace,
            ticks=ticks,
            total_cost_usd=total_cost,
            halted=halted,
            halt_reason=halt_reason,
            stopped_by_signal=stopped_by_signal,
        )

    return DaemonResult(
        ticks=ticks,
        total_cost_usd=total_cost,
        halted=halted,
        halt_reason=halt_reason,
        stopped_by_signal=stopped_by_signal,
    )


async def _post_final_message(
    workspace: Path,
    *,
    ticks: int,
    total_cost_usd: float,
    halted: bool,
    halt_reason: Optional[str],
    stopped_by_signal: bool,
) -> None:
    """Post a 'daemon stopped' notification through the same webhook the
    agent uses. Best-effort: failures are logged but don't propagate."""
    try:
        from hyperherd import watch
        from hyperherd.config import load_config

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
        log.info("Posted daemon-stopped notification.")
    except Exception as e:
        log.warning("Failed to post daemon-stopped notification: %s", e)
