"""Per-tick prompt assembly: system prompt (the skill markdown) plus a
human-friendly summary of the TickState that the agent reads as the user
turn."""

from pathlib import Path

from hyperherd.monitor_agent.state import TickState


def system_prompt() -> str:
    """The skill markdown — same on every tick (cached by the SDK)."""
    return (Path(__file__).parent / "monitor.md").read_text()


def render_state(state: TickState) -> str:
    """Render a TickState as the per-tick user-message turn.

    Compact-ish — the agent will call `read_state()` for the full dict if
    it needs structure. This rendering is the headline summary so the
    agent can decide what to do without burning through tokens parsing.
    """
    lines: list[str] = []
    lines.append(f"## Tick: {state.trigger}")
    lines.append("")
    lines.append(f"Sweep: **{state.sweep_name}** at `{state.workspace}`")
    lines.append("")

    totals = state.totals or {}
    if totals:
        order = ["ready", "submitted", "queued", "running",
                 "completed", "failed", "cancelled"]
        parts = [f"{totals.get(k, 0)} {k}" for k in order if totals.get(k)]
        parts.append(f"{totals.get('total', len(state.trials))} total")
        lines.append("Totals: " + ", ".join(parts))
    else:
        lines.append("Totals: (no trials yet)")
    lines.append("")

    if state.newly_failed:
        lines.append(f"**Newly failed since last tick: {len(state.newly_failed)}**")
        for f in state.newly_failed[:5]:
            tail = " | ".join((f.stderr_tail or [])[-2:])
            lines.append(f"  - idx {f.index} ({f.experiment_name or '?'}) "
                         f"slurm_state={f.slurm_state or '?'} — {tail or '(no stderr captured)'}")
        if len(state.newly_failed) > 5:
            lines.append(f"  ...and {len(state.newly_failed) - 5} more — "
                         f"call read_state() for the full list with full stderr tails")
        lines.append("")

    if state.newly_completed:
        lines.append(f"Newly completed since last tick: indices {state.newly_completed}")
        lines.append("")

    if state.newly_pruned:
        lines.append(f"Newly pruned since last tick: indices {state.newly_pruned}")
        lines.append("")

    if state.inbox:
        lines.append(f"**Inbox: {len(state.inbox)} new message(s) from the user**")
        for m in state.inbox:
            text = m.text.strip().replace("\n", " ")
            if len(text) > 200:
                text = text[:197] + "..."
            lines.append(f"  - [{m.timestamp}] {m.author}: {text}")
        lines.append("")

    if state.chat_history:
        lines.append("Recent conversation (excludes per-tick heartbeats):")
        for c in state.chat_history:
            text = c.text.strip().replace("\n", " ")
            if len(text) > 200:
                text = text[:197] + "..."
            who = c.author or c.role
            lines.append(f"  - [{c.timestamp}] {c.role}/{who}: {text}")
        lines.append("")

    lines.append("---")
    if state.plan:
        lines.append("Plan (from MONITOR_PLAN.md):")
        lines.append("")
        lines.append(state.plan)
    else:
        lines.append("**No MONITOR_PLAN.md exists yet — this is the first tick.** "
                     "Either run the setup interview (if a user message in the inbox is "
                     "asking you to configure things) or, for an autonomous boot, write a "
                     "minimal plan with sensible defaults: notify-only remediation, no "
                     "wandb integration yet, Phase: not-started. Then proceed with the "
                     "phased rollout.")

    lines.append("")
    lines.append("Decide, take 1-3 actions via tools, post one `tick_summary` "
                 "for the heartbeat (and a separate `msg` if there's user "
                 "conversation to address), call `schedule_next` with the "
                 "cadence-table delay, and end your turn.")

    return "\n".join(lines)
