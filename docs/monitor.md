# Autonomous monitor

`herd monitor` runs a Claude-powered agent as a long-lived daemon that operates your sweep for you. Start it once, walk away, and direct it from Discord. **Recommended for any sweep longer than a few minutes.**

## What it does

- **Staged rollout.** Trial 0 first as a canary, then trials 1–2, then the rest. One bad launcher script doesn't waste the whole array.
- **Failure triage.** On host OOM the agent bumps `slurm.mem` and resubmits. On TIMEOUT it bumps `slurm.time`. CUDA OOM is notified, not retried (`slurm.mem` doesn't help GPU memory). Recurring identical exceptions halt the loop — likely a code bug worth your attention.
- **Two-way Discord channel** per sweep — auto-created on startup. Mention the bot for free-form questions; use slash commands for deterministic actions.
- **Wakes on activity.** Trial failures, completions, and Discord replies trigger a tick within seconds, not on a polling timer.
- **NaN/inf-only kill policy** for live trials. Anything else suspicious gets a warning, not an action.

## Setup (one-time)

Python 3.10+ is required for the daemon. Trial training code can still run on older Python — they're separate environments.

```bash
pip install 'hyperherd[monitor]'
```

Then create a Discord bot once by walking through [Discord setup](discord-setup.md). After that, you'll have:

- a `DISCORD_BOT_TOKEN` env var on the machine that runs the daemon
- a Discord server (guild) ID you'll add to each sweep:

```yaml
# hyperherd.yaml
discord:
  guild_id: "1234567890123456789"
```

## Run

From your sweep workspace:

```bash
herd monitor
```

The daemon connects to Discord, creates a `#<sweep-name>` channel inside your server, and posts there. If your workspace doesn't have a manifest yet (you just wrote the YAML), the daemon materializes one for you — no `herd run --dry-run` step required up front.

The daemon runs in the foreground. Wrap in `tmux` or `nohup` to outlive a logout:

```bash
tmux new -s monitor 'herd monitor'
# Ctrl-b d to detach; `tmux attach -t monitor` to come back.
```

## The setup interview

On a fresh sweep, the agent asks 3 short questions in the channel, one per message. Reply by `@`-mentioning the bot or replying to its message:

1. **What are you optimizing?** `maximize <metric>` / `minimize <metric>` / `none`.
2. **On failures: `remediate` or `notify`?** Whether to auto-bump mem/time and resubmit, or just alert you.
3. **Where do I read the metric?** `wandb` / `results-json` / `none`.

Reply `defaults` / `skip` to bypass the rest of the interview and start immediately with safe defaults (`metric=none`, `remediation=notify`).

If trials are already running when you launch the daemon, the agent recognizes the **hot-reload** case: it asks only questions 1 and 2, skips the canary, and jumps straight to operating the in-flight sweep. If every trial is already terminal, you get a **postmortem** instead — summary post, ask "rerun anything, or halt?", end.

## Operating from Discord

Two interaction styles in your sweep's channel.

### Slash commands — deterministic, no agent involvement

Type `/` and Discord shows the autocomplete list:

| Command | What it does |
|---|---|
| `/status` | Sweep totals + per-trial table |
| `/stats` | Per-trial timing + memory usage |
| `/params` | Sweep config: parameters, grid, every trial combo |
| `/info` | Daemon metadata: phase, uptime, tick count, total cost |
| `/plan` | The agent's `MONITOR_PLAN.md` contents |
| `/run <index>` | Submit (or resubmit) one trial |
| `/run_all` | Submit every ready trial |
| `/cancel <index>` | Cancel one trial |
| `/cancel_all` | Cancel every live trial |
| `/tail <index> [lines]` | Last N lines of a trial's stderr |
| `/stop` | Stop the daemon entirely |
| `/help` | List of these commands |

These run locally against the workspace and post the answer in the channel. No model call, no spend.

### Mentions and replies — free-form, wakes the agent

For anything that needs reasoning (cadence changes, remediation overrides, status questions), `@`-mention the bot or reply to one of its messages:

```
@HerdDog please bump mem to 16G and resubmit failed trials
@HerdDog why is idx 3 stuck?
HerdDog: pause until tomorrow
```

The bot reacts 👀 to confirm receipt; the agent processes on its next tick (usually within a second or two). While the agent is working, Discord shows "BotName is typing…" at the bottom of the channel.

The agent understands phrases like:

| You say | Effect |
|---|---|
| `pause` / `stop` / `halt` | Halts the loop. Daemon exits. |
| `bump mem to 16G` / `give it more time` | Edits `slurm.mem` / `slurm.time`, resubmits affected trials. |
| `set metric to val_acc` | Updates the plan's metric source. |
| `how's it going?` | Posts a fresh status summary. |
| anything unclear | Asks one specific question and waits. |

## What you'll see during a sweep

Each tick posts a one-line summary, even quiet ones — silence means the daemon crashed:

```
tick clean — 4 running, 5 completed, 0 failed. Next tick in 30 min.
bumped slurm.time 1h→1h30m after 1 TIMEOUT (idx 3); resubmitting. Next tick in 5 min.
```

Tick cadence is adaptive: 2–5 min during rollout and just after activity, 30–60 min steady-state. Every message ends with a `Next tick in <duration>` countdown.

## Failure triage

| SLURM cause | Cause type | If `remediate` | If `notify` |
|---|---|---|---|
| `OUT_OF_MEMORY` | Host RAM exceeded | Bump `slurm.mem` 50%, resubmit | Notify only |
| `RuntimeError: CUDA out of memory` | GPU VRAM | **Notify only**, even in `remediate` mode | Notify |
| `TIMEOUT` | Wall-clock too short | Bump `slurm.time` 50%, resubmit | Notify |
| `NODE_FAIL` / preemption | Infrastructure | Resubmit unchanged | Notify |
| Same exception ≥ 2 trials | Code/env bug | **Halt the loop** | Same |

Auto-bumps are capped at one per failure class per sweep. If a 50% mem bump still OOMs, that class switches to notify mode for the rest of the sweep.

## Stop, restart, override

`/stop` from Discord stops the daemon and posts a final summary. From the terminal: SIGINT (Ctrl-C in the foreground) or SIGTERM does the same.

To re-run the setup interview from scratch: `rm <workspace>/.hyperherd/MONITOR_PLAN.md` then `herd monitor`.

The daemon writes an audit log of every tool call to `<workspace>/.hyperherd/agent_log.jsonl` if you need to retrace what the agent did.

## Tweaking the agent's behavior

The agent's playbook lives in `hyperherd/monitor_agent/prompt/monitor.md` inside the package. If you want to change cadence, triage policy, or interview questions, edit it and restart the daemon.
