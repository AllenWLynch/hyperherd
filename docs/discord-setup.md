# Discord setup for the monitor daemon

The monitor daemon can use Discord as both its outbound notification channel
and its inbound message inbox. Per sweep, it auto-creates a text channel
named after the sweep (e.g. `mnist-sweep`) inside a server you provision.
You and the agent talk in that channel; the agent reads your replies on its
next tick.

This page is the one-time setup. After it's done you only need to set the
guild ID in each sweep's `hyperherd.yaml`.

## 1. Create the bot

1. Go to <https://discord.com/developers/applications> and click
   **New Application**. Pick a name (it's your call — "HyperHerd" works).
2. In the left nav, open **Bot**. Click **Reset Token**, then **Copy** the
   token. You'll paste it into your shell environment in step 4.
3. On the same page, scroll to **Privileged Gateway Intents** and turn on
   **MESSAGE CONTENT INTENT**. The bot needs this to read your replies.
4. Save.

## 2. Invite the bot to your server

1. In the left nav, open **OAuth2 → URL Generator**.
2. Under **Scopes**, check **`bot`** and **`applications.commands`**
   (the latter enables `/status`, `/stop`, etc.).
3. Under **Bot Permissions**, check:
    - View Channels
    - Send Messages
    - Read Message History
    - Add Reactions  *(for the 👀 ack on inbound messages)*
    - Manage Channels  *(for auto-creating a channel per sweep)*
    - Manage Messages  *(for pinning the live dashboard message — optional; without it the dashboard still updates in place, just isn't pinned to the top)*
4. Copy the generated URL, paste it in your browser, pick the server, and
   authorize.

## 3. Get the server (guild) ID

1. In Discord, open **User Settings → Advanced** and turn on **Developer
   Mode**.
2. Right-click the server name in the sidebar → **Copy Server ID**.

That's the value you'll put in `hyperherd.yaml` as `discord.guild_id`.

## 4. Set up the daemon's environment

Set the bot token in the environment where the daemon runs (don't put it in
YAML):

```bash
export DISCORD_BOT_TOKEN='your-bot-token-here'
```

Persisting it in your shell rc, a `.env` file, or your job-scheduler
environment is fine — wherever you'd put any other secret.

## 5. Configure the sweep

In the workspace's `hyperherd.yaml`:

```yaml
discord:
  guild_id: "1234567890123456789"
```

Optional overrides:

```yaml
discord:
  guild_id: "1234567890123456789"
  channel_name: "my-experiment"     # otherwise derived from the sweep name
  channel_id: "9876543210987654321" # pin to an existing channel; skips auto-create
```

## 6. Run

```bash
herd monitor my-sweep/
```

The daemon connects, finds-or-creates the channel, posts a startup-style
tick from the agent, and starts listening for your replies in that channel.

## What you'll see

- A new text channel appears (e.g. `#mnist-sweep`) the first time you run.
- A pinned 📊 **live dashboard** message appears at the top of the channel
  and updates in place every minute with current totals, daemon
  metadata, and the per-trial table — so you don't have to keep typing
  `/status`. Disable with `discord.dashboard_refresh_seconds: 0` in
  `hyperherd.yaml`. If the bot lacks `Manage Messages` permission, the
  dashboard still updates in place, it just isn't pinned to the top.
- Each tick, the agent posts a summary message. You can reply at any
  time — the bot adds a 👀 reaction to your message immediately so you
  know it was received, and the daemon wakes to run a `user_message`
  tick within a second or two. While the agent is working on a tick,
  Discord shows a "Bot is typing…" indicator at the bottom of the
  channel.
- Every post wears an emoji prefix so you can tell at a glance who's
  speaking:
    - 🐕 the agent (decisions, replies, tick summaries — every model-
      driven post)
    - ▶️ trial started running, ✅ trial completed, ⚠️ trial failed
      (free, posted by the SLURM poller the moment a transition is
      detected)
    - 🐾 periodic daemon heartbeat with current totals (every 5 min)
    - 🛑 final daemon-stopped notification
- Three ways to address the bot:
    1. **Discord-resolved mention** — type `@` and pick the bot from the
       autocomplete dropdown. Renders as a clickable link.
    2. **Reply** to one of the bot's messages.
    3. **Plain-text prefix** — `@HerdDog ...`, `HerdDog: ...`, or
       `HerdDog, ...` (case-insensitive). Useful on mobile or when the
       autocomplete doesn't fire.
  Plain channel chatter that doesn't address the bot is ignored.
- When the agent halts (sweep complete, recurring failure, or you said so),
  the daemon posts a final "stopped" message and exits.

## Slash commands (deterministic — no agent in the loop)

These commands run locally against the workspace and post the answer in
the channel. No model call, no rate-limit window consumed.

| Command | What it does |
|---|---|
| `/status` | Sweep totals + per-trial table |
| `/stats` | Per-trial timing + memory usage |
| `/params` | Parameter grid: spec + every trial combo |
| `/info` | Daemon metadata: phase, uptime, tick count, total cost |
| `/plan` | The agent's `MONITOR_PLAN.md` contents |
| `/run <index>` | Submit (or resubmit) one trial |
| `/run_all` | Submit every ready trial |
| `/cancel <index>` | Cancel one trial |
| `/cancel_all` | Cancel every live trial |
| `/tail <index> [lines]` | Last N lines of a trial's stderr (default 20) |
| `/stop` | Stop the monitor daemon entirely |
| `/help` | List of these commands |

Discord auto-completes the names and validates parameter types. They're
guild-scoped, so they appear instantly when the daemon connects.

## Common phrases the agent understands

When you @-mention the bot, it wakes the agent for a model-driven reply.

| You say | Effect |
|---|---|
| `pause` / `stop` / `halt` | Agent halts; daemon exits. (Or use `/stop`.) |
| `how's it going?` | Posts a fresh status summary on the next tick. (Or use `/status`.) |
| `bump mem to 32G` | Bumps `slurm.mem` and resubmits affected trials. |
| `give it more time` | Bumps `slurm.time`. |
| `set metric to val_acc` / `watch wandb run XYZ` | Updates the plan's metric source. |
| anything else | Agent acknowledges and asks for clarification if needed. |

## Troubleshooting

**"Bot can't see guild"** — the bot wasn't invited to the server. Re-run
the OAuth invite from step 2.

**"Failed to create channel"** — the bot lacks the *Manage Channels*
permission. Re-invite with the right boxes checked, or pre-create the
channel manually and pin it via `discord.channel_id`.

**Daemon prints "DISCORD_BOT_TOKEN is not in the environment"** — the
env var didn't propagate to the daemon's shell. Source it in the same
shell you launch `herd monitor` from.

**No reply lands in the inbox** — the bot doesn't have the *MESSAGE
CONTENT INTENT* enabled in the Developer Portal (step 1.3). Discord
silently strips message content for bots without it.
