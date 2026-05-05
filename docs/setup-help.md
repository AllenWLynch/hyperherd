# Set up HyperHerd with Claude Code

> Paste this document (or just the URL) into Claude Code and Claude will walk you through HyperHerd setup end-to-end — installing the package, scaffolding a workspace, authoring `hyperherd.yaml` + `launch.sh`, validating, and (optionally) bringing up the autonomous monitor with Discord.

The rest of this page is written in second person to **Claude**. If you're a human reading this, the [Getting started](getting-started.md) page is the friendlier read.

---

## Your role, Claude

You're helping the user get HyperHerd running on their cluster. They may be brand-new to the package, may already have a trainer they want to sweep, and may or may not want the autonomous monitor. Drive the conversation — don't dump everything at once.

**Do not skip Phase 0.** The user's environment dictates almost every later choice (Python version, container runtime, trainer harness, monitor opt-in). Ask all the questions in one message, wait for answers.

Defer to the user's existing setup. If they already have a `train.py` and a working environment, your job is to wire HyperHerd around it — don't rewrite their trainer.

When in doubt about HyperHerd specifics, fetch the relevant page from `https://allenwlynch.github.io/hyperherd/` (e.g. `getting-started/`, `configuration/`, `monitor/`, `discord-setup/`, `launcher/`) rather than guessing.

---

## Phase 0 — Setup interview (ask all of these in one message)

1. **What's the cluster?** SLURM partition name, typical resources for one trial (GPUs, memory, walltime). Is `sbatch`/`sacct`/`squeue` already on `$PATH`?
2. **What's the trainer?** Path to the existing training script (e.g. `python train.py`), and which arg style: Hydra-style `name=value` overrides, `--flags`, or something else.
3. **What's the environment?** Container (Apptainer/Singularity, Docker), conda/mamba env, `module load`, plain `pip`, or `uv`. You'll need to know how the user normally invokes their trainer on a compute node.
4. **What parameters do they want to sweep?** Names + types (discrete values vs continuous range), and whether it's a full grid, a partial grid (sweep some, hold others fixed), or one-at-a-time around a baseline.
5. **Do they want the autonomous monitor?** That's the agent daemon that ramps trials, diagnoses failures, and chats over Discord. It needs Python ≥ 3.10 and an Anthropic API key (or a Claude Code subscription). If they say no, skip Phase 5.

---

## Phase 1 — Install

Pick the right install line based on Phase 0.5:

```bash
# Base CLI only — Python ≥ 3.8
pip install hyperherd

# With the autonomous monitor — Python ≥ 3.10
pip install 'hyperherd[monitor]'
```

After install, verify:

```bash
herd --help
```

Then install the bundled Claude Code skill so the user (and you) get deeper config help in future sessions:

```bash
herd install-skill
```

This drops a skill into `~/.claude/skills/hyperherd-config/`. Tell the user that future `hyperherd.yaml` editing sessions will pick it up automatically.

---

## Phase 2 — Scaffold the workspace

```bash
herd init <workspace_name>
```

This creates two files: `hyperherd.yaml` (declarative sweep) and `launch.sh` (bash entry point). **Open both and edit them with the user.**

### `hyperherd.yaml` essentials

A minimal config:

```yaml
name: my_sweep                    # used as the SLURM job name and Discord channel name
launcher: ./launch.sh             # path is resolved relative to this file
parameters:
  learning_rate:
    type: continuous
    abbrev: lr
    low: 1e-5
    high: 1e-2
    scale: log
    steps: 5
  optimizer:
    type: discrete
    abbrev: opt
    values: [adam, sgd]
grid: all                          # full Cartesian product of the above
slurm:
  partition: gpu
  time: "04:00:00"
  mem: 16G
  cpus_per_task: 4
  gres: "gpu:1"
```

Key decisions to walk through:

- **Grid mode.** `grid: all` (Cartesian), `grid: [param1, param2]` (sweep these, fix others at their `default:`), or omit `grid` for one-at-a-time.
- **Discrete vs continuous.** Continuous needs `low`/`high`/`scale`/`steps`. Log scale requires `low > 0`.
- **`abbrev`.** Short, distinct token used in trial names like `lr-0.001_opt-adam`. **Required** when the parameter name has anything outside `[A-Za-z0-9._-]`.
- **Conditions.** If parameters interact (e.g. `optimizer=adam` should never use `momentum`), use `conditions:` — fetch `configuration/` and `conditions/` if needed.

For the full reference, fetch `https://allenwlynch.github.io/hyperherd/configuration/`.

### `launch.sh` contract

The script is invoked as `bash launch.sh "<overrides>"`. `$1` is a space-separated `name=value` string. Available env vars inside the script:

- `$HYPERHERD_WORKSPACE` — absolute workspace path
- `$HYPERHERD_SWEEP_NAME` — `name:` from `hyperherd.yaml` (shared across trials)
- `$HYPERHERD_TRIAL_ID` — array task index
- `$HYPERHERD_TRIAL_NAME` — auto-generated per-trial id (e.g. `lr-0.001_opt-adam`)

For Hydra trainers, the launcher is a one-liner because Hydra consumes the override string natively:

```bash
#!/bin/bash
set -euo pipefail
OVERRIDES="$1"
apptainer exec --nv container.sif python train.py $OVERRIDES
```

For non-Hydra trainers, parse the string. Either with the `parse_overrides()` helper:

```python
from hyperherd import parse_overrides
parsed = parse_overrides(sys.argv[1])  # → {"learning_rate": "0.001", "optimizer": "adam"}
```

…or with bash word-splitting + the user's CLI conventions. Don't invent a parser if one already exists in their trainer.

For container/conda/module patterns, fetch `https://allenwlynch.github.io/hyperherd/launcher/`.

---

## Phase 3 — Validate before submitting

```bash
# List every trial the YAML produces (status-agnostic):
herd ls <workspace>

# Submission preview — pending indices + the sbatch script:
herd run <workspace> --dry-run

# Run a single trial locally (no SLURM) to sanity-check the launcher + trainer:
herd test <workspace> 0

# Hydra users: print the resolved config for trial 0 without training:
herd test <workspace> 0 --cfg-job
```

`herd ls` answers "what's in the sweep?"; `herd run --dry-run` answers "what would `herd run` do right now?". Read the dry-run output carefully — it prints the exact bash that will run on the compute node. If anything looks off (wrong container path, missing module load, wrong override key), fix before submitting.

---

## Phase 4 — Submit

```bash
herd run <workspace>
```

Then track:

```bash
herd status         # one-shot status table
herd tail 3         # last 20 lines of trial 3's stdout/stderr
herd stats          # sacct accounting once trials finish
herd res            # TSV of params + logged metrics
```

If a trial fails, fix the issue and re-run `herd run` — it's idempotent and only resubmits `ready`/`failed`/`cancelled` trials.

To log per-trial metrics from the trainer, add this to the training code:

```python
from hyperherd import log_result
log_result("val_accuracy", 0.94, step=epoch)
log_result("final_loss", 0.12)
```

For PyTorch Lightning users, the bundled logger forwards every `pl_module.log()` call automatically:

```python
from hyperherd.integrations.lightning import HyperHerdLogger
trainer = pl.Trainer(logger=[wandb_logger, HyperHerdLogger()])
```

---

## Phase 5 — Autonomous monitor (optional)

Skip this section if the user said no in Phase 0.5.

The monitor is a long-running daemon that:
- Watches the sweep, posts state-change events to Discord
- Runs the staged-rollout / failure-triage / pruning policy via a Claude agent loop
- Accepts slash commands (`/status`, `/tail`, `/run`, `/cancel`, `/prune`, `/metrics`, `/stop`) and free-form `@`-mentions

### One-time Discord bot setup

This is a multi-step walkthrough — fetch `https://allenwlynch.github.io/hyperherd/discord-setup/` and run it interactively. The summary:

1. Create an application + bot at https://discord.com/developers/applications
2. Enable the **MESSAGE CONTENT** privileged gateway intent
3. Generate an invite URL with scopes `bot` + `applications.commands` and the permissions `View Channels`, `Send Messages`, `Read Message History`, `Manage Channels`
4. Invite the bot to the user's server
5. Copy the bot token → `DISCORD_BOT_TOKEN` env var
6. Right-click the server → Copy Server ID → `discord.guild_id` in `hyperherd.yaml`

If the user plans to run multiple sweeps in parallel, create one bot per workspace — Discord allows only one gateway connection per token, so two daemons sharing a token will kick each other off. The daemon's startup preflight detects this via a per-channel heartbeat marker and refuses to start. The marker is cleared on clean shutdown so a normal restart isn't blocked; pass `--force-discord` if a previous daemon was killed uncleanly and the stale heartbeat hasn't aged out yet (~18 min).

### Anthropic credentials

Either:

- **API console billing:** `export ANTHROPIC_API_KEY=sk-ant-...`
- **Claude Code subscription:** the user runs `claude /login` once; the daemon picks up the OAuth token automatically

### `hyperherd.yaml` additions

```yaml
discord:
  guild_id: 1234567890123456789

# Optional — external MCP servers the agent should have access to
mcp_servers:
  - name: wandb
    command: npx
    args: ["-y", "@wandb/mcp-server"]
    env:
      WANDB_API_KEY: ${WANDB_API_KEY}
```

### Per-workspace `.env`

Drop secrets in `<workspace>/.env` so the daemon picks them up without leaking them to git:

```bash
DISCORD_BOT_TOKEN=...
ANTHROPIC_API_KEY=...
WANDB_API_KEY=...
```

The daemon auto-loads `<workspace>/.env` at startup and only fills in keys not already set in the environment.

### Run

```bash
herd monitor <workspace>
```

The daemon connects to Discord, creates a channel for the sweep, runs a short setup interview (metric, remediation policy, metric source), then operates the sweep autonomously. Wrap in `tmux`/`screen` to survive disconnects.

If the user wants the dashboard + slash-command surface but not the agent-driven cost, suggest:

```bash
herd monitor <workspace> --no-agent     # passive mode — no token spend
```

For the full picture, fetch `https://allenwlynch.github.io/hyperherd/monitor/`.

---

## Common pitfalls

- **`HYPERHERD_TRIAL_NAME` confusion.** This is the auto-generated *per-trial* id; `HYPERHERD_SWEEP_NAME` is the *shared* sweep name. Older code may reference `HYPERHERD_EXPERIMENT_NAME` — that's a legacy alias for `TRIAL_NAME`, still set by HyperHerd but don't write new code against it.
- **Idempotent training.** Trials may be resubmitted (after SLURM-side failures or `scancel`). Use `$HYPERHERD_TRIAL_NAME` for a stable output dir, resume from checkpoint on startup, and don't fail on existing output dirs.
- **Compute nodes without Python.** HyperHerd doesn't need Python on the compute node — the per-trial values are baked into the sbatch `case` statement at submission time. Only `bash` is required outside the container.
- **`abbrev` collisions.** Two parameters with the same `abbrev` will produce ambiguous trial names; the validator catches this but the error can be cryptic — pick distinct short tokens.
- **`launcher:` path.** Resolved relative to the `hyperherd.yaml` file's directory, not the cwd. Use `./launch.sh` and keep them in the same workspace dir.

---

## When something breaks

- **`herd run --dry-run`** is the first thing to try — it does the same validation `herd run` does without submitting.
- **`herd test <workspace> 0`** runs trial 0 locally (no SLURM) so the launcher / trainer / overrides can be debugged in isolation.
- **`.hyperherd/logs/<idx>.out` and `.err`** capture each trial's stdout/stderr.
- **`herd snapshot <workspace>`** prints a JSON bundle of the whole sweep state — useful for debugging or for handing the user a diff.

For anything beyond this, point the user at the relevant docs page or fetch it directly:

| Topic | URL |
|-------|-----|
| Sweep config reference | `https://allenwlynch.github.io/hyperherd/configuration/` |
| Conditions | `https://allenwlynch.github.io/hyperherd/conditions/` |
| Launcher patterns | `https://allenwlynch.github.io/hyperherd/launcher/` |
| Command reference | `https://allenwlynch.github.io/hyperherd/commands/` |
| Autonomous monitor | `https://allenwlynch.github.io/hyperherd/monitor/` |
| Discord setup | `https://allenwlynch.github.io/hyperherd/discord-setup/` |
| Workspace layout | `https://allenwlynch.github.io/hyperherd/workspace/` |
| Results & logging | `https://allenwlynch.github.io/hyperherd/results/` |
