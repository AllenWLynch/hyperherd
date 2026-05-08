---
description: Author or edit a HyperHerd sweep configuration (hyperherd.yaml) and its companion launch.sh. Use when the user is creating, editing, or debugging a hyperherd.yaml file, scaffolding a sweep workspace, designing parameter conditions, or asking how to express a particular sweep shape (full grid, partial grid, one-at-a-time, conditional overrides).
---

# HyperHerd Configuration

HyperHerd runs SLURM hyperparameter sweeps from two files in a workspace directory:

- `hyperherd.yaml` — declarative sweep config (parameters, grid mode, SLURM resources, conditions, static overrides, optional Discord channel for the autonomous monitor)
- `launch.sh` — bash script that receives a `name=value` override string as **the first positional argument (`$1`)**. HyperHerd does not edit, template, or string-substitute into this file; it just invokes `bash launch.sh "<override-string>"` per trial. The script must read `$1` (typically `OVERRIDES="$1"`) and forward those tokens to its training command — there is no magic placeholder to splice into the file. The string format is whatever the launcher chooses to do with it — Hydra trainers consume it natively; non-Hydra trainers can `parse_overrides()` it (see `from hyperherd import parse_overrides`).

The full reference lives at `docs/configuration.md` in this repo. **Read it before writing a non-trivial config** — this skill is a checklist and a set of patterns, not a substitute for the doc.

## When to use this skill

- Creating a new `hyperherd.yaml` from scratch
- Editing an existing one (adding parameters, changing grid mode, adding/refactoring conditions)
- Debugging Pydantic validation errors from `herd run` / `herd test`
- Designing the launcher script

## Checklist before writing

1. **What's the sweep shape?**
   - Full Cartesian product → `grid: all`, no defaults required
   - Sweep a subset, hold others fixed → `grid: [param1, param2]`, defaults required for non-grid params
   - One-at-a-time around a baseline → omit `grid`, defaults required for **all** params
2. **For each parameter:**
   - Pick `discrete` (explicit `values:`) or `continuous` (`low`/`high`/`scale`/`steps`)
   - Optional `abbrev` (short, distinct — used in experiment names like `lr-0.001_opt-adam`); defaults to the parameter name when omitted. **Required** when the parameter name contains anything outside `[A-Za-z0-9._-]` (spaces, `/`, `+`, `~`, etc.) — the token ends up in a directory path.
   - For log-scale continuous, **`low` must be > 0**
   - If not in the grid (per rule 1), include a `default`
   - If a discrete value contains `/` (file paths, urls), provide a parallel `labels:` list (same length as `values`) with short, unique display tokens. The full value is still passed to Hydra; only the experiment name uses the label.
3. **Conditions** (formerly `constraints`, both keys still parse):
   - Need to filter out illegal combinations? Use `exclude`
   - Need to pin a parameter when another is set? Use `force`
   - Need to inject **non-parameter** overrides (e.g. `scheduler.warmup_steps`)? Use `set`
4. **Launcher:** does it need a container, conda env, or modules? Most launchers are 5–10 lines.
5. **Autonomous monitor** (optional but recommended for sweeps > a few minutes): set `discord.guild_id` so `herd monitor` has a channel to post in. See "Discord block" below.

## `when` matchers (the flexible part)

Each value in a condition's `when:` block can be:

| Form | Example | Meaning |
|------|---------|---------|
| scalar | `optimizer: sgd` | exact match |
| list | `optimizer: [sgd, momentum_sgd]` | OR — any element |
| operator map | `learning_rate: {gt: 0.01}` | `eq`, `ne`, `gt`, `ge`, `lt`, `le`, `in`, `not_in` |
| `expr` | `expr: "opt == 'adam' and lr > 0.01"` | free-form expression over params |

Multiple keys in one `when` are AND'd (including `expr` — it combines with structured matchers). Use a list to avoid duplicating a condition for several values of the same parameter.

### Expressions in `when` and `set`

`when.expr` and `set.<key>.expr` evaluate a small whitelisted expression language against the trial's parameters. Use it when structured matchers can't say what you mean (predicate over multiple params) or when an extra override should be **computed** from sweep values.

```yaml
parameters:
  y: {type: discrete, abbrev: y, values: [1, 3, 5]}

conditions:
  - name: x_from_y
    when:
      expr: "y >= 3"           # predicate over swept params
    set:
      x: {expr: "20 * y"}      # computed extra override
      scheduler.warmup: 100    # literals still work
```

Allowed: arithmetic (`+ - * / // % **`), comparisons (chained too: `0 < x < 10`), `and`/`or`/`not`, `in` / `not in` against tuple/list literals, `x if cond else y`, numbers/strings/booleans/`None`, names of swept parameters, and the bounding helpers `min(...)` / `max(...)` (e.g. `min(max(y, 0), 10)` to clamp).

Rejected at config-load time: any other function call, attribute access, subscripting, lambdas, comprehensions, f-strings, keyword arguments, any unknown name.

`+`/`~`/`++` prefixes are stripped in the expression namespace, so `+experiment` is referenced as `experiment` inside an expr. If two parameters would collide after stripping (e.g. both `foo` and `+foo`), config validation fails.

`set.<key>.expr` is evaluated **after** `force:`, so the expression sees forced values, not pre-force ones.

## Hydra-flavored `+key` / `~key` overrides

For Hydra trainers, you sometimes need to *add* a new key (`+experiment=foo`) or *delete* one (`~foo`). Put the prefix directly in the parameter name:

```yaml
parameters:
  +experiment:
    type: discrete
    abbrev: exp
    values: [small, large]
```

This emits `+experiment=small` in the override string. The `abbrev` keeps the `+` out of `experiment_name` (which would otherwise break Hydra). Same pattern works for `~foo` (delete) and `++foo` (force add/replace), and inside `force` / `set` keys.

## Override ordering (memorize this)

The override string is built left-to-right, last wins:

1. `experiment_name=<auto>`
2. swept parameter `name=value` pairs
3. `static_overrides`
4. condition `set` extras

So `set` **overrides** `static_overrides`. Use `static_overrides` for the always-on baseline; use `set` for conditional adjustments.

## Passing extra config to every trial

Use the top-level `static_overrides` list. Each entry is a literal `name=value` token appended to every trial.

```yaml
static_overrides:
  - "data.root=/scratch/imagenet"
    - "trainer.max_epochs=90"
    - "trainer.seed=42"
```

These are **not** swept and **not** validated against the parameter list — they're free-form `name=value` tokens passed verbatim to the launcher (which forwards them as-is to the trainer). Hydra trainers consume them as overrides; non-Hydra trainers can `parse_overrides()` them. Use this for fixed paths, seeds, dataset roots, logger config, etc. that differ from your trainer's defaults but don't change across the sweep.

If the value should depend on a swept parameter, use a condition with `set:` instead.

## Pre-written instructions to the monitor (`PROMPT.md`)

Drop a `PROMPT.md` file in the workspace root (next to `hyperherd.yaml`) to give `herd monitor` standing instructions for the sweep. The file is read fresh on every tick and rendered into the agent's user-message context, so edits take effect on the next wake-up — no daemon restart.

Use it for things you'd otherwise have to repeat in chat:

- The success metric and direction (`maximize val/acc`, `minimize val/loss`) — saves the interview round-trip on greenfield boots.
- Domain hints the agent can't infer from logs alone ("loss below 0.5 by step 1000 is the threshold for 'looks promising'").
- Remediation policy ("don't bump time past 12h", "prune anything diverging past step 5000").
- Pointers to dashboards or external tooling the agent should mention to the user.
- "Quiet hours" or cadence preferences ("don't ping me between 10pm-8am unless something fails").

Keep it short — it's read every tick and counts toward the prompt budget. A dozen lines is plenty.

`PROMPT.md` is **complementary** to `MONITOR_PLAN.md`: PROMPT.md is the user's stable preamble, MONITOR_PLAN.md is the agent's evolving working memory (it rewrites it each tick). Don't duplicate state into PROMPT.md.

## Discord block (autonomous monitor)

`herd monitor` operates the sweep through a per-sweep Discord channel — the agent posts status, the user replies via mention or slash commands. To enable it, add a `discord:` block with a server ID. The bot token comes from the `DISCORD_BOT_TOKEN` env var (don't put secrets in YAML), set up once per server per `docs/discord-setup.md`.

```yaml
discord:
  guild_id: "1234567890123456789"
```

Optional fields:

- `channel_id`: pin to a specific existing channel (skips auto-create)
- `channel_name`: override the sweep-derived channel name
- `dashboard_refresh_seconds`: how often the live-dashboard message updates (default 60, set 0 to disable)

If `discord.guild_id` is set but `DISCORD_BOT_TOKEN` is unset, `herd monitor` refuses to start with an error pointing at the missing env var. Don't paper this over by removing the discord block "just to make it run" — the user explicitly opted in.

## External MCP servers (vendor logger integrations)

If the user wants the autonomous monitor to talk to wandb, mlflow, ClickUp, or any other MCP-capable service, add an `mcp_servers:` list. Each entry is the SDK's external-MCP shape; tools appear to the agent as `mcp__<name>__*`.

```yaml
mcp_servers:
  - name: wandb
    command: uvx
    args:
      - --from
      - git+https://github.com/wandb/wandb-mcp-server
      - wandb-mcp-server
    env:
      WANDB_API_KEY: ${WANDB_API_KEY}
```

`${VAR}` references are expanded from the daemon's environment at startup. See `docs/mcp-integrations.md`. Most users don't need this — the agent's built-in `compute_metric` tool aggregates `log_result` streams from disk and covers ~95% of monitoring queries. Only add an MCP if the user explicitly asks for vendor-tool access.

## Streaming `log_result` for the monitor

If the user wants `herd monitor` to be able to prune diverging trials early, the trainer needs to call `log_result(name, value, step=N)` periodically (in addition to the bare `log_result(name, value)` for final summary metrics). The monitor's `compute_metric` tool reads these streams. See `docs/results.md` for framework-specific patterns (Lightning callback, HuggingFace TrainerCallback, plain PyTorch loop).

This is independent of the YAML — the YAML doesn't need a config knob for it. Just mention it when the user asks about pruning or early-stopping.

## Environment variables in the launcher

HyperHerd exports four environment variables before invoking `launch.sh`. Use them inside the launcher (or pass them through to the training script) to give every trial a stable identity.

| Variable | Value | Typical use |
|----------|-------|-------------|
| `HYPERHERD_WORKSPACE` | absolute path to the workspace directory | resolving paths under `.hyperherd/`, locating the manifest |
| `HYPERHERD_SWEEP_NAME` | the sweep's `name:` from yaml (shared across trials) | wandb project name, parent output directory |
| `HYPERHERD_TRIAL_ID` | the SLURM array task index (same as `$SLURM_ARRAY_TASK_ID`) | per-trial output subdir, `log_result()` keying |
| `HYPERHERD_TRIAL_NAME` | the auto-generated per-trial identifier (e.g. `lr-0.001_opt-adam_bs-64`) | wandb run name, output directory, checkpoint path |

The training code can read these directly:

```python
import os
trial_name = os.environ["HYPERHERD_TRIAL_NAME"]
output_dir = f"./outputs/{trial_name}"        # idempotent, stable across resubmissions
```

Or pass them through Hydra by referencing them in `static_overrides` (Hydra resolves `${env:VAR}` if you have OmegaConf env-var resolution enabled), but reading them directly in Python is usually simpler.

**Idempotency reminder:** because `herd run` resubmits failed/cancelled trials with the same array indices and parameters, your training script must use a *deterministic* output path (driven by `HYPERHERD_TRIAL_NAME` or `HYPERHERD_TRIAL_ID`) and resume from checkpoint on startup.

## Common patterns

### Full grid, simple

```yaml
name: my_sweep
launcher: ./launch.sh

slurm:
  partition: gpu
  time: "04:00:00"
  mem: "16G"
  cpus_per_task: 4
  gres: "gpu:1"

grid: all

parameters:
  learning_rate:
    abbrev: lr
    type: continuous
    low: 1e-4
    high: 1e-1
    scale: log
    steps: 4
  optimizer:
    abbrev: opt
    type: discrete
    values: [adam, sgd, adamw]
```

### Partial grid (sweep two, hold others)

```yaml
grid: [learning_rate, weight_decay]

parameters:
  learning_rate:
    abbrev: lr
    type: continuous
    low: 1e-4
    high: 1e-1
    scale: log
    steps: 4
    default: 0.001
  weight_decay:
    abbrev: wd
    type: continuous
    low: 0.0
    high: 0.01
    steps: 3
    default: 0.001
  optimizer:           # not in grid → needs default
    abbrev: opt
    type: discrete
    values: [adam, adamw]
    default: adamw
  batch_size:          # not in grid → needs default
    abbrev: bs
    type: discrete
    values: [64, 128, 256]
    default: 128
```

### One-at-a-time (omit grid; all params need defaults)

```yaml
# (no `grid:` field)

parameters:
  learning_rate: { abbrev: lr, type: continuous, low: 1e-4, high: 1e-1, scale: log, steps: 4, default: 0.001 }
  optimizer:     { abbrev: opt, type: discrete, values: [adam, sgd, adamw], default: adamw }
  batch_size:    { abbrev: bs, type: discrete, values: [64, 128, 256], default: 128 }
```

### Conditions

```yaml
conditions:
  # OR-match across multiple optimizer values in one rule.
  - name: sgd_family_no_high_lr
    when:
      optimizer: [sgd, momentum_sgd]
      learning_rate: {gt: 0.01}
    exclude:
      learning_rate: [0.05, 0.1]

  # Pin a parameter conditionally.
  - name: adamw_fixed_wd
    when:
      optimizer: adamw
    force:
      weight_decay: 0.01

  # Inject non-parameter overrides (the `set` field).
  # Keys are arbitrary name=value tokens; not validated as parameters.
  - name: adamw_warmup
    when:
      optimizer: adamw
    set:
      scheduler.type: cosine
      scheduler.warmup_steps: 1000
```

### Launcher (Apptainer)

`$1` is the override string HyperHerd passes in. `$OVERRIDES` is then expanded (unquoted, on purpose) into space-separated tokens for the training command — `python train.py lr=0.001 opt=adam`. Don't search for or modify any placeholder inside this file; HyperHerd never reads it.

```bash
#!/bin/bash
set -euo pipefail

OVERRIDES="$1"
CONTAINER="/path/to/container.sif"

apptainer exec --nv --bind "/scratch:/scratch" "$CONTAINER" \
    python train.py $OVERRIDES
```

### Launcher (conda)

```bash
#!/bin/bash
set -euo pipefail

OVERRIDES="$1"
source /opt/conda/etc/profile.d/conda.sh
conda activate myenv

python train.py $OVERRIDES
```

## Pitfalls

- **`abbrev` collisions** silently produce ambiguous experiment names. Keep them distinct.
- **Discrete values containing `/`** (paths, URLs) are rejected unless `labels:` is provided. Labels themselves may not contain `/` and must be unique.
- **Log scale with `low: 0`** is rejected — log requires `low > 0`.
- **Continuous `default` outside `[low, high]`** is rejected at parse time.
- **Discrete `default` not in `values`** is rejected at parse time.
- **`exclude` / `force` referencing an unknown parameter** is rejected at parse time. (`set` keys are *not* validated — they're free-form Hydra paths.)
- **Trial dedup on params only** — if a `force` collapses two combos to the same params, they merge. Extras (`set`) are deterministic from params, so they don't break this.
- **Idempotent training is required** for `herd run` resubmission to work. Use `$HYPERHERD_TRIAL_NAME` for a stable output dir and resume from checkpoint on startup.
- **Launcher path** in `hyperherd.yaml` is resolved relative to the config file's directory, not the cwd.

## Throttling concurrent jobs

Set `slurm.max_concurrent: N` to cap simultaneously running array tasks. HyperHerd appends `%N` to the SLURM array spec (e.g. `--array=0-49%5`). Use this when:

- The cluster has a per-user concurrent-job limit
- You want to avoid hammering a shared filesystem
- You're sanity-checking a sweep — start with `max_concurrent: 1` to serialize

Override per invocation with `herd run --max-concurrent N` (CLI wins over the config value).

## Workflow

After writing or editing the config, suggest the user run:

```bash
herd run <workspace> --dry-run    # validate config, preview trials & sbatch script
herd test <workspace>             # run trial 0 with --cfg job (Hydra config validation only)
herd local <workspace>            # run trial 0 end-to-end locally — full pre-flight, no SLURM
herd run <workspace>              # actually submit
```

`herd local` runs the launcher exactly like SLURM would, with `HYPERHERD_*` env vars set. It refuses any index that has ever been submitted to SLURM (would clobber outputs).

## Authoring discipline

- Don't invent fields. The full set is in `docs/configuration.md`. Top-level fields today are: `name`, `grid`, `launcher`, `slurm`, `parameters`, `conditions`, `static_overrides`, `discord` (autonomous monitor), `mcp_servers` (external MCP integrations). Anything else is wrong.
- Don't recommend `constraints:` for new configs — `conditions:` is the canonical key (the legacy alias is for backward compat only).
- Don't recommend a `watch:` block. It used to exist for the legacy webhook-poster (`herd watch`), which has been removed. The monitor's chat surface is now the `discord:` block.
- Don't recommend a top-level `hydra:` block. There used to be one (a wrapper around `static_overrides`), now it's just `static_overrides`. The legacy alias still parses but it's not the canonical form.
- Pick `abbrev` values that read well in filenames: `lr`, `opt`, `wd`, `bs`, `nl`, `do` (dropout), `sd` (seed), etc.
- When the user asks for a sweep, default to **partial grid** if they name 2–3 swept params and the rest are fixed; default to **full grid** only if they explicitly want a Cartesian product.
