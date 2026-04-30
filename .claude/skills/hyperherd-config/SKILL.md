---
description: Author or edit a HyperHerd sweep configuration (hyperherd.yaml) and its companion launch.sh. Use when the user is creating, editing, or debugging a hyperherd.yaml file, scaffolding a sweep workspace, designing parameter conditions, or asking how to express a particular sweep shape (full grid, partial grid, one-at-a-time, conditional overrides).
---

# HyperHerd Configuration

HyperHerd runs SLURM hyperparameter sweeps from two files in a workspace directory:

- `hyperherd.yaml` — declarative sweep config (parameters, grid mode, SLURM resources, conditions, static Hydra overrides)
- `launch.sh` — bash script that receives a Hydra override string as `$1` and runs the training command

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
   - Provide an `abbrev` (short, distinct — used in experiment names like `lr-0.001_opt-adam`)
   - For log-scale continuous, **`low` must be > 0**
   - If not in the grid (per rule 1), include a `default`
   - If a discrete value contains `/` (file paths, urls), provide a parallel `labels:` list (same length as `values`) with short, unique display tokens. The full value is still passed to Hydra; only the experiment name uses the label.
3. **Conditions** (formerly `constraints`, both keys still parse):
   - Need to filter out illegal combinations? Use `exclude`
   - Need to pin a parameter when another is set? Use `force`
   - Need to inject **non-parameter** Hydra overrides (e.g. `scheduler.warmup_steps`)? Use `set`
4. **Launcher:** does it need a container, conda env, or modules? Most launchers are 5–10 lines.

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

## Hydra `+key` / `~key` overrides

To inject a Hydra override that *adds* a new key (`+experiment=foo`) or *removes* one (`~foo`), put the prefix directly in the parameter name:

```yaml
parameters:
  +experiment:
    type: discrete
    abbrev: exp
    values: [small, large]
```

This emits `+experiment=small` in the override string. The `abbrev` keeps the `+` out of `experiment_name` (which would otherwise break Hydra). Same pattern works for `~foo` (delete) and `++foo` (force add/replace), and inside `force` / `set` keys.

## Override ordering (memorize this)

The Hydra override string is built left-to-right, last wins:

1. `experiment_name=<auto>`
2. swept parameter `name=value` pairs
3. `hydra.static_overrides`
4. condition `set` extras

So `set` **overrides** `static_overrides`. Use `static_overrides` for the always-on baseline; use `set` for conditional adjustments.

## Passing extra Hydra config to every trial

Use the top-level `hydra` block. Each entry is a literal Hydra override string appended to every trial.

```yaml
hydra:
  static_overrides:
    - "data.root=/scratch/imagenet"
    - "trainer.max_epochs=90"
    - "trainer.seed=42"
```

These are **not** swept and **not** validated against the parameter list — they're free-form Hydra paths, same as `set` keys. Use this for fixed paths, seeds, dataset roots, logger config, etc. that differ from your Hydra defaults but don't change across the sweep.

If the value should depend on a swept parameter, use a condition with `set:` instead.

## Environment variables in the launcher

HyperHerd exports three environment variables before invoking `launch.sh`. Use them inside the launcher (or pass them through to the training script) to give every trial a stable identity.

| Variable | Value | Typical use |
|----------|-------|-------------|
| `HYPERHERD_WORKSPACE` | absolute path to the workspace directory | resolving paths under `.hyperherd/`, locating the manifest |
| `HYPERHERD_TRIAL_ID` | the SLURM array task index (same as `$SLURM_ARRAY_TASK_ID`) | per-trial output subdir, `log_result()` keying |
| `HYPERHERD_EXPERIMENT_NAME` | the auto-generated name (e.g. `lr-0.001_opt-adam_bs-64`) | wandb run name, output directory, checkpoint path |

The training code can read these directly:

```python
import os
exp_name = os.environ["HYPERHERD_EXPERIMENT_NAME"]
output_dir = f"./outputs/{exp_name}"          # idempotent, stable across resubmissions
```

Or pass them through Hydra by referencing them in `static_overrides` (Hydra resolves `${env:VAR}` if you have OmegaConf env-var resolution enabled), but reading them directly in Python is usually simpler.

**Idempotency reminder:** because `herd run` resubmits failed/cancelled trials with the same array indices and parameters, your training script must use a *deterministic* output path (driven by `HYPERHERD_EXPERIMENT_NAME` or `HYPERHERD_TRIAL_ID`) and resume from checkpoint on startup.

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

  # Inject non-parameter Hydra overrides (the `set` field).
  # Keys are arbitrary Hydra paths; not validated as parameters.
  - name: adamw_warmup
    when:
      optimizer: adamw
    set:
      scheduler.type: cosine
      scheduler.warmup_steps: 1000
```

### Launcher (Apptainer)

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
- **Idempotent training is required** for `herd run` resubmission to work. Use `$HYPERHERD_EXPERIMENT_NAME` for a stable output dir and resume from checkpoint on startup.
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

- Don't invent fields. The full set is in `docs/configuration.md`. If the user asks for something outside the supported expr language (function calls, attribute access, etc.), say so and suggest restructuring with `if/else` or a structured matcher.
- Don't recommend `constraints:` for new configs — `conditions:` is the canonical key (the legacy alias is for backward compat only).
- Pick `abbrev` values that read well in filenames: `lr`, `opt`, `wd`, `bs`, `nl`, `do` (dropout), `sd` (seed), etc.
- When the user asks for a sweep, default to **partial grid** if they name 2–3 swept params and the rest are fixed; default to **full grid** only if they explicitly want a Cartesian product.
