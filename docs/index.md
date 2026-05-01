# HyperHerd

**Launch and monitor hyperparameter optimization job arrays on SLURM.**

HyperHerd takes a YAML configuration file describing your hyperparameter search space and submits it as a SLURM job array. Each array task receives a string of `name=value` overrides for one parameter combination. A user-provided launcher script handles container setup, environment modules, and invokes your training command however it likes.

[Hydra](https://hydra.cc/) is the recommended trainer harness — its CLI consumes `name=value` overrides natively, so the string passes through unchanged — but you're free to parse the arguments however you want. The only Hydra-specific feature in HyperHerd is `herd test`, which appends `--cfg job` to validate a Hydra config without running training.

## What you write

Two files in a workspace directory:

- **`hyperherd.yaml`** — declarative sweep config: parameters, grid mode, SLURM resources, conditions, static Hydra overrides.
- **`launch.sh`** — a bash script that receives a `name=value` override string as `$1` and runs your training command in whatever environment you need (container, conda, modules, etc.).

## What HyperHerd does

- Generates the full set of trials from your sweep definition (full grid, partial grid, or one-at-a-time).
- Applies declarative `conditions` to filter or modify combinations.
- Writes a manifest of `(trial_index, params, experiment_name)` records.
- Generates and submits a SLURM array job; each task resolves its parameters from the manifest and invokes your launcher.
- Tracks state across runs: re-running the same workspace resubmits only failed/pending trials.
- Lets you **edit `hyperherd.yaml` mid-sweep** — new trials are appended on the next `herd run`; completed trials keep their results.
- Reads back per-trial metrics logged via `from hyperherd import log_result`.
- Optional [`herd watch`](commands.md#herd-watch) daemon posts trial state changes to Slack / Discord / ntfy / any webhook.

## Scope

HyperHerd is opinionated. It assumes:

1. SLURM job arrays as the dispatch mechanism.
2. `name=value` overrides as the parameter contract.
3. A bash launcher script as the integration point.

Hydra is recommended (its CLI consumes the overrides directly), but the launcher is a free-form bash script — parse the override string however you want.

## Where to next

<div class="grid cards" markdown>

- **[Getting started](getting-started.md)** — install, scaffold, run your first sweep.
- **[Sweep config reference](configuration.md)** — every field in `hyperherd.yaml`.
- **[Command reference](commands.md)** — every `herd` subcommand.
- **[Claude Code skill](claude-skill.md)** — generate configs by asking Claude.

</div>
