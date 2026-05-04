# HyperHerd

!!! warning "Pre-release / actively developed"
    HyperHerd is in soft launch — the YAML schema, CLI flags, and Python API may change without notice between versions. Pin to an exact version (`hyperherd==X.Y.Z`) if you build on top of it, and expect breaking changes until a tagged 1.0.

**Hyperparameter sweeps on SLURM, run by an autonomous agent.** Declare your search in YAML, hand over a one-line launcher script, and walk away — [`herd monitor`](monitor.md) submits trials in stages, diagnoses failures, retries when SLURM can fix the problem, and posts only when it can't.

!!! tip "Want to skip ahead?"
    The repo ships [a complete MNIST sweep](example.md) you can clone and run as-is. PyTorch Lightning + Hydra trainer, 11 trials, all four condition forms in use. Two minutes from `git clone` to trials on the queue.

!!! example "Have Claude Code set you up"
    Open Claude Code in your project directory and paste the block below — Claude will walk you through install, config authoring, validation, and (if you want it) the autonomous monitor end-to-end. Full guide at [Set up with Claude Code](setup-help.md).

    ```
    Help me set up HyperHerd. Read the setup guide at
    https://raw.githubusercontent.com/AllenWLynch/hyperherd/main/docs/setup-help.md
    and follow it — start with the Phase 0 interview, then drive the rest.
    ```

## What you write

Two files in a workspace directory:

- **`hyperherd.yaml`** — your sweep declaratively: parameters, grid mode, SLURM resources, conditions.
- **`launch.sh`** — a one-line bash script that receives a `name=value` override string as `$1` and runs your training command in whatever environment you need (container, conda, uv, modules).

## What you get

- **One-command sweeps.** No sbatch boilerplate, no manual resubmits — `herd run` generates and submits the array, tracks state, and resumes failed/pending trials on rerun.
- **An agent that actually operates the sweep.** [`herd monitor`](monitor.md) ramps trials in stages, diagnoses failures, bumps memory or wall-time when that's the right fix, and pings you only when it can't.
- **Two-way Discord control.** A dedicated channel per sweep with deterministic slash commands (`/status`, `/run`, `/cancel`, `/tail`, …) and free-form mentions for the agent.
- **Edit your sweep mid-run.** Bump a parameter range or add a value; the next `herd run` appends new trials without touching the ones already running.
- **Configs you don't have to memorize.** The bundled [Claude Code skill](claude-skill.md) writes `hyperherd.yaml` for you from a one-paragraph description.
- **An audit trail.** Every trial's parameters, status, and logged metrics live in `.hyperherd/` and come out as TSV via `herd res` or JSON via `herd snapshot`.

[Hydra](https://hydra.cc/) is the recommended trainer harness — its CLI consumes `name=value` overrides natively, so the string passes through unchanged — but the launcher is free-form bash, so parse the arguments however you want.

## Scope

HyperHerd is opinionated. It assumes:

1. SLURM job arrays as the dispatch mechanism.
2. `name=value` overrides as the parameter contract.
3. A bash launcher script as the integration point.

## Where to next

<div class="grid cards" markdown>

- **[Try the MNIST example](example.md)** — the fastest way to see HyperHerd work. Clone, install, run.
- **[Autonomous monitor](monitor.md)** — start here if you want the agent runner. Setup, Discord channel, slash commands, failure triage.
- **[Discord setup](discord-setup.md)** — one-time bot creation walkthrough.
- **[Getting started](getting-started.md)** — install, scaffold, run your first sweep.
- **[Sweep config reference](configuration.md)** — every field in `hyperherd.yaml`.
- **[Command reference](commands.md)** — every `herd` subcommand.
- **[Claude Code skill](claude-skill.md)** — generate configs by asking Claude.

</div>
