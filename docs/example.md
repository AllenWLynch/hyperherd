# Try the bundled MNIST example

The repo ships a complete end-to-end sweep at `examples/mnist_training/`. PyTorch Lightning + Hydra trainer, 11 trials sweeping learning rate × optimizer, conditional rules that prune unstable combinations and inject extras for `adamw` and `sgd`. Clone, install, run — you can have it on the queue in two minutes.

## What the example does

A 4-step learning rate (1e-4 → 1e-1, log-scaled) crossed with three optimizers (`adam`, `sgd`, `adamw`) — 12 combinations, reduced to 11 after a condition drops `sgd` at `lr > 0.05` (it's unstable). Each trial trains a small MLP on MNIST for 50 epochs and logs:

- **Final** `test_acc` / `test_loss` / `best_val_acc` via plain `log_result(name, value)` — these show up in `herd res`.
- **Streaming** `train_loss` / `train_acc` every training step via the `HyperHerdLogger` Lightning logger (good for `/plot` curves).
- **Epoch-keyed** `val_loss` / `val_acc`, logged once per epoch from `on_validation_epoch_end` via `log_result(name, value, step=self.current_epoch)`. The integer-per-epoch step is what [successive halving](#successive-halving) compares trials on, and what the monitor's `compute_metric` reads for pruning decisions.

The config showcases all four condition forms — programmatic predicate, literal-extra injection, expression-computed extra, structured force — plus a `successive_halving:` block, so you can see what the YAML looks like beyond the bare minimum.

## Run it

```bash
# Install HyperHerd from PyPI:
pip install hyperherd

# Clone the repo to get the example workspace (the example files
# aren't shipped in the wheel — only the CLI is):
git clone https://github.com/AllenWLynch/hyperherd.git
cd hyperherd

# Install the trial-side training deps (PyTorch + Lightning + Hydra):
pip install -r examples/mnist_training/requirements.txt

# Edit examples/mnist_training/hyperherd.yaml: change `slurm.partition`
# to one your cluster has. Defaults to `short`.

# List every trial the YAML produces — no SLURM, status-agnostic:
herd ls examples/mnist_training/

# Preview the submission plan (sbatch script + pending indices):
herd run examples/mnist_training/ --dry-run

# Submit:
herd run examples/mnist_training/

# Watch the run by hand:
herd status examples/mnist_training/
herd stats examples/mnist_training/
herd tail examples/mnist_training/ 0
```

When trials finish:

```bash
herd res examples/mnist_training/
```

prints a TSV of every trial's parameters and final metrics.

## Or hand it to the monitor

If you've done the [Discord setup](discord-setup.md) once, point the daemon at the example workspace:

```bash
pip install '.[monitor]'
herd monitor examples/mnist_training/
```

The daemon will:

1. Auto-init the manifest if you haven't run `herd run` yet.
2. Connect to your Discord server, create a `#mnist-sweep` channel.
3. Walk you through the 3-question setup interview in that channel.
4. Run the canary (trial 0), then phase 2 (trials 1–2), then the rest.
5. Diagnose failures, post heartbeats, summarize the result when it's done.

You drive it from the channel — `/status`, `/run 5`, `/stop 3`, `/tail 7`, or `@HerdDog please bump mem to 4G`.

## Successive halving

The example's `hyperherd.yaml` includes a `successive_halving:` block (objective `val_loss`, rungs at epochs `[5, 10, 20, 40]`). Run it by hand against an in-flight sweep:

```bash
# Preview what SH would do right now (no changes):
herd sh examples/mnist_training/ --dry-run

# Apply: prune the provably-worst trials, pause the undecidable, promote the rest:
herd sh examples/mnist_training/
```

Run it on a loop (`watch -n 300 herd sh examples/mnist_training/`) or just let the monitor's `run_sh` tool handle it. Because the trainer raises `hyperherd.TrialPruned` at its next logged epoch and `train.py` catches it at the top level, a pruned/paused trial stops cleanly (its `last.ckpt` is preserved, so a resumed trial picks up where it left off). Paused trials show up `paused` in `herd status`; SH may resume them automatically once enough peers reach the same rung.

## Things to play with

Once you have it running, edit `examples/mnist_training/hyperherd.yaml` and re-run — HyperHerd reconciles:

- Add `0.5` to `dropout.values` → 4 new trials get appended on the next `herd run`, existing ones stay.
- Change `grid` from `[learning_rate, optimizer]` to `all` → full Cartesian (with constraint pruning) — many more trials.
- Add a `static_overrides: ["max_epochs=10"]` → faster iteration for debugging. (Lower `successive_halving.budget` to match.)
- Tweak `successive_halving.min_steps` / `eta` → prune earlier or more aggressively, then watch `herd sh --dry-run`.
- Edit a condition to invert it (`> 0.05` → `< 0.05`) → see the constraint engine prune a different chunk of the grid.

Anything you change persists on subsequent runs — no manifest regeneration needed.

## What's in the workspace after a run

```
examples/mnist_training/
├── hyperherd.yaml             # the sweep config
├── launch.sh                  # launcher: invokes train.py with overrides
├── train.py                   # PyTorch Lightning trainer (calls log_result)
├── requirements.txt           # trial-side deps
├── data/                      # MNIST download
└── .hyperherd/                # HyperHerd state (you can `herd clean -a` this)
    ├── manifest.json          # the sweep's source of truth
    ├── job.sbatch             # generated SLURM script
    ├── results/<idx>.json     # per-trial metric files (test_acc, test_loss, …)
    └── logs/<idx>.{out,err}   # per-trial stdout / stderr from SLURM
```

If you also ran `herd monitor`, you'll see `MONITOR_PLAN.md`, `chat-history.jsonl`, `agent_log.jsonl`, and a few snapshot files alongside.
