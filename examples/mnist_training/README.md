# MNIST Training Example

A complete example training an MNIST digit classifier using PyTorch Lightning, Hydra, and HyperHerd. Demonstrates idempotent training with checkpoint resume and result logging.

## What it does

`train.py` is a Hydra-configured PyTorch Lightning training script that:

- Trains a simple 3-layer MLP on MNIST (28x28 -> hidden -> hidden -> 10)
- **Checkpoints deterministically** to `outputs/<experiment_name>/` using the HyperHerd experiment name
- **Resumes from checkpoint** automatically if `last.ckpt` exists (idempotent)
- **Tests using the best checkpoint** (by validation accuracy) after training
- **Logs final metrics** via `hyperherd.log_result()` for collection with `herd res`
- **Supports GPU or CPU** via Lightning's `accelerator=auto`

## Prerequisites

```bash
pip install pytorch-lightning torchvision hydra-core
pip install -e /path/to/hyperherd  # for log_result support
```

## Hyperparameters

The sweep is configured as a partial grid over `learning_rate` and `optimizer`, with other parameters held at defaults:

| Parameter       | Values                    | Default |
|----------------|---------------------------|---------|
| `learning_rate` | 1e-4 to 1e-1 (4 log steps) | 0.001   |
| `optimizer`     | adam, sgd, adamw          | adam    |
| `batch_size`    | 32, 64, 128               | 64      |
| `hidden_dim`    | 64, 128, 256              | 128     |
| `dropout`       | 0.0, 0.2, 0.5            | 0.2     |

With `grid: [learning_rate, optimizer]`, this produces 4 x 3 = 12 trials (minus 1 from the SGD high-LR constraint = 11 trials).

The `conditions:` block in `hyperherd.yaml` shows off four constraint forms in one place:

1. **`when.expr`** — programmatic predicate (`optimizer == 'sgd' and learning_rate > 0.05`).
2. **`set` (literals)** — inject Hydra overrides not declared as parameters (`weight_decay`, `gradient_clip_val`).
3. **`set.expr`** — compute an extra from swept params (`weight_decay = 0.1 * learning_rate`).
4. **`force`** — pin a swept parameter under a condition (`dropout = 0` for tiny LRs).

The trainer reads `weight_decay` and `gradient_clip_val` so the per-trial effects are visible in `herd run -n` and in the actual training logs.

## Running locally (single trial)

```bash
cd examples/mnist_training

# Run one trial directly:
python train.py learning_rate=0.001 optimizer=adam batch_size=64 \
    hidden_dim=128 dropout=0.0 experiment_name=test_run max_epochs=3

# Re-run the same command — resumes from checkpoint:
python train.py learning_rate=0.001 optimizer=adam batch_size=64 \
    hidden_dim=128 dropout=0.0 experiment_name=test_run max_epochs=3
```

## Running with HyperHerd

```bash
cd examples/mnist_training

# Validate Hydra config:
herd test .

# Preview the sweep:
herd run . --dry-run

# Submit to SLURM:
herd run .

# Monitor progress:
herd status .

# SLURM accounting (runtime, peak/avg memory) for every trial — or one:
herd stats .
herd stats . 3

# Tail a trial's log:
herd tail . 0

# Cancel one trial, or every live trial:
herd stop . 3
herd stop . --all

# Re-run to resubmit failed trials (they resume from checkpoint).
# If you edit hyperherd.yaml between runs, herd reconciles the manifest:
# new values append as fresh trials, removed ones drop (or, if any already
# ran/completed, --force keeps them as orphans).
herd run .

# Collect results:
herd res .

# Clean up:
herd clean . --all
```

## Idempotency

The training script is idempotent:

1. **Deterministic output path**: `outputs/<experiment_name>/` is derived from the parameter abbreviations (e.g. `outputs/lr-0.001_opt-adam_bs-64_hd-128_do-0/`).
2. **Checkpoint resume**: If `last.ckpt` exists in the output dir, Lightning resumes training from that checkpoint.
3. **Reproducible splits**: The train/val split uses a fixed seed (`torch.Generator().manual_seed(42)`).

If a trial fails mid-training and HyperHerd resubmits it, training picks up from the last saved epoch.

## Results

After all trials complete, `herd res .` prints:

```
trial_id  experiment_name              learning_rate  optimizer  test_acc  test_loss  best_val_acc
0         lr-0.0001_opt-adam_bs-64...  0.0001         adam       0.978     0.071      0.981
1         lr-0.001_opt-adam_bs-64...   0.001          adam       0.983     0.052      0.985
...
```
