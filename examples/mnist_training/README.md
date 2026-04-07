# MNIST Training Example

A complete example training an MNIST digit classifier using PyTorch Lightning, Hydra, and HyperWhip. Demonstrates idempotent training with checkpoint resume and result logging.

## What it does

`train.py` is a Hydra-configured PyTorch Lightning training script that:

- Trains a simple 3-layer MLP on MNIST (28x28 -> hidden -> hidden -> 10)
- **Checkpoints deterministically** to `outputs/<experiment_name>/` using the HyperWhip experiment name
- **Resumes from checkpoint** automatically if `last.ckpt` exists (idempotent)
- **Tests using the best checkpoint** (by validation accuracy) after training
- **Logs final metrics** via `hyperwhip.log_result()` for collection with `whip res`
- **Supports GPU or CPU** via Lightning's `accelerator=auto`

## Prerequisites

```bash
pip install pytorch-lightning torchvision hydra-core
pip install -e /path/to/hyperwhip  # for log_result support
```

## Hyperparameters

The sweep is configured as a partial grid over `learning_rate` and `optimizer`, with other parameters held at defaults:

| Parameter       | Values                    | Default |
|----------------|---------------------------|---------|
| `learning_rate` | 1e-4 to 1e-1 (4 log steps) | 0.001   |
| `optimizer`     | adam, sgd, adamw          | adam    |
| `batch_size`    | 32, 64, 128               | 64      |
| `hidden_dim`    | 64, 128, 256              | 128     |
| `dropout`       | 0.0, 0.2, 0.5            | 0.0     |

With `grid: [learning_rate, optimizer]`, this produces 4 x 3 = 12 trials (minus 1 from the SGD high-LR constraint = 11 trials).

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

## Running with HyperWhip

```bash
cd examples/mnist_training

# Validate Hydra config:
whip test .

# Preview the sweep:
whip run . --dry-run

# Submit to SLURM:
whip run .

# Monitor progress:
whip status .

# Tail a trial's log:
whip tail . 0

# Re-run to resubmit failed trials (they resume from checkpoint):
whip run .

# Collect results:
whip res .

# Clean up:
whip clean . --all
```

## Idempotency

The training script is idempotent:

1. **Deterministic output path**: `outputs/<experiment_name>/` is derived from the parameter abbreviations (e.g. `outputs/lr-0.001_opt-adam_bs-64_hd-128_do-0/`).
2. **Checkpoint resume**: If `last.ckpt` exists in the output dir, Lightning resumes training from that checkpoint.
3. **Reproducible splits**: The train/val split uses a fixed seed (`torch.Generator().manual_seed(42)`).

If a trial fails mid-training and HyperWhip resubmits it, training picks up from the last saved epoch.

## Results

After all trials complete, `whip res .` prints:

```
trial_id  experiment_name              learning_rate  optimizer  test_acc  test_loss  best_val_acc
0         lr-0.0001_opt-adam_bs-64...  0.0001         adam       0.978     0.071      0.981
1         lr-0.001_opt-adam_bs-64...   0.001          adam       0.983     0.052      0.985
...
```
