# Results & logging

HyperHerd ships a lightweight API for logging metrics from inside your training code. Two modes — call it once at the end of training for a summary metric, or repeatedly during training to give the [autonomous monitor](monitor.md) signal it can use to make pruning / alert decisions.

## Logging from training code

```python
from hyperherd import log_result

# Streaming mode — call as often as you want during training. Each call
# appends one (step, value) point to a per-metric JSONL file. The
# autonomous monitor's `compute_metric` tool aggregates these to decide
# whether a trial is diverging.
for step, batch in enumerate(train_loader):
    loss = train_step(batch)
    if step % 100 == 0:
        log_result("val_loss", val_loss(), step=step)

# Final-summary mode — call once at the end of training/evaluation.
# Writes to a flat per-trial JSON file consumed by `herd res`.
log_result("test_accuracy", 0.95)
log_result("final_loss", 0.12)
log_result("epochs_completed", 50)
```

Final-summary mode writes to `.hyperherd/results/<trial_id>.json` (a flat `{name: value}` dict, last write wins per name). Streaming mode writes to `.hyperherd/results/<trial_id>/stream/<name>.jsonl` (one line per call, append-only).

Both resolve the workspace from `HYPERHERD_WORKSPACE` and `HYPERHERD_TRIAL_ID` environment variables, which `herd run` and `herd test` set automatically.

- Values must be JSON-serializable (numbers, strings, booleans, lists, dicts).
- Streaming mode is opt-in — only worth wiring up if you're using `herd monitor` and want the agent making metric-based decisions. If you're not, just call `log_result(name, value)` without `step` at the end.
- The two modes are independent: a value logged via streaming does NOT also appear in the final-summary JSON, and vice versa.

## Streaming from common frameworks

The `step` argument can be any monotonic counter — global step, epoch, batch index. Pick whatever your trainer exposes; the agent treats it opaquely. Patterns by framework:

**PyTorch Lightning** — pass a callback to your `Trainer`:

```python
class HyperHerdStreamCallback(pl.Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        if os.environ.get("HYPERHERD_TRIAL_ID") is None:
            return
        from hyperherd import log_result
        step = int(trainer.global_step)
        for name in ("val_loss", "val_acc"):
            v = trainer.callback_metrics.get(name)
            if v is not None:
                log_result(name, v.item() if hasattr(v, "item") else v, step=step)
```

The MNIST [example](example.md) ships this callback wired in.

**HuggingFace Transformers** — subclass `TrainerCallback`:

```python
from transformers import TrainerCallback

class HyperHerdStreamCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if os.environ.get("HYPERHERD_TRIAL_ID") is None:
            return
        from hyperherd import log_result
        for k, v in (metrics or {}).items():
            log_result(k, v, step=state.global_step)
```

Pass `callbacks=[HyperHerdStreamCallback()]` to `Trainer(...)`.

**Plain PyTorch** — call directly from your eval loop:

```python
from hyperherd import log_result

for step, batch in enumerate(train_loader):
    ...
    if step % 500 == 0:
        val_loss = evaluate(model, val_loader)
        log_result("val_loss", val_loss, step=step)
```

**Frameworks that hide the loop (Megatron, DeepSpeed, etc.)** — these typically expose `iteration` or `global_step` in their hook callbacks. Hook the equivalent of "end of validation" or "end of N steps" and call `log_result` from there. If the framework gives you no callback at all, write to `.hyperherd/results/<trial_id>/stream/<name>.jsonl` directly — it's just a plain JSONL file with one `{"step": int, "value": ..., "ts": float}` object per line.

## Viewing results

```bash
herd res
```

Prints a TSV with every trial's parameters and logged metrics:

```
trial_id  experiment_name          learning_rate  optimizer  test_acc  test_loss
0         lr-0.01_opt-adam_bs-32   0.01           adam       0.92      0.31
1         lr-0.01_opt-adam_bs-64   0.01           adam       0.87      0.45
2         lr-0.01_opt-sgd_bs-32    0.01           sgd
3         lr-0.01_opt-sgd_bs-64    0.01           sgd
```

Trials without results show empty cells. Pipe to `column -t` for aligned display, or redirect to a file for pandas / Excel.

```bash
herd res > results.tsv
herd res | column -t -s $'\t' | less
```

## Custom downstream parsing

Each `.hyperherd/results/<trial_id>.json` is a flat JSON object you can read directly:

```python
import json, glob, os

workspace = "my_experiment"
results = {}
for path in glob.glob(f"{workspace}/.hyperherd/results/*.json"):
    trial_id = int(os.path.splitext(os.path.basename(path))[0])
    with open(path) as f:
        results[trial_id] = json.load(f)
```

For a richer join with the parameter manifest, read both:

```python
with open(f"{workspace}/.hyperherd/manifest.json") as f:
    manifest = {t["index"]: t for t in json.load(f)}
```
