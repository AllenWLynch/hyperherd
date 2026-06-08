"""MNIST classifier with PyTorch Lightning + Hydra.

Demonstrates idempotent training with HyperHerd:
- Checkpoints to a deterministic path based on experiment_name
- Resumes from checkpoint automatically on re-run
- **Streams every Lightning-logged metric via `HyperHerdLogger`** (from
  `hyperherd.integrations.lightning`), so the autonomous monitor's
  `compute_metric` tool can see training trajectories and make pruning
  decisions. The logger forwards each `self.log(...)` call to
  `hyperherd.log_result(...)` automatically.
- Logs final test metrics via `log_result()` at the end of the run.
"""

import os

import hydra
import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from hyperherd import TrialPruned
from hyperherd.integrations.lightning import HyperHerdLogger


class MNISTClassifier(pl.LightningModule):
    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        learning_rate: float,
        optimizer: str,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 10),
        )

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        return loss, acc

    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        # logger=False keeps these out of HyperHerdLogger's per-global-step
        # stream; we re-log them per EPOCH in on_validation_epoch_end so the
        # `successive_halving` rungs in hyperherd.yaml are expressed in epochs
        # (clean integers) rather than batch counts. Lightning still routes
        # them to callback_metrics for the prog bar + ModelCheckpoint.
        self.log("val_loss", loss, prog_bar=True, logger=False)
        self.log("val_acc", acc, prog_bar=True, logger=False)

    def on_validation_epoch_end(self):
        # Epoch-indexed validation metrics for successive halving (`herd sh`):
        # one integer step per epoch regardless of batch size / dataset size.
        # `log_result` raises `hyperherd.TrialPruned` if the pruner has
        # signalled this trial — that propagates up to train.py's top-level
        # handler for a clean, checkpoint-preserving exit.
        metrics = self.trainer.callback_metrics
        try:
            from hyperherd import log_result
            for name in ("val_loss", "val_acc"):
                value = metrics.get(name)
                if value is not None:
                    log_result(name, float(value), step=self.current_epoch)
        except (ImportError, RuntimeError):
            # Not running inside a HyperHerd trial (local dev) — skip.
            pass

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def configure_optimizers(self):
        name = self.hparams.optimizer
        lr = self.hparams.learning_rate
        wd = self.hparams.weight_decay
        if name == "adam":
            return torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        elif name == "adamw":
            return torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=wd)
        elif name == "sgd":
            return torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        else:
            raise ValueError(f"Unknown optimizer: {name}")


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

    def prepare_data(self):
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.train_set, self.val_set = random_split(
                full, [55000, 5000],
                generator=torch.Generator().manual_seed(42),
            )
        if stage == "test" or stage is None:
            self.test_set = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size,
                          num_workers=self.num_workers)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Deterministic output dir based on experiment name
    exp_name = cfg.experiment_name
    output_dir = os.path.join("outputs", exp_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Experiment: {exp_name}")
    print(f"Output dir: {output_dir}")
    print(f"  lr={cfg.learning_rate}, opt={cfg.optimizer}, bs={cfg.batch_size}")
    print(f"  hidden_dim={cfg.hidden_dim}, dropout={cfg.dropout}")
    print(f"  weight_decay={cfg.weight_decay}, gradient_clip_val={cfg.gradient_clip_val}")
    print(f"  max_epochs={cfg.max_epochs}, accelerator={cfg.accelerator}")

    # Seed for reproducibility
    pl.seed_everything(42, workers=True)

    # Model
    model = MNISTClassifier(
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
        learning_rate=cfg.learning_rate,
        optimizer=cfg.optimizer,
        weight_decay=cfg.weight_decay,
    )

    # Data
    datamodule = MNISTDataModule(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )

    # Checkpoint callback — saves to output_dir for idempotency
    checkpoint_cb = ModelCheckpoint(
        dirpath=output_dir,
        filename="best",
        monitor="val_acc",
        mode="max",
        save_last=True,  # always save last.ckpt for resume
    )

    # Resume from checkpoint if it exists
    last_ckpt = os.path.join(output_dir, "last.ckpt")
    resume_from = last_ckpt if os.path.isfile(last_ckpt) else None
    if resume_from:
        print(f"  Resuming from: {resume_from}")

    # Trainer. HyperHerdLogger streams every self.log(...) call to
    # .hyperherd/results/<idx>/stream/<name>.jsonl, and no-ops outside a trial.
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=cfg.accelerator,
        devices=1,
        callbacks=[checkpoint_cb],
        logger=[HyperHerdLogger()],
        default_root_dir=output_dir,
        enable_progress_bar=True,
        gradient_clip_val=cfg.gradient_clip_val,
    )

    # Train
    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_from)

    # Test using the best checkpoint
    best_ckpt = checkpoint_cb.best_model_path or last_ckpt
    test_results = trainer.test(model, datamodule=datamodule, ckpt_path=best_ckpt)

    # Log results to HyperHerd
    if test_results:
        metrics = test_results[0]
        try:
            from hyperherd import log_result
            log_result("test_acc", metrics.get("test_acc", 0.0))
            log_result("test_loss", metrics.get("test_loss", 0.0))
            log_result("best_val_acc", float(checkpoint_cb.best_model_score or 0.0))
            print(f"\nResults logged to HyperHerd workspace.")
        except (ImportError, RuntimeError):
            # hyperherd not installed or not running inside a herd trial
            print(f"\nTest accuracy: {metrics.get('test_acc', 0.0):.4f}")
            print(f"Test loss: {metrics.get('test_loss', 0.0):.4f}")


if __name__ == "__main__":
    try:
        main()
    except TrialPruned as e:
        # Successive-halving (`herd sh`) asked this trial to stop. The
        # HyperHerdLogger raises TrialPruned at the next streamed metric;
        # we unwind here and exit cleanly. Lightning's `save_last`
        # checkpoint means a resumed ('paused') trial picks up where it
        # left off. The terminal manifest status is owned by `herd sh`,
        # so a clean exit-0 is correct — this isn't a failure.
        print(f"\n[hyperherd] {e} — stopping cleanly (action={e.action}).")
        raise SystemExit(0)
