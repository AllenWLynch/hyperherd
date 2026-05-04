"""Lightning Logger that streams metrics through ``hyperherd.log_result``.

Add to your trainer once and every ``pl_module.log(name, value)`` call
flows through to the trial's per-metric stream files (the same ones
``herd res`` and the monitor agent read)::

    from hyperherd.integrations.lightning import HyperHerdLogger

    trainer = pl.Trainer(logger=[wandb_logger, HyperHerdLogger()])

When run outside a HyperHerd trial (``HYPERHERD_WORKSPACE`` /
``HYPERHERD_TRIAL_ID`` unset), the logger no-ops, so the same trainer
code works for local dev and sweep runs.

Requires Lightning. Install with ``pip install hyperherd[lightning]``.
"""

from __future__ import annotations

import logging
import math
import os
from typing import Any, Dict, Mapping, Optional

# Lightning ships under two PyPI names with identical APIs: the modern
# `lightning` umbrella (import path `lightning.pytorch.*`) and the legacy
# `pytorch-lightning` (`pytorch_lightning.*`). Accept either so users
# don't have to switch distributions to use the integration.
try:
    from lightning.pytorch.loggers import Logger
    from lightning.pytorch.utilities.rank_zero import rank_zero_only
except ImportError:
    try:
        from pytorch_lightning.loggers import Logger
        from pytorch_lightning.utilities.rank_zero import rank_zero_only
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "hyperherd.integrations.lightning requires Lightning. Install "
            "either `lightning` (recommended) or `pytorch-lightning`, e.g. "
            "`pip install hyperherd[lightning]`."
        ) from e

from hyperherd.logging import log_result

logger = logging.getLogger(__name__)


def _coerce_scalar(value: Any) -> Optional[float]:
    """Best-effort scalar extraction. Returns None for tensors with >1 element,
    non-numeric values, NaN, or Inf — those are silently dropped."""
    if value is None:
        return None
    # Avoid a hard torch dependency: duck-type on tensor-like objects.
    item = getattr(value, "item", None)
    numel = getattr(value, "numel", None)
    if callable(item) and callable(numel):
        try:
            if numel() != 1:
                return None
            value = item()
        except Exception:
            return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


class HyperHerdLogger(Logger):
    """Lightning Logger that forwards metrics to ``hyperherd.log_result``.

    Streaming mode (per-step) on every ``log_metrics`` call; final-summary
    mode on ``finalize`` — the latter writes ``final_<name>`` keys plus a
    ``status`` field to the flat ``<trial>.json`` that ``herd res`` reads.

    No-ops when ``HYPERHERD_WORKSPACE`` / ``HYPERHERD_TRIAL_ID`` aren't set,
    so the same trainer code works locally and in a sweep.
    """

    def __init__(self, name: str = "hyperherd", version: Optional[str] = None) -> None:
        super().__init__()
        self._name = name
        self._version = (
            version
            or os.environ.get("HYPERHERD_EXPERIMENT_NAME")
            or os.environ.get("HYPERHERD_TRIAL_ID")
            or "local"
        )
        self._enabled = bool(
            os.environ.get("HYPERHERD_WORKSPACE")
            and os.environ.get("HYPERHERD_TRIAL_ID") is not None
        )
        self._latest: Dict[str, float] = {}
        if not self._enabled:
            logger.info(
                "HyperHerdLogger: HYPERHERD_WORKSPACE / HYPERHERD_TRIAL_ID not set, disabling."
            )
        else:
            logger.info(
                f"HyperHerdLogger initialized for trial {self._version} in workspace "
                f"{os.environ.get('HYPERHERD_WORKSPACE')}."
            )

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @rank_zero_only
    def log_hyperparams(self, params: Any, *args: Any, **kwargs: Any) -> None:
        # Sweep parameters are already known to HyperHerd from the override string;
        # no need to re-record them per-trial.
        return

    @rank_zero_only
    def log_metrics(self, metrics: Mapping[str, Any], step: Optional[int] = None) -> None:
        if not self._enabled:
            return
        s = 0 if step is None else int(step)
        for name, raw_value in metrics.items():
            value = _coerce_scalar(raw_value)
            if value is None:
                continue
            try:
                # log_result honors '/' as nested subdirs under stream/, so
                # Lightning-style names like 'train/loss' are preserved.
                log_result(name, value, step=s)
                self._latest[name] = value
            except Exception as e:
                logger.warning("HyperHerdLogger: log_result(%s) failed: %s", name, e)

    @rank_zero_only
    def save(self) -> None:
        return

    @rank_zero_only
    def finalize(self, status: str) -> None:
        if not self._enabled:
            return
        try:
            for name, value in self._latest.items():
                log_result(f"final_{name}", value)
            log_result("status", status)
        except Exception as e:
            logger.warning("HyperHerdLogger: finalize failed: %s", e)
