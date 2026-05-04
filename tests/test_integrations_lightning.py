"""Tests for `hyperherd.integrations.lightning`.

The integration is a Lightning `Logger` that forwards `pl_module.log()`
calls into HyperHerd's per-trial streams. Tests cover:

- `_coerce_scalar` — extraction from native Python numbers, single-
  element tensor-likes, NaN/Inf rejection, multi-element rejection.
- `HyperHerdLogger` — no-ops when env vars missing; writes streams
  (including slash-nested names) when env present; finalize writes
  `final_<name>` summary entries plus a `status` field.

Skipped if Lightning isn't installed.
"""

import os
import shutil
import tempfile
import unittest

from hyperherd import manifest
from hyperherd.logging import (
    list_metric_streams,
    load_metric_stream,
    load_trial_results,
)

try:
    from hyperherd.integrations.lightning import (
        HyperHerdLogger,
        _coerce_scalar,
    )
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False


@unittest.skipUnless(LIGHTNING_AVAILABLE, "Lightning not installed")
class TestCoerceScalar(unittest.TestCase):
    def test_python_float(self):
        self.assertEqual(_coerce_scalar(0.5), 0.5)

    def test_python_int(self):
        self.assertEqual(_coerce_scalar(7), 7.0)

    def test_python_bool(self):
        # bools coerce to 0.0 / 1.0 — fine for monitoring.
        self.assertEqual(_coerce_scalar(True), 1.0)

    def test_none_returns_none(self):
        self.assertIsNone(_coerce_scalar(None))

    def test_nan_returns_none(self):
        self.assertIsNone(_coerce_scalar(float("nan")))

    def test_inf_returns_none(self):
        self.assertIsNone(_coerce_scalar(float("inf")))
        self.assertIsNone(_coerce_scalar(float("-inf")))

    def test_string_returns_none(self):
        self.assertIsNone(_coerce_scalar("not a number"))

    def test_single_element_tensor(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        t = torch.tensor(0.42)
        self.assertAlmostEqual(_coerce_scalar(t), 0.42, places=5)

    def test_multi_element_tensor_returns_none(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch not installed")
        t = torch.tensor([1.0, 2.0, 3.0])
        self.assertIsNone(_coerce_scalar(t))


@unittest.skipUnless(LIGHTNING_AVAILABLE, "Lightning not installed")
class TestHyperHerdLoggerDisabled(unittest.TestCase):
    """When `HYPERHERD_*` env vars are unset, the logger should no-op
    quietly so the same trainer code works for local dev."""

    def setUp(self):
        # Belt-and-suspenders: ensure neither var is set.
        self._saved = {
            k: os.environ.pop(k, None)
            for k in ("HYPERHERD_WORKSPACE", "HYPERHERD_TRIAL_ID")
        }

    def tearDown(self):
        for k, v in self._saved.items():
            if v is not None:
                os.environ[k] = v

    def test_logger_disabled_when_env_unset(self):
        log = HyperHerdLogger()
        # No exception raised; calls are silent no-ops.
        log.log_metrics({"val_loss": 0.5}, step=10)
        log.finalize("success")
        # Nothing to assert on disk (no workspace), but the calls
        # mustn't raise.

    def test_version_falls_back_to_local(self):
        log = HyperHerdLogger()
        self.assertEqual(log.version, "local")


@unittest.skipUnless(LIGHTNING_AVAILABLE, "Lightning not installed")
class TestHyperHerdLoggerEnabled(unittest.TestCase):
    """When the env is set, log_metrics must write per-step entries to
    the per-metric stream files, including slash-nested names."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        manifest.init_workspace(self.tmp)
        os.environ["HYPERHERD_WORKSPACE"] = self.tmp
        os.environ["HYPERHERD_TRIAL_ID"] = "5"

    def tearDown(self):
        shutil.rmtree(self.tmp)
        os.environ.pop("HYPERHERD_WORKSPACE", None)
        os.environ.pop("HYPERHERD_TRIAL_ID", None)
        os.environ.pop("HYPERHERD_EXPERIMENT_NAME", None)

    def test_log_metrics_writes_streams(self):
        log = HyperHerdLogger()
        log.log_metrics({"val_loss": 0.9, "val_acc": 0.5}, step=0)
        log.log_metrics({"val_loss": 0.7, "val_acc": 0.7}, step=100)
        log.log_metrics({"val_loss": 0.5, "val_acc": 0.85}, step=200)

        loss = load_metric_stream(self.tmp, 5, "val_loss")
        self.assertEqual([p["step"] for p in loss], [0, 100, 200])
        self.assertAlmostEqual(loss[-1]["value"], 0.5)

        acc = load_metric_stream(self.tmp, 5, "val_acc")
        self.assertEqual(len(acc), 3)

    def test_log_metrics_with_slash_nested_names(self):
        """Lightning trainers commonly use 'train/loss' / 'val/loss' so
        the underlying log_result must accept the slashes verbatim."""
        log = HyperHerdLogger()
        log.log_metrics({"train/loss": 0.42, "val/loss": 0.31}, step=10)

        train = load_metric_stream(self.tmp, 5, "train/loss")
        self.assertEqual(len(train), 1)
        self.assertAlmostEqual(train[0]["value"], 0.42)

        names = list_metric_streams(self.tmp, 5)
        self.assertIn("train/loss", names)
        self.assertIn("val/loss", names)

    def test_skips_nan_inf_and_non_numeric(self):
        log = HyperHerdLogger()
        log.log_metrics({
            "ok": 0.5,
            "nan_metric": float("nan"),
            "inf_metric": float("inf"),
            "stringy": "not a number",
            "noney": None,
        }, step=0)

        names = set(list_metric_streams(self.tmp, 5))
        self.assertEqual(names, {"ok"})

    def test_step_none_treated_as_zero(self):
        # Lightning sometimes calls log_metrics(step=None). We must
        # accept it and still produce a valid stream entry.
        log = HyperHerdLogger()
        log.log_metrics({"val_loss": 0.5}, step=None)
        stream = load_metric_stream(self.tmp, 5, "val_loss")
        self.assertEqual(stream[0]["step"], 0)

    def test_finalize_writes_summary_and_status(self):
        log = HyperHerdLogger()
        log.log_metrics({"val_loss": 0.7, "val_acc": 0.85}, step=100)
        log.log_metrics({"val_loss": 0.5, "val_acc": 0.9}, step=200)
        log.finalize("success")

        summary = load_trial_results(self.tmp, 5)
        # Last seen value of each metric becomes a `final_<name>` entry.
        self.assertIn("final_val_loss", summary)
        self.assertAlmostEqual(summary["final_val_loss"], 0.5)
        self.assertIn("final_val_acc", summary)
        self.assertAlmostEqual(summary["final_val_acc"], 0.9)
        self.assertEqual(summary["status"], "success")

    def test_version_uses_experiment_name_when_set(self):
        os.environ["HYPERHERD_EXPERIMENT_NAME"] = "lr-0.001_opt-adam"
        log = HyperHerdLogger()
        self.assertEqual(log.version, "lr-0.001_opt-adam")

    def test_version_falls_back_to_trial_id(self):
        # No HYPERHERD_EXPERIMENT_NAME — falls through to trial_id.
        os.environ.pop("HYPERHERD_EXPERIMENT_NAME", None)
        log = HyperHerdLogger()
        self.assertEqual(log.version, "5")


if __name__ == "__main__":
    unittest.main()
