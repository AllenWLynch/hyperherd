"""Tests for result logging and override parsing."""

import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from hyperherd import manifest
from hyperherd.logging import (
    list_metric_streams,
    load_all_results,
    load_metric_stream,
    load_trial_results,
    log_result,
    parse_overrides,
    read_trial_progress,
)


class TestLogResult(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        manifest.init_workspace(self.tmpdir)
        # `log_result(strict=True)` requires the manifest to exist so the
        # bind-mount check passes. Drop a stub.
        with open(os.path.join(
            self.tmpdir, manifest.WORKSPACE_DIR, manifest.MANIFEST_FILE,
        ), "w") as f:
            f.write("{}")
        # Set env vars as mush would
        os.environ["HYPERHERD_WORKSPACE"] = self.tmpdir
        os.environ["HYPERHERD_TRIAL_ID"] = "3"

    def tearDown(self):
        shutil.rmtree(self.tmpdir)
        os.environ.pop("HYPERHERD_WORKSPACE", None)
        os.environ.pop("HYPERHERD_TRIAL_ID", None)

    def test_log_single_metric(self):
        log_result("accuracy", 0.95)
        results = load_trial_results(self.tmpdir, 3)
        self.assertEqual(results["accuracy"], 0.95)

    def test_log_multiple_metrics(self):
        log_result("accuracy", 0.95)
        log_result("loss", 0.12)
        log_result("epochs", 50)
        results = load_trial_results(self.tmpdir, 3)
        self.assertEqual(results["accuracy"], 0.95)
        self.assertEqual(results["loss"], 0.12)
        self.assertEqual(results["epochs"], 50)

    def test_overwrite_metric(self):
        log_result("accuracy", 0.5)
        log_result("accuracy", 0.95)
        results = load_trial_results(self.tmpdir, 3)
        self.assertEqual(results["accuracy"], 0.95)

    def test_missing_env_vars(self):
        os.environ.pop("HYPERHERD_WORKSPACE")
        with self.assertRaises(RuntimeError):
            log_result("x", 1)

    def test_missing_trial_id(self):
        os.environ.pop("HYPERHERD_TRIAL_ID")
        with self.assertRaises(RuntimeError):
            log_result("x", 1)

    def test_streaming_appends_per_step(self):
        log_result("val_loss", 0.9, step=0)
        log_result("val_loss", 0.7, step=100)
        log_result("val_loss", 0.5, step=200)

        stream = load_metric_stream(self.tmpdir, 3, "val_loss")
        self.assertEqual(len(stream), 3)
        self.assertEqual(stream[0]["step"], 0)
        self.assertEqual(stream[2]["value"], 0.5)

        # Stream and final-summary modes don't share storage.
        results = load_trial_results(self.tmpdir, 3)
        self.assertNotIn("val_loss", results)

    def test_list_metric_streams(self):
        log_result("val_loss", 0.5, step=0)
        log_result("train_loss", 0.6, step=0)
        names = list_metric_streams(self.tmpdir, 3)
        self.assertEqual(set(names), {"val_loss", "train_loss"})

    def test_list_metric_streams_empty(self):
        # No streaming calls yet.
        log_result("test_acc", 0.9)
        self.assertEqual(list_metric_streams(self.tmpdir, 3), [])

    # --- slash-nested metric names ------------------------------------

    def test_slash_metric_writes_nested_file(self):
        """Lightning-style names like 'train/loss' are stored at the
        matching nested path under stream/."""
        log_result("train/loss", 0.5, step=0)

        nested = os.path.join(
            self.tmpdir, ".hyperherd", "results", "3",
            "stream", "train", "loss.jsonl",
        )
        self.assertTrue(os.path.isfile(nested),
                        f"expected nested stream at {nested}")

    def test_slash_metric_round_trips_through_load(self):
        log_result("train/loss", 0.5, step=0)
        log_result("train/loss", 0.4, step=100)
        log_result("val/loss", 0.6, step=0)

        train = load_metric_stream(self.tmpdir, 3, "train/loss")
        self.assertEqual(len(train), 2)
        self.assertEqual(train[1]["value"], 0.4)

        val = load_metric_stream(self.tmpdir, 3, "val/loss")
        self.assertEqual(len(val), 1)

    def test_list_metric_streams_recurses_with_slashes(self):
        """Nested streams appear in the list with slashes preserved
        (POSIX style regardless of host OS)."""
        log_result("train/loss", 0.5, step=0)
        log_result("train/acc", 0.9, step=0)
        log_result("val/loss", 0.4, step=0)
        log_result("flat_metric", 0.1, step=0)

        names = list_metric_streams(self.tmpdir, 3)
        self.assertEqual(
            set(names),
            {"train/loss", "train/acc", "val/loss", "flat_metric"},
        )

    def test_double_slash_paths_round_trip(self):
        """Multi-level nesting (a/b/c) works too."""
        log_result("system/gpu/memory", 1024, step=0)
        stream = load_metric_stream(self.tmpdir, 3, "system/gpu/memory")
        self.assertEqual(len(stream), 1)
        self.assertEqual(stream[0]["value"], 1024)
        self.assertIn("system/gpu/memory", list_metric_streams(self.tmpdir, 3))

    def test_rejects_dotdot_traversal(self):
        with self.assertRaises(ValueError):
            log_result("../escape", 0.0, step=0)
        with self.assertRaises(ValueError):
            log_result("ok/../escape", 0.0, step=0)

    def test_rejects_absolute_path(self):
        with self.assertRaises(ValueError):
            log_result("/etc/passwd", 0.0, step=0)

    def test_rejects_dotdot_with_backslash_separator(self):
        # Even though we use '/', a malicious user might try '..\foo' on a
        # Windows-style host. The validator normalizes '\' → '/' before
        # checking for '..' components.
        with self.assertRaises(ValueError):
            log_result("ok\\..\\escape", 0.0, step=0)


class TestReadTrialProgress(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        manifest.init_workspace(self.tmpdir)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _write_stream(self, idx, name, records):
        d = os.path.join(
            self.tmpdir, manifest.WORKSPACE_DIR, "results", str(idx), "stream",
        )
        os.makedirs(os.path.dirname(os.path.join(d, name + ".jsonl")), exist_ok=True)
        with open(os.path.join(d, name + ".jsonl"), "w") as f:
            for step, value, ts in records:
                f.write(json.dumps({"step": step, "value": value, "ts": ts}) + "\n")

    def test_no_streams_returns_none(self):
        self.assertEqual(read_trial_progress(self.tmpdir, 7), (None, None))

    def test_current_step_and_rate(self):
        # 10 steps over 60s → 6 of those intervals span 50 steps in 50s = 1/s.
        recs = [(s, 0.5, float(s)) for s in range(0, 101, 10)]  # ts == step
        self._write_stream(3, "val_loss", recs)
        step, spm = read_trial_progress(self.tmpdir, 3)
        self.assertEqual(step, 100)
        # 100 steps over 100s (ts==step) → 60 steps/min.
        self.assertAlmostEqual(spm, 60.0, places=3)

    def test_picks_highest_step_across_metrics(self):
        self._write_stream(3, "val_loss", [(0, 1.0, 0.0), (50, 0.5, 50.0)])
        self._write_stream(3, "train/loss", [(0, 2.0, 0.0), (120, 0.2, 120.0)])
        step, spm = read_trial_progress(self.tmpdir, 3)
        self.assertEqual(step, 120)

    def test_single_record_has_no_rate(self):
        self._write_stream(3, "val_loss", [(42, 0.5, 100.0)])
        step, spm = read_trial_progress(self.tmpdir, 3)
        self.assertEqual(step, 42)
        self.assertIsNone(spm)

    def test_reads_only_tail_window(self):
        # 5000 records; with window=10 we should still report the last step
        # and a finite rate without reading everything into a list.
        recs = [(s, 0.1, float(s)) for s in range(5000)]
        self._write_stream(3, "val_loss", recs)
        step, spm = read_trial_progress(self.tmpdir, 3, window=10)
        self.assertEqual(step, 4999)
        self.assertIsNotNone(spm)


class TestLoadAllResults(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        manifest.init_workspace(self.tmpdir)
        with open(os.path.join(
            self.tmpdir, manifest.WORKSPACE_DIR, manifest.MANIFEST_FILE,
        ), "w") as f:
            f.write("{}")
        os.environ["HYPERHERD_WORKSPACE"] = self.tmpdir

    def tearDown(self):
        shutil.rmtree(self.tmpdir)
        os.environ.pop("HYPERHERD_WORKSPACE", None)
        os.environ.pop("HYPERHERD_TRIAL_ID", None)

    def test_load_multiple_trials(self):
        os.environ["HYPERHERD_TRIAL_ID"] = "0"
        log_result("acc", 0.9)
        os.environ["HYPERHERD_TRIAL_ID"] = "1"
        log_result("acc", 0.8)
        os.environ["HYPERHERD_TRIAL_ID"] = "2"
        log_result("acc", 0.95)

        all_results = load_all_results(self.tmpdir)
        self.assertEqual(len(all_results), 3)
        self.assertEqual(all_results[0]["acc"], 0.9)
        self.assertEqual(all_results[2]["acc"], 0.95)

    def test_empty_results(self):
        all_results = load_all_results(self.tmpdir)
        self.assertEqual(all_results, {})


class TestParseOverrides(unittest.TestCase):
    """Round-trip values from the override-string format into Python types."""

    def test_basic_kv_pairs(self):
        out = parse_overrides("lr=0.001 batch_size=64 optimizer=adam")
        self.assertEqual(out, {"lr": 0.001, "batch_size": 64, "optimizer": "adam"})

    def test_bool_and_null(self):
        out = parse_overrides("use_amp=true verbose=false note=null other=None")
        self.assertEqual(
            out, {"use_amp": True, "verbose": False, "note": None, "other": None}
        )

    def test_float_with_exponent(self):
        out = parse_overrides("lr=1e-3 wd=2.5e-4")
        self.assertAlmostEqual(out["lr"], 0.001)
        self.assertAlmostEqual(out["wd"], 2.5e-4)

    def test_signed_int_and_float(self):
        out = parse_overrides("seed=-1 momentum=-0.9")
        self.assertEqual(out["seed"], -1)
        self.assertIsInstance(out["seed"], int)
        self.assertAlmostEqual(out["momentum"], -0.9)

    def test_string_value_unchanged(self):
        out = parse_overrides("optimizer=adam exp_name=lr-0.001_opt-adam")
        self.assertEqual(out["optimizer"], "adam")
        self.assertEqual(out["exp_name"], "lr-0.001_opt-adam")

    def test_skips_non_kv_tokens(self):
        # Tokens without `=` (e.g. trailing `--cfg job` from herd test --cfg-job)
        # are silently dropped — parser is meant to extract params, not flags.
        out = parse_overrides("lr=0.001 --cfg job optimizer=adam")
        self.assertEqual(out, {"lr": 0.001, "optimizer": "adam"})

    def test_empty_string(self):
        self.assertEqual(parse_overrides(""), {})

    def test_reads_sys_argv_when_omitted(self):
        with mock.patch.object(sys, "argv", ["train.py", "lr=0.5 epochs=3"]):
            out = parse_overrides()
        self.assertEqual(out, {"lr": 0.5, "epochs": 3})

    def test_raises_when_no_arg_and_argv_empty(self):
        with mock.patch.object(sys, "argv", ["train.py"]):
            with self.assertRaises(RuntimeError):
                parse_overrides()


class TestPruneSignal(unittest.TestCase):
    """`log_result(step=...)` self-terminates a cooperative trial when `herd sh`
    has written a prune/pause signal for it."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        manifest.init_workspace(self.tmpdir)
        with open(os.path.join(
            self.tmpdir, manifest.WORKSPACE_DIR, manifest.MANIFEST_FILE,
        ), "w") as f:
            f.write("{}")
        os.environ["HYPERHERD_WORKSPACE"] = self.tmpdir
        os.environ["HYPERHERD_TRIAL_ID"] = "3"

    def tearDown(self):
        shutil.rmtree(self.tmpdir)
        os.environ.pop("HYPERHERD_WORKSPACE", None)
        os.environ.pop("HYPERHERD_TRIAL_ID", None)

    def test_no_signal_streaming_is_silent(self):
        log_result("loss", 0.5, step=1)  # must not raise

    def test_streaming_raises_when_signalled(self):
        from hyperherd.logging import TrialPruned, write_prune_signal
        write_prune_signal(self.tmpdir, 3, "pause")
        with self.assertRaises(TrialPruned) as cm:
            log_result("loss", 0.4, step=2)
        self.assertEqual(cm.exception.action, "pause")
        self.assertEqual(cm.exception.index, 3)
        # The metric was still recorded before the raise.
        self.assertEqual(len(load_metric_stream(self.tmpdir, 3, "loss")), 1)

    def test_prune_action_propagates(self):
        from hyperherd.logging import TrialPruned, write_prune_signal
        write_prune_signal(self.tmpdir, 3, "prune")
        with self.assertRaises(TrialPruned) as cm:
            log_result("loss", 0.4, step=2)
        self.assertEqual(cm.exception.action, "prune")

    def test_final_summary_mode_never_raises(self):
        from hyperherd.logging import write_prune_signal
        write_prune_signal(self.tmpdir, 3, "prune")
        log_result("test_acc", 0.9)  # no step → must NOT raise

    def test_clear_signal_stops_raising(self):
        from hyperherd.logging import (
            TrialPruned, write_prune_signal, clear_prune_signal,
        )
        write_prune_signal(self.tmpdir, 3, "pause")
        with self.assertRaises(TrialPruned):
            log_result("loss", 0.4, step=2)
        clear_prune_signal(self.tmpdir, 3)
        log_result("loss", 0.3, step=3)  # must not raise

    def test_signal_is_per_trial(self):
        from hyperherd.logging import write_prune_signal
        # Signal targets trial 7, but we're running as trial 3 → no raise.
        write_prune_signal(self.tmpdir, 7, "prune")
        log_result("loss", 0.4, step=2)

    def test_invalid_action_rejected(self):
        from hyperherd.logging import write_prune_signal
        with self.assertRaises(ValueError):
            write_prune_signal(self.tmpdir, 3, "bogus")


if __name__ == "__main__":
    unittest.main()
