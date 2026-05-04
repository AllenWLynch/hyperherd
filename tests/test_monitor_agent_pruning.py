"""Tests for the agent-driven pruning surface: compute_metric aggregation,
prune_index manifest behavior, and the env-expansion in mcp_servers.

The prune_index test stops short of the actual `herd stop` subprocess —
that path is exercised in the existing CLI tests. Here we just verify
the manifest gets stamped `pruned` and that herd run skips it.
"""

import asyncio
import json
import math
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hyperherd import manifest
from hyperherd.logging import log_result
from hyperherd.monitor_agent import tools as tools_mod


class TestComputeMetric(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        manifest.init_workspace(self.tmp)
        # Bind the tool context the way tick.run_tick would.
        tools_mod.set_context(
            workspace=Path(self.tmp),
            sweep_name="test_sweep",
            last_state_json="{}",
        )
        os.environ["HYPERHERD_WORKSPACE"] = self.tmp
        os.environ["HYPERHERD_TRIAL_ID"] = "3"

    def tearDown(self):
        shutil.rmtree(self.tmp)
        os.environ.pop("HYPERHERD_WORKSPACE", None)
        os.environ.pop("HYPERHERD_TRIAL_ID", None)

    def _run(self, **args):
        result = asyncio.run(tools_mod.compute_metric.handler(args))
        return json.loads(result["content"][0]["text"])

    def test_returns_n_zero_for_missing_stream(self):
        out = self._run(index=99, metric="val_loss")
        self.assertEqual(out["n"], 0)

    def test_aggregates_streamed_values(self):
        log_result("val_loss", 0.9, step=0)
        log_result("val_loss", 0.7, step=100)
        log_result("val_loss", 0.5, step=200)
        log_result("val_loss", 0.4, step=300)

        out = self._run(index=3, metric="val_loss")
        self.assertEqual(out["n"], 4)
        self.assertEqual(out["last"], 0.4)
        self.assertAlmostEqual(out["min"], 0.4)
        self.assertAlmostEqual(out["max"], 0.9)
        self.assertAlmostEqual(out["mean"], 0.625)
        self.assertAlmostEqual(out["median"], 0.6)
        self.assertFalse(out["has_nan_or_inf"])

    def test_detects_nan_and_inf(self):
        log_result("val_loss", 0.5, step=0)
        log_result("val_loss", float("nan"), step=100)
        log_result("val_loss", float("inf"), step=200)

        out = self._run(index=3, metric="val_loss")
        self.assertTrue(out["has_nan_or_inf"])
        # Aggregates over the finite subset.
        self.assertEqual(out["mean"], 0.5)

    def test_recent_trail_capped_at_eight(self):
        for step in range(20):
            log_result("val_loss", 1.0 - step * 0.01, step=step)
        out = self._run(index=3, metric="val_loss")
        # Recent surfaces the trailing few values for trend inspection.
        self.assertEqual(len(out["recent"]), 8)
        self.assertEqual(out["recent"][-1], out["last"])

    def test_window_last_n(self):
        for step in range(20):
            log_result("val_loss", 1.0 - step * 0.01, step=step)
        out = self._run(index=3, metric="val_loss", last_n=5)
        self.assertEqual(out["n"], 5)
        self.assertEqual(out["n_total"], 20)
        self.assertEqual(out["step_first"], 15)
        self.assertEqual(out["step_last"], 19)

    def test_window_step_range(self):
        for step in [0, 100, 200, 300, 400]:
            log_result("val_loss", step * 0.001, step=step)
        out = self._run(index=3, metric="val_loss",
                        step_min=100, step_max=300)
        self.assertEqual(out["n"], 3)
        self.assertEqual(out["step_first"], 100)
        self.assertEqual(out["step_last"], 300)

    def test_window_since_seconds(self):
        # All entries written just now — `since_seconds=10` keeps them all.
        for step in range(5):
            log_result("val_loss", 0.5, step=step)
        out = self._run(index=3, metric="val_loss", since_seconds=10)
        self.assertEqual(out["n"], 5)

    def test_window_since_excludes_old_entries(self):
        # Hand-write a stream file with stale timestamps to test the cutoff
        # without sleeping in the test.
        import time as _time
        stream_dir = Path(self.tmp) / ".hyperherd" / "results" / "3" / "stream"
        stream_dir.mkdir(parents=True, exist_ok=True)
        with open(stream_dir / "val_loss.jsonl", "w") as f:
            for i, age in enumerate([3600, 1800, 5, 1]):  # seconds ago
                f.write(json.dumps({
                    "step": i, "value": 0.1 * i, "ts": _time.time() - age,
                }) + "\n")
        out = self._run(index=3, metric="val_loss", since_seconds=60)
        self.assertEqual(out["n"], 2)  # only the two with age < 60 sec

    def test_empty_window_returns_n_zero(self):
        for step in range(5):
            log_result("val_loss", 0.5, step=step)
        out = self._run(index=3, metric="val_loss",
                        step_min=1000, step_max=2000)
        self.assertEqual(out["n"], 0)
        self.assertEqual(out["n_total"], 5)
        self.assertIn("note", out)


class TestPrunedStatus(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        manifest.init_workspace(self.tmp)
        # Hand-write a small manifest in mixed states. Going under
        # `append_trials` to avoid coupling the test to the trial-record
        # constructor.
        manifest_path = os.path.join(self.tmp, ".hyperherd", "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump([
                {"index": 0, "status": "completed", "params": {"lr": 0.01},
                 "experiment_name": "lr-0.01"},
                {"index": 1, "status": "running", "params": {"lr": 0.02},
                 "experiment_name": "lr-0.02"},
                {"index": 2, "status": "ready", "params": {"lr": 0.03},
                 "experiment_name": "lr-0.03"},
            ], f)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_pruned_excluded_from_pending(self):
        # Mark idx 1 as pruned.
        manifest.update_trial_status(self.tmp, 1, "pruned")
        pending = manifest.get_pending_indices(self.tmp)
        # idx 0 is completed (skipped), idx 2 is ready (resubmittable),
        # idx 1 is pruned — must NOT appear in pending.
        self.assertIn(2, pending)
        self.assertNotIn(0, pending)
        self.assertNotIn(1, pending)


class TestMcpEnvExpansion(unittest.TestCase):
    def test_expands_env_var_references(self):
        from hyperherd.monitor_agent.tick import _resolve_env

        with mock.patch.dict(os.environ, {"FOO": "bar", "BAZ": "qux"}, clear=False):
            out = _resolve_env({
                "TOKEN": "${FOO}",
                "URL": "https://example.com/${BAZ}",
                "PLAIN": "no-vars",
            })
        self.assertEqual(out["TOKEN"], "bar")
        self.assertEqual(out["URL"], "https://example.com/qux")
        self.assertEqual(out["PLAIN"], "no-vars")

    def test_missing_env_var_becomes_empty(self):
        from hyperherd.monitor_agent.tick import _resolve_env

        os.environ.pop("HOPEFULLY_NOT_SET_XYZ", None)
        out = _resolve_env({"X": "${HOPEFULLY_NOT_SET_XYZ}"})
        self.assertEqual(out["X"], "")


if __name__ == "__main__":
    unittest.main()
