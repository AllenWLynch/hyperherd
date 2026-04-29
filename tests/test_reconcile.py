"""Tests for manifest reconciliation against config edits."""

import json
import os
import shutil
import tempfile
import unittest

from hyperherd import manifest
from hyperherd.constraints import Trial


def _trial(params, extras=None):
    return Trial(params=params, extras=extras or {})


class TestTrialHash(unittest.TestCase):
    def test_stable_across_key_order(self):
        h1 = manifest.trial_hash({"a": 1, "b": 2})
        h2 = manifest.trial_hash({"b": 2, "a": 1})
        self.assertEqual(h1, h2)

    def test_extras_change_hash(self):
        h1 = manifest.trial_hash({"a": 1}, {})
        h2 = manifest.trial_hash({"a": 1}, {"sched.warmup": 100})
        self.assertNotEqual(h1, h2)

    def test_param_value_change_changes_hash(self):
        self.assertNotEqual(
            manifest.trial_hash({"lr": 0.001}),
            manifest.trial_hash({"lr": 0.01}),
        )


class TestReconcile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        manifest.init_workspace(self.tmp)
        self.abbrevs = {"lr": "lr", "opt": "opt"}

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _seed(self, combos):
        return manifest.create_manifest(
            self.tmp, [_trial(c) for c in combos], self.abbrevs
        )

    def test_clean_when_unchanged(self):
        existing = self._seed([{"lr": 0.1}, {"lr": 0.01}])
        diff = manifest.reconcile_manifest(
            existing, [_trial({"lr": 0.1}), _trial({"lr": 0.01})]
        )
        self.assertTrue(diff.is_clean)
        self.assertEqual(len(diff.kept), 2)

    def test_added_only(self):
        existing = self._seed([{"lr": 0.1}])
        diff = manifest.reconcile_manifest(
            existing, [_trial({"lr": 0.1}), _trial({"lr": 0.01})]
        )
        self.assertEqual(len(diff.kept), 1)
        self.assertEqual(len(diff.added), 1)
        self.assertEqual(diff.added[0].params, {"lr": 0.01})
        self.assertEqual(diff.removed, [])

    def test_removed_only(self):
        existing = self._seed([{"lr": 0.1}, {"lr": 0.01}])
        diff = manifest.reconcile_manifest(existing, [_trial({"lr": 0.1})])
        self.assertEqual(len(diff.kept), 1)
        self.assertEqual(diff.added, [])
        self.assertEqual(len(diff.removed), 1)
        self.assertEqual(diff.removed[0]["params"], {"lr": 0.01})

    def test_extras_distinguish_trials(self):
        existing = self._seed([{"lr": 0.1}])
        # Same params but new extras → considered a different trial.
        diff = manifest.reconcile_manifest(
            existing, [_trial({"lr": 0.1}, {"sched": "cosine"})]
        )
        self.assertEqual(len(diff.added), 1)
        self.assertEqual(len(diff.removed), 1)


class TestAppendDrop(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        manifest.init_workspace(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_append_assigns_fresh_indices_past_history(self):
        manifest.create_manifest(
            self.tmp, [_trial({"lr": 0.1}), _trial({"lr": 0.01})], {"lr": "lr"}
        )
        # Simulate idx 1 having been submitted (so job_ids.json references it),
        # then dropped from the manifest. Reusing idx 1 would cause
        # _sync_slurm_status to apply the old SLURM state to a brand-new trial.
        manifest.record_job_submission(self.tmp, "12345", [0, 1])
        manifest.drop_trials(self.tmp, [1])
        manifest.append_trials(
            self.tmp, [_trial({"lr": 0.001})], {"lr": "lr"}, None
        )
        trials = manifest.load_manifest(self.tmp)
        indices = sorted(t["index"] for t in trials)
        self.assertEqual(indices, [0, 2])

    def test_append_freezes_experiment_name(self):
        manifest.create_manifest(self.tmp, [_trial({"lr": 0.1})], {"lr": "lr"})
        # Append with a *different* abbrev mapping — kept trial keeps its old name.
        manifest.append_trials(
            self.tmp, [_trial({"lr": 0.01})], {"lr": "learning_rate"}, None
        )
        trials = manifest.load_manifest(self.tmp)
        self.assertEqual(trials[0]["experiment_name"], "lr-0.1")
        self.assertEqual(trials[1]["experiment_name"], "learning_rate-0.01")

    def test_drop_trials(self):
        manifest.create_manifest(
            self.tmp, [_trial({"a": i}) for i in range(4)], {"a": "a"}
        )
        manifest.drop_trials(self.tmp, [1, 3])
        trials = manifest.load_manifest(self.tmp)
        self.assertEqual([t["index"] for t in trials], [0, 2])


class TestHashBackfill(unittest.TestCase):
    """load_manifest must lazy-add `hash` to manifests written before this feature."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        manifest.init_workspace(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_legacy_manifest_gets_hash_on_load(self):
        # Hand-write a manifest with no `hash` field, simulating an old workspace.
        legacy = [
            {
                "index": 0,
                "params": {"lr": 0.1},
                "extras": {},
                "experiment_name": "lr-0.1",
                "status": "completed",
            }
        ]
        with open(manifest.manifest_path(self.tmp), "w") as f:
            json.dump(legacy, f)

        loaded = manifest.load_manifest(self.tmp)
        self.assertIn("hash", loaded[0])
        self.assertEqual(
            loaded[0]["hash"], manifest.trial_hash({"lr": 0.1}, {})
        )

    def test_legacy_manifest_reconciles(self):
        legacy = [
            {
                "index": 0,
                "params": {"lr": 0.1},
                "extras": {},
                "experiment_name": "lr-0.1",
                "status": "completed",
            }
        ]
        with open(manifest.manifest_path(self.tmp), "w") as f:
            json.dump(legacy, f)

        existing = manifest.load_manifest(self.tmp)
        diff = manifest.reconcile_manifest(
            existing, [_trial({"lr": 0.1}), _trial({"lr": 0.01})]
        )
        self.assertEqual(len(diff.kept), 1)
        self.assertEqual(len(diff.added), 1)


if __name__ == "__main__":
    unittest.main()
