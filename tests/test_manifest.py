"""Tests for manifest and workspace management."""

import json
import os
import shutil
import tempfile
import unittest

from hyperwhip import manifest


class TestManifest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_init_workspace(self):
        manifest.init_workspace(self.tmpdir)
        self.assertTrue(os.path.isdir(manifest.workspace_path(self.tmpdir)))
        self.assertTrue(os.path.isdir(manifest.logs_path(self.tmpdir)))

    def test_create_and_load_manifest(self):
        manifest.init_workspace(self.tmpdir)
        combos = [
            {"lr": 0.001, "opt": "adam"},
            {"lr": 0.01, "opt": "sgd"},
        ]
        trials = manifest.create_manifest(self.tmpdir, combos)
        self.assertEqual(len(trials), 2)
        self.assertEqual(trials[0]["index"], 0)
        self.assertEqual(trials[0]["status"], "pending")

        loaded = manifest.load_manifest(self.tmpdir)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[1]["params"]["opt"], "sgd")

    def test_update_status(self):
        manifest.init_workspace(self.tmpdir)
        manifest.create_manifest(self.tmpdir, [{"a": 1}, {"a": 2}])

        manifest.update_trial_status(self.tmpdir, 0, "running")
        trials = manifest.load_manifest(self.tmpdir)
        self.assertEqual(trials[0]["status"], "running")
        self.assertEqual(trials[1]["status"], "pending")

    def test_bulk_update(self):
        manifest.init_workspace(self.tmpdir)
        manifest.create_manifest(self.tmpdir, [{"a": 1}, {"a": 2}, {"a": 3}])

        manifest.bulk_update_status(self.tmpdir, {0: "completed", 2: "failed"})
        trials = manifest.load_manifest(self.tmpdir)
        self.assertEqual(trials[0]["status"], "completed")
        self.assertEqual(trials[1]["status"], "pending")
        self.assertEqual(trials[2]["status"], "failed")

    def test_get_pending_indices(self):
        manifest.init_workspace(self.tmpdir)
        manifest.create_manifest(self.tmpdir, [{"a": i} for i in range(5)])
        manifest.bulk_update_status(self.tmpdir, {0: "completed", 2: "running", 4: "failed"})
        pending = manifest.get_pending_indices(self.tmpdir)
        # pending + failed = 1, 3, 4
        self.assertEqual(sorted(pending), [1, 3, 4])

    def test_resolve_overrides(self):
        manifest.init_workspace(self.tmpdir)
        manifest.create_manifest(self.tmpdir, [
            {"lr": 0.001, "opt": "adam", "bs": 32},
        ])
        overrides = manifest.resolve_overrides(self.tmpdir, 0)
        self.assertIn("lr=", overrides)
        self.assertIn("opt=adam", overrides)
        self.assertIn("bs=32", overrides)

    def test_resolve_overrides_with_static(self):
        manifest.init_workspace(self.tmpdir)
        manifest.create_manifest(self.tmpdir, [{"lr": 0.001}])
        overrides = manifest.resolve_overrides(self.tmpdir, 0, ["data.path=/tmp"])
        self.assertIn("data.path=/tmp", overrides)

    def test_job_id_tracking(self):
        manifest.init_workspace(self.tmpdir)
        manifest.record_job_submission(self.tmpdir, "12345", [0, 1, 2])
        manifest.record_job_submission(self.tmpdir, "12346", [3, 4])

        records = manifest.get_job_ids(self.tmpdir)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["slurm_job_id"], "12345")
        self.assertEqual(records[1]["indices"], [3, 4])


class TestWorkspaceExists(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_not_exists(self):
        self.assertFalse(manifest.workspace_exists(self.tmpdir))

    def test_exists_after_create(self):
        manifest.init_workspace(self.tmpdir)
        manifest.create_manifest(self.tmpdir, [{"a": 1}])
        self.assertTrue(manifest.workspace_exists(self.tmpdir))


if __name__ == "__main__":
    unittest.main()
