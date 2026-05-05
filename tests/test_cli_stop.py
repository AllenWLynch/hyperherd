"""Tests for `herd stop`: cancelling a single trial in a job array."""

import argparse
import os
import shutil
import tempfile
import unittest
from unittest import mock

from hyperherd import manifest
from hyperherd.cli import cmd_stop


def _write_config(base: str) -> None:
    cfg = (
        "name: t\n"
        f"workspace: {base}\n"
        f"launcher: {os.path.join(base, 'launch.sh')}\n"
        "grid: all\n"
        "parameters:\n"
        "  lr:\n"
        "    type: discrete\n"
        "    abbrev: lr\n"
        "    values: [0.1, 0.01]\n"
        "slurm:\n"
        "  partition: p\n"
        "  time: '00:10:00'\n"
        "  mem: 1G\n"
        "  cpus_per_task: 1\n"
    )
    with open(os.path.join(base, "hyperherd.yaml"), "w") as f:
        f.write(cfg)
    with open(os.path.join(base, "launch.sh"), "w") as f:
        f.write("#!/bin/bash\n")
    os.chmod(os.path.join(base, "launch.sh"), 0o755)


class TestCmdStop(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        _write_config(self.tmpdir)
        manifest.init_workspace(self.tmpdir)
        manifest.create_manifest(
            self.tmpdir,
            [{"lr": 0.1}, {"lr": 0.01}],
            abbrevs={"lr": "lr"},
        )

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def _args(self, index, all=False):
        return argparse.Namespace(workspace=self.tmpdir, index=index, all=all)

    def test_no_workspace(self):
        # Wipe the workspace dir to simulate a fresh project.
        shutil.rmtree(manifest.workspace_path(self.tmpdir))
        with mock.patch("hyperherd.cli.slurm.cancel_array_task") as cancel:
            rc = cmd_stop(self._args(0))
        self.assertEqual(rc, 1)
        cancel.assert_not_called()

    def test_unknown_index(self):
        manifest.bulk_update_status(self.tmpdir, {0: "running"})
        manifest.record_job_submission(self.tmpdir, "12345", [0, 1])
        with mock.patch("hyperherd.cli.slurm.cancel_array_task") as cancel, \
             mock.patch("hyperherd.cli._sync_slurm_status"):
            rc = cmd_stop(self._args(99))
        self.assertEqual(rc, 1)
        cancel.assert_not_called()

    def test_refuses_when_not_running(self):
        # Trial is still in 'ready' — nothing to cancel.
        manifest.record_job_submission(self.tmpdir, "12345", [0, 1])
        with mock.patch("hyperherd.cli.slurm.cancel_array_task") as cancel, \
             mock.patch("hyperherd.cli._sync_slurm_status"):
            rc = cmd_stop(self._args(0))
        self.assertEqual(rc, 1)
        cancel.assert_not_called()

    def test_no_job_id_recorded(self):
        # Status is 'submitted' but we somehow have no job_ids record.
        manifest.bulk_update_status(self.tmpdir, {0: "submitted"})
        with mock.patch("hyperherd.cli.slurm.cancel_array_task") as cancel, \
             mock.patch("hyperherd.cli._sync_slurm_status"):
            rc = cmd_stop(self._args(0))
        self.assertEqual(rc, 1)
        cancel.assert_not_called()

    def test_cancels_running_trial(self):
        manifest.bulk_update_status(self.tmpdir, {0: "running", 1: "running"})
        manifest.record_job_submission(self.tmpdir, "12345", [0, 1])

        with mock.patch("hyperherd.cli.slurm.cancel_array_task") as cancel, \
             mock.patch("hyperherd.cli._sync_slurm_status"):
            rc = cmd_stop(self._args(0))

        self.assertEqual(rc, 0)
        cancel.assert_called_once_with("12345", 0)
        trials = manifest.load_manifest(self.tmpdir)
        self.assertEqual(trials[0]["status"], "cancelled")
        # Other trial untouched.
        self.assertEqual(trials[1]["status"], "running")

    def test_uses_most_recent_submission(self):
        # Trial 0 submitted, failed, resubmitted under a new job id.
        manifest.record_job_submission(self.tmpdir, "12345", [0, 1])
        manifest.record_job_submission(self.tmpdir, "67890", [0])
        manifest.bulk_update_status(self.tmpdir, {0: "running"})

        with mock.patch("hyperherd.cli.slurm.cancel_array_task") as cancel, \
             mock.patch("hyperherd.cli._sync_slurm_status"):
            rc = cmd_stop(self._args(0))

        self.assertEqual(rc, 0)
        cancel.assert_called_once_with("67890", 0)

    def test_all_cancels_only_live_trials(self):
        # Mix of statuses: 0 running, 1 completed, plus ready ones from setUp.
        manifest.bulk_update_status(self.tmpdir, {0: "running", 1: "completed"})
        manifest.record_job_submission(self.tmpdir, "12345", [0, 1])

        with mock.patch("hyperherd.cli.slurm.cancel_array_task") as cancel, \
             mock.patch("hyperherd.cli._sync_slurm_status"):
            rc = cmd_stop(self._args(None, all=True))

        self.assertEqual(rc, 0)
        cancel.assert_called_once_with("12345", 0)
        trials = manifest.load_manifest(self.tmpdir)
        by_idx = {t["index"]: t["status"] for t in trials}
        self.assertEqual(by_idx[0], "cancelled")
        self.assertEqual(by_idx[1], "completed")  # untouched

    def test_all_no_live_trials(self):
        # Everything completed/failed already — nothing to do.
        manifest.bulk_update_status(self.tmpdir, {0: "completed", 1: "failed"})
        with mock.patch("hyperherd.cli.slurm.cancel_array_task") as cancel, \
             mock.patch("hyperherd.cli._sync_slurm_status"):
            rc = cmd_stop(self._args(None, all=True))
        self.assertEqual(rc, 0)
        cancel.assert_not_called()

    def test_neither_index_nor_all_errors(self):
        with mock.patch("hyperherd.cli.slurm.cancel_array_task") as cancel, \
             mock.patch("hyperherd.cli._sync_slurm_status"):
            rc = cmd_stop(self._args(None, all=False))
        self.assertEqual(rc, 1)
        cancel.assert_not_called()

    def test_both_index_and_all_errors(self):
        with mock.patch("hyperherd.cli.slurm.cancel_array_task") as cancel, \
             mock.patch("hyperherd.cli._sync_slurm_status"):
            rc = cmd_stop(self._args(0, all=True))
        self.assertEqual(rc, 1)
        cancel.assert_not_called()

    def test_cancels_queued_trial(self):
        manifest.bulk_update_status(self.tmpdir, {0: "queued"})
        manifest.record_job_submission(self.tmpdir, "12345", [0, 1])

        with mock.patch("hyperherd.cli.slurm.cancel_array_task") as cancel, \
             mock.patch("hyperherd.cli._sync_slurm_status"):
            rc = cmd_stop(self._args(0))

        self.assertEqual(rc, 0)
        cancel.assert_called_once_with("12345", 0)
        trials = manifest.load_manifest(self.tmpdir)
        self.assertEqual(trials[0]["status"], "cancelled")


class TestSyncStickyCancelled(unittest.TestCase):
    """Regression: a trial cancelled while queued must stay 'cancelled'
    even if sacct still reports PENDING for it (sacct lag). Without the
    sticky guard, `herd run` would resync to 'queued', and that's
    excluded from `get_pending_indices` — the trial would never be
    resubmitted."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        manifest.init_workspace(self.tmpdir)
        manifest.create_manifest(
            self.tmpdir, [{"lr": 0.1}, {"lr": 0.01}], abbrevs={"lr": "lr"},
        )
        manifest.record_job_submission(self.tmpdir, "12345", [0, 1])

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_cancelled_status_resists_pending_sacct_row(self):
        from hyperherd.cli import _sync_slurm_status
        # Post-cancel: idx 0 was cancelled, sacct still reports the
        # array as PENDING (scancel issued but accounting lagging).
        manifest.bulk_update_status(self.tmpdir, {0: "cancelled", 1: "queued"})
        with mock.patch(
            "hyperherd.cli.slurm.query_job_status",
            return_value={("12345", 0): "PENDING", ("12345", 1): "PENDING"},
        ):
            _sync_slurm_status(self.tmpdir)
        trials = manifest.load_manifest(self.tmpdir)
        by_idx = {t["index"]: t["status"] for t in trials}
        self.assertEqual(by_idx[0], "cancelled")  # protected
        self.assertEqual(by_idx[1], "queued")     # normal sync

    def test_cancelled_index_still_resubmittable(self):
        # End-to-end shape of the user's bug: cancel + sync (with sacct
        # lag) must leave idx 0 in `get_pending_indices`.
        from hyperherd.cli import _sync_slurm_status
        manifest.bulk_update_status(self.tmpdir, {0: "cancelled", 1: "running"})
        with mock.patch(
            "hyperherd.cli.slurm.query_job_status",
            return_value={("12345", 0): "PENDING", ("12345", 1): "RUNNING"},
        ):
            _sync_slurm_status(self.tmpdir)
        self.assertIn(0, manifest.get_pending_indices(self.tmpdir))


if __name__ == "__main__":
    unittest.main()
