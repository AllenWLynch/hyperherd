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

    def _args(self, index):
        return argparse.Namespace(workspace=self.tmpdir, index=index)

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


if __name__ == "__main__":
    unittest.main()
