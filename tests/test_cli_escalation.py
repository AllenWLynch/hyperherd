"""Tests for the cooperative-stop backstop: `_sync_slurm_status` force-cancels a
trial that ignored its prune/pause signal past `prune_grace_seconds`."""

import argparse
import os
import shutil
import tempfile
import time
import unittest
from unittest import mock

from hyperherd import manifest
from hyperherd.cli import _sync_slurm_status
from hyperherd.logging import (
    write_prune_signal,
    signal_path,
    signal_escalated,
)


def _write_config(base: str, grace: int = 600) -> None:
    cfg = (
        "name: t\n"
        f"workspace: {base}\n"
        f"launcher: {os.path.join(base, 'launch.sh')}\n"
        "grid: all\n"
        f"prune_grace_seconds: {grace}\n"
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


class TestEscalateUnresponsiveSignals(unittest.TestCase):
    JOB_ID = "12345"

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        _write_config(self.tmp)
        manifest.init_workspace(self.tmp)
        manifest.create_manifest(
            self.tmp, [{"lr": 0.1}, {"lr": 0.01}], abbrevs={"lr": "lr"}
        )
        manifest.record_job_submission(self.tmp, self.JOB_ID, [0, 1])

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _signal(self, idx, action, age_seconds):
        """Write a prune/pause signal and backdate its mtime by `age_seconds`."""
        write_prune_signal(self.tmp, idx, action)
        p = signal_path(self.tmp, idx)
        old = time.time() - age_seconds
        os.utime(p, (old, old))

    def _sync(self, slurm_states):
        """Run a sync with a stubbed sacct snapshot; return the cancel mock."""
        statuses = {(self.JOB_ID, idx): st for idx, st in slurm_states.items()}
        with mock.patch(
            "hyperherd.cli.slurm.query_job_status", return_value=statuses
        ), mock.patch("hyperherd.cli.slurm.cancel_array_task") as cancel:
            _sync_slurm_status(self.tmp)
        return cancel

    def test_cancels_overdue_paused_trial(self):
        manifest.bulk_update_status(self.tmp, {0: "paused"})
        self._signal(0, "pause", age_seconds=1000)  # > 600 grace
        cancel = self._sync({0: "RUNNING"})
        cancel.assert_called_once_with(self.JOB_ID, 0)
        self.assertTrue(signal_escalated(self.tmp, 0))

    def test_cancels_overdue_pruned_trial(self):
        manifest.bulk_update_status(self.tmp, {0: "pruned"})
        self._signal(0, "prune", age_seconds=1000)
        cancel = self._sync({0: "RUNNING"})
        cancel.assert_called_once_with(self.JOB_ID, 0)

    def test_within_grace_not_cancelled(self):
        manifest.bulk_update_status(self.tmp, {0: "paused"})
        self._signal(0, "pause", age_seconds=10)  # < 600 grace
        cancel = self._sync({0: "RUNNING"})
        cancel.assert_not_called()
        self.assertFalse(signal_escalated(self.tmp, 0))

    def test_cooperated_trial_not_cancelled(self):
        # Signal is old, but the trial already exited — sacct shows COMPLETED.
        manifest.bulk_update_status(self.tmp, {0: "pruned"})
        self._signal(0, "prune", age_seconds=1000)
        cancel = self._sync({0: "COMPLETED"})
        cancel.assert_not_called()

    def test_escalated_once_not_repeated(self):
        manifest.bulk_update_status(self.tmp, {0: "paused"})
        self._signal(0, "pause", age_seconds=1000)
        self._sync({0: "RUNNING"}).assert_called_once_with(self.JOB_ID, 0)
        # Second sync (sacct still lagging at RUNNING) must not re-scancel.
        self._sync({0: "RUNNING"}).assert_not_called()

    def test_grace_zero_disables_backstop(self):
        _write_config(self.tmp, grace=0)
        manifest.bulk_update_status(self.tmp, {0: "paused"})
        self._signal(0, "pause", age_seconds=99999)
        cancel = self._sync({0: "RUNNING"})
        cancel.assert_not_called()

    def test_no_signal_no_cancel(self):
        # Paused via manifest but no signal file written → nothing to escalate.
        manifest.bulk_update_status(self.tmp, {0: "paused"})
        cancel = self._sync({0: "RUNNING"})
        cancel.assert_not_called()


if __name__ == "__main__":
    unittest.main()
