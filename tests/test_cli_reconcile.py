"""End-to-end tests for `herd run` reconciling a config edit against an existing manifest."""

import argparse
import os
import shutil
import tempfile
import unittest
from unittest import mock

from hyperherd import manifest
from hyperherd.cli import cmd_launch


def _write_config(base: str, values):
    cfg = (
        "name: t\n"
        f"workspace: {base}\n"
        f"launcher: {os.path.join(base, 'launch.sh')}\n"
        "grid: all\n"
        "parameters:\n"
        "  lr:\n"
        "    type: discrete\n"
        "    abbrev: lr\n"
        f"    values: {list(values)}\n"
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


def _args(workspace, **overrides):
    base = dict(
        workspace=workspace,
        dry_run=True,
        max_concurrent=None,
        indices=None,
        force=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


class TestReconcileOnRun(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        # First run with two trials seeds the manifest.
        _write_config(self.tmp, [0.1, 0.01])
        with mock.patch("hyperherd.cli.run_preflight", return_value=[]):
            rc = cmd_launch(_args(self.tmp))
        self.assertEqual(rc, 0)
        self.initial_trials = manifest.load_manifest(self.tmp)
        self.assertEqual(len(self.initial_trials), 2)

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_added_value_appends_with_fresh_index(self):
        _write_config(self.tmp, [0.1, 0.01, 0.001])
        with mock.patch("hyperherd.cli.run_preflight", return_value=[]):
            rc = cmd_launch(_args(self.tmp))
        self.assertEqual(rc, 0)

        trials = manifest.load_manifest(self.tmp)
        self.assertEqual(len(trials), 3)
        self.assertEqual(sorted(t["index"] for t in trials), [0, 1, 2])
        # New trial got the highest index, not a recycled one.
        new = next(t for t in trials if t["params"]["lr"] == 0.001)
        self.assertEqual(new["index"], 2)

    def test_removed_pending_value_drops_silently(self):
        _write_config(self.tmp, [0.1])
        with mock.patch("hyperherd.cli.run_preflight", return_value=[]):
            rc = cmd_launch(_args(self.tmp))
        self.assertEqual(rc, 0)
        trials = manifest.load_manifest(self.tmp)
        self.assertEqual(len(trials), 1)
        self.assertEqual(trials[0]["params"]["lr"], 0.1)

    def test_removed_completed_trial_refused_without_force(self):
        # Mark one trial completed, then drop it from the config.
        manifest.bulk_update_status(self.tmp, {1: "completed"})
        _write_config(self.tmp, [0.1])
        with mock.patch("hyperherd.cli.run_preflight", return_value=[]):
            rc = cmd_launch(_args(self.tmp))
        self.assertEqual(rc, 1)
        # Manifest unchanged: completed trial is still there.
        trials = manifest.load_manifest(self.tmp)
        self.assertEqual(len(trials), 2)

    def test_removed_completed_trial_kept_as_orphan_with_force(self):
        manifest.bulk_update_status(self.tmp, {1: "completed"})
        _write_config(self.tmp, [0.1])
        with mock.patch("hyperherd.cli.run_preflight", return_value=[]):
            rc = cmd_launch(_args(self.tmp, force=True))
        self.assertEqual(rc, 0)
        trials = manifest.load_manifest(self.tmp)
        # Completed trial preserved; not in pending set.
        self.assertEqual(len(trials), 2)
        completed = next(t for t in trials if t["index"] == 1)
        self.assertEqual(completed["status"], "completed")

    def test_kept_trial_retains_experiment_name_after_abbrev_edit(self):
        # Edit just the abbreviation. With our freezing rule, the existing
        # trials' names should not change.
        cfg_path = os.path.join(self.tmp, "hyperherd.yaml")
        with open(cfg_path) as f:
            content = f.read()
        content = content.replace("abbrev: lr", "abbrev: learning_rate")
        with open(cfg_path, "w") as f:
            f.write(content)

        with mock.patch("hyperherd.cli.run_preflight", return_value=[]):
            rc = cmd_launch(_args(self.tmp))
        self.assertEqual(rc, 0)

        trials = manifest.load_manifest(self.tmp)
        for orig, now in zip(self.initial_trials, trials):
            self.assertEqual(orig["experiment_name"], now["experiment_name"])


if __name__ == "__main__":
    unittest.main()
