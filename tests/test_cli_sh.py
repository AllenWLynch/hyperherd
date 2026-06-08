"""Tests for `herd sh`: the successive-halving command that reads sweep state
and applies PRUNE/PAUSE/SUBMIT side effects (manifest + signals + SLURM submit)."""

import argparse
import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

from hyperherd import manifest
from hyperherd.cli import cmd_sh
from hyperherd.logging import read_prune_signal, signal_path


def _write_config(base, *, with_sh=True):
    sh = (
        "successive_halving:\n"
        "  metric: val_loss\n"
        "  direction: min\n"
        "  min_steps: 10\n"
        "  budget: 80\n"
        "  eta: 2\n"
    ) if with_sh else ""
    cfg = (
        "name: t\n"
        f"workspace: {base}\n"
        f"launcher: {os.path.join(base, 'launch.sh')}\n"
        "grid: [lr]\n"
        "parameters:\n"
        "  lr:\n"
        "    type: discrete\n"
        "    abbrev: lr\n"
        "    values: [0.1, 0.2, 0.3, 0.4]\n"
        "slurm:\n"
        "  partition: p\n"
        "  time: '00:10:00'\n"
        "  mem: 1G\n"
        "  cpus_per_task: 1\n"
        + sh
    )
    with open(os.path.join(base, "hyperherd.yaml"), "w") as f:
        f.write(cfg)
    with open(os.path.join(base, "launch.sh"), "w") as f:
        f.write("#!/bin/bash\n")
    os.chmod(os.path.join(base, "launch.sh"), 0o755)


def _stream(base, idx, *pairs):
    d = os.path.join(base, ".hyperherd", "results", str(idx), "stream")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "val_loss.jsonl"), "w") as f:
        for s, v in pairs:
            f.write(json.dumps({"step": s, "value": v, "ts": s}) + "\n")


class TestCmdSh(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        _write_config(self.tmp)
        manifest.init_workspace(self.tmp)
        manifest.create_manifest(
            self.tmp,
            [{"lr": 0.1}, {"lr": 0.2}, {"lr": 0.3}, {"lr": 0.4}],
            abbrevs={"lr": "lr"},
        )

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def _args(self, **over):
        base = dict(
            workspace=self.tmp, dry_run=False, json_output=False,
            metric=None, direction=None, min_steps=None, budget=None,
            eta=None, max_concurrent=None,
        )
        base.update(over)
        return argparse.Namespace(**base)

    def _status(self):
        return {t["index"]: t["status"]
                for t in manifest.load_manifest(self.tmp)}

    def test_no_workspace_errors(self):
        shutil.rmtree(manifest.workspace_path(self.tmp))
        rc = cmd_sh(self._args())
        self.assertEqual(rc, 1)

    def test_missing_config_errors(self):
        _write_config(self.tmp, with_sh=False)
        with mock.patch("hyperherd.cli._sync_slurm_status"):
            rc = cmd_sh(self._args())
        self.assertEqual(rc, 1)

    def test_cli_flags_supply_missing_config(self):
        _write_config(self.tmp, with_sh=False)
        manifest.bulk_update_status(
            self.tmp, {0: "running", 1: "running", 2: "running", 3: "running"})
        for i, v in enumerate([0.1, 0.2, 0.3, 0.4]):
            _stream(self.tmp, i, (10, v))
        with mock.patch("hyperherd.cli._sync_slurm_status"):
            rc = cmd_sh(self._args(
                metric="val_loss", direction="min", min_steps=10, budget=80))
        self.assertEqual(rc, 0)
        st = self._status()
        self.assertEqual(st[2], "pruned")
        self.assertEqual(st[3], "pruned")

    def test_prunes_bottom_half_and_writes_signals(self):
        manifest.bulk_update_status(
            self.tmp, {0: "running", 1: "running", 2: "running", 3: "running"})
        for i, v in enumerate([0.1, 0.2, 0.3, 0.4]):
            _stream(self.tmp, i, (10, v))
        with mock.patch("hyperherd.cli._sync_slurm_status"):
            rc = cmd_sh(self._args())
        self.assertEqual(rc, 0)
        st = self._status()
        self.assertEqual(st[0], "running")   # top half untouched
        self.assertEqual(st[1], "running")
        self.assertEqual(st[2], "pruned")
        self.assertEqual(st[3], "pruned")
        # Cooperative signals written for the stopped trials.
        self.assertEqual(read_prune_signal(self.tmp, 2), "prune")
        self.assertEqual(read_prune_signal(self.tmp, 3), "prune")
        self.assertIsNone(read_prune_signal(self.tmp, 0))

    def test_pauses_ambiguous_trial(self):
        # idx0 reached rung0; others not yet → idx0 ambiguous → paused.
        manifest.bulk_update_status(
            self.tmp, {0: "running", 1: "running", 2: "running", 3: "running"})
        _stream(self.tmp, 0, (10, 0.1))
        _stream(self.tmp, 1, (5, 0.2))
        _stream(self.tmp, 2, (5, 0.3))
        _stream(self.tmp, 3, (5, 0.4))
        with mock.patch("hyperherd.cli._sync_slurm_status"):
            rc = cmd_sh(self._args())
        self.assertEqual(rc, 0)
        st = self._status()
        self.assertEqual(st[0], "paused")
        self.assertEqual(read_prune_signal(self.tmp, 0), "pause")

    def test_dry_run_makes_no_changes(self):
        manifest.bulk_update_status(
            self.tmp, {0: "running", 1: "running", 2: "running", 3: "running"})
        for i, v in enumerate([0.1, 0.2, 0.3, 0.4]):
            _stream(self.tmp, i, (10, v))
        with mock.patch("hyperherd.cli._sync_slurm_status"):
            rc = cmd_sh(self._args(dry_run=True))
        self.assertEqual(rc, 0)
        # Nothing mutated.
        self.assertEqual(self._status()[2], "running")
        self.assertFalse(os.path.exists(signal_path(self.tmp, 2)))

    def test_submits_ready_trials_in_one_array(self):
        # All ready → one batched submission, statuses become 'submitted'.
        with mock.patch("hyperherd.cli._sync_slurm_status"), \
             mock.patch("hyperherd.cli.slurm.generate_sbatch_script",
                        return_value="#sbatch"), \
             mock.patch("hyperherd.cli.slurm.submit_job",
                        return_value="999") as submit:
            rc = cmd_sh(self._args())
        self.assertEqual(rc, 0)
        submit.assert_called_once()
        st = self._status()
        self.assertTrue(all(v == "submitted" for v in st.values()))
        # One job_ids record covering all four indices.
        recs = manifest.get_job_ids(self.tmp)
        self.assertEqual(recs[-1]["slurm_job_id"], "999")
        self.assertEqual(sorted(recs[-1]["indices"]), [0, 1, 2, 3])

    def test_resume_clears_signal(self):
        # idx0 paused with a stale signal but now provably top-half → resume.
        manifest.bulk_update_status(
            self.tmp, {0: "paused", 1: "running", 2: "running", 3: "running"})
        from hyperherd.logging import write_prune_signal
        write_prune_signal(self.tmp, 0, "pause")
        _stream(self.tmp, 0, (10, 0.05))   # best
        _stream(self.tmp, 1, (10, 0.2))
        _stream(self.tmp, 2, (10, 0.3))
        _stream(self.tmp, 3, (10, 0.4))
        with mock.patch("hyperherd.cli._sync_slurm_status"), \
             mock.patch("hyperherd.cli.slurm.generate_sbatch_script",
                        return_value="#sbatch"), \
             mock.patch("hyperherd.cli.slurm.submit_job", return_value="1001"):
            rc = cmd_sh(self._args())
        self.assertEqual(rc, 0)
        self.assertEqual(self._status()[0], "submitted")
        # Stale signal cleared so the resumed trial doesn't self-terminate.
        self.assertIsNone(read_prune_signal(self.tmp, 0))

    def test_json_output(self):
        manifest.bulk_update_status(
            self.tmp, {0: "running", 1: "running", 2: "running", 3: "running"})
        for i, v in enumerate([0.1, 0.2, 0.3, 0.4]):
            _stream(self.tmp, i, (10, v))
        buf = io.StringIO()
        with mock.patch("hyperherd.cli._sync_slurm_status"), redirect_stdout(buf):
            rc = cmd_sh(self._args(json_output=True))
        self.assertEqual(rc, 0)
        payload = json.loads(buf.getvalue())
        self.assertEqual(payload["rungs"], [10, 20, 40, 80])
        self.assertEqual(sorted(payload["pruned"]), [2, 3])
        self.assertEqual(len(payload["decisions"]), 4)


class TestPausedSticky(unittest.TestCase):
    """A paused trial must survive a SLURM sync that still reports it RUNNING
    (the trial self-terminates cooperatively; sacct lags)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        manifest.init_workspace(self.tmp)
        manifest.create_manifest(
            self.tmp, [{"lr": 0.1}, {"lr": 0.2}], abbrevs={"lr": "lr"})
        manifest.record_job_submission(self.tmp, "12345", [0, 1])

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def test_paused_resists_stale_running_row(self):
        from hyperherd.cli import _sync_slurm_status
        manifest.bulk_update_status(self.tmp, {0: "paused", 1: "running"})
        with mock.patch(
            "hyperherd.cli.slurm.query_job_status",
            return_value={("12345", 0): "RUNNING", ("12345", 1): "RUNNING"},
        ), mock.patch(
            "hyperherd.cli.slurm.query_squeue_live",
            return_value={("12345", 0): "RUNNING", ("12345", 1): "RUNNING"},
        ):
            _sync_slurm_status(self.tmp)
        st = {t["index"]: t["status"] for t in manifest.load_manifest(self.tmp)}
        self.assertEqual(st[0], "paused")   # sticky
        self.assertEqual(st[1], "running")


if __name__ == "__main__":
    unittest.main()
