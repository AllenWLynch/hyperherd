"""End-to-end tests for the `herd run` / `herd stop` positional grammar.

`tests/test_argspec.py` covers the classifier in isolation. These drive the real
`main()` via `sys.argv`, which is the only place the argparse/classifier seam is
actually exercised — argparse's greedy positional matcher is precisely what the
classifier exists to work around, so a unit test of the classifier alone would
not catch a regression in the wiring.
"""

import argparse
import io
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

from hyperherd import manifest
from hyperherd.cli import main

_YAML = """\
name: t
workspace: {ws}
launcher: {ws}/launch.sh
grid: all
parameters:
  lr:
    type: discrete
    abbrev: lr
    values: [0.1, 0.01]
  bs:
    type: discrete
    abbrev: bs
    values: [32, 64]
slurm:
  partition: p
  time: '00:10:00'
  mem: 1G
  cpus_per_task: 1
"""


class _CliCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        with open(os.path.join(self.tmp, "hyperherd.yaml"), "w") as f:
            f.write(_YAML.format(ws=self.tmp))
        launcher = os.path.join(self.tmp, "launch.sh")
        with open(launcher, "w") as f:
            f.write("#!/bin/bash\n")
        os.chmod(launcher, 0o755)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_cli(self, *argv):
        """Invoke main() with argv; return (exit_code, stdout, stderr)."""
        out, err = io.StringIO(), io.StringIO()
        code = 0
        with mock.patch("sys.argv", ["herd", *argv]), \
                mock.patch("hyperherd.cli.run_preflight", return_value=[]), \
                redirect_stdout(out), redirect_stderr(err):
            try:
                main()
            except SystemExit as e:
                code = e.code or 0
        return code, out.getvalue(), err.getvalue()

    def seed_manifest(self):
        """Create the 4-trial manifest via a dry run."""
        code, _, err = self.run_cli("run", self.tmp, "-n")
        self.assertEqual(code, 0, err)
        return manifest.load_manifest(self.tmp)


class TestRunPositionals(_CliCase):
    def test_workspace_only(self):
        trials = self.seed_manifest()
        self.assertEqual(len(trials), 4)  # 2 lr x 2 bs

    def test_indices_and_override_positionally(self):
        # The headline case: `herd run <ws> 1-2 bs=128`.
        self.seed_manifest()
        code, _, err = self.run_cli("run", self.tmp, "1-2", "bs=128", "-n")
        self.assertEqual(code, 0, err)
        trials = {t["index"]: t for t in manifest.load_manifest(self.tmp)}
        self.assertEqual(trials[1]["overrides"], {"bs": "128"})
        self.assertEqual(trials[2]["overrides"], {"bs": "128"})
        # Untargeted trials are untouched.
        self.assertEqual(trials[0]["overrides"], {})
        self.assertEqual(trials[3]["overrides"], {})

    def test_order_of_positionals_does_not_matter(self):
        self.seed_manifest()
        code, _, err = self.run_cli("run", "bs=128", "1", self.tmp, "-n")
        self.assertEqual(code, 0, err)
        trials = {t["index"]: t for t in manifest.load_manifest(self.tmp)}
        self.assertEqual(trials[1]["overrides"], {"bs": "128"})

    def test_deprecated_dash_i_still_works(self):
        # The monitor agent shells `herd run --json -i <spec> <ws>`.
        self.seed_manifest()
        code, _, err = self.run_cli("run", "-i", "1-2", self.tmp, "-n")
        self.assertEqual(code, 0, err)

    def test_indices_both_ways_rejected(self):
        self.seed_manifest()
        code, _, err = self.run_cli("run", self.tmp, "1-2", "-i", "3", "-n")
        self.assertEqual(code, 2)
        self.assertIn("not both", err)

    def test_override_without_a_selector_is_refused(self):
        # `herd run bs=128` would otherwise silently rewrite the whole sweep.
        self.seed_manifest()
        code, _, err = self.run_cli("run", self.tmp, "bs=128", "-n")
        self.assertEqual(code, 1)
        self.assertIn("refusing to apply overrides to the whole sweep", err.lower())

    def test_override_with_all_flag_is_allowed(self):
        self.seed_manifest()
        code, _, err = self.run_cli("run", self.tmp, "bs=128", "--all", "-n")
        self.assertEqual(code, 0, err)
        trials = manifest.load_manifest(self.tmp)
        self.assertTrue(all(t["overrides"] == {"bs": "128"} for t in trials))

    def test_where_narrows_override_target(self):
        self.seed_manifest()
        code, _, err = self.run_cli(
            "run", self.tmp, "--where", "lr=0.1", "ckpt=/x", "-n")
        self.assertEqual(code, 0, err)
        trials = {t["index"]: t for t in manifest.load_manifest(self.tmp)}
        hit = [i for i, t in trials.items() if t["overrides"]]
        self.assertTrue(hit)
        for i in hit:
            self.assertEqual(trials[i]["params"]["lr"], 0.1)

    def test_deprecated_pin_merges_into_where_and_warns(self):
        self.seed_manifest()
        code, _, err = self.run_cli("run", self.tmp, "-p", "lr=0.1", "-n")
        self.assertEqual(code, 0, err)
        self.assertIn("-p/--pin is deprecated", err)


class TestOverridePersistence(_CliCase):
    def test_override_reaches_the_sbatch_script(self):
        self.seed_manifest()
        code, out, err = self.run_cli("run", self.tmp, "1", "bs=128", "-n")
        self.assertEqual(code, 0, err)
        # The dry run prints the sbatch script, which bakes each trial's
        # override string into a bash `case`. The override must come LAST so it
        # wins over the swept param of the same name.
        self.assertIn("bs=128", out)
        import re
        plain = re.sub(r"\x1b\[[0-9;]*m", "", out)   # the dry run dims its output
        line = [ln for ln in plain.splitlines()
                if "OVERRIDES=" in ln and "bs=128" in ln]
        self.assertTrue(line, "no OVERRIDES line carried the override")
        # The swept `bs=64` and the override `bs=128` both appear; Hydra applies
        # left-to-right, so the override MUST come last to win.
        self.assertTrue(
            line[0].rstrip("'\" ").endswith("bs=128"),
            f"override is not last in the string: {line[0]!r}",
        )

    def test_override_survives_a_later_bare_run(self):
        # This is the failure mode a submission-scoped override would have had:
        # anything that resubmits the trial (SH resume, the agent, a plain
        # `herd run`) must not silently drop it.
        self.seed_manifest()
        self.run_cli("run", self.tmp, "1", "bs=128", "-n")
        self.run_cli("run", self.tmp, "-n")
        trials = {t["index"]: t for t in manifest.load_manifest(self.tmp)}
        self.assertEqual(trials[1]["overrides"], {"bs": "128"})

    def test_clear_overrides(self):
        self.seed_manifest()
        self.run_cli("run", self.tmp, "1", "bs=128", "-n")
        code, _, err = self.run_cli("run", self.tmp, "1", "--clear-overrides", "-n")
        self.assertEqual(code, 0, err)
        trials = {t["index"]: t for t in manifest.load_manifest(self.tmp)}
        self.assertEqual(trials[1]["overrides"], {})

    def test_swept_override_renames_the_trial(self):
        # `bs` is swept, so overriding it moves the trial to a new output dir
        # rather than overwriting the original run's results.
        trials = {t["index"]: t for t in self.seed_manifest()}
        original = trials[1]["experiment_name"]
        self.run_cli("run", self.tmp, "1", "bs=128", "-n")
        after = {t["index"]: t for t in manifest.load_manifest(self.tmp)}
        self.assertNotEqual(after[1]["experiment_name"], original)
        self.assertIn("bs-128", after[1]["experiment_name"])

    def test_override_to_an_existing_config_does_not_collide(self):
        # REGRESSION. "Re-run trial 0 at the batch size that worked for trial 1"
        # is the natural way to use this feature. If the override SUBSTITUTED
        # into the name (lr-0.1_bs-32 -> lr-0.1_bs-64) it would land on trial
        # 1's name, and trial 0 would write straight into trial 1's output
        # directory — destroying the result we were comparing against, and
        # racing it if trial 1 were still running. The suffix form keeps the
        # original (unique) params as the base, so names stay unique.
        self.seed_manifest()
        self.run_cli("run", self.tmp, "0", "bs=64", "-n")   # trial 1 IS bs=64
        trials = manifest.load_manifest(self.tmp)
        names = [t["experiment_name"] for t in trials]
        self.assertEqual(len(names), len(set(names)), f"name collision: {names}")
        by_idx = {t["index"]: t for t in trials}
        self.assertNotEqual(
            by_idx[0]["experiment_name"], by_idx[1]["experiment_name"])

    def test_effective_params_drive_where_and_results(self):
        # REGRESSION. `params` keeps the pre-override value, so anything that
        # *interprets* a trial (which bracket it's in, what hyperparameters to
        # attribute its metric to, whether --where matches) must read the
        # EFFECTIVE config or it reports the trial under a config it never ran.
        from hyperherd.cli import _filter_trials_by_where
        self.seed_manifest()
        self.run_cli("run", self.tmp, "0", "bs=64", "-n")
        trials = manifest.load_manifest(self.tmp)
        eff = manifest.effective_params(trials[0]["params"], trials[0]["overrides"])
        self.assertEqual(eff["bs"], 64)
        # Trial 0 trains at bs=64, so `--where bs=64` must find it.
        self.assertIn(0, [t["index"] for t in _filter_trials_by_where(trials, {"bs": 64})])
        # ...and must NOT find it under the value it no longer runs.
        self.assertNotIn(0, [t["index"] for t in _filter_trials_by_where(trials, {"bs": 32})])

    def test_non_swept_override_keeps_the_name(self):
        # `ckpt` isn't a swept param — it doesn't identify a different point in
        # the search space, so the trial re-runs in place (checkpoint resume).
        trials = {t["index"]: t for t in self.seed_manifest()}
        original = trials[1]["experiment_name"]
        self.run_cli("run", self.tmp, "1", "ckpt=/scratch/last.ckpt", "-n")
        after = {t["index"]: t for t in manifest.load_manifest(self.tmp)}
        self.assertEqual(after[1]["experiment_name"], original)

    def test_override_does_not_change_trial_hash(self):
        # The hash is the reconciliation identity. If an override changed it,
        # the next `herd run` would think the config edit replaced the trial.
        trials = {t["index"]: t for t in self.seed_manifest()}
        before = trials[1]["hash"]
        self.run_cli("run", self.tmp, "1", "bs=128", "-n")
        after = {t["index"]: t for t in manifest.load_manifest(self.tmp)}
        self.assertEqual(after[1]["hash"], before)

    def test_override_survives_reconciliation(self):
        # Edit the config to add a trial; the overridden trial keeps its index
        # AND its overrides.
        self.seed_manifest()
        self.run_cli("run", self.tmp, "1", "bs=128", "-n")
        with open(os.path.join(self.tmp, "hyperherd.yaml"), "w") as f:
            f.write(_YAML.format(ws=self.tmp).replace(
                "values: [0.1, 0.01]", "values: [0.1, 0.01, 0.001]"))
        code, _, err = self.run_cli("run", self.tmp, "-n")
        self.assertEqual(code, 0, err)
        trials = {t["index"]: t for t in manifest.load_manifest(self.tmp)}
        self.assertEqual(trials[1]["overrides"], {"bs": "128"})
        self.assertEqual(len(trials), 6)  # 3 lr x 2 bs

    def test_verbatim_value_not_reformatted(self):
        # `lr=1e-3` must reach the trainer as `1e-3`, not be round-tripped
        # through the float formatter into `0.001`.
        self.seed_manifest()
        code, out, _ = self.run_cli("run", self.tmp, "1", "lr=1e-3", "-n")
        self.assertEqual(code, 0)
        self.assertIn("lr=1e-3", out)


class TestReRunCompleted(_CliCase):
    def _complete(self, index):
        manifest.bulk_update_status(self.tmp, {index: "completed"})

    def test_bare_rerun_of_completed_still_needs_force(self):
        self.seed_manifest()
        self._complete(1)
        code, _, err = self.run_cli("run", self.tmp, "1", "-n")
        self.assertEqual(code, 1)
        self.assertIn("already running/completed", err)

    def test_rerun_completed_with_an_override_is_permitted(self):
        # An explicit index plus an override is unambiguous re-run intent, and
        # the swept-param override renames the output dir so the original
        # result survives. No --force needed.
        self.seed_manifest()
        self._complete(1)
        code, _, err = self.run_cli("run", self.tmp, "1", "bs=128", "-n")
        self.assertEqual(code, 0, err)
        trials = {t["index"]: t for t in manifest.load_manifest(self.tmp)}
        self.assertEqual(trials[1]["overrides"], {"bs": "128"})

    def test_rerun_live_trial_still_needs_force_even_with_override(self):
        # A second array task for a running index would race the first over the
        # same output files — that guard is not negotiable.
        self.seed_manifest()
        manifest.bulk_update_status(self.tmp, {1: "running"})
        code, _, err = self.run_cli("run", self.tmp, "1", "bs=128", "-n")
        self.assertEqual(code, 1)
        self.assertIn("running", err)


class TestStopPositionals(_CliCase):
    def setUp(self):
        super().setUp()
        self.seed_manifest()
        manifest.record_job_submission(self.tmp, "999", [0, 1, 2, 3])

    def test_stop_a_range(self):
        manifest.bulk_update_status(
            self.tmp, {0: "running", 1: "running", 2: "running"})
        with mock.patch("hyperherd.cli._sync_slurm_status"), \
                mock.patch("hyperherd.slurm.cancel_array_task") as cancel:
            code, out, err = self.run_cli("stop", self.tmp, "0-2")
        self.assertEqual(code, 0, err)
        self.assertEqual(cancel.call_count, 3)
        trials = {t["index"]: t for t in manifest.load_manifest(self.tmp)}
        self.assertEqual(trials[0]["status"], "cancelled")
        self.assertEqual(trials[2]["status"], "cancelled")

    def test_range_skips_non_live_instead_of_failing(self):
        manifest.bulk_update_status(
            self.tmp, {0: "running", 1: "completed", 2: "running"})
        with mock.patch("hyperherd.cli._sync_slurm_status"), \
                mock.patch("hyperherd.slurm.cancel_array_task") as cancel:
            code, out, err = self.run_cli("stop", self.tmp, "0-2")
        self.assertEqual(code, 0, err)
        self.assertEqual(cancel.call_count, 2)   # 1 was skipped, not cancelled
        self.assertIn("Skipped", out)
        trials = {t["index"]: t for t in manifest.load_manifest(self.tmp)}
        self.assertEqual(trials[1]["status"], "completed")

    def test_single_non_live_index_is_still_a_hard_error(self):
        # Naming exactly one trial that can't be cancelled should say so, not
        # exit 0 having done nothing.
        manifest.bulk_update_status(self.tmp, {1: "completed"})
        with mock.patch("hyperherd.cli._sync_slurm_status"):
            code, _, err = self.run_cli("stop", self.tmp, "1")
        self.assertEqual(code, 1)
        self.assertIn("nothing to cancel", err)

    def test_stop_all_still_works(self):
        manifest.bulk_update_status(self.tmp, {0: "running", 1: "running"})
        with mock.patch("hyperherd.cli._sync_slurm_status"), \
                mock.patch("hyperherd.slurm.cancel_array_task") as cancel:
            code, _, err = self.run_cli("stop", self.tmp, "--all")
        self.assertEqual(code, 0, err)
        self.assertEqual(cancel.call_count, 2)

    def test_stop_rejects_overrides(self):
        code, _, err = self.run_cli("stop", self.tmp, "bs=128")
        self.assertEqual(code, 2)
        self.assertIn("--where", err)


if __name__ == "__main__":
    unittest.main()
