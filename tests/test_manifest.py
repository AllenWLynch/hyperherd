"""Tests for manifest and workspace management."""

import json
import os
import shutil
import tempfile
import unittest

from hyperherd import manifest


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
        self.assertEqual(trials[0]["status"], "ready")

        loaded = manifest.load_manifest(self.tmpdir)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[1]["params"]["opt"], "sgd")

    def test_update_status(self):
        manifest.init_workspace(self.tmpdir)
        manifest.create_manifest(self.tmpdir, [{"a": 1}, {"a": 2}])

        manifest.update_trial_status(self.tmpdir, 0, "running")
        trials = manifest.load_manifest(self.tmpdir)
        self.assertEqual(trials[0]["status"], "running")
        self.assertEqual(trials[1]["status"], "ready")

    def test_bulk_update(self):
        manifest.init_workspace(self.tmpdir)
        manifest.create_manifest(self.tmpdir, [{"a": 1}, {"a": 2}, {"a": 3}])

        manifest.bulk_update_status(self.tmpdir, {0: "completed", 2: "failed"})
        trials = manifest.load_manifest(self.tmpdir)
        self.assertEqual(trials[0]["status"], "completed")
        self.assertEqual(trials[1]["status"], "ready")
        self.assertEqual(trials[2]["status"], "failed")

    def test_get_pending_indices(self):
        manifest.init_workspace(self.tmpdir)
        manifest.create_manifest(self.tmpdir, [{"a": i} for i in range(5)])
        manifest.bulk_update_status(self.tmpdir, {0: "completed", 2: "running", 4: "failed"})
        pending = manifest.get_pending_indices(self.tmpdir)
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

    def test_resolve_overrides_none_becomes_null(self):
        manifest.init_workspace(self.tmpdir)
        manifest.create_manifest(self.tmpdir, [{"ckpt": None, "lr": 0.001}])
        overrides = manifest.resolve_overrides(self.tmpdir, 0)
        # Hydra reads bare `None` as the string "None"; `null` resolves to Python None.
        self.assertIn("ckpt=null", overrides)
        self.assertNotIn("ckpt=None", overrides)

    def test_job_id_tracking(self):
        manifest.init_workspace(self.tmpdir)
        manifest.record_job_submission(self.tmpdir, "12345", [0, 1, 2])
        manifest.record_job_submission(self.tmpdir, "12346", [3, 4])

        records = manifest.get_job_ids(self.tmpdir)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["slurm_job_id"], "12345")
        self.assertEqual(records[1]["indices"], [3, 4])


class TestExperimentName(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_build_experiment_name(self):
        params = {"learning_rate": 0.001, "optimizer": "adam", "batch_size": 64}
        abbrevs = {"learning_rate": "lr", "optimizer": "opt", "batch_size": "bs"}
        name = manifest.build_experiment_name(params, abbrevs)
        self.assertEqual(name, "lr-0.001_opt-adam_bs-64")

    def test_build_experiment_name_float_formatting(self):
        params = {"lr": 0.00012345}
        abbrevs = {"lr": "lr"}
        name = manifest.build_experiment_name(params, abbrevs)
        self.assertEqual(name, "lr-0.0001234")

    def test_manifest_stores_experiment_name(self):
        manifest.init_workspace(self.tmpdir)
        combos = [{"lr": 0.001, "opt": "adam"}]
        abbrevs = {"lr": "lr", "opt": "opt"}
        trials = manifest.create_manifest(self.tmpdir, combos, abbrevs)
        self.assertEqual(trials[0]["experiment_name"], "lr-0.001_opt-adam")

        loaded = manifest.load_manifest(self.tmpdir)
        self.assertEqual(loaded[0]["experiment_name"], "lr-0.001_opt-adam")

    def test_resolve_overrides_includes_experiment_name(self):
        manifest.init_workspace(self.tmpdir)
        combos = [{"lr": 0.001, "opt": "adam"}]
        abbrevs = {"lr": "lr", "opt": "opt"}
        manifest.create_manifest(self.tmpdir, combos, abbrevs)

        overrides = manifest.resolve_overrides(self.tmpdir, 0)
        self.assertIn("experiment_name=lr-0.001_opt-adam", overrides)

    def test_resolve_overrides_extras_win_over_statics(self):
        from hyperherd.constraints import Trial
        manifest.init_workspace(self.tmpdir)
        trials = [Trial(params={"lr": 0.001}, extras={"scheduler.type": "cosine"})]
        manifest.create_manifest(self.tmpdir, trials, {"lr": "lr"})

        overrides = manifest.resolve_overrides(
            self.tmpdir, 0, static_overrides=["scheduler.type=linear", "data.root=/x"]
        )
        # Extras must appear AFTER statics so Hydra (last-wins) lets `set`
        # override `static_overrides`.
        self.assertLess(
            overrides.index("scheduler.type=linear"),
            overrides.index("scheduler.type=cosine"),
        )
        # Params come before statics
        self.assertLess(
            overrides.index("lr=0.001"),
            overrides.index("data.root=/x"),
        )

    def test_experiment_name_uses_labels(self):
        params = {"pretrained": "/scratch/ckpts/a.ckpt", "lr": 0.001}
        abbrevs = {"pretrained": "pre", "lr": "lr"}
        labels = {"pretrained": {"/scratch/ckpts/a.ckpt": "a"}}
        name = manifest.build_experiment_name(params, abbrevs, labels)
        self.assertEqual(name, "pre-a_lr-0.001")

    def test_manifest_persists_extras(self):
        from hyperherd.constraints import Trial
        manifest.init_workspace(self.tmpdir)
        trials = [Trial(params={"opt": "adamw"}, extras={"scheduler.warmup": 1000})]
        manifest.create_manifest(self.tmpdir, trials, {"opt": "opt"})

        loaded = manifest.load_manifest(self.tmpdir)
        self.assertEqual(loaded[0]["extras"], {"scheduler.warmup": 1000})


class TestDiscreteLabels(unittest.TestCase):
    def test_slash_value_without_labels_rejected(self):
        from hyperherd.config import DiscreteParameter
        with self.assertRaises(Exception):
            DiscreteParameter(
                type="discrete",
                abbrev="m",
                values=["/scratch/a.ckpt", "/scratch/b.ckpt"],
            )

    def test_slash_value_with_labels_accepted(self):
        from hyperherd.config import DiscreteParameter
        p = DiscreteParameter(
            type="discrete",
            abbrev="m",
            values=["/scratch/a.ckpt", "/scratch/b.ckpt"],
            labels=["a", "b"],
        )
        self.assertEqual(p.label_for("/scratch/a.ckpt"), "a")

    def test_label_with_slash_rejected(self):
        from hyperherd.config import DiscreteParameter
        with self.assertRaises(Exception):
            DiscreteParameter(
                type="discrete",
                abbrev="m",
                values=["a", "b"],
                labels=["x/y", "z"],
            )

    def test_labels_length_mismatch_rejected(self):
        from hyperherd.config import DiscreteParameter
        with self.assertRaises(Exception):
            DiscreteParameter(
                type="discrete",
                abbrev="m",
                values=["a", "b", "c"],
                labels=["x", "y"],
            )

    def test_duplicate_labels_rejected(self):
        from hyperherd.config import DiscreteParameter
        with self.assertRaises(Exception):
            DiscreteParameter(
                type="discrete",
                abbrev="m",
                values=["a", "b"],
                labels=["x", "x"],
            )

    def test_config_labels_property_exposes_mapping(self):
        from hyperherd.config import Config
        cfg = Config.model_validate({
            "name": "t",
            "workspace": "/tmp",
            "parameters": {
                "ckpt": {
                    "type": "discrete",
                    "abbrev": "c",
                    "values": ["/p/a", "/p/b"],
                    "labels": ["a", "b"],
                },
                "lr": {
                    "type": "discrete",
                    "abbrev": "lr",
                    "values": [0.1, 0.01],
                },
            },
            "grid": "all",
            "slurm": {"partition": "short"},
        })
        self.assertEqual(cfg.labels, {"ckpt": {"/p/a": "a", "/p/b": "b"}})


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


class TestAtomicWrites(unittest.TestCase):
    """Hammer concurrent readers + writers to verify a reader never sees
    an empty/partial manifest. Before the atomic-rename fix, the
    `open("w")` truncate window would surface as `JSONDecodeError`
    every few seconds in production."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        manifest.init_workspace(self.tmpdir)
        # Seed with 50 trials so writes have nontrivial content.
        big = [{"lr": 0.001 * i, "opt": "adam"} for i in range(50)]
        manifest.create_manifest(self.tmpdir, big)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_concurrent_reads_never_see_partial_writes(self):
        import threading

        stop = threading.Event()
        bad_reads = []

        def writer():
            i = 0
            while not stop.is_set():
                manifest.update_trial_status(self.tmpdir, i % 50, "running")
                i += 1

        def reader():
            while not stop.is_set():
                try:
                    trials = manifest.load_manifest(self.tmpdir)
                    if len(trials) != 50:
                        bad_reads.append(("len", len(trials)))
                except json.JSONDecodeError as e:
                    bad_reads.append(("json", str(e)))
                except OSError as e:
                    bad_reads.append(("os", str(e)))

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        # Run for a brief window — long enough to reproduce the
        # truncate-then-write race that the old code hit reliably.
        import time
        time.sleep(0.5)
        stop.set()
        for t in threads:
            t.join()

        self.assertEqual(
            bad_reads, [],
            f"saw {len(bad_reads)} partial reads: {bad_reads[:3]}",
        )


if __name__ == "__main__":
    unittest.main()
