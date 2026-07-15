"""Tests for config-declared derived trials (the `derived:` block).

A derived trial mints a NEW trial that inherits a base grid trial's params and
folds in extra overrides, with its own identity/name/checkpoint. It must be:
additive (base grid untouched), reconcile-stable (survives `herd run`), and it
must deliver the override to the launcher.
"""

import shutil
import tempfile
import unittest

from hyperherd import manifest
from hyperherd.config import Config, DerivedSpec
from hyperherd.constraints import Trial, apply_constraints, apply_derived
from hyperherd.search import build_trials


def _make_config(derived=None):
    raw = {
        "name": "t",
        "workspace": "/tmp/test_ws",
        "launcher": "./launch.sh",
        "slurm": {"partition": "short"},
        "grid": ["lr"],
        "parameters": {
            "lr": {"abbrev": "lr", "type": "discrete",
                   "values": [0.1, 0.2, 0.3], "default": 0.1},
        },
    }
    if derived is not None:
        raw["derived"] = derived
    return Config.model_validate(raw)


class TestConfigParsing(unittest.TestCase):
    def test_from_alias_and_overrides(self):
        c = _make_config([{"from": 2, "overrides": {"chunked": True}}])
        self.assertEqual(len(c.derived), 1)
        self.assertEqual(c.derived[0].from_index, 2)
        self.assertEqual(c.derived[0].overrides, {"chunked": True})

    def test_default_is_empty(self):
        self.assertEqual(_make_config().derived, [])

    def test_empty_overrides_rejected(self):
        with self.assertRaises(Exception):
            DerivedSpec(**{"from": 0, "overrides": {}})

    def test_whitespace_in_value_rejected(self):
        with self.assertRaises(Exception):
            DerivedSpec(**{"from": 0, "overrides": {"k": "a b"}})

    def test_bad_key_rejected(self):
        with self.assertRaises(Exception):
            DerivedSpec(**{"from": 0, "overrides": {"a=b": 1}})


class TestApplyDerived(unittest.TestCase):
    def test_appends_without_touching_base(self):
        base = apply_constraints([{"lr": 0.1}, {"lr": 0.2}, {"lr": 0.3}], [])
        out = apply_derived(
            base,
            [DerivedSpec(**{"from": 2, "overrides": {"chunked": True}})],
            abbrevs={"lr": "lr"},
        )
        self.assertEqual(len(out), 4)
        # base grid unchanged
        self.assertEqual([t.params for t in out[:3]], [t.params for t in base])
        derived = out[3]
        self.assertEqual(derived.params, {"lr": 0.3})
        self.assertEqual(derived.extras, {"chunked": True})
        self.assertEqual(derived.derived_overrides, {"chunked": True})

    def test_extras_merge_with_constraint_extras(self):
        base = [Trial(params={"lr": 0.3}, extras={"warmup": 100})]
        out = apply_derived(
            base, [DerivedSpec(**{"from": 0, "overrides": {"chunked": True}})], {}
        )
        self.assertEqual(out[1].extras, {"warmup": 100, "chunked": True})

    def test_out_of_range_from_raises(self):
        base = apply_constraints([{"lr": 0.1}], [])
        with self.assertRaises(ValueError):
            apply_derived(base, [DerivedSpec(**{"from": 5, "overrides": {"x": 1}})], {})

    def test_no_derived_is_identity(self):
        base = apply_constraints([{"lr": 0.1}], [])
        self.assertIs(apply_derived(base, [], {}), base)

    def test_build_trials_includes_derived(self):
        c = _make_config([
            {"from": 2, "overrides": {"strip_functional_tags": True}},
            {"from": 2, "overrides": {"chunked": True}},
        ])
        trials = build_trials(c)
        self.assertEqual(len(trials), 5)  # 3 grid + 2 derived


class TestManifestIntegration(unittest.TestCase):
    def setUp(self):
        self.ws = tempfile.mkdtemp()
        manifest.init_workspace(self.ws)

    def tearDown(self):
        shutil.rmtree(self.ws)

    def _trials(self):
        c = _make_config([
            {"from": 2, "overrides": {"strip_functional_tags": True}},
            {"from": 2, "overrides": {"chunked": True}},
        ])
        return c, build_trials(c)

    def test_distinct_name_and_hash(self):
        c, trials = self._trials()
        records = manifest.create_manifest(self.ws, trials, c.abbrevs, c.labels)
        names = [r["experiment_name"] for r in records]
        self.assertEqual(names[2], "lr-0.3")
        # Derived trials inherit the base name plus an `_ov_` suffix.
        self.assertTrue(names[3].startswith("lr-0.3_ov_strip_functional_tags-"))
        self.assertTrue(names[4].startswith("lr-0.3_ov_chunked-"))
        # A distinct name means a distinct checkpoint dir (no clobber of base).
        self.assertNotEqual(names[2], names[3])
        self.assertNotEqual(names[3], names[4])
        # Distinct identity hashes.
        self.assertEqual(len({r["hash"] for r in records}), 5)

    def test_override_reaches_launcher(self):
        c, trials = self._trials()
        manifest.create_manifest(self.ws, trials, c.abbrevs, c.labels)
        ov = manifest.resolve_overrides(self.ws, 3)
        self.assertIn("strip_functional_tags=true", ov)

    def test_reconcile_keeps_derived(self):
        """The whole point: a second `herd run` must not orphan derived trials."""
        c, trials = self._trials()
        manifest.create_manifest(self.ws, trials, c.abbrevs, c.labels)
        existing = manifest.load_manifest(self.ws)
        rec = manifest.reconcile_manifest(existing, build_trials(c))
        self.assertTrue(rec.is_clean)
        self.assertEqual(len(rec.kept), 5)
        self.assertEqual(rec.added, [])
        self.assertEqual(rec.removed, [])

    def test_adding_a_derivation_is_purely_additive(self):
        c, trials = self._trials()
        manifest.create_manifest(self.ws, trials, c.abbrevs, c.labels)
        existing = manifest.load_manifest(self.ws)
        c.derived.append(DerivedSpec(**{"from": 1, "overrides": {"chunked": True}}))
        rec = manifest.reconcile_manifest(existing, build_trials(c))
        self.assertEqual(len(rec.kept), 5)
        self.assertEqual(len(rec.added), 1)
        self.assertEqual(rec.removed, [])


if __name__ == "__main__":
    unittest.main()
