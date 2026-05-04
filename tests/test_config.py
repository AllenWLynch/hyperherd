"""Tests for config-level concerns: parsing, defaults, back-compat."""

import unittest

from hyperherd.config import Config


def _minimal(**overrides):
    raw = {
        "name": "t",
        "workspace": "/tmp/ws",
        "launcher": "./launch.sh",
        "parameters": {
            "lr": {"abbrev": "lr", "type": "discrete", "values": [0.1, 0.01]},
        },
        "grid": "all",
        "slurm": {"partition": "short"},
    }
    raw.update(overrides)
    return Config.model_validate(raw)


class TestStaticOverridesTopLevel(unittest.TestCase):
    """`static_overrides` lives at the top level (was `hydra.static_overrides`)."""

    def test_top_level_field_set(self):
        cfg = _minimal(static_overrides=["data.path=/scratch", "max_epochs=10"])
        self.assertEqual(cfg.static_overrides, ["data.path=/scratch", "max_epochs=10"])

    def test_top_level_default_empty(self):
        cfg = _minimal()
        self.assertEqual(cfg.static_overrides, [])


class TestHydraBackCompat(unittest.TestCase):
    """Old configs using `hydra.static_overrides` still parse for one version."""

    def test_legacy_hydra_section_lifted(self):
        cfg = _minimal(hydra={"static_overrides": ["a=1", "b=2"]})
        self.assertEqual(cfg.static_overrides, ["a=1", "b=2"])

    def test_top_level_wins_over_legacy(self):
        # If a config has both — perhaps mid-migration — the top-level field is
        # authoritative; the nested one is ignored, not appended.
        cfg = _minimal(
            static_overrides=["new=1"],
            hydra={"static_overrides": ["old=1"]},
        )
        self.assertEqual(cfg.static_overrides, ["new=1"])

    def test_empty_legacy_section_does_not_break(self):
        cfg = _minimal(hydra={})
        self.assertEqual(cfg.static_overrides, [])


class TestAbbrevSafety(unittest.TestCase):
    """`abbrev` must be required when the parameter name isn't filename-safe."""

    def _build(self, params, **overrides):
        raw = {
            "name": "t",
            "workspace": "/tmp/ws",
            "launcher": "./launch.sh",
            "parameters": params,
            "grid": "all",
            "slurm": {"partition": "short"},
        }
        raw.update(overrides)
        return Config.model_validate(raw)

    def test_safe_param_name_without_abbrev_ok(self):
        cfg = self._build({"learning_rate": {"type": "discrete", "values": [0.1]}})
        # No abbrev — falls back to name.
        self.assertEqual(cfg.abbrevs["learning_rate"], "learning_rate")

    def test_dotted_param_name_without_abbrev_ok(self):
        # Hydra-style dotted paths are permitted in the safe set.
        cfg = self._build({"model.lr": {"type": "discrete", "values": [0.1]}})
        self.assertEqual(cfg.abbrevs["model.lr"], "model.lr")

    def test_param_name_with_slash_requires_abbrev(self):
        with self.assertRaisesRegex(Exception, "unsafe for filenames"):
            self._build({"foo/bar": {"type": "discrete", "values": [0.1]}})

    def test_param_name_with_slash_with_abbrev_ok(self):
        cfg = self._build(
            {"foo/bar": {"abbrev": "fb", "type": "discrete", "values": [0.1]}}
        )
        self.assertEqual(cfg.abbrevs["foo/bar"], "fb")

    def test_param_name_with_space_requires_abbrev(self):
        with self.assertRaisesRegex(Exception, "unsafe for filenames"):
            self._build({"learning rate": {"type": "discrete", "values": [0.1]}})

    def test_explicit_abbrev_is_also_validated(self):
        with self.assertRaisesRegex(Exception, "abbrev"):
            self._build(
                {"lr": {"abbrev": "bad/abbrev", "type": "discrete", "values": [0.1]}}
            )


class TestParamNameOverrideSafety(unittest.TestCase):
    """Param names with shell/syntax metachars are rejected even with an abbrev."""

    def _build(self, params):
        raw = {
            "name": "t",
            "workspace": "/tmp/ws",
            "launcher": "./launch.sh",
            "parameters": params,
            "grid": "all",
            "slurm": {"partition": "short"},
        }
        return Config.model_validate(raw)

    def test_quote_in_name_rejected_even_with_abbrev(self):
        # The abbrev makes the experiment_name safe, but the param name still
        # ends up in the override string as `foo'bar=value` and would corrupt
        # the single-quote wrapping of the OVERRIDES bash assignment.
        with self.assertRaisesRegex(Exception, "forbidden characters"):
            self._build(
                {"foo'bar": {"abbrev": "fb", "type": "discrete", "values": [0.1]}}
            )

    def test_equals_in_name_rejected(self):
        with self.assertRaisesRegex(Exception, "forbidden characters"):
            self._build(
                {"foo=bar": {"abbrev": "fb", "type": "discrete", "values": [0.1]}}
            )

    def test_space_with_abbrev_still_rejected(self):
        with self.assertRaisesRegex(Exception, "forbidden characters"):
            self._build(
                {"foo bar": {"abbrev": "fb", "type": "discrete", "values": [0.1]}}
            )

    def test_hydra_prefixes_allowed(self):
        # +foo / ++foo / ~foo are legitimate Hydra override prefixes.
        cfg = self._build(
            {
                "+experiment": {"abbrev": "exp", "type": "discrete", "values": ["a"]},
                "++mode":      {"abbrev": "m",   "type": "discrete", "values": ["x"]},
                "~legacy":     {"abbrev": "lg",  "type": "discrete", "values": [1]},
            }
        )
        self.assertIn("+experiment", cfg.parameters)


if __name__ == "__main__":
    unittest.main()
