"""Tests for positional-token classification (`argspec.py`).

Every row of the ambiguity table gets a case here: these are the invocations
where "is this a workspace, an index, or an override?" is genuinely unclear, and
the answers are the CLI's contract.
"""

import os
import tempfile
import unittest

from hyperherd.argspec import (
    Positionals,
    TokenError,
    classify_positionals,
    looks_like_index_spec,
    shadowed_directory,
)


class TestBasicClassification(unittest.TestCase):
    def test_no_tokens(self):
        p = classify_positionals([])
        self.assertEqual(p, Positionals(workspace=".", indices=None, overrides=[]))

    def test_dot_is_a_workspace(self):
        p = classify_positionals(["."])
        self.assertEqual(p.workspace, ".")
        self.assertIsNone(p.indices)

    def test_workspace_only(self):
        p = classify_positionals(["myws"])
        self.assertEqual(p.workspace, "myws")

    def test_indices_only(self):
        p = classify_positionals(["1-4"])
        self.assertEqual(p.workspace, ".")
        self.assertEqual(p.indices, "1-4")

    def test_the_headline_case(self):
        # `herd run 1-4 batch_size=32` — the invocation this module exists for.
        p = classify_positionals(["1-4", "batch_size=32"])
        self.assertEqual(p.workspace, ".")
        self.assertEqual(p.indices, "1-4")
        self.assertEqual(p.overrides, ["batch_size=32"])

    def test_workspace_indices_overrides(self):
        p = classify_positionals(["myws", "1-4,7", "batch_size=32", "lr=0.01"])
        self.assertEqual(p.workspace, "myws")
        self.assertEqual(p.indices, "1-4,7")
        self.assertEqual(p.overrides, ["batch_size=32", "lr=0.01"])

    def test_order_independent(self):
        # Classification is per-token, so any order works.
        p = classify_positionals(["batch_size=32", "1-4", "./ws"])
        self.assertEqual(p.workspace, "./ws")
        self.assertEqual(p.indices, "1-4")
        self.assertEqual(p.overrides, ["batch_size=32"])

    def test_multiple_index_tokens_union_merge(self):
        p = classify_positionals(["1", "3", "5-7"])
        self.assertEqual(p.indices, "1,3,5-7")


class TestOverrideValues(unittest.TestCase):
    """The `=`-first rule makes awkward values safe without any extra logic."""

    def test_value_with_hyphen(self):
        p = classify_positionals(["lr=1e-3"])
        self.assertEqual(p.overrides, ["lr=1e-3"])
        self.assertIsNone(p.indices)

    def test_value_with_commas(self):
        p = classify_positionals(["tags=a,b,c"])
        self.assertEqual(p.overrides, ["tags=a,b,c"])
        self.assertIsNone(p.indices)

    def test_value_that_is_itself_an_index_spec(self):
        # `resume_from=1-4` must not be mistaken for indices.
        p = classify_positionals(["resume_from=1-4"])
        self.assertEqual(p.overrides, ["resume_from=1-4"])
        self.assertIsNone(p.indices)

    def test_path_value(self):
        p = classify_positionals(["ckpt=/scratch/run-7/last.ckpt"])
        self.assertEqual(p.overrides, ["ckpt=/scratch/run-7/last.ckpt"])

    def test_hydra_prefixed_keys(self):
        p = classify_positionals(["+extra=1", "~dropout=null"])
        self.assertEqual(p.overrides, ["+extra=1", "~dropout=null"])

    def test_empty_value_is_allowed(self):
        # `key=` is a legitimate Hydra "set to empty string".
        p = classify_positionals(["note="])
        self.assertEqual(p.overrides, ["note="])

    def test_empty_key_rejected(self):
        with self.assertRaisesRegex(TokenError, "empty key"):
            classify_positionals(["=32"])

    def test_whitespace_in_value_rejected(self):
        # The launcher word-splits $OVERRIDES, so this would silently become two
        # overrides. Better to refuse than to mangle the trial config.
        with self.assertRaisesRegex(TokenError, "whitespace"):
            classify_positionals(["msg=hello world"])

    def test_illegal_key_char_rejected(self):
        with self.assertRaisesRegex(TokenError, "illegal character"):
            classify_positionals(["ba'd=1"])


class TestWorkspaceIndexAmbiguity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cwd = os.getcwd()
        os.chdir(self.tmp)

    def tearDown(self):
        os.chdir(self.cwd)

    def test_numeric_directory_is_read_as_an_index(self):
        # BEHAVIOR CHANGE: a directory literally named `3` no longer wins.
        # Rule 2 (index spec) fires before any isdir check.
        os.mkdir("3")
        p = classify_positionals(["3"])
        self.assertEqual(p.indices, "3")
        self.assertEqual(p.workspace, ".")

    def test_dot_slash_escapes_a_numeric_directory(self):
        os.mkdir("3")
        p = classify_positionals(["./3"])
        self.assertEqual(p.workspace, "./3")
        self.assertIsNone(p.indices)

    def test_shadowed_directory_is_reported(self):
        os.mkdir("3")
        tokens = ["3"]
        p = classify_positionals(tokens)
        self.assertEqual(shadowed_directory(p, tokens), "3")

    def test_shadowed_directory_none_when_no_collision(self):
        tokens = ["3"]
        p = classify_positionals(tokens)
        self.assertIsNone(shadowed_directory(p, tokens))

    def test_range_like_directory_errors_with_a_hint(self):
        # `2024-01` parses as an index range whose start > end. Without the
        # start<=end check, slurm._parse_array_range would silently yield [].
        os.mkdir("2024-01")
        with self.assertRaises(TokenError) as ctx:
            classify_positionals(["2024-01"])
        self.assertIn("2024 > 1", str(ctx.exception))
        self.assertIn("./2024-01", str(ctx.exception))

    def test_date_like_directory_with_two_hyphens_is_a_workspace(self):
        # The regex allows at most one hyphen per term, so `2024-01-05` doesn't
        # match at all and falls through to the workspace rule.
        self.assertFalse(looks_like_index_spec("2024-01-05"))
        p = classify_positionals(["2024-01-05"])
        self.assertEqual(p.workspace, "2024-01-05")

    def test_nonexistent_workspace_still_binds_as_workspace(self):
        # No isdir gate on rule 3 — this must reach load_config so the user gets
        # "config file not found", not a token-classification error.
        p = classify_positionals(["newdir"])
        self.assertEqual(p.workspace, "newdir")

    def test_two_workspaces_rejected(self):
        with self.assertRaisesRegex(TokenError, "more than one workspace"):
            classify_positionals(["ws1", "ws2"])


class TestDisallowedKinds(unittest.TestCase):
    def test_overrides_rejected_when_disallowed(self):
        # `herd ls lr=0.1` — the user meant --where.
        with self.assertRaisesRegex(TokenError, "did you mean --where"):
            classify_positionals(["lr=0.1"], allow_overrides=False)

    def test_indices_rejected_when_disallowed(self):
        with self.assertRaisesRegex(TokenError, "takes no trial indices"):
            classify_positionals(["1-4"], allow_indices=False)

    def test_stop_accepts_indices_but_not_overrides(self):
        p = classify_positionals(["1-4"], allow_overrides=False)
        self.assertEqual(p.indices, "1-4")


if __name__ == "__main__":
    unittest.main()
