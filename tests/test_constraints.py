"""Tests for constraint evaluation and filtering."""

import unittest

from hyperherd.config import Constraint
from hyperherd.constraints import apply_constraints


class TestNumericStringCoercion(unittest.TestCase):
    """YAML 1.1 reads `1e-3` (no decimal/sign) as a string. Without coercion
    at config-load time, comparison matchers silently fail to match."""

    def test_le_with_yaml_style_scientific_notation(self):
        # Same shape as a YAML loader would produce for `{le: 1e-3}`.
        c = Constraint(
            name="t",
            when={"lr": {"le": "1e-3"}},
            force={"clamped": 1},
        )
        out = apply_constraints([{"lr": 0.0001}, {"lr": 0.01}], [c])
        self.assertEqual(out[0].params.get("clamped"), 1)
        self.assertNotIn("clamped", out[1].params)

    def test_exclude_list_with_string_scientific(self):
        c = Constraint(
            name="t",
            when={"lr": "0.1"},
            exclude={"lr": ["1e-1"]},
        )
        out = apply_constraints([{"lr": 0.1}, {"lr": 0.01}], [c])
        # 0.1 should be excluded by the string-scientific entry.
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].params["lr"], 0.01)

    def test_genuinely_string_value_left_alone(self):
        c = Constraint(
            name="t",
            when={"opt": "adam"},
            exclude={"opt": ["sgd"]},
        )
        # "adam" / "sgd" are not float-parseable; they must remain strings.
        self.assertEqual(c.when["opt"], "adam")
        self.assertEqual(c.exclude["opt"], ["sgd"])


class TestExcludeConstraint(unittest.TestCase):
    def test_basic_exclude(self):
        combos = [
            {"opt": "sgd", "lr": 0.1},
            {"opt": "sgd", "lr": 0.01},
            {"opt": "adam", "lr": 0.1},
            {"opt": "adam", "lr": 0.01},
        ]
        constraints = [
            Constraint(
                name="no_high_lr_for_sgd",
                when={"opt": "sgd"},
                exclude={"lr": [0.1]},
            )
        ]
        result = apply_constraints(combos, constraints)
        self.assertEqual(len(result), 3)
        for t in result:
            self.assertFalse(t.params["opt"] == "sgd" and t.params["lr"] == 0.1)

    def test_exclude_no_match(self):
        combos = [
            {"opt": "adam", "lr": 0.1},
            {"opt": "adam", "lr": 0.01},
        ]
        constraints = [
            Constraint(name="test", when={"opt": "sgd"}, exclude={"lr": [0.1]})
        ]
        result = apply_constraints(combos, constraints)
        self.assertEqual(len(result), 2)


class TestForceConstraint(unittest.TestCase):
    def test_basic_force(self):
        combos = [
            {"opt": "adamw", "lr": 0.1, "wd": 0.0},
            {"opt": "adamw", "lr": 0.01, "wd": 0.0},
            {"opt": "sgd", "lr": 0.1, "wd": 0.0},
        ]
        constraints = [
            Constraint(
                name="force_wd",
                when={"opt": "adamw"},
                force={"wd": 0.01},
            )
        ]
        result = apply_constraints(combos, constraints)
        for t in result:
            if t.params["opt"] == "adamw":
                self.assertEqual(t.params["wd"], 0.01)
        sgd = [t for t in result if t.params["opt"] == "sgd"]
        self.assertEqual(len(sgd), 1)
        self.assertEqual(sgd[0].params["wd"], 0.0)

    def test_force_deduplication(self):
        combos = [
            {"opt": "adamw", "lr": 0.1, "wd": 0.0},
            {"opt": "adamw", "lr": 0.1, "wd": 0.1},
        ]
        constraints = [
            Constraint(name="force", when={"opt": "adamw"}, force={"wd": 0.01})
        ]
        result = apply_constraints(combos, constraints)
        self.assertEqual(len(result), 1)


class TestMultipleConstraints(unittest.TestCase):
    def test_chained(self):
        combos = [
            {"opt": "sgd", "lr": 0.1, "wd": 0.0},
            {"opt": "sgd", "lr": 0.01, "wd": 0.0},
            {"opt": "adamw", "lr": 0.1, "wd": 0.0},
            {"opt": "adamw", "lr": 0.01, "wd": 0.0},
        ]
        constraints = [
            Constraint(name="c1", when={"opt": "sgd"}, exclude={"lr": [0.1]}),
            Constraint(name="c2", when={"opt": "adamw"}, force={"wd": 0.01}),
        ]
        result = apply_constraints(combos, constraints)
        self.assertEqual(len(result), 3)


class TestWhenListMatcher(unittest.TestCase):
    """A list value in `when` is OR-match across the listed values."""

    def test_or_match_excludes_multiple_optimizers(self):
        combos = [
            {"opt": "sgd", "lr": 0.1},
            {"opt": "momentum_sgd", "lr": 0.1},
            {"opt": "adam", "lr": 0.1},
        ]
        constraints = [
            Constraint(
                name="no_high_lr_for_sgd_family",
                when={"opt": ["sgd", "momentum_sgd"]},
                exclude={"lr": [0.1]},
            )
        ]
        result = apply_constraints(combos, constraints)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].params["opt"], "adam")


class TestWhenOperatorMatcher(unittest.TestCase):
    def test_gt_filter(self):
        combos = [
            {"opt": "sgd", "lr": 0.001},
            {"opt": "sgd", "lr": 0.05},
            {"opt": "sgd", "lr": 0.1},
        ]
        constraints = [
            Constraint(
                name="sgd_no_high_lr",
                when={"opt": "sgd", "lr": {"gt": 0.01}},
                exclude={"lr": [0.05, 0.1]},
            )
        ]
        result = apply_constraints(combos, constraints)
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0].params["lr"], 0.001)

    def test_in_operator(self):
        combos = [
            {"bs": 32}, {"bs": 64}, {"bs": 128},
        ]
        constraints = [
            Constraint(
                name="big_batch_only",
                when={"bs": {"in": [64, 128]}},
                force={"bs": 256},
            )
        ]
        result = apply_constraints(combos, constraints)
        # bs=32 untouched, bs=64 and bs=128 both forced to 256, dedup to one
        bs_values = sorted(t.params["bs"] for t in result)
        self.assertEqual(bs_values, [32, 256])

    def test_invalid_operator_rejected(self):
        with self.assertRaises(Exception):
            Constraint(
                name="bad",
                when={"lr": {"approx": 0.01}},
                exclude={"lr": [0.1]},
            )

    def test_two_operator_keys_rejected(self):
        with self.assertRaises(Exception):
            Constraint(
                name="bad",
                when={"lr": {"gt": 0.01, "lt": 0.1}},
                exclude={"lr": [0.5]},
            )


class TestSetField(unittest.TestCase):
    def test_set_injects_extras(self):
        combos = [
            {"opt": "adamw", "lr": 0.001},
            {"opt": "sgd", "lr": 0.001},
        ]
        constraints = [
            Constraint(
                name="adamw_warmup",
                when={"opt": "adamw"},
                set={"scheduler.type": "cosine", "scheduler.warmup_steps": 1000},
            )
        ]
        result = apply_constraints(combos, constraints)
        self.assertEqual(len(result), 2)
        adamw = next(t for t in result if t.params["opt"] == "adamw")
        sgd = next(t for t in result if t.params["opt"] == "sgd")
        self.assertEqual(adamw.extras["scheduler.type"], "cosine")
        self.assertEqual(adamw.extras["scheduler.warmup_steps"], 1000)
        self.assertEqual(sgd.extras, {})

    def test_set_only_constraint_allowed(self):
        # `set` alone (no exclude/force) should be valid
        c = Constraint(
            name="just_set",
            when={"opt": "adamw"},
            set={"foo.bar": 1},
        )
        self.assertEqual(c.set, {"foo.bar": 1})

    def test_no_action_rejected(self):
        with self.assertRaises(Exception):
            Constraint(name="empty", when={"opt": "adamw"})

    def test_multiple_constraints_accumulate_extras(self):
        combos = [{"opt": "adamw", "stage": "pretrain"}]
        constraints = [
            Constraint(name="c1", when={"opt": "adamw"}, set={"a": 1}),
            Constraint(name="c2", when={"stage": "pretrain"}, set={"b": 2}),
        ]
        result = apply_constraints(combos, constraints)
        self.assertEqual(result[0].extras, {"a": 1, "b": 2})


if __name__ == "__main__":
    unittest.main()
