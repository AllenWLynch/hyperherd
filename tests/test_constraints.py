"""Tests for constraint evaluation and filtering."""

import unittest

from hyperwhip.config import Constraint
from hyperwhip.constraints import apply_constraints


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
        # sgd+0.1 should be gone
        for c in result:
            self.assertFalse(c["opt"] == "sgd" and c["lr"] == 0.1)

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
        for c in result:
            if c["opt"] == "adamw":
                self.assertEqual(c["wd"], 0.01)
        # sgd combo should be unchanged
        sgd = [c for c in result if c["opt"] == "sgd"]
        self.assertEqual(len(sgd), 1)
        self.assertEqual(sgd[0]["wd"], 0.0)

    def test_force_deduplication(self):
        combos = [
            {"opt": "adamw", "lr": 0.1, "wd": 0.0},
            {"opt": "adamw", "lr": 0.1, "wd": 0.1},
        ]
        constraints = [
            Constraint(name="force", when={"opt": "adamw"}, force={"wd": 0.01})
        ]
        result = apply_constraints(combos, constraints)
        # Both collapse to {opt=adamw, lr=0.1, wd=0.01}, deduplicated to 1
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
        # sgd+0.1 excluded, adamw gets wd=0.01
        self.assertEqual(len(result), 3)


if __name__ == "__main__":
    unittest.main()
