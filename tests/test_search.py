"""Tests for search space generation."""

import math
import unittest

from hyperwhip.config import Config
from hyperwhip.search import generate_combinations, _discretize_continuous, _get_param_values


def _make_config(parameters, grid=None, defaults=None):
    raw = {
        "name": "test",
        "workspace": "/tmp/test_ws",
        "parameters": parameters,
        "launcher": "./launch.sh",
    }
    if grid is not None:
        raw["grid"] = grid
    if defaults is not None:
        raw["defaults"] = defaults
    return Config.model_validate(raw)


class TestDiscretizeContinuous(unittest.TestCase):
    def test_linear(self):
        config = _make_config(
            {"x": {"abbrev": "x", "type": "continuous", "low": 0.0, "high": 1.0, "steps": 5}},
            grid="all",
        )
        vals = _get_param_values(config.parameters["x"])
        self.assertEqual(len(vals), 5)
        self.assertAlmostEqual(vals[0], 0.0)
        self.assertAlmostEqual(vals[4], 1.0)
        self.assertAlmostEqual(vals[2], 0.5)

    def test_log(self):
        config = _make_config(
            {"lr": {"abbrev": "lr", "type": "continuous", "low": 1e-4, "high": 1e-2, "scale": "log", "steps": 3}},
            grid="all",
        )
        vals = _get_param_values(config.parameters["lr"])
        self.assertEqual(len(vals), 3)
        self.assertAlmostEqual(vals[0], 1e-4)
        self.assertAlmostEqual(vals[1], 1e-3)
        self.assertAlmostEqual(vals[2], 1e-2)

    def test_single_step(self):
        config = _make_config(
            {"x": {"abbrev": "x", "type": "continuous", "low": 5.0, "high": 10.0, "steps": 1}},
            grid="all",
        )
        vals = _get_param_values(config.parameters["x"])
        self.assertEqual(vals, [5.0])


class TestFullGrid(unittest.TestCase):
    def test_all_discrete(self):
        config = _make_config(
            {
                "a": {"abbrev": "a", "type": "discrete", "values": [1, 2]},
                "b": {"abbrev": "b", "type": "discrete", "values": ["x", "y", "z"]},
            },
            grid="all",
        )
        combos = generate_combinations(config)
        self.assertEqual(len(combos), 6)  # 2 * 3

    def test_mixed_types(self):
        config = _make_config(
            {
                "lr": {"abbrev": "lr", "type": "continuous", "low": 0.1, "high": 1.0, "steps": 3},
                "opt": {"abbrev": "opt", "type": "discrete", "values": ["a", "b"]},
            },
            grid="all",
        )
        combos = generate_combinations(config)
        self.assertEqual(len(combos), 6)  # 3 * 2

    def test_single_param(self):
        config = _make_config(
            {"x": {"abbrev": "x", "type": "discrete", "values": [10, 20, 30]}},
            grid="all",
        )
        combos = generate_combinations(config)
        self.assertEqual(len(combos), 3)


class TestPartialGrid(unittest.TestCase):
    def test_grid_subset(self):
        config = _make_config(
            {
                "a": {"abbrev": "a", "type": "discrete", "values": [1, 2, 3]},
                "b": {"abbrev": "b", "type": "discrete", "values": ["x", "y"]},
                "c": {"abbrev": "c", "type": "discrete", "values": [10, 20]},
            },
            grid=["a", "b"],
            defaults={"a": 1, "b": "x", "c": 10},
        )
        combos = generate_combinations(config)
        # 3 * 2 = 6, c always = 10
        self.assertEqual(len(combos), 6)
        for c in combos:
            self.assertEqual(c["c"], 10)

    def test_single_grid_param(self):
        config = _make_config(
            {
                "a": {"abbrev": "a", "type": "discrete", "values": [1, 2, 3]},
                "b": {"abbrev": "b", "type": "discrete", "values": ["x", "y"]},
            },
            grid=["a"],
            defaults={"a": 1, "b": "x"},
        )
        combos = generate_combinations(config)
        # 3 combos, b always = "x"
        self.assertEqual(len(combos), 3)
        for c in combos:
            self.assertEqual(c["b"], "x")


class TestOneAtATime(unittest.TestCase):
    def test_basic(self):
        config = _make_config(
            {
                "a": {"abbrev": "a", "type": "discrete", "values": [1, 2, 3]},
                "b": {"abbrev": "b", "type": "discrete", "values": ["x", "y"]},
            },
            defaults={"a": 1, "b": "x"},
        )
        combos = generate_combinations(config)
        # base: {a=1, b=x}
        # vary a: {a=2, b=x}, {a=3, b=x}
        # vary b: {a=1, b=y}
        # Total: 4
        self.assertEqual(len(combos), 4)
        self.assertEqual(combos[0], {"a": 1, "b": "x"})

    def test_no_duplicates(self):
        config = _make_config(
            {
                "a": {"abbrev": "a", "type": "discrete", "values": [1, 2]},
                "b": {"abbrev": "b", "type": "discrete", "values": [10, 20]},
            },
            defaults={"a": 1, "b": 10},
        )
        combos = generate_combinations(config)
        # base + 1 for a + 1 for b = 3
        self.assertEqual(len(combos), 3)


if __name__ == "__main__":
    unittest.main()
