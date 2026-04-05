"""Tests for search space generation."""

import math
import unittest

from hyperwhip.config import Config, HydraConfig, ParameterSpec, SearchConfig, SlurmConfig
from hyperwhip.search import generate_combinations, _discretize_continuous


def _make_config(parameters, mode="grid", defaults=None):
    return Config(
        name="test",
        workspace="/tmp/test_ws",
        search=SearchConfig(mode=mode, defaults=defaults),
        slurm=SlurmConfig(),
        hydra=HydraConfig(),
        launcher="./launch.sh",
        parameters=parameters,
        constraints=[],
    )


class TestDiscretizeContinuous(unittest.TestCase):
    def test_linear(self):
        p = ParameterSpec(name="x", type="continuous", low=0.0, high=1.0, scale="linear", steps=5)
        vals = _discretize_continuous(p)
        self.assertEqual(len(vals), 5)
        self.assertAlmostEqual(vals[0], 0.0)
        self.assertAlmostEqual(vals[4], 1.0)
        self.assertAlmostEqual(vals[2], 0.5)

    def test_log(self):
        p = ParameterSpec(name="lr", type="continuous", low=1e-4, high=1e-2, scale="log", steps=3)
        vals = _discretize_continuous(p)
        self.assertEqual(len(vals), 3)
        self.assertAlmostEqual(vals[0], 1e-4)
        self.assertAlmostEqual(vals[1], 1e-3)
        self.assertAlmostEqual(vals[2], 1e-2)

    def test_single_step(self):
        p = ParameterSpec(name="x", type="continuous", low=5.0, high=10.0, scale="linear", steps=1)
        vals = _discretize_continuous(p)
        self.assertEqual(vals, [5.0])


class TestGridSearch(unittest.TestCase):
    def test_all_discrete(self):
        params = [
            ParameterSpec(name="a", type="discrete", values=[1, 2]),
            ParameterSpec(name="b", type="discrete", values=["x", "y", "z"]),
        ]
        config = _make_config(params)
        combos = generate_combinations(config)
        self.assertEqual(len(combos), 6)  # 2 * 3
        # Check all present
        a_vals = {c["a"] for c in combos}
        b_vals = {c["b"] for c in combos}
        self.assertEqual(a_vals, {1, 2})
        self.assertEqual(b_vals, {"x", "y", "z"})

    def test_mixed_types(self):
        params = [
            ParameterSpec(name="lr", type="continuous", low=0.1, high=1.0, scale="linear", steps=3),
            ParameterSpec(name="opt", type="discrete", values=["a", "b"]),
        ]
        config = _make_config(params)
        combos = generate_combinations(config)
        self.assertEqual(len(combos), 6)  # 3 * 2

    def test_single_param(self):
        params = [ParameterSpec(name="x", type="discrete", values=[10, 20, 30])]
        config = _make_config(params)
        combos = generate_combinations(config)
        self.assertEqual(len(combos), 3)


class TestAxesSearch(unittest.TestCase):
    def test_basic_axes(self):
        params = [
            ParameterSpec(name="a", type="discrete", values=[1, 2, 3]),
            ParameterSpec(name="b", type="discrete", values=["x", "y"]),
        ]
        defaults = {"a": 1, "b": "x"}
        config = _make_config(params, mode="axes", defaults=defaults)
        combos = generate_combinations(config)
        # base: {a=1, b=x}
        # vary a: {a=2, b=x}, {a=3, b=x}
        # vary b: {a=1, b=y}
        # Total: 1 + 2 + 1 = 4
        self.assertEqual(len(combos), 4)
        self.assertEqual(combos[0], {"a": 1, "b": "x"})

    def test_axes_no_duplicates(self):
        params = [
            ParameterSpec(name="a", type="discrete", values=[1, 2]),
            ParameterSpec(name="b", type="discrete", values=[10, 20]),
        ]
        defaults = {"a": 1, "b": 10}
        config = _make_config(params, mode="axes", defaults=defaults)
        combos = generate_combinations(config)
        # base + 1 for a + 1 for b = 3
        self.assertEqual(len(combos), 3)


if __name__ == "__main__":
    unittest.main()
