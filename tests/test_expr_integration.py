"""Integration tests: when.expr and set.expr inside Constraint + apply_constraints."""

import unittest

from hyperherd.config import Config, Constraint
from hyperherd.constraints import apply_constraints


class TestWhenExpr(unittest.TestCase):
    def test_expr_filters_trials(self):
        combos = [
            {"opt": "adam", "lr": 0.1},
            {"opt": "adam", "lr": 0.001},
            {"opt": "sgd", "lr": 0.1},
            {"opt": "sgd", "lr": 0.001},
        ]
        # exclude SGD when lr is small
        constraints = [
            Constraint(
                name="no_tiny_lr_for_sgd",
                when={"expr": "opt == 'sgd' and lr < 0.01"},
                exclude={"lr": [0.001]},
            )
        ]
        result = apply_constraints(combos, constraints)
        self.assertEqual(len(result), 3)
        for t in result:
            self.assertFalse(t.params["opt"] == "sgd" and t.params["lr"] == 0.001)

    def test_expr_anded_with_structured(self):
        combos = [
            {"opt": "sgd", "lr": 0.1},
            {"opt": "sgd", "lr": 0.001},
            {"opt": "adam", "lr": 0.1},
        ]
        # Both must match: opt=sgd AND lr<0.01.
        constraints = [
            Constraint(
                name="combined",
                when={"opt": "sgd", "expr": "lr < 0.01"},
                exclude={"lr": [0.001]},
            )
        ]
        result = apply_constraints(combos, constraints)
        # Only (sgd, 0.001) should be excluded.
        self.assertEqual(
            sorted((t.params["opt"], t.params["lr"]) for t in result),
            [("adam", 0.1), ("sgd", 0.1)],
        )


class TestSetExpr(unittest.TestCase):
    def test_set_expr_computes_extra(self):
        combos = [{"y": 1}, {"y": 3}, {"y": 5}]
        constraints = [
            Constraint(
                name="x_from_y",
                when={"y": {"ge": 0}},
                set={"x": {"expr": "20 * y"}},
            )
        ]
        result = apply_constraints(combos, constraints)
        self.assertEqual([t.extras["x"] for t in result], [20, 60, 100])

    def test_set_expr_uses_post_force_value(self):
        combos = [{"opt": "adam"}]
        constraints = [
            Constraint(
                name="adamw_warmup",
                when={"opt": "adam"},
                force={"opt": "adamw"},
                set={"sched.tag": {"expr": "opt + '-tag'"}},
            )
        ]
        result = apply_constraints(combos, constraints)
        # `force` runs before `set`, so the expression sees opt='adamw'.
        self.assertEqual(result[0].extras["sched.tag"], "adamw-tag")

    def test_set_literal_dict_passes_through(self):
        # A non-{expr} dict value is preserved verbatim.
        combos = [{"y": 1}]
        constraints = [
            Constraint(
                name="literal",
                when={"y": 1},
                set={"thing": {"foo": "bar"}},
            )
        ]
        result = apply_constraints(combos, constraints)
        self.assertEqual(result[0].extras["thing"], {"foo": "bar"})


class TestPlusPrefixInExpr(unittest.TestCase):
    def test_plus_prefix_param_exposed_unprefixed(self):
        combos = [{"+experiment": "small"}, {"+experiment": "large"}]
        constraints = [
            Constraint(
                name="filter_small",
                when={"expr": "experiment == 'small'"},
                set={"note": {"expr": "experiment + '-tagged'"}},
            )
        ]
        result = apply_constraints(combos, constraints)
        self.assertEqual(len(result), 2)
        small = next(t for t in result if t.params["+experiment"] == "small")
        large = next(t for t in result if t.params["+experiment"] == "large")
        self.assertEqual(small.extras.get("note"), "small-tagged")
        # large was not matched by `when`, so no note.
        self.assertNotIn("note", large.extras)


class TestConfigValidation(unittest.TestCase):
    def _cfg(self, **conditions):
        return {
            "name": "t",
            "workspace": "/tmp",
            "grid": "all",
            "parameters": {
                "y": {"type": "discrete", "abbrev": "y", "values": [1, 3, 5]},
                "opt": {"type": "discrete", "abbrev": "opt", "values": ["adam", "sgd"]},
            },
            "conditions": [conditions],
        }

    def test_valid_when_expr_accepted(self):
        Config.model_validate(
            self._cfg(name="ok", when={"expr": "y > 1"}, exclude={"y": [3]})
        )

    def test_valid_set_expr_accepted(self):
        Config.model_validate(
            self._cfg(
                name="ok",
                when={"y": {"ge": 0}},
                set={"x": {"expr": "20 * y"}},
            )
        )

    def test_unknown_name_in_when_expr_rejected(self):
        with self.assertRaises(Exception) as ctx:
            Config.model_validate(
                self._cfg(name="bad", when={"expr": "z > 1"}, exclude={"y": [3]})
            )
        self.assertIn("Unknown name", str(ctx.exception))

    def test_disallowed_call_in_set_expr_rejected(self):
        with self.assertRaises(Exception) as ctx:
            Config.model_validate(
                self._cfg(
                    name="bad",
                    when={"y": {"ge": 0}},
                    set={"x": {"expr": "abs(y)"}},
                )
            )
        self.assertIn("Disallowed function", str(ctx.exception))

    def test_min_max_in_set_expr_accepted(self):
        Config.model_validate(
            self._cfg(
                name="ok",
                when={"y": {"ge": 0}},
                set={"x": {"expr": "min(max(y, 0), 10)"}},
            )
        )

    def test_collision_rejected_at_load(self):
        cfg = {
            "name": "t",
            "workspace": "/tmp",
            "grid": "all",
            "parameters": {
                "experiment": {
                    "type": "discrete", "abbrev": "e", "values": ["a"],
                },
                "+experiment": {
                    "type": "discrete", "abbrev": "ep", "values": ["b"],
                },
            },
        }
        with self.assertRaises(Exception) as ctx:
            Config.model_validate(cfg)
        self.assertIn("collision", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
