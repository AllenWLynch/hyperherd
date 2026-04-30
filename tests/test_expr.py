"""Tests for the whitelisted expression evaluator."""

import unittest

from hyperherd.expr import (
    ExprError,
    eval_expr,
    sanitized_namespace,
    validate_expr,
    validate_namespace_keys,
)


class TestValidate(unittest.TestCase):
    NAMES = {"x", "y", "opt"}

    def test_arithmetic_ok(self):
        validate_expr("20 * y + 1", self.NAMES)

    def test_comparison_ok(self):
        validate_expr("x > y and opt == 'adam'", self.NAMES)

    def test_membership_ok(self):
        validate_expr("opt in ('adam', 'sgd')", self.NAMES)

    def test_conditional_ok(self):
        validate_expr("x if y > 0 else -x", self.NAMES)

    def test_unknown_name_rejected(self):
        with self.assertRaises(ExprError) as ctx:
            validate_expr("z + 1", self.NAMES)
        self.assertIn("Unknown name", str(ctx.exception))

    def test_min_max_allowed(self):
        validate_expr("max(0, x)", self.NAMES)
        validate_expr("min(x, max(y, 1))", self.NAMES)

    def test_other_call_rejected(self):
        with self.assertRaises(ExprError):
            validate_expr("abs(x)", self.NAMES)

    def test_kwarg_call_rejected(self):
        with self.assertRaises(ExprError):
            validate_expr("min(x, default=0)", self.NAMES)

    def test_attribute_rejected(self):
        with self.assertRaises(ExprError):
            validate_expr("x.real", self.NAMES)

    def test_subscript_rejected(self):
        with self.assertRaises(ExprError):
            validate_expr("x[0]", self.NAMES)

    def test_lambda_rejected(self):
        with self.assertRaises(ExprError):
            validate_expr("(lambda v: v + 1)(x)", self.NAMES)

    def test_walrus_rejected(self):
        # NamedExpr is not in the allowed list.
        with self.assertRaises(ExprError):
            validate_expr("(z := x)", self.NAMES)

    def test_syntax_error(self):
        with self.assertRaises(ExprError):
            validate_expr("x +", self.NAMES)


class TestEval(unittest.TestCase):
    def test_basic_arithmetic(self):
        self.assertEqual(eval_expr("20 * y", {"y": 3}), 60)

    def test_boolean(self):
        self.assertTrue(eval_expr("x > 0 and y < 10", {"x": 1, "y": 3}))

    def test_membership(self):
        self.assertTrue(eval_expr("opt in ('adam', 'sgd')", {"opt": "adam"}))

    def test_conditional(self):
        self.assertEqual(eval_expr("x if x > 0 else -x", {"x": -5}), 5)

    def test_min_max_eval(self):
        self.assertEqual(eval_expr("max(0, x)", {"x": -3}), 0)
        self.assertEqual(eval_expr("min(x, 10)", {"x": 42}), 10)

    def test_no_builtins(self):
        # Even if user smuggled `abs` past validate_expr (they can't, but verify
        # the runtime sandbox), eval_expr would fail because builtins are gone.
        with self.assertRaises(NameError):
            eval_expr("abs(x)", {"x": -1})


class TestNamespaceSanitization(unittest.TestCase):
    def test_strips_plus_prefix(self):
        ns = sanitized_namespace({"+experiment": "foo", "lr": 0.01})
        self.assertEqual(ns, {"experiment": "foo", "lr": 0.01})

    def test_strips_double_plus(self):
        self.assertEqual(
            sanitized_namespace({"++foo": 1}), {"foo": 1}
        )

    def test_strips_tilde(self):
        self.assertEqual(sanitized_namespace({"~foo": None}), {"foo": None})

    def test_collision_rejected(self):
        with self.assertRaises(ExprError):
            validate_namespace_keys(["foo", "+foo"])

    def test_no_collision_when_disjoint(self):
        # Same prefix on the same name is fine — only different originals collide.
        validate_namespace_keys(["+foo", "+bar", "baz"])


if __name__ == "__main__":
    unittest.main()
