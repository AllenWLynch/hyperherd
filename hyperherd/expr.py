"""Whitelisted AST evaluator for `when.expr` and `set.<key>.expr` expressions.

The evaluator is deliberately minimal: arithmetic, comparisons, boolean logic,
membership tests, and conditional expressions over the trial's parameter
namespace. No function calls, no attribute access, no subscripting — anything
that could load arbitrary code is rejected at validation time.
"""

import ast
from typing import Any, Dict, Iterable, Set


_ALLOWED_NODES = (
    ast.Expression,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.UnaryOp,
    ast.BinOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.Tuple,
    ast.List,
)

_ALLOWED_OPS = (
    # arithmetic
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow, ast.FloorDiv,
    # unary
    ast.UAdd, ast.USub, ast.Not,
    # boolean
    ast.And, ast.Or,
    # comparison
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
)

# Whitelisted callables — bounding/clamping is the natural use case.
# Kept tiny on purpose; expand only when a clear pattern emerges.
_ALLOWED_CALLS = {"min": min, "max": max}


class ExprError(ValueError):
    """Raised when an expression fails validation or evaluation."""


def _sanitize_key(name: str) -> str:
    """Strip Hydra override prefixes so a param name becomes a valid Python identifier.

    `+foo` / `++foo` / `~foo` all expose as `foo` in the expression namespace.
    """
    s = name.lstrip("+").lstrip("~")
    return s


def sanitized_namespace(params: Dict[str, Any]) -> Dict[str, Any]:
    """Build the eval namespace from a trial's params, stripping `+`/`~` prefixes.

    Caller is responsible for ensuring no two params collide after sanitization
    (validate_namespace_keys raises at config-load time if they do).
    """
    return {_sanitize_key(k): v for k, v in params.items()}


def validate_namespace_keys(param_names: Iterable[str]) -> None:
    """Raise if two parameter names would collide after prefix-stripping.

    e.g. `experiment` and `+experiment` both expose as `experiment` — ambiguous
    in expressions, so reject at config-load time rather than at eval time.
    """
    seen: Dict[str, str] = {}
    for name in param_names:
        s = _sanitize_key(name)
        if s in seen and seen[s] != name:
            raise ExprError(
                f"Parameter name collision in expression namespace: "
                f"{seen[s]!r} and {name!r} both expose as {s!r}. "
                f"Rename one to avoid ambiguity in `when.expr` / `set.expr`."
            )
        seen[s] = name


def validate_expr(expr_str: str, allowed_names: Set[str]) -> None:
    """Parse and AST-check an expression string. Names must be in `allowed_names`.

    Raises ExprError on syntax error, disallowed node type, or unknown name.
    """
    try:
        tree = ast.parse(expr_str, mode="eval")
    except SyntaxError as e:
        raise ExprError(f"Invalid expression {expr_str!r}: {e.msg}") from e

    for node in ast.walk(tree):
        if isinstance(node, _ALLOWED_NODES):
            continue
        if isinstance(node, _ALLOWED_OPS):
            continue
        if isinstance(node, ast.Call):
            # Only bare-name calls to whitelisted functions; no kwargs/starargs.
            if not isinstance(node.func, ast.Name):
                raise ExprError(
                    f"Only direct calls to {sorted(_ALLOWED_CALLS)} are allowed "
                    f"in expression {expr_str!r}"
                )
            if node.func.id not in _ALLOWED_CALLS:
                raise ExprError(
                    f"Disallowed function {node.func.id!r} in expression "
                    f"{expr_str!r} (allowed: {sorted(_ALLOWED_CALLS)})"
                )
            if node.keywords:
                raise ExprError(
                    f"Keyword arguments are not allowed in expression {expr_str!r}"
                )
            continue
        raise ExprError(
            f"Disallowed syntax in expression {expr_str!r}: "
            f"{type(node).__name__} is not permitted "
            f"(allowed: arithmetic, comparison, boolean, membership, "
            f"{sorted(_ALLOWED_CALLS)})"
        )

    callable_names = set(_ALLOWED_CALLS)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id not in allowed_names \
                and node.id not in callable_names:
            raise ExprError(
                f"Unknown name {node.id!r} in expression {expr_str!r}. "
                f"Available names: {sorted(allowed_names)}"
            )


def eval_expr(expr_str: str, namespace: Dict[str, Any]) -> Any:
    """Evaluate a pre-validated expression against the given namespace.

    Caller must have already invoked validate_expr() at config-load time —
    eval_expr does NOT re-check the AST. Globals/builtins are stripped to
    block any sneaky access via shadowed names.
    """
    code = compile(expr_str, "<hyperherd-expr>", "eval")
    # Whitelisted callables ride alongside params; param names take precedence
    # if a sweep ever has a param literally called "min" or "max".
    scope = {**_ALLOWED_CALLS, **namespace}
    return eval(code, {"__builtins__": {}}, scope)
