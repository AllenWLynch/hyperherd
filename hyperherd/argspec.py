"""Positional-token classification for the `herd` CLI.

Commands like `run` and `stop` accept up to three kinds of positional argument
in any order::

    herd run myws 1-4,7 batch_size=32 lr=0.01
           ^ws  ^indices ^overrides

argparse cannot disambiguate these itself: with `workspace nargs="?"` +
`indices nargs="?"` its positional matcher is greedy, so `herd run 1-4 bs=32`
binds ``workspace="1-4"`` before any of our code runs. So the parsers take a
single catch-all `pos nargs="*"` and hand the raw tokens here.

The classification order is load-bearing:

1. A token containing ``=`` is an **override**. Checking this *first* is what
   makes override values containing ``-`` or ``,`` safe for free (``lr=1e-3``,
   ``tags=a,b,c``) — the index-spec regex never sees them, because those
   characters only ever appear after the ``=``.
2. A token matching :data:`INDEX_SPEC_RE` is an **index spec**. Multiple index
   tokens union-merge, so ``herd stop 1 3 5-7`` works.
3. Anything else is the **workspace**. Deliberately *not* gated on
   ``os.path.isdir`` — a nonexistent path must still reach `load_config`, whose
   "config file not found" error is the useful one.

Rule 2 before rule 3 means a directory literally named ``3`` is read as a trial
index, inverting the old behavior. ``./3`` disambiguates, and the caller is
expected to warn (see :func:`shadowed_directory`).
"""

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

# One or more comma-separated terms, each `N` or `N-M`. Note a term admits at
# most one hyphen, so a date-like directory (`2024-01-05`) does NOT match and
# falls through to the workspace rule.
INDEX_SPEC_RE = re.compile(r"^\d+(-\d+)?(,\d+(-\d+)?)*$")

# Characters that cannot appear in an override *key*. Mirrors the parameter-name
# rule in config.py `_validate_param_name_chars`: these break the space-separated
# `key=value` override string the launcher receives as `$1`.
_FORBIDDEN_KEY_CHARS = set("'\"`=\\\n\r\t ")


class TokenError(ValueError):
    """A positional token that can't be classified, or one that conflicts."""


@dataclass(frozen=True)
class Positionals:
    """The classified positional arguments of a command."""

    workspace: str                # "." when not given
    indices: Optional[str]        # normalized SLURM-style spec, or None
    overrides: List[str]          # raw "KEY=VALUE" tokens, in given order


def looks_like_index_spec(token: str) -> bool:
    return bool(INDEX_SPEC_RE.match(token))


def shadowed_directory(pos: Positionals, tokens: Sequence[str]) -> Optional[str]:
    """The token we read as an index spec that is *also* an existing directory.

    `herd run 3` means "trial 3" now, even from a parent directory containing a
    `3/` workspace. Callers surface this as a note pointing at `./3`.
    """
    if not pos.indices:
        return None
    for tok in tokens:
        if "=" not in tok and looks_like_index_spec(tok) and os.path.isdir(tok):
            return tok
    return None


def _validate_override(token: str) -> None:
    key, value = token.split("=", 1)
    if not key:
        raise TokenError(f"override {token!r} has an empty key")
    bad = sorted(set(key) & _FORBIDDEN_KEY_CHARS)
    if bad:
        raise TokenError(
            f"override key {key!r} contains illegal character(s) {''.join(bad)!r}"
        )
    # The launcher receives the override string as one shell word and splits it
    # on whitespace, so a space inside a value would silently become two
    # overrides. Reject it rather than mangle the trial's config.
    if any(c.isspace() for c in value):
        raise TokenError(
            f"override {token!r} has whitespace in its value; "
            f"the launcher splits overrides on whitespace, so this cannot be passed through"
        )


def _validate_index_spec(token: str) -> None:
    for part in token.split(","):
        if "-" not in part:
            continue
        start, end = (int(x) for x in part.split("-", 1))
        if start > end:
            raise TokenError(
                f"invalid index range {part!r} in {token!r}: {start} > {end}"
                + (f" (for the directory, write ./{token})"
                   if os.path.isdir(token) else "")
            )


def classify_positionals(
    tokens: Sequence[str],
    *,
    allow_indices: bool = True,
    allow_overrides: bool = True,
) -> Positionals:
    """Sort `tokens` into a workspace, an index spec, and override tokens.

    Raises `TokenError` with a user-facing message on anything ambiguous or
    disallowed; callers pass that straight to `parser.error`.
    """
    workspace: Optional[str] = None
    index_tokens: List[str] = []
    overrides: List[str] = []

    for tok in tokens:
        if "=" in tok:
            if not allow_overrides:
                raise TokenError(
                    f"unexpected override {tok!r} — did you mean --where {tok}?"
                )
            _validate_override(tok)
            overrides.append(tok)
        elif looks_like_index_spec(tok):
            if not allow_indices:
                raise TokenError(f"this command takes no trial indices (got {tok!r})")
            _validate_index_spec(tok)
            index_tokens.append(tok)
        elif workspace is not None:
            raise TokenError(
                f"more than one workspace path given: {workspace!r} and {tok!r}"
            )
        else:
            workspace = tok

    indices = ",".join(index_tokens) if index_tokens else None
    return Positionals(
        workspace=workspace if workspace is not None else ".",
        indices=indices,
        overrides=overrides,
    )
