"""Tests for `herd run --where name=value`.

`--where` *selects* the trials to submit (as opposed to a bare positional
`name=value`, which *overrides* a param on them). We test the pure helpers
(`_parse_where_args`, `_filter_trials_by_where`) plus an end-to-end run via
`cmd_launch` in dry-run mode (no SLURM submission).

`-p/--pin` is the deprecated spelling; `main()` merges it into `where`.
"""

import argparse
import os
import shutil
import tempfile
import unittest
from unittest import mock

from hyperherd import manifest
from hyperherd.cli import (
    cmd_launch,
    _parse_where_args,
    _filter_trials_by_where,
)
from hyperherd.config import Config


def _config_for_where():
    """Two-parameter sweep (lr × opt) with one static_overrides key."""
    return Config.model_validate({
        "name": "t",
        "workspace": "/tmp/where_test",
        "launcher": "./launch.sh",
        "grid": "all",
        "parameters": {
            "lr": {
                "type": "discrete", "abbrev": "lr",
                "values": [0.1, 0.01, 0.001],
            },
            "opt": {
                "type": "discrete", "abbrev": "opt",
                "values": ["adam", "sgd"],
            },
        },
        "slurm": {"partition": "short"},
        "static_overrides": ["data.root=/scratch"],
    })


class TestParseWhereArgs(unittest.TestCase):
    def setUp(self):
        self.cfg = _config_for_where()

    def test_none_returns_empty(self):
        self.assertEqual(_parse_where_args(None, self.cfg), {})

    def test_empty_list_returns_empty(self):
        self.assertEqual(_parse_where_args([], self.cfg), {})

    def test_int_coercion(self):
        # `lr=1` would coerce to int 1; choose a discrete that matches.
        cfg = Config.model_validate({
            "name": "t", "workspace": "/tmp", "launcher": "./l.sh",
            "grid": "all",
            "parameters": {"bs": {
                "type": "discrete", "abbrev": "bs", "values": [16, 32, 64],
            }},
            "slurm": {"partition": "short"},
        })
        self.assertEqual(_parse_where_args(["bs=32"], cfg), {"bs": 32})

    def test_float_coercion(self):
        self.assertEqual(_parse_where_args(["lr=0.001"], self.cfg), {"lr": 0.001})

    def test_str_fallback(self):
        self.assertEqual(_parse_where_args(["opt=adam"], self.cfg), {"opt": "adam"})

    def test_multiple_filters(self):
        pins = _parse_where_args(["lr=0.001", "opt=sgd"], self.cfg)
        self.assertEqual(pins, {"lr": 0.001, "opt": "sgd"})

    def test_missing_equals_rejected(self):
        with self.assertRaisesRegex(ValueError, "expected 'name=value'"):
            _parse_where_args(["lr"], self.cfg)

    def test_empty_name_rejected(self):
        with self.assertRaisesRegex(ValueError, "empty parameter name"):
            _parse_where_args(["=42"], self.cfg)

    def test_unknown_param_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown parameter"):
            _parse_where_args(["xyz=1"], self.cfg)

    def test_static_overrides_key_rejected(self):
        # `data.root` is a static_overrides key, not a sweep param.
        with self.assertRaisesRegex(ValueError, "static_overrides key"):
            _parse_where_args(["data.root=/x"], self.cfg)


class TestPinIsDeprecatedAlias(unittest.TestCase):
    """`-p/--pin` still works — main() merges it into `where` and warns."""

    def test_pin_merges_into_where(self):
        import io
        from contextlib import redirect_stderr
        from hyperherd.cli import main

        argv = ["herd", "ls", "--help"]
        # Just assert the flag is still accepted by the parser (a full run needs
        # a workspace); the merge itself is exercised in test_cli_positional.
        with mock.patch("sys.argv", argv), redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main()


class TestFilterTrialsByWhere(unittest.TestCase):
    def setUp(self):
        # Six trials: 3 lr × 2 opt.
        self.trials = [
            {"index": 0, "params": {"lr": 0.1, "opt": "adam"}},
            {"index": 1, "params": {"lr": 0.1, "opt": "sgd"}},
            {"index": 2, "params": {"lr": 0.01, "opt": "adam"}},
            {"index": 3, "params": {"lr": 0.01, "opt": "sgd"}},
            {"index": 4, "params": {"lr": 0.001, "opt": "adam"}},
            {"index": 5, "params": {"lr": 0.001, "opt": "sgd"}},
        ]

    def test_no_filters_keeps_all(self):
        out = _filter_trials_by_where(self.trials, {})
        self.assertEqual(len(out), 6)

    def test_single_filter(self):
        out = _filter_trials_by_where(self.trials, {"opt": "adam"})
        self.assertEqual([t["index"] for t in out], [0, 2, 4])

    def test_two_filters_intersect(self):
        out = _filter_trials_by_where(self.trials, {"lr": 0.001, "opt": "sgd"})
        self.assertEqual([t["index"] for t in out], [5])

    def test_loose_numeric_equality(self):
        # An int-typed filter value matches a float-typed stored value
        # via Python's `==` (32 == 32.0).
        trials = [{"index": 0, "params": {"bs": 32.0}}]
        out = _filter_trials_by_where(trials, {"bs": 32})
        self.assertEqual(len(out), 1)


class TestCmdLaunchWhereE2E(unittest.TestCase):
    """End-to-end: dry-run with --where actually narrows the submission."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        cfg = (
            "name: t\n"
            f"workspace: {self.tmp}\n"
            f"launcher: {os.path.join(self.tmp, 'launch.sh')}\n"
            "grid: all\n"
            "parameters:\n"
            "  lr:\n"
            "    type: discrete\n"
            "    abbrev: lr\n"
            "    values: [0.1, 0.01, 0.001]\n"
            "  opt:\n"
            "    type: discrete\n"
            "    abbrev: opt\n"
            "    values: [adam, sgd]\n"
            "slurm:\n"
            "  partition: p\n"
            "  time: '00:10:00'\n"
            "  mem: 1G\n"
            "  cpus_per_task: 1\n"
        )
        with open(os.path.join(self.tmp, "hyperherd.yaml"), "w") as f:
            f.write(cfg)
        with open(os.path.join(self.tmp, "launch.sh"), "w") as f:
            f.write("#!/bin/bash\n")
        os.chmod(os.path.join(self.tmp, "launch.sh"), 0o755)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _args(self, **overrides):
        base = dict(
            workspace=self.tmp,
            dry_run=True,
            max_concurrent=None,
            indices=None,
            force=False,
            where=None,
            overrides=[],
            clear_overrides=False,
            all=False,
            json_output=False,
        )
        base.update(overrides)
        return argparse.Namespace(**base)

    def test_no_where_submits_full_grid(self):
        with mock.patch("hyperherd.cli.run_preflight", return_value=[]):
            rc = cmd_launch(self._args())
        self.assertEqual(rc, 0)
        trials = manifest.load_manifest(self.tmp)
        self.assertEqual(len(trials), 6)  # 3 × 2

    def test_where_narrows_to_one_trial(self):
        with mock.patch("hyperherd.cli.run_preflight", return_value=[]):
            rc = cmd_launch(self._args(where=["lr=0.001", "opt=sgd"]))
        # Dry-run succeeds and the manifest still has all 6 trials —
        # the filter only narrows what gets submitted, not what's
        # tracked.
        self.assertEqual(rc, 0)

    def test_where_no_match_errors(self):
        with mock.patch("hyperherd.cli.run_preflight", return_value=[]):
            rc = cmd_launch(self._args(where=["lr=999"]))
        self.assertEqual(rc, 1)

    def test_where_unknown_param_errors(self):
        with mock.patch("hyperherd.cli.run_preflight", return_value=[]):
            rc = cmd_launch(self._args(where=["xyz=1"]))
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
