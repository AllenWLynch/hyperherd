"""Tests for preflight validation checks."""

import os
import shutil
import stat
import tempfile
import unittest

from hyperwhip.config import Config
from hyperwhip.preflight import PreflightError, run_preflight


def _make_config(parameters=None, constraints=None, launcher=None, workspace=None, grid="all"):
    if parameters is None:
        parameters = {"lr": {"abbrev": "lr", "type": "discrete", "values": [0.1, 0.01]}}
    if constraints is None:
        constraints = []

    raw = {
        "name": "test",
        "workspace": workspace or "/tmp/hyperwhip_preflight_test",
        "parameters": parameters,
        "constraints": constraints,
        "launcher": launcher or "",
    }
    if grid is not None:
        raw["grid"] = grid
    return Config.model_validate(raw)


class TestLauncherCheck(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_missing_launcher(self):
        config = _make_config(launcher="")
        with self.assertRaises(PreflightError) as ctx:
            run_preflight(config)
        self.assertIn("No launcher script", str(ctx.exception))

    def test_nonexistent_launcher(self):
        config = _make_config(launcher="/tmp/nonexistent_launcher_xyz.sh")
        with self.assertRaises(PreflightError) as ctx:
            run_preflight(config)
        self.assertIn("not found", str(ctx.exception))

    def test_non_executable_launcher(self):
        launcher = os.path.join(self.tmpdir, "launch.sh")
        with open(launcher, "w") as f:
            f.write("#!/bin/bash\n")
        os.chmod(launcher, stat.S_IRUSR | stat.S_IWUSR)

        config = _make_config(launcher=launcher, workspace=self.tmpdir)
        with self.assertRaises(PreflightError) as ctx:
            run_preflight(config)
        self.assertIn("not executable", str(ctx.exception))

    def test_valid_launcher(self):
        launcher = os.path.join(self.tmpdir, "launch.sh")
        with open(launcher, "w") as f:
            f.write("#!/bin/bash\n")
        os.chmod(launcher, stat.S_IRWXU)

        config = _make_config(launcher=launcher, workspace=self.tmpdir)
        run_preflight(config)


class TestWorkspaceCheck(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.launcher = os.path.join(self.tmpdir, "launch.sh")
        with open(self.launcher, "w") as f:
            f.write("#!/bin/bash\n")
        os.chmod(self.launcher, stat.S_IRWXU)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_nonexistent_parent(self):
        config = _make_config(
            launcher=self.launcher,
            workspace="/tmp/no_such_parent_xyz/no_such_child/ws",
        )
        with self.assertRaises(PreflightError) as ctx:
            run_preflight(config)
        self.assertIn("parent directory", str(ctx.exception))

    def test_existing_writable_workspace(self):
        ws = os.path.join(self.tmpdir, "ws")
        os.makedirs(ws)
        config = _make_config(launcher=self.launcher, workspace=ws)
        run_preflight(config)


class TestDefaultsCheck(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.launcher = os.path.join(self.tmpdir, "launch.sh")
        with open(self.launcher, "w") as f:
            f.write("#!/bin/bash\n")
        os.chmod(self.launcher, stat.S_IRWXU)

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_default_not_in_values_fails_at_parse(self):
        """Pydantic validates default is in values at parse time."""
        from pydantic import ValidationError
        with self.assertRaises((ValidationError, Exception)):
            _make_config(
                launcher=self.launcher, workspace=self.tmpdir,
                parameters={"opt": {"abbrev": "opt", "type": "discrete", "values": ["adam", "sgd"], "default": "rmsprop"}},
                grid=None,
            )

    def test_continuous_default_out_of_range_fails_at_parse(self):
        from pydantic import ValidationError
        with self.assertRaises((ValidationError, Exception)):
            _make_config(
                launcher=self.launcher, workspace=self.tmpdir,
                parameters={"lr": {"abbrev": "lr", "type": "continuous", "low": 0.001, "high": 0.1, "steps": 3, "default": 999.0}},
                grid=None,
            )

    def test_valid_defaults(self):
        config = _make_config(
            launcher=self.launcher, workspace=self.tmpdir,
            parameters={
                "lr": {"abbrev": "lr", "type": "continuous", "low": 0.001, "high": 0.1, "steps": 3, "default": 0.01},
                "opt": {"abbrev": "opt", "type": "discrete", "values": ["adam", "sgd"], "default": "adam"},
            },
            grid=None,
        )
        run_preflight(config)


class TestConstraintValidation(unittest.TestCase):
    def test_unknown_when_param(self):
        from pydantic import ValidationError
        with self.assertRaises((ValidationError, Exception)):
            _make_config(
                parameters={"lr": {"abbrev": "lr", "type": "discrete", "values": [0.1]}},
                constraints=[{"name": "c1", "when": {"nonexistent": "val"}, "exclude": {"lr": [0.1]}}],
            )

    def test_unknown_exclude_param(self):
        from pydantic import ValidationError
        with self.assertRaises((ValidationError, Exception)):
            _make_config(
                parameters={"lr": {"abbrev": "lr", "type": "discrete", "values": [0.1]}},
                constraints=[{"name": "c1", "when": {"lr": 0.1}, "exclude": {"bogus": [1]}}],
            )

    def test_valid_constraint_refs(self):
        config = _make_config(
            parameters={
                "lr": {"abbrev": "lr", "type": "discrete", "values": [0.1, 0.01]},
                "opt": {"abbrev": "opt", "type": "discrete", "values": ["adam", "sgd"]},
            },
            constraints=[{"name": "c1", "when": {"opt": "sgd"}, "exclude": {"lr": [0.1]}}],
        )
        self.assertEqual(len(config.conditions), 1)


if __name__ == "__main__":
    unittest.main()
