"""Tests for the init scaffolding."""

import os
import shutil
import stat
import tempfile
import unittest

from hyperwhip.init import scaffold


class TestScaffold(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir)

    def test_creates_files(self):
        config_path, launcher_path = scaffold(self.tmpdir, name="test_exp")
        self.assertTrue(os.path.isfile(config_path))
        self.assertTrue(os.path.isfile(launcher_path))
        self.assertTrue(config_path.endswith("hyperwhip.yaml"))
        self.assertTrue(launcher_path.endswith("launch.sh"))

    def test_config_contains_name(self):
        scaffold(self.tmpdir, name="my_sweep")
        with open(os.path.join(self.tmpdir, "hyperwhip.yaml")) as f:
            content = f.read()
        self.assertIn("name: my_sweep", content)

    def test_config_contains_grid(self):
        scaffold(self.tmpdir, name="test", grid="all")
        with open(os.path.join(self.tmpdir, "hyperwhip.yaml")) as f:
            content = f.read()
        self.assertIn("grid: all", content)

    def test_config_contains_gres(self):
        scaffold(self.tmpdir, name="test", gres="gpu:1")
        with open(os.path.join(self.tmpdir, "hyperwhip.yaml")) as f:
            content = f.read()
        self.assertIn('gres: "gpu:1"', content)

    def test_launcher_is_executable(self):
        _, launcher_path = scaffold(self.tmpdir, name="test")
        mode = os.stat(launcher_path).st_mode
        self.assertTrue(mode & stat.S_IEXEC)

    def test_launcher_contains_default_command(self):
        scaffold(self.tmpdir, name="test")
        with open(os.path.join(self.tmpdir, "launch.sh")) as f:
            content = f.read()
        self.assertIn("python train.py $OVERRIDES", content)

    def test_refuses_overwrite_by_default(self):
        scaffold(self.tmpdir, name="test")
        with self.assertRaises(FileExistsError):
            scaffold(self.tmpdir, name="test")

    def test_force_overwrites(self):
        scaffold(self.tmpdir, name="test")
        scaffold(self.tmpdir, name="test_v2", overwrite=True)
        with open(os.path.join(self.tmpdir, "hyperwhip.yaml")) as f:
            content = f.read()
        self.assertIn("name: test_v2", content)

    def test_default_name_from_directory(self):
        subdir = os.path.join(self.tmpdir, "cool_experiment")
        os.makedirs(subdir)
        scaffold(subdir)
        with open(os.path.join(subdir, "hyperwhip.yaml")) as f:
            content = f.read()
        self.assertIn("name: cool_experiment", content)

    def test_custom_slurm_settings(self):
        scaffold(self.tmpdir, name="test", partition="a100", time="12:00:00", mem="64G", cpus=8)
        with open(os.path.join(self.tmpdir, "hyperwhip.yaml")) as f:
            content = f.read()
        self.assertIn("partition: a100", content)
        self.assertIn('time: "12:00:00"', content)
        self.assertIn('mem: "64G"', content)
        self.assertIn("cpus_per_task: 8", content)


if __name__ == "__main__":
    unittest.main()
