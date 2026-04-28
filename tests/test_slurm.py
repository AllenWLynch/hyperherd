"""Tests for SLURM utilities (no actual SLURM interaction)."""

import unittest

from hyperwhip.config import Config
from hyperwhip.slurm import _indices_to_array_spec, generate_sbatch_script


class TestIndicesToArraySpec(unittest.TestCase):
    def test_contiguous(self):
        self.assertEqual(_indices_to_array_spec([0, 1, 2, 3]), "0-3")

    def test_single(self):
        self.assertEqual(_indices_to_array_spec([5]), "5")

    def test_gaps(self):
        self.assertEqual(_indices_to_array_spec([0, 1, 3, 5, 6, 7]), "0-1,3,5-7")

    def test_all_individual(self):
        self.assertEqual(_indices_to_array_spec([1, 3, 5]), "1,3,5")

    def test_unsorted(self):
        self.assertEqual(_indices_to_array_spec([5, 3, 1, 2]), "1-3,5")

    def test_duplicates(self):
        self.assertEqual(_indices_to_array_spec([1, 1, 2, 2, 3]), "1-3")

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            _indices_to_array_spec([])


def _make_config(max_concurrent=None):
    raw = {
        "name": "t",
        "workspace": "/tmp/ws",
        "launcher": "/tmp/ws/launch.sh",
        "parameters": {"lr": {"type": "discrete", "abbrev": "lr", "values": [0.1, 0.01]}},
        "grid": "all",
        "slurm": {"partition": "p", "time": "00:10:00", "mem": "1G", "cpus_per_task": 1},
    }
    if max_concurrent is not None:
        raw["slurm"]["max_concurrent"] = max_concurrent
    return Config.model_validate(raw)


class TestArrayThrottle(unittest.TestCase):
    def test_no_throttle(self):
        cfg = _make_config()
        script = generate_sbatch_script(cfg, [0, 1, 2])
        self.assertIn("#SBATCH --array=0-2\n", script)

    def test_config_throttle(self):
        cfg = _make_config(max_concurrent=4)
        script = generate_sbatch_script(cfg, [0, 1, 2, 3, 4])
        self.assertIn("#SBATCH --array=0-4%4\n", script)

    def test_cli_overrides_config(self):
        cfg = _make_config(max_concurrent=4)
        script = generate_sbatch_script(cfg, [0, 1, 2, 3, 4], max_concurrent=2)
        self.assertIn("#SBATCH --array=0-4%2\n", script)

    def test_cli_only(self):
        cfg = _make_config()
        script = generate_sbatch_script(cfg, [0, 1, 2], max_concurrent=1)
        self.assertIn("#SBATCH --array=0-2%1\n", script)


if __name__ == "__main__":
    unittest.main()
