"""Tests for SLURM utilities (no actual SLURM interaction)."""

import unittest

from hyperwhip.slurm import _indices_to_array_spec


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


if __name__ == "__main__":
    unittest.main()
