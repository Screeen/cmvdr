import numpy as np
import unittest

# replace these imports with wherever you’ve defined them
from cmvdr.modulator import Modulator


class TestToleranceClustering(unittest.TestCase):

    def check_equivalence(self, x, tol=1e-2):
        u1, inv1 = Modulator.unique_with_relative_tolerance(x, tol=tol, return_inverse=True)
        u2, inv2 = Modulator.unique_with_relative_tolerance_fast(x, tol=tol, return_inverse=True)

        self.assertTrue(np.allclose(u1, u2), f"Unique values differ:\n slow={u1}\nfast={u2}")
        self.assertTrue(np.array_equal(inv1, inv2), f"Inverse indices differ:\n slow={inv1}\nfast={inv2}")

    def test_small_floats_with_zeros(self):
        x = np.array([0.1, 0.1005, 0.2, 0.1999, 0.0, 0.0])
        self.check_equivalence(x)

    def test_mixed_signs(self):
        x = np.array([-1.0, -1.009, -0.5, -0.495, 0.5, 0.495])
        self.check_equivalence(x)

    def test_random_near_duplicates(self):
        x = np.concatenate([
            np.linspace(1, 2, 5),
            np.linspace(1, 2, 5) * (1 + 5e-3),
            np.zeros(3),
        ])
        self.check_equivalence(x)