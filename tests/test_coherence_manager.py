import numpy as np
import unittest

from cmvdr.coherence_manager import CoherenceManager


class TestCoherenceEquivalence(unittest.TestCase):

    def check_equivalence(self, mod, mod_c, psds, alpha, cc0, delta_f, fs, tol=1e-12):
        rho_slow = CoherenceManager.compute_coherence_internal(
            mod, mod_c, psds, alpha, cc0, delta_f, fs
        )
        rho_fast = CoherenceManager.compute_coherence_internal_fast(
            mod, mod_c, psds, alpha, cc0, delta_f, fs
        )

        self.assertTrue(
            np.allclose(rho_slow, rho_fast, atol=tol, rtol=0),
            msg=f"Mismatch in coherence arrays:\nslow={rho_slow}\nfast={rho_fast}"
        )

    def test_synthetic_data_case1(self):
        P_sum = 8
        kk_max = 16
        frames = 10

        mod = (np.random.randn(P_sum, kk_max, frames) +
               1j * np.random.randn(P_sum, kk_max, frames))
        mod_c = np.conj(mod)

        psds = np.mean(np.abs(mod) ** 2, axis=-1)

        alpha = np.linspace(10, 1000, P_sum)
        cc0 = 3
        delta_f = 5.0
        fs = 16000

        self.check_equivalence(mod, mod_c, psds, alpha, cc0, delta_f, fs)

    def test_zero_modulation(self):
        P_sum = 4
        kk_max = 8
        frames = 6

        mod = np.zeros((P_sum, kk_max, frames), dtype=np.complex128)
        mod_c = np.conj(mod)

        psds = np.ones((P_sum, kk_max))  # avoid divide-by-zero

        alpha = np.array([100, 200, 300, 400])
        cc0 = 1
        delta_f = 10.0
        fs = 8000

        self.check_equivalence(mod, mod_c, psds, alpha, cc0, delta_f, fs)

    def test_random_noise_case(self):
        np.random.seed(0)
        P_sum = 16
        kk_max = 32
        frames = 12

        mod = (np.random.randn(P_sum, kk_max, frames) +
               1j * np.random.randn(P_sum, kk_max, frames))
        mod_c = np.conj(mod)

        psds = np.mean(np.abs(mod) ** 2, axis=-1)

        alpha = np.sort(np.random.uniform(0, 1000, P_sum))
        cc0 = 5
        delta_f = 2.5
        fs = 22050

        self.check_equivalence(mod, mod_c, psds, alpha, cc0, delta_f, fs)


if __name__ == '__main__':
    unittest.main()
