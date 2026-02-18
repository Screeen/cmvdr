import numpy as np
import unittest

from cmvdr.estimation.coherence_manager import CoherenceManager


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


class TestCalculateHarmonicInfo(unittest.TestCase):
    """Tests for calculate_harmonic_info_from_coherence function."""

    def test_zero_always_at_first_position(self):
        """Test that zero modulation is always at the FIRST position in modulation sets."""
        # Create a simple coherence matrix where alpha=0 has high coherence
        alpha_vec_hz = np.array([0, -100, -200, -300, -400])
        P_sum = len(alpha_vec_hz)
        kk_max = 10
        
        # Create coherence matrix with varying values
        # Make sure alpha=0 (index 0) has high coherence
        rho = np.zeros((P_sum, kk_max))
        rho[0, :] = 0.95  # alpha=0 has high coherence
        rho[1, 5] = 0.85  # Some other modulations have high coherence at specific bins
        rho[2, 5] = 0.75
        rho[3, 5] = 0.65
        
        thr = 0.6
        P_max_cfg = 3
        nfft_real = kk_max
        
        # This should not raise a warning
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            harm_info = CoherenceManager.calculate_harmonic_info_from_coherence(
                alpha_vec_hz, rho, thr, P_max_cfg, nfft_real
            )
            
            # Check that no warning was raised about missing zero
            warning_messages = [str(warning.message) for warning in w]
            zero_warnings = [msg for msg in warning_messages if "0 should always be selected" in msg]
            self.assertEqual(len(zero_warnings), 0, 
                           f"Should not warn about missing zero when it's properly selected. Warnings: {zero_warnings}")
        
        # Verify that zero is in the modulation sets AND at first position
        for mod_set in harm_info.alpha_mods_sets:
            self.assertIn(0, mod_set, "Zero should be in all modulation sets")
            self.assertEqual(mod_set[0], 0, "Zero must be at the FIRST position (required by Modulator)")

    def test_zero_forced_at_first_position_when_low_coherence(self):
        """Test that zero is forced to first position even when it has low coherence."""
        # Create a scenario where alpha=0 doesn't have high enough coherence
        alpha_vec_hz = np.array([0, -100, -200, -300, -400])
        P_sum = len(alpha_vec_hz)
        kk_max = 10
        
        # Create coherence matrix where alpha=0 has LOW coherence
        rho = np.zeros((P_sum, kk_max))
        rho[0, :] = 0.3   # alpha=0 has LOW coherence (below threshold)
        rho[1, 5] = 0.95  # Other modulations have HIGH coherence
        rho[2, 5] = 0.85
        rho[3, 5] = 0.75
        rho[4, 5] = 0.65
        
        thr = 0.6  # Threshold above alpha=0's coherence
        P_max_cfg = 3
        nfft_real = kk_max
        
        # This SHOULD raise a warning since alpha=0 won't be naturally selected
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            harm_info = CoherenceManager.calculate_harmonic_info_from_coherence(
                alpha_vec_hz, rho, thr, P_max_cfg, nfft_real
            )
            
            # Check that warning WAS raised about missing zero
            warning_messages = [str(warning.message) for warning in w]
            zero_warnings = [msg for msg in warning_messages if "0 should always be selected" in msg]
            self.assertGreater(len(zero_warnings), 0, 
                             "Should warn when zero is not selected due to low coherence")
        
        # Verify that zero is STILL in the modulation sets at first position (forced inclusion)
        for mod_set in harm_info.alpha_mods_sets:
            self.assertIn(0, mod_set, "Zero should be in all modulation sets (forced)")
            self.assertEqual(mod_set[0], 0, "Zero must be at the FIRST position even when forced")

    def test_reordering_puts_zero_first(self):
        """Test that reordering always puts zero at first position regardless of coherence order."""
        # Test case where highest coherence is NOT at index 0
        alpha_vec_hz = np.array([0, -100, -200, -300])
        P_sum = len(alpha_vec_hz)
        kk_max = 5
        
        rho = np.zeros((P_sum, kk_max))
        # Make index 2 (-200) have highest coherence, but 0 also above threshold
        rho[2, :] = 0.99  # highest
        rho[0, :] = 0.85  # zero has second-highest
        rho[1, :] = 0.75
        rho[3, :] = 0.65
        
        thr = 0.6
        P_max_cfg = 3
        nfft_real = kk_max
        
        harm_info = CoherenceManager.calculate_harmonic_info_from_coherence(
            alpha_vec_hz, rho, thr, P_max_cfg, nfft_real
        )
        
        # Zero should be at first position despite not having highest coherence
        for mod_set in harm_info.alpha_mods_sets:
            self.assertEqual(mod_set[0], 0, 
                           "Zero must be at first position even when another frequency has higher coherence")
            # The code sorts indices, so the order is determined by the original alpha_vec_hz array
            # For alpha_vec_hz = [0, -100, -200, -300], selecting indices [2, 0, 1] sorted becomes [0, 1, 2]
            # which gives values [0, -100, -200] after reordering to put 0 first


if __name__ == '__main__':
    unittest.main()
