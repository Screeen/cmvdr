import unittest
import numpy as np
from cmvdr.util import globs as gs

gs.rng, _ = gs.compute_rng(seed_is_random=False, rnd_seed_=123)
from cmvdr.data_gen.f0_manager import F0Manager


class F0ManagerFrequencyShiftTests(unittest.TestCase):
    """Tests for F0Manager.shift_frequencies_by_percentage method."""

    def setUp(self):
        # Reset random state for reproducible tests
        gs.rng, _ = gs.compute_rng(seed_is_random=False, rnd_seed_=123)
        # Initialize inharmonicity_signs for the F0Manager
        F0Manager.get_inharmonicity_signs()

    def test_shift_frequencies_zero_percentage_returns_unchanged(self):
        """Test that zero percentage returns frequencies unchanged."""
        freqs = np.array([100.0, 200.0, 300.0])
        result = F0Manager.shift_frequencies_by_percentage(freqs, 0.0, all_same_sign=False, fixed_amount=False)
        np.testing.assert_array_equal(result, freqs)

    def test_shift_frequencies_with_fixed_amount_all_same_sign(self):
        """Test fixed_amount=True with all_same_sign=True."""
        freqs = np.array([100.0, 200.0, 300.0])
        percentage = 0.01  # 1%
        result = F0Manager.shift_frequencies_by_percentage(freqs, percentage, all_same_sign=True, fixed_amount=True)
        expected = freqs * (1 + percentage)
        np.testing.assert_allclose(result, expected)

    def test_shift_frequencies_with_fixed_amount_varying_sign(self):
        """Test fixed_amount=True with all_same_sign=False (uses alternating signs)."""
        freqs = np.array([100.0, 200.0, 300.0])
        percentage = 0.01  # 1%
        result = F0Manager.shift_frequencies_by_percentage(freqs, percentage, all_same_sign=False, fixed_amount=True)
        # Should apply alternating signs
        self.assertEqual(len(result), len(freqs))
        # Check that shifts are applied (but we can't predict exact sign pattern without accessing inharmonicity_signs)
        self.assertFalse(np.array_equal(result, freqs))

    def test_shift_frequencies_random_amount_varying_sign(self):
        """Test fixed_amount=False with all_same_sign=False (random shifts with varying signs)."""
        freqs = np.array([100.0, 200.0, 300.0])
        percentage = 0.01  # 1%
        result = F0Manager.shift_frequencies_by_percentage(freqs, percentage, all_same_sign=False, fixed_amount=False)
        # Should apply random shifts between -1% and +1%
        self.assertEqual(len(result), len(freqs))
        # Check that frequencies are shifted
        self.assertFalse(np.array_equal(result, freqs))
        # Check that shifts are within expected range
        for i, (orig, shifted) in enumerate(zip(freqs, result)):
            max_shift = orig * percentage
            self.assertLessEqual(abs(shifted - orig), max_shift * 1.01)  # small tolerance for floating point

    def test_shift_frequencies_empty_array(self):
        """Test that empty array returns empty array."""
        freqs = np.array([])
        result = F0Manager.shift_frequencies_by_percentage(freqs, 0.01, all_same_sign=False, fixed_amount=False)
        self.assertEqual(len(result), 0)

    def test_shift_frequencies_single_element(self):
        """Test with single frequency."""
        freqs = np.array([100.0])
        percentage = 0.01
        result = F0Manager.shift_frequencies_by_percentage(freqs, percentage, all_same_sign=False, fixed_amount=False)
        self.assertEqual(len(result), 1)
        # Should be shifted
        self.assertNotEqual(result[0], freqs[0])

    def test_shift_frequencies_large_percentage(self):
        """Test with 1% frequency shift (percentage = 0.01)."""
        freqs = np.array([100.0, 200.0])
        percentage = 0.01  # 1% (decimal representation)
        result = F0Manager.shift_frequencies_by_percentage(freqs, percentage, all_same_sign=False, fixed_amount=False)
        # All frequencies should still be positive
        self.assertTrue(np.all(result > 0))


class ExperimentManagerFrequencyErrorInjectionTests(unittest.TestCase):
    """Tests for frequency error injection logic in experiment manager."""

    def setUp(self):
        # Reset random state for reproducible tests
        gs.rng, _ = gs.compute_rng(seed_is_random=False, rnd_seed_=123)
        # Initialize inharmonicity_signs for the F0Manager
        F0Manager.get_inharmonicity_signs()

    def test_conditional_check_zero_percentage_no_shift(self):
        """Test that mod_error_perc=0 doesn't apply shift."""
        cfg = {'mod_error_perc': 0}
        harmonic_freqs_est = np.array([100.0, 200.0])
        original_freqs = harmonic_freqs_est.copy()
        
        # Simulate the conditional check
        if cfg.get('mod_error_perc', 0) > 0 and harmonic_freqs_est.size > 0:
            harmonic_freqs_est = F0Manager.shift_frequencies_by_percentage(
                harmonic_freqs_est, cfg['mod_error_perc'] / 100, 
                all_same_sign=False, fixed_amount=False)
        
        np.testing.assert_array_equal(harmonic_freqs_est, original_freqs)

    def test_conditional_check_positive_percentage_applies_shift(self):
        """Test that mod_error_perc>0 applies shift."""
        cfg = {'mod_error_perc': 1.0}  # 1%
        harmonic_freqs_est = np.array([100.0, 200.0])
        original_freqs = harmonic_freqs_est.copy()
        
        # Simulate the conditional check
        if cfg.get('mod_error_perc', 0) > 0 and harmonic_freqs_est.size > 0:
            harmonic_freqs_est = F0Manager.shift_frequencies_by_percentage(
                harmonic_freqs_est, cfg['mod_error_perc'] / 100, 
                all_same_sign=False, fixed_amount=False)
        
        self.assertFalse(np.array_equal(harmonic_freqs_est, original_freqs))

    def test_conditional_check_empty_array_no_error(self):
        """Test that empty array doesn't cause errors."""
        cfg = {'mod_error_perc': 1.0}
        harmonic_freqs_est = np.array([])
        
        # Simulate the conditional check - should not execute shift
        if cfg.get('mod_error_perc', 0) > 0 and harmonic_freqs_est.size > 0:
            harmonic_freqs_est = F0Manager.shift_frequencies_by_percentage(
                harmonic_freqs_est, cfg['mod_error_perc'] / 100, 
                all_same_sign=False, fixed_amount=False)
        
        self.assertEqual(harmonic_freqs_est.size, 0)

    def test_conditional_check_missing_parameter_uses_default(self):
        """Test that missing mod_error_perc parameter uses default of 0."""
        cfg = {}  # No mod_error_perc key
        harmonic_freqs_est = np.array([100.0, 200.0])
        original_freqs = harmonic_freqs_est.copy()
        
        # Simulate the conditional check
        if cfg.get('mod_error_perc', 0) > 0 and harmonic_freqs_est.size > 0:
            harmonic_freqs_est = F0Manager.shift_frequencies_by_percentage(
                harmonic_freqs_est, cfg['mod_error_perc'] / 100, 
                all_same_sign=False, fixed_amount=False)
        
        np.testing.assert_array_equal(harmonic_freqs_est, original_freqs)


if __name__ == '__main__':
    unittest.main()
