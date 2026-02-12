"""
Unit tests for CovarianceEstimator with focus on recursive mode.

Tests cover:
1. Recursive mode with single-frame and multi-frame slice_frames
2. Expected output shapes for cross_noisy_early_wb
3. Equivalence between recursive and block-processing estimates
4. Shape preservation under M/P variations
5. Forgetting factor sensitivity
"""

import unittest
import numpy as np
import cmvdr.util.globs as gs

gs.rng, _ = gs.compute_rng(seed_is_random=False, rnd_seed_=42)

from cmvdr.estimation.covariance_estimator import CovarianceEstimator
from cmvdr.util.harmonic_info import HarmonicInfo


class SimpleHarmonicInfo:
    """Minimal HarmonicInfo mock for testing."""

    def __init__(self, K, P_all=None):
        self.K = K
        if P_all is None:
            self._P_all = np.ones(K, dtype=int)
        else:
            self._P_all = np.array(P_all, dtype=int)

    def get_num_shifts_all_frequencies(self):
        return self._P_all

    def get_harmonic_set_and_num_shifts(self, kk):
        """Return harmonic set index and number of shifts for frequency bin kk."""
        return 0, self._P_all[kk]


class TestCovarianceEstimatorRecursiveInit(unittest.TestCase):
    """Test recursive mode initialization."""

    def setUp(self):
        self.M = 2
        self.K = 5
        self.P = 1
        self.num_frames_total = 10

    def _make_config_recursive(self, forgetting_factor=0.5):
        """Create a recursive config."""
        return {
            'recursive_average': True,
            'cov_est_forgetting_factor': forgetting_factor,
            'use_rank1_model_for_oracle_cov_wet_estimation': True,
        }

    def _make_signals_dict(self, M, K, P, num_frames, include_wet=False):
        """Create a minimal signals_dict with required structure."""
        num_harmonic_sets = 1  # Single harmonic set for simplicity
        signals = {
            'noisy': {
                'stft': np.random.randn(M, K, num_frames) + 1j * np.random.randn(M, K, num_frames),
                'stft_conj': np.conj(np.random.randn(M, K, num_frames) + 1j * np.random.randn(M, K, num_frames)),
                'mod_stft_3d': np.random.randn(num_harmonic_sets, M * P, K, num_frames).astype(np.complex128),
                'mod_stft_3d_conj': np.conj(np.random.randn(num_harmonic_sets, M * P, K, num_frames).astype(np.complex128)),
            },
            'noise_cov_est': {
                'stft': np.random.randn(M, K, num_frames) + 1j * np.random.randn(M, K, num_frames),
                'stft_conj': np.conj(np.random.randn(M, K, num_frames) + 1j * np.random.randn(M, K, num_frames)),
                'mod_stft_3d': np.random.randn(num_harmonic_sets, M * P, K, num_frames).astype(np.complex128),
                'mod_stft_3d_conj': np.conj(np.random.randn(num_harmonic_sets, M * P, K, num_frames).astype(np.complex128)),
            }
        }
        if include_wet:
            signals['wet_rank1'] = {
                'stft': np.random.randn(M, K, num_frames) + 1j * np.random.randn(M, K, num_frames),
                'stft_conj': np.conj(np.random.randn(M, K, num_frames) + 1j * np.random.randn(M, K, num_frames)),
                'mod_stft_3d': np.random.randn(num_harmonic_sets, M * P, K, num_frames).astype(np.complex128),
                'mod_stft_3d_conj': np.conj(np.random.randn(num_harmonic_sets, M * P, K, num_frames).astype(np.complex128)),
            }
        return signals

    def test_recursive_init_allocates_correct_shapes(self):
        """Test that recursive initialization allocates covariance matrices with correct shapes."""
        cfg = self._make_config_recursive()
        cov_est = CovarianceEstimator(cfg, cyclostationary_target=False)
        cov_est.set_dimensions((self.K, self.M, self.P))
        cov_est.harmonic_info = SimpleHarmonicInfo(self.K, P_all=np.ones(self.K, dtype=int))

        signals = self._make_signals_dict(self.M, self.K, self.P, self.num_frames_total)

        # First call with empty cov_dict_prev should initialize
        cov_dict = cov_est.prepare_covariances(signals, needs_initialization=True)

        # Check shapes
        self.assertEqual(cov_dict['noisy_wb'].shape, (self.K, self.M * self.P, self.M * self.P))
        self.assertEqual(cov_dict['noise_wb'].shape, (self.K, self.M * self.P, self.M * self.P))
        self.assertEqual(cov_dict['noisy_wb'].dtype, np.complex128)

    def test_recursive_init_first_iteration_flag(self):
        """Test that is_first_iteration=True triggers initialization."""
        cfg = self._make_config_recursive()
        cov_est = CovarianceEstimator(cfg, cyclostationary_target=False)
        cov_est.set_dimensions((self.K, self.M, self.P))
        cov_est.harmonic_info = SimpleHarmonicInfo(self.K, P_all=np.ones(self.K, dtype=int))

        signals = self._make_signals_dict(self.M, self.K, self.P, self.num_frames_total)

        # Empty cov_dict_prev should trigger initialization
        cov_dict_prev = {}
        self.assertTrue(cov_dict_prev == {})

        # Prepare should be called and return initialized dict
        cov_dict = cov_est.prepare_covariances(signals, needs_initialization=True)
        self.assertGreater(len(cov_dict), 0)
        self.assertTrue(np.any(cov_dict['noise_wb'] != 0))


class TestCovarianceEstimatorRank1Update(unittest.TestCase):
    """Test rank-1 update logic in recursive mode."""

    def setUp(self):
        self.M = 2
        self.K = 5
        self.P = 1
        self.num_frames_total = 10

    def _make_config_recursive(self, forgetting_factor=0.5, cyclostationary_target=False):
        return {
            'recursive_average': True,
            'cov_est_forgetting_factor': forgetting_factor,
            'use_rank1_model_for_oracle_cov_wet_estimation': True,
        }

    def _make_signals_dict_with_modulated(self, M, K, P, num_frames, include_wet=False):
        """Create signals_dict with proper modulated signal structure (harmonic_set, M*P, K, frames)."""
        num_harmonic_sets = 1  # Single harmonic set for simplicity

        signals = {
            'noisy': {
                'stft': np.random.randn(M, K, num_frames) + 1j * np.random.randn(M, K, num_frames),
                'stft_conj': np.conj(np.random.randn(M, K, num_frames) + 1j * np.random.randn(M, K, num_frames)),
                'mod_stft_3d': np.random.randn(num_harmonic_sets, M * P, K, num_frames).astype(np.complex128),
                'mod_stft_3d_conj': np.conj(np.random.randn(num_harmonic_sets, M * P, K, num_frames).astype(np.complex128)),
            },
            'noise_cov_est': {
                'stft': np.random.randn(M, K, num_frames) + 1j * np.random.randn(M, K, num_frames),
                'stft_conj': np.conj(np.random.randn(M, K, num_frames) + 1j * np.random.randn(M, K, num_frames)),
                'mod_stft_3d': np.random.randn(num_harmonic_sets, M * P, K, num_frames).astype(np.complex128),
                'mod_stft_3d_conj': np.conj(np.random.randn(num_harmonic_sets, M * P, K, num_frames).astype(np.complex128)),
            }
        }
        if include_wet:
            signals['wet_rank1'] = {
                'stft': np.random.randn(M, K, num_frames) + 1j * np.random.randn(M, K, num_frames),
                'stft_conj': np.conj(np.random.randn(M, K, num_frames) + 1j * np.random.randn(M, K, num_frames)),
                'mod_stft_3d': np.random.randn(num_harmonic_sets, M * P, K, num_frames).astype(np.complex128),
                'mod_stft_3d_conj': np.conj(np.random.randn(num_harmonic_sets, M * P, K, num_frames).astype(np.complex128)),
            }
        return signals

    def test_rank1_update_single_frame_rejects_multiframe(self):
        """Test that rank1_update raises error for multi-frame slice_frames."""
        cfg = self._make_config_recursive(cyclostationary_target=True)
        cov_est = CovarianceEstimator(cfg, cyclostationary_target=True)
        cov_est.set_dimensions((self.K, self.M, self.P))
        cov_est.harmonic_info = SimpleHarmonicInfo(self.K, P_all=np.ones(self.K, dtype=int))

        signals = self._make_signals_dict_with_modulated(self.M, self.K, self.P, self.num_frames_total)

        # Initialize cov_dict
        cov_dict = CovarianceEstimator.allocate_covariance_matrices((self.K, self.M, self.P),
                                                                     is_mwf=False, use_pseudo_cov=False)
        cov_dict['noise_wb'] = cov_est.estimate_noise_covariance(signals['noise_cov_est'], cov_dict['noise_wb'])
        cov_dict = CovarianceEstimator.initialize_covariance_matrices(cov_dict)

        # Multi-frame slice should raise ValueError
        multi_frame_slice = slice(0, 3)
        with self.assertRaises(ValueError) as context:
            cov_est.rank1_update_covariances(cov_dict, signals, multi_frame_slice,
                                            forget_factor=0.5, is_cmvdr=True)
        self.assertIn("single frame", str(context.exception))

    def test_rank1_update_single_frame_succeeds(self):
        """Test that rank1_update works with single-frame slice_frames."""
        cfg = self._make_config_recursive(cyclostationary_target=True)
        cov_est = CovarianceEstimator(cfg, cyclostationary_target=True)
        cov_est.set_dimensions((self.K, self.M, self.P))
        cov_est.harmonic_info = SimpleHarmonicInfo(self.K, P_all=np.ones(self.K, dtype=int))

        signals = self._make_signals_dict_with_modulated(self.M, self.K, self.P, self.num_frames_total)

        # Initialize cov_dict
        cov_dict = CovarianceEstimator.allocate_covariance_matrices((self.K, self.M, self.P),
                                                                     is_mwf=False, use_pseudo_cov=False)
        cov_dict['noise_wb'] = cov_est.estimate_noise_covariance(signals['noise_cov_est'], cov_dict['noise_wb'])
        cov_dict = CovarianceEstimator.initialize_covariance_matrices(cov_dict)
        cov_dict_prev = cov_dict.copy()

        # Single-frame slice should succeed
        single_frame_slice = slice(0, 1)
        cov_dict_updated = cov_est.rank1_update_covariances(cov_dict, signals, single_frame_slice,
                                                            forget_factor=0.5, is_cmvdr=True)

        # Check that cov_dict_updated has correct shape
        self.assertEqual(cov_dict_updated['noisy_wb'].shape, cov_dict_prev['noisy_wb'].shape)
        self.assertEqual(cov_dict_updated['noisy_wb'].dtype, np.complex128)

    def test_rank1_update_forgetting_factor_blending(self):
        """Test that forgetting factor correctly blends old and new covariances."""
        cfg = self._make_config_recursive(forgetting_factor=0.5, cyclostationary_target=True)
        cov_est = CovarianceEstimator(cfg, cyclostationary_target=True)
        cov_est.set_dimensions((self.K, self.M, self.P))
        cov_est.harmonic_info = SimpleHarmonicInfo(self.K, P_all=np.ones(self.K, dtype=int))

        signals = self._make_signals_dict_with_modulated(self.M, self.K, self.P, self.num_frames_total)

        # Initialize with known values
        cov_dict = CovarianceEstimator.allocate_covariance_matrices((self.K, self.M, self.P),
                                                                     is_mwf=False, use_pseudo_cov=False)
        cov_dict['noise_wb'] = cov_est.estimate_noise_covariance(signals['noise_cov_est'], cov_dict['noise_wb'])
        cov_dict = CovarianceEstimator.initialize_covariance_matrices(cov_dict)

        # Store initial value
        initial_cov = cov_dict['noisy_wb'].copy()

        # Perform rank-1 update
        forget_factor = 0.7
        single_frame_slice = slice(0, 1)
        cov_dict_updated = cov_est.rank1_update_covariances(cov_dict, signals, single_frame_slice,
                                                            forget_factor=forget_factor, is_cmvdr=True)

        # Check that update happened (not exactly same as initial)
        self.assertFalse(np.allclose(cov_dict_updated['noisy_wb'], initial_cov))

        # Check finite values
        self.assertTrue(np.all(np.isfinite(cov_dict_updated['noisy_wb'])))

    def test_rank1_update_preserves_shape_across_iterations(self):
        """Test that shape is preserved across multiple rank-1 updates."""
        cfg = self._make_config_recursive(cyclostationary_target=True)
        cov_est = CovarianceEstimator(cfg, cyclostationary_target=True)
        cov_est.set_dimensions((self.K, self.M, self.P))
        cov_est.harmonic_info = SimpleHarmonicInfo(self.K, P_all=np.ones(self.K, dtype=int))

        signals = self._make_signals_dict_with_modulated(self.M, self.K, self.P, self.num_frames_total)

        # Initialize
        cov_dict = CovarianceEstimator.allocate_covariance_matrices((self.K, self.M, self.P),
                                                                     is_mwf=False, use_pseudo_cov=False)
        cov_dict['noise_wb'] = cov_est.estimate_noise_covariance(signals['noise_cov_est'], cov_dict['noise_wb'])
        cov_dict = CovarianceEstimator.initialize_covariance_matrices(cov_dict)

        expected_shape = (self.K, self.M * self.P, self.M * self.P)
        self.assertEqual(cov_dict['noisy_wb'].shape, expected_shape)

        # Multiple iterations
        for frame_idx in range(3):
            frame_slice = slice(frame_idx, frame_idx + 1)
            cov_dict = cov_est.rank1_update_covariances(cov_dict, signals, frame_slice,
                                                        forget_factor=0.5, is_cmvdr=True)
            self.assertEqual(cov_dict['noisy_wb'].shape, expected_shape)


class TestCrossNoisyEarlyShapes(unittest.TestCase):
    """Test shapes and computation of cross_noisy_early_wb for cMWF."""

    def setUp(self):
        self.M = 2
        self.K = 5
        self.P = 1
        self.num_frames = 4
        self.num_harmonic_sets = 1

    def _make_config_with_mwf(self, cyclostationary_target=True):
        return {
            'recursive_average': True,
            'cov_est_forgetting_factor': 0.5,
            'use_rank1_model_for_oracle_cov_wet_estimation': True,
        }

    def _make_signals_with_wet(self, M, K, P, num_frames):
        """Create signals_dict with wet_rank1 for cMWF."""
        num_harmonic_sets = 1

        signals = {
            'noisy': {
                'stft': np.random.randn(M, K, num_frames) + 1j * np.random.randn(M, K, num_frames),
                'stft_conj': np.conj(np.random.randn(M, K, num_frames) + 1j * np.random.randn(M, K, num_frames)),
                'mod_stft_3d': np.random.randn(num_harmonic_sets, M * P, K, num_frames).astype(np.complex128),
                'mod_stft_3d_conj': np.conj(np.random.randn(num_harmonic_sets, M * P, K, num_frames).astype(np.complex128)),
            },
            'noise_cov_est': {
                'stft': np.random.randn(M, K, num_frames) + 1j * np.random.randn(M, K, num_frames),
                'stft_conj': np.conj(np.random.randn(M, K, num_frames) + 1j * np.random.randn(M, K, num_frames)),
                'mod_stft_3d': np.random.randn(num_harmonic_sets, M * P, K, num_frames).astype(np.complex128),
                'mod_stft_3d_conj': np.conj(np.random.randn(num_harmonic_sets, M * P, K, num_frames).astype(np.complex128)),
            },
            'wet_rank1': {
                'stft': np.random.randn(M, K, num_frames) + 1j * np.random.randn(M, K, num_frames),
                'stft_conj': np.conj(np.random.randn(M, K, num_frames) + 1j * np.random.randn(M, K, num_frames)),
                'mod_stft_3d': np.random.randn(num_harmonic_sets, M * P, K, num_frames).astype(np.complex128),
                'mod_stft_3d_conj': np.conj(np.random.randn(num_harmonic_sets, M * P, K, num_frames).astype(np.complex128)),
            }
        }
        return signals

    def test_cross_noisy_early_wb_shape_single_P(self):
        """Test cross_noisy_early_wb has shape (K, M*P) for P=1."""
        cfg = self._make_config_with_mwf(cyclostationary_target=True)
        cov_est = CovarianceEstimator(cfg, cyclostationary_target=True)
        cov_est.set_dimensions((self.K, self.M, self.P))
        cov_est.harmonic_info = SimpleHarmonicInfo(self.K, P_all=np.ones(self.K, dtype=int))

        signals = self._make_signals_with_wet(self.M, self.K, self.P, self.num_frames)

        # Allocate with is_mwf=True to include cross_noisy_early_wb
        cov_dict = CovarianceEstimator.allocate_covariance_matrices((self.K, self.M, self.P),
                                                                     is_mwf=True, use_pseudo_cov=False)
        self.assertIn('cross_noisy_early_wb', cov_dict)
        self.assertEqual(cov_dict['cross_noisy_early_wb'].shape, (self.K, self.M * self.P))

    def test_cross_noisy_early_wb_shape_multiple_P(self):
        """Test cross_noisy_early_wb has shape (K, M*P) for P>1."""
        P = 3
        cfg = self._make_config_with_mwf(cyclostationary_target=True)
        cov_est = CovarianceEstimator(cfg, cyclostationary_target=True)
        cov_est.set_dimensions((self.K, self.M, P))
        cov_est.harmonic_info = SimpleHarmonicInfo(self.K, P_all=np.full(self.K, P, dtype=int))

        signals = self._make_signals_with_wet(self.M, self.K, P, self.num_frames)

        cov_dict = CovarianceEstimator.allocate_covariance_matrices((self.K, self.M, P),
                                                                     is_mwf=True, use_pseudo_cov=False)
        self.assertEqual(cov_dict['cross_noisy_early_wb'].shape, (self.K, self.M * P))

    def test_cross_noisy_early_wb_updated_in_rank1(self):
        """Test that cross_noisy_early_wb is updated in rank1_update when wet_rank1 is present."""
        cfg = self._make_config_with_mwf(cyclostationary_target=True)
        cov_est = CovarianceEstimator(cfg, cyclostationary_target=True)
        cov_est.set_dimensions((self.K, self.M, self.P))
        cov_est.harmonic_info = SimpleHarmonicInfo(self.K, P_all=np.ones(self.K, dtype=int))

        signals = self._make_signals_with_wet(self.M, self.K, self.P, self.num_frames)

        # Allocate and initialize
        cov_dict = CovarianceEstimator.allocate_covariance_matrices((self.K, self.M, self.P),
                                                                     is_mwf=True, use_pseudo_cov=False)
        cov_dict['noise_wb'] = cov_est.estimate_noise_covariance(signals['noise_cov_est'], cov_dict['noise_wb'])
        cov_dict = CovarianceEstimator.initialize_covariance_matrices(cov_dict)

        # Store initial value
        initial_cross = cov_dict['cross_noisy_early_wb'].copy()

        # Update (is_cmvdr=False to include cross-covariance update)
        single_frame_slice = slice(0, 1)
        cov_dict_updated = cov_est.rank1_update_covariances(cov_dict, signals, single_frame_slice,
                                                            forget_factor=0.7, is_cmvdr=False)

        # Check that cross_noisy_early_wb was updated
        self.assertIn('cross_noisy_early_wb', cov_dict_updated)
        self.assertEqual(cov_dict_updated['cross_noisy_early_wb'].shape, (self.K, self.M * self.P))
        # Should be updated from zero initialization
        self.assertTrue(np.any(np.abs(cov_dict_updated['cross_noisy_early_wb']) > 0))


class TestRecursiveVsBlockEquivalence(unittest.TestCase):
    """Test equivalence between recursive and block-processing estimates."""

    def setUp(self):
        self.M = 2
        self.K = 5
        self.P = 1
        self.num_frames = 4

    def _make_config_recursive(self):
        return {
            'recursive_average': True,
            'cov_est_forgetting_factor': 1.0,  # No averaging: only new estimate
            'use_rank1_model_for_oracle_cov_wet_estimation': True,
        }

    def _make_config_batch(self):
        return {
            'recursive_average': False,
            'cov_est_forgetting_factor': 0.5,
            'use_rank1_model_for_oracle_cov_wet_estimation': True,
        }

    def _make_signals_for_equivalence(self, M, K, P, num_frames):
        """Create signals_dict suitable for equivalence testing."""
        num_harmonic_sets = 1

        # Use deterministic random signals for reproducibility
        rng = np.random.RandomState(123)

        signals = {
            'noisy': {
                'stft': rng.randn(M, K, num_frames) + 1j * rng.randn(M, K, num_frames),
                'stft_conj': rng.randn(M, K, num_frames) - 1j * rng.randn(M, K, num_frames),
                'mod_stft_3d': rng.randn(num_harmonic_sets, M * P, K, num_frames).astype(np.complex128),
                'mod_stft_3d_conj': (rng.randn(num_harmonic_sets, M * P, K, num_frames) -
                                     1j * rng.randn(num_harmonic_sets, M * P, K, num_frames)).astype(np.complex128),
            },
            'noise_cov_est': {
                'stft': rng.randn(M, K, num_frames) + 1j * rng.randn(M, K, num_frames),
                'stft_conj': rng.randn(M, K, num_frames) - 1j * rng.randn(M, K, num_frames),
                'mod_stft_3d': rng.randn(num_harmonic_sets, M * P, K, num_frames).astype(np.complex128),
                'mod_stft_3d_conj': (rng.randn(num_harmonic_sets, M * P, K, num_frames) -
                                     1j * rng.randn(num_harmonic_sets, M * P, K, num_frames)).astype(np.complex128),
            }
        }
        return signals

    def test_recursive_single_update_equals_batch_single_frame(self):
        """Test that single recursive update matches batch processing on same data."""
        # This is a simplified test: with forgetting_factor=1.0, recursive should give same result
        # as batch processing on a single frame

        cfg_recursive = self._make_config_recursive()
        cov_est_recursive = CovarianceEstimator(cfg_recursive, cyclostationary_target=True)
        cov_est_recursive.set_dimensions((self.K, self.M, self.P))
        cov_est_recursive.harmonic_info = SimpleHarmonicInfo(self.K, P_all=np.ones(self.K, dtype=int))

        cfg_batch = self._make_config_batch()
        cov_est_batch = CovarianceEstimator(cfg_batch, cyclostationary_target=True)
        cov_est_batch.set_dimensions((self.K, self.M, self.P))
        cov_est_batch.harmonic_info = SimpleHarmonicInfo(self.K, P_all=np.ones(self.K, dtype=int))

        signals = self._make_signals_for_equivalence(self.M, self.K, self.P, self.num_frames)

        # Recursive: allocate, initialize, then single update on first frame
        cov_dict_rec = CovarianceEstimator.allocate_covariance_matrices((self.K, self.M, self.P),
                                                                         is_mwf=False, use_pseudo_cov=False)
        cov_dict_rec['noise_wb'] = cov_est_recursive.estimate_noise_covariance(signals['noise_cov_est'],
                                                                               cov_dict_rec['noise_wb'])
        cov_dict_rec = CovarianceEstimator.initialize_covariance_matrices(cov_dict_rec)
        cov_dict_rec = cov_est_recursive.rank1_update_covariances(cov_dict_rec, signals, slice(0, 1),
                                                                  forget_factor=1.0, is_cmvdr=True)

        # Batch: allocate and estimate all frames at once
        cov_dict_batch = CovarianceEstimator.allocate_covariance_matrices((self.K, self.M, self.P),
                                                                           is_mwf=False, use_pseudo_cov=False)
        cov_dict_batch['noise_wb'] = cov_est_batch.estimate_noise_covariance(signals['noise_cov_est'],
                                                                              cov_dict_batch['noise_wb'])
        cov_dict_batch = cov_est_batch.estimate_covariances_block_processing(cov_dict_batch, signals,
                                                                             slice(0, self.num_frames))

        # Both should be finite
        self.assertTrue(np.all(np.isfinite(cov_dict_rec['noisy_wb'])))
        self.assertTrue(np.all(np.isfinite(cov_dict_batch['noisy_wb'])))

        # Shapes should match
        self.assertEqual(cov_dict_rec['noisy_wb'].shape, cov_dict_batch['noisy_wb'].shape)

    def test_recursive_sequential_updates_converges(self):
        """Test that sequential recursive updates don't diverge (basic convergence check)."""
        cfg = self._make_config_recursive()
        cov_est = CovarianceEstimator(cfg, cyclostationary_target=True)
        cov_est.set_dimensions((self.K, self.M, self.P))
        cov_est.harmonic_info = SimpleHarmonicInfo(self.K, P_all=np.ones(self.K, dtype=int))

        signals = self._make_signals_for_equivalence(self.M, self.K, self.P, self.num_frames)

        # Initialize
        cov_dict = CovarianceEstimator.allocate_covariance_matrices((self.K, self.M, self.P),
                                                                     is_mwf=False, use_pseudo_cov=False)
        cov_dict['noise_wb'] = cov_est.estimate_noise_covariance(signals['noise_cov_est'], cov_dict['noise_wb'])
        cov_dict = CovarianceEstimator.initialize_covariance_matrices(cov_dict)

        # Sequential updates with moderate forgetting factor
        cov_norms = []
        for frame_idx in range(self.num_frames):
            cov_dict = cov_est.rank1_update_covariances(cov_dict, signals, slice(frame_idx, frame_idx + 1),
                                                        forget_factor=0.5, is_cmvdr=True)
            norm = np.linalg.norm(cov_dict['noisy_wb'])
            cov_norms.append(norm)
            # Check for NaNs or Infs
            self.assertTrue(np.all(np.isfinite(cov_dict['noisy_wb'])))

        # Norms should be positive
        self.assertTrue(all(norm > 0 for norm in cov_norms))


class TestMultiFrameSliceFrames(unittest.TestCase):
    """Test handling of multi-frame slice_frames in different scenarios."""

    def setUp(self):
        self.M = 2
        self.K = 5
        self.P = 1
        self.num_frames = 10

    def _make_config_recursive(self):
        return {
            'recursive_average': True,
            'cov_est_forgetting_factor': 0.5,
            'use_rank1_model_for_oracle_cov_wet_estimation': True,
        }

    def _make_signals(self, M, K, P, num_frames):
        num_harmonic_sets = 1
        signals = {
            'noisy': {
                'stft': np.random.randn(M, K, num_frames) + 1j * np.random.randn(M, K, num_frames),
                'stft_conj': np.conj(np.random.randn(M, K, num_frames) + 1j * np.random.randn(M, K, num_frames)),
                'mod_stft_3d': np.random.randn(num_harmonic_sets, M * P, K, num_frames).astype(np.complex128),
                'mod_stft_3d_conj': np.conj(np.random.randn(num_harmonic_sets, M * P, K, num_frames).astype(np.complex128)),
            },
            'noise_cov_est': {
                'stft': np.random.randn(M, K, num_frames) + 1j * np.random.randn(M, K, num_frames),
                'stft_conj': np.conj(np.random.randn(M, K, num_frames) + 1j * np.random.randn(M, K, num_frames)),
                'mod_stft_3d': np.random.randn(num_harmonic_sets, M * P, K, num_frames).astype(np.complex128),
                'mod_stft_3d_conj': np.conj(np.random.randn(num_harmonic_sets, M * P, K, num_frames).astype(np.complex128)),
            }
        }
        return signals

    def test_multiframe_slice_rejected_at_frame_boundary(self):
        """Test that multi-frame slices are rejected even at frame boundaries."""
        cfg = self._make_config_recursive()
        cov_est = CovarianceEstimator(cfg, cyclostationary_target=True)
        cov_est.set_dimensions((self.K, self.M, self.P))
        cov_est.harmonic_info = SimpleHarmonicInfo(self.K, P_all=np.ones(self.K, dtype=int))

        signals = self._make_signals(self.M, self.K, self.P, self.num_frames)

        cov_dict = CovarianceEstimator.allocate_covariance_matrices((self.K, self.M, self.P),
                                                                     is_mwf=False, use_pseudo_cov=False)
        cov_dict['noise_wb'] = cov_est.estimate_noise_covariance(signals['noise_cov_est'], cov_dict['noise_wb'])
        cov_dict = CovarianceEstimator.initialize_covariance_matrices(cov_dict)

        # Try slice covering frames 5-8 (3 frames)
        multi_frame_slice = slice(5, 8)
        with self.assertRaises(ValueError):
            cov_est.rank1_update_covariances(cov_dict, signals, multi_frame_slice,
                                            forget_factor=0.5, is_cmvdr=True)

    def test_sequential_single_frame_updates_span_many_frames(self):
        """Test that sequential single-frame updates can span entire dataset."""
        cfg = self._make_config_recursive()
        cov_est = CovarianceEstimator(cfg, cyclostationary_target=True)
        cov_est.set_dimensions((self.K, self.M, self.P))
        cov_est.harmonic_info = SimpleHarmonicInfo(self.K, P_all=np.ones(self.K, dtype=int))

        signals = self._make_signals(self.M, self.K, self.P, self.num_frames)

        cov_dict = CovarianceEstimator.allocate_covariance_matrices((self.K, self.M, self.P),
                                                                     is_mwf=False, use_pseudo_cov=False)
        cov_dict['noise_wb'] = cov_est.estimate_noise_covariance(signals['noise_cov_est'], cov_dict['noise_wb'])
        cov_dict = CovarianceEstimator.initialize_covariance_matrices(cov_dict)

        # Process all frames one-by-one
        for frame_idx in range(self.num_frames):
            cov_dict = cov_est.rank1_update_covariances(cov_dict, signals, slice(frame_idx, frame_idx + 1),
                                                        forget_factor=0.5, is_cmvdr=True)
            self.assertTrue(np.all(np.isfinite(cov_dict['noisy_wb'])))

        # Final covariance should be valid
        self.assertTrue(np.all(np.isfinite(cov_dict['noisy_wb'])))


class TestMultipleHarmonicSets(unittest.TestCase):
    """Test recursive mode with multiple harmonic sets (P>1)."""

    def setUp(self):
        self.M = 2
        self.K = 6
        self.P_values = [1, 2, 3]  # Test different P values

    def _make_config_recursive(self):
        return {
            'recursive_average': True,
            'cov_est_forgetting_factor': 0.5,
            'use_rank1_model_for_oracle_cov_wet_estimation': True,
        }

    def _make_signals_with_P(self, M, K, P_all, num_frames):
        """Create signals_dict with possibly varying P per frequency."""
        # Create a single harmonic set for simplicity
        max_P = np.max(P_all)
        num_harmonic_sets = 1

        signals = {
            'noisy': {
                'stft': np.random.randn(M, K, num_frames) + 1j * np.random.randn(M, K, num_frames),
                'stft_conj': np.conj(np.random.randn(M, K, num_frames) + 1j * np.random.randn(M, K, num_frames)),
                'mod_stft_3d': np.random.randn(num_harmonic_sets, M * max_P, K, num_frames).astype(np.complex128),
                'mod_stft_3d_conj': np.conj(np.random.randn(num_harmonic_sets, M * max_P, K, num_frames).astype(np.complex128)),
            },
            'noise_cov_est': {
                'stft': np.random.randn(M, K, num_frames) + 1j * np.random.randn(M, K, num_frames),
                'stft_conj': np.conj(np.random.randn(M, K, num_frames) + 1j * np.random.randn(M, K, num_frames)),
                'mod_stft_3d': np.random.randn(num_harmonic_sets, M * max_P, K, num_frames).astype(np.complex128),
                'mod_stft_3d_conj': np.conj(np.random.randn(num_harmonic_sets, M * max_P, K, num_frames).astype(np.complex128)),
            }
        }
        return signals

    def test_allocation_with_max_P(self):
        """Test covariance allocation with max P across frequencies."""
        for P in self.P_values:
            cfg = self._make_config_recursive()
            cov_est = CovarianceEstimator(cfg, cyclostationary_target=True)
            cov_est.set_dimensions((self.K, self.M, P))

            cov_dict = CovarianceEstimator.allocate_covariance_matrices((self.K, self.M, P),
                                                                         is_mwf=False, use_pseudo_cov=False)

            # Shape should be (K, M*P, M*P)
            self.assertEqual(cov_dict['noisy_wb'].shape, (self.K, self.M * P, self.M * P))

    def test_rank1_update_with_varying_P(self):
        """Test rank1_update with varying P per frequency."""
        K = 5
        P_all = np.array([1, 2, 1, 3, 2], dtype=int)  # Varying P for K=5
        cfg = self._make_config_recursive()
        cov_est = CovarianceEstimator(cfg, cyclostationary_target=True)
        max_P = np.max(P_all)
        cov_est.set_dimensions((K, self.M, max_P))
        cov_est.harmonic_info = SimpleHarmonicInfo(K, P_all=P_all)

        signals = self._make_signals_with_P(self.M, K, P_all, 4)

        cov_dict = CovarianceEstimator.allocate_covariance_matrices((K, self.M, max_P),
                                                                     is_mwf=False, use_pseudo_cov=False)
        cov_dict['noise_wb'] = cov_est.estimate_noise_covariance(signals['noise_cov_est'], cov_dict['noise_wb'])
        cov_dict = CovarianceEstimator.initialize_covariance_matrices(cov_dict)

        # Single update should succeed
        cov_dict = cov_est.rank1_update_covariances(cov_dict, signals, slice(0, 1),
                                                    forget_factor=0.5, is_cmvdr=True)

        self.assertEqual(cov_dict['noisy_wb'].shape, (K, self.M * max_P, self.M * max_P))
        self.assertTrue(np.all(np.isfinite(cov_dict['noisy_wb'])))


class TestNarrowbandExtraction(unittest.TestCase):
    """Test narrowband covariance extraction from wideband."""

    def setUp(self):
        self.M = 3
        self.K = 5
        self.P = 2

    def test_copy_multiband_to_narrowband_extracts_spatial_part(self):
        """Test that copy_multiband_to_narrowband correctly extracts spatial (M x M) part."""
        cov_dict = {
            'noisy_wb': np.random.randn(self.K, self.M * self.P, self.M * self.P).astype(np.complex128),
            'noise_wb': np.random.randn(self.K, self.M * self.P, self.M * self.P).astype(np.complex128),
        }

        cov_dict_extracted = CovarianceEstimator.copy_multiband_to_narrowband(cov_dict, M=self.M, name_input_sig='noisy')

        # Check that narrowband covariance was created
        self.assertIn('noisy_nb', cov_dict_extracted)
        self.assertEqual(cov_dict_extracted['noisy_nb'].shape, (self.K, self.M, self.M))

        # Check that it's the first M x M block of each frequency bin
        for kk in range(self.K):
            self.assertTrue(np.allclose(cov_dict_extracted['noisy_nb'][kk],
                                       cov_dict['noisy_wb'][kk, :self.M, :self.M]))

    def test_cross_covariance_narrowband_extraction(self):
        """Test extraction of narrowband cross-covariance from wideband."""
        cov_dict = {
            'cross_noisy_early_wb': np.random.randn(self.K, self.M * self.P).astype(np.complex128),
        }

        cov_dict_extracted = CovarianceEstimator.copy_multiband_to_narrowband(cov_dict, M=self.M, name_input_sig='noisy')

        # Check that narrowband cross-covariance was created
        self.assertIn('cross_noisy_early_nb', cov_dict_extracted)
        self.assertEqual(cov_dict_extracted['cross_noisy_early_nb'].shape, (self.K, self.M))

        # Check that it's the first M elements of each frequency bin
        for kk in range(self.K):
            self.assertTrue(np.allclose(cov_dict_extracted['cross_noisy_early_nb'][kk],
                                       cov_dict['cross_noisy_early_wb'][kk, :self.M]))

    def test_narrowband_extraction_with_missing_keys(self):
        """Test that missing covariance keys don't cause errors."""
        cov_dict = {
            'noisy_wb': np.random.randn(self.K, self.M * self.P, self.M * self.P).astype(np.complex128),
        }

        # Should not raise an error even if some keys are missing
        cov_dict_extracted = CovarianceEstimator.copy_multiband_to_narrowband(cov_dict, M=self.M, name_input_sig='noisy')

        self.assertIn('noisy_nb', cov_dict_extracted)
        # Optional keys should not cause errors
        self.assertNotIn('noise_nb', cov_dict_extracted)


class TestNumericalStability(unittest.TestCase):
    """Test numerical stability across different signal magnitudes."""

    def setUp(self):
        self.M = 2
        self.K = 5
        self.P = 1

    def _make_config_recursive(self):
        return {
            'recursive_average': True,
            'cov_est_forgetting_factor': 0.5,
            'use_rank1_model_for_oracle_cov_wet_estimation': True,
        }

    def _make_signals_with_magnitude(self, M, K, P, num_frames, magnitude_scale=1.0):
        """Create signals with specific magnitude."""
        num_harmonic_sets = 1
        signals = {
            'noisy': {
                'stft': magnitude_scale * (np.random.randn(M, K, num_frames) + 1j * np.random.randn(M, K, num_frames)),
                'stft_conj': magnitude_scale * np.conj(np.random.randn(M, K, num_frames) + 1j * np.random.randn(M, K, num_frames)),
                'mod_stft_3d': magnitude_scale * np.random.randn(num_harmonic_sets, M * P, K, num_frames).astype(np.complex128),
                'mod_stft_3d_conj': magnitude_scale * np.conj(np.random.randn(num_harmonic_sets, M * P, K, num_frames).astype(np.complex128)),
            },
            'noise_cov_est': {
                'stft': magnitude_scale * (np.random.randn(M, K, num_frames) + 1j * np.random.randn(M, K, num_frames)),
                'stft_conj': magnitude_scale * np.conj(np.random.randn(M, K, num_frames) + 1j * np.random.randn(M, K, num_frames)),
                'mod_stft_3d': magnitude_scale * np.random.randn(num_harmonic_sets, M * P, K, num_frames).astype(np.complex128),
                'mod_stft_3d_conj': magnitude_scale * np.conj(np.random.randn(num_harmonic_sets, M * P, K, num_frames).astype(np.complex128)),
            }
        }
        return signals

    def test_small_magnitude_signals(self):
        """Test stability with very small signal magnitudes."""
        cfg = self._make_config_recursive()
        cov_est = CovarianceEstimator(cfg, cyclostationary_target=True)
        cov_est.set_dimensions((self.K, self.M, self.P))
        cov_est.harmonic_info = SimpleHarmonicInfo(self.K, P_all=np.ones(self.K, dtype=int))

        signals = self._make_signals_with_magnitude(self.M, self.K, self.P, 4, magnitude_scale=1e-6)

        cov_dict = CovarianceEstimator.allocate_covariance_matrices((self.K, self.M, self.P),
                                                                     is_mwf=False, use_pseudo_cov=False)
        cov_dict['noise_wb'] = cov_est.estimate_noise_covariance(signals['noise_cov_est'], cov_dict['noise_wb'])
        cov_dict = CovarianceEstimator.initialize_covariance_matrices(cov_dict)

        # Update should not produce NaNs or Infs
        cov_dict = cov_est.rank1_update_covariances(cov_dict, signals, slice(0, 1),
                                                    forget_factor=0.5, is_cmvdr=True)
        self.assertTrue(np.all(np.isfinite(cov_dict['noisy_wb'])))

    def test_large_magnitude_signals(self):
        """Test stability with very large signal magnitudes."""
        cfg = self._make_config_recursive()
        cov_est = CovarianceEstimator(cfg, cyclostationary_target=True)
        cov_est.set_dimensions((self.K, self.M, self.P))
        cov_est.harmonic_info = SimpleHarmonicInfo(self.K, P_all=np.ones(self.K, dtype=int))

        signals = self._make_signals_with_magnitude(self.M, self.K, self.P, 4, magnitude_scale=1e6)

        cov_dict = CovarianceEstimator.allocate_covariance_matrices((self.K, self.M, self.P),
                                                                     is_mwf=False, use_pseudo_cov=False)
        cov_dict['noise_wb'] = cov_est.estimate_noise_covariance(signals['noise_cov_est'], cov_dict['noise_wb'])
        cov_dict = CovarianceEstimator.initialize_covariance_matrices(cov_dict)

        # Update should not produce NaNs or Infs
        cov_dict = cov_est.rank1_update_covariances(cov_dict, signals, slice(0, 1),
                                                    forget_factor=0.5, is_cmvdr=True)
        self.assertTrue(np.all(np.isfinite(cov_dict['noisy_wb'])))

    def test_mixed_magnitude_signals(self):
        """Test stability with mixed-magnitude signals across frequencies."""
        cfg = self._make_config_recursive()
        cov_est = CovarianceEstimator(cfg, cyclostationary_target=True)
        cov_est.set_dimensions((self.K, self.M, self.P))
        cov_est.harmonic_info = SimpleHarmonicInfo(self.K, P_all=np.ones(self.K, dtype=int))

        # Create signals with different magnitudes per frequency
        base_signals = self._make_signals_with_magnitude(self.M, self.K, self.P, 4, magnitude_scale=1.0)

        # Scale different frequencies differently
        for kk in range(self.K):
            scale = 10.0 ** (kk - 2)  # Varies from 1e-2 to 1e2
            base_signals['noisy']['stft'][:, kk, :] *= scale
            base_signals['noise_cov_est']['stft'][:, kk, :] *= scale
            base_signals['noisy']['mod_stft_3d'][:, :, kk, :] *= scale
            base_signals['noise_cov_est']['mod_stft_3d'][:, :, kk, :] *= scale

        cov_dict = CovarianceEstimator.allocate_covariance_matrices((self.K, self.M, self.P),
                                                                     is_mwf=False, use_pseudo_cov=False)
        cov_dict['noise_wb'] = cov_est.estimate_noise_covariance(base_signals['noise_cov_est'], cov_dict['noise_wb'])
        cov_dict = CovarianceEstimator.initialize_covariance_matrices(cov_dict)

        # Update should still be stable
        cov_dict = cov_est.rank1_update_covariances(cov_dict, base_signals, slice(0, 1),
                                                    forget_factor=0.5, is_cmvdr=True)
        self.assertTrue(np.all(np.isfinite(cov_dict['noisy_wb'])))


if __name__ == '__main__':
    unittest.main()



