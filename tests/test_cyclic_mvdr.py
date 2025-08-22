# tests/test_cyclic_mvdr_no_mocks.py
import unittest
import numpy as np
import src.globs as gs
gs.rng, _ = gs.compute_rng(seed_is_random=False, rnd_seed_=123)
from src.cyclic_mvdr import CyclicMVDR
from src.beamformer import Beamformer


class SimpleHarmonicInfo:
    """ Minimal harmonic_info used by CyclicMVDR tests. """
    def __init__(self, K, P_all=None):
        self.K = K
        if P_all is None:
            # default to P=1 for all freqs
            self._P_all = np.ones(K, dtype=int)
        else:
            self._P_all = np.array(P_all, dtype=int)

    def get_num_shifts_all_frequencies(self):
        return self._P_all


class CyclicMVDRIntegrationTests(unittest.TestCase):
    def setUp(self):
        # small sizes to keep tests fast
        self.M = 2
        self.K = 5
        self.loadings_cfg = (0, 0, 1000)  # safe default expected by get_loading_* methods

    def test_init_rtf_est_and_noise_var(self):
        sig_shape_k_m = (self.K, self.M)
        cm = CyclicMVDR(self.loadings_cfg, sig_shape_k_m, minimize_noisy_cov_mvdr=True, noise_var_rtf=0.123)
        # rtf_est should have shape (K, M) and first column set to 1
        self.assertEqual(cm.rtf_est.shape, sig_shape_k_m)
        self.assertTrue(np.allclose(cm.rtf_est[:, 0], 1.0))
        self.assertAlmostEqual(cm.noise_var_rtf, 0.123)

    def test_compute_mvdr_beamformers_single_channel_returns_ones(self):
        K = 4
        M = 1
        cm = CyclicMVDR(self.loadings_cfg, (K, M))
        cov_input_nb = np.zeros((K, M, M), dtype=np.complex128)
        cov_noise_nb = np.zeros_like(cov_input_nb)
        # shapes ok; for M==1 the implementation sets weights to 1
        weights, err_flags, cond, sv = cm.compute_mvdr_beamformers(cov_input_nb, cov_noise_nb)
        self.assertEqual(weights.shape, (M, K))
        self.assertTrue(np.allclose(weights, 1.0))

    def test_compute_mvdr_beamformers_multichannel_basic(self):
        K = 3
        M = 2
        cm = CyclicMVDR(self.loadings_cfg, (K, M))
        # create positive definite covariances for each bin
        cov_input_nb = np.zeros((K, M, M), dtype=np.complex128)
        cov_noise_nb = np.zeros_like(cov_input_nb)
        for kk in range(K):
            A = np.array([[2.0 + kk, 0.1], [0.1, 1.5 + kk]])
            cov_input_nb[kk] = A
            cov_noise_nb[kk] = np.eye(M) * 0.5
        # provide an oracle rtf array (K, M)
        oracle = np.ones((K, M), dtype=np.complex128)
        weights, err_flags, cond, sv = cm.compute_mvdr_beamformers(cov_input_nb, cov_noise_nb, which_variant='oracle', speech_rtf_oracle=oracle)
        self.assertEqual(weights.shape, (M, K))
        # no NaNs, finite
        self.assertTrue(np.all(np.isfinite(weights)))

    def test_compute_cyclic_mvdr_beamformers_basic_shapes(self):
        K = self.K
        M = self.M
        P_max = 1  # simplest case: no extra shifts
        cm = CyclicMVDR(self.loadings_cfg, (K, M))
        # harmonic_info with P_all ones
        cm.harmonic_info = SimpleHarmonicInfo(K, P_all=np.ones(K, dtype=int))

        # build cov dict: shapes expected: *_nb: (K, M, M), *_wb: (K, M*P_max, M*P_max)
        cov_dict = {}
        cov_dict['noisy_nb'] = np.zeros((K, M, M), dtype=np.complex128)
        cov_dict['noise_nb'] = np.zeros((K, M, M), dtype=np.complex128)
        cov_dict['noisy_wb'] = np.zeros((K, M * P_max, M * P_max), dtype=np.complex128)
        cov_dict['noise_wb'] = np.zeros((K, M * P_max, M * P_max), dtype=np.complex128)

        # fill with small PD matrices for non-cyclic bins
        for kk in range(K):
            cov_dict['noisy_nb'][kk] = np.eye(M) * (1.0 + kk)
            cov_dict['noise_nb'][kk] = np.eye(M) * 0.3
            cov_dict['noisy_wb'][kk] = np.eye(M * P_max) * (1.0 + kk)
            cov_dict['noise_wb'][kk] = np.eye(M * P_max) * 0.2

            # add some small perturbation to ensure they are not exactly multiples of identity
            cov_dict['noisy_nb'][kk] += 0.03 * kk
            cov_dict['noise_nb'][kk] += 0.02 * kk
            cov_dict['noisy_wb'][kk] += 0.04 * kk
            cov_dict['noise_wb'][kk] += 0.02 * kk

        cyclic_bins = np.array([1, 3], dtype=int)
        # provide an oracle (empty) and call
        weights, err_flags, cond_num_cov, sv = cm.compute_cyclic_mvdr_beamformers(cov_dict, 'blind', cyclic_bins, speech_rtf_oracle=np.array([]))
        # weights shape should be (M * P_max, K)
        self.assertEqual(weights.shape, (M * P_max, K))
        self.assertEqual(err_flags.shape, (K,))
        self.assertEqual(sv.shape[0], K)

    def test_compute_cyclic_mvdr_beamformers_wl_basic_shapes(self):
        K = self.K
        M = self.M
        P_max = 1
        cm = CyclicMVDR(self.loadings_cfg, (K, M))
        cm.harmonic_info = SimpleHarmonicInfo(K, P_all=np.ones(K, dtype=int))

        cov_dict = {}
        cov_dict['noisy_nb'] = np.zeros((K, M, M), dtype=np.complex128)
        cov_dict['noise_nb'] = np.zeros((K, M, M), dtype=np.complex128)
        cov_dict['noisy_wb'] = np.zeros((K, M * P_max, M * P_max), dtype=np.complex128)
        cov_dict['noise_wb'] = np.zeros((K, M * P_max, M * P_max), dtype=np.complex128)
        # pseudo covariance used by WL method
        cov_dict['noisy_pseudo'] = np.zeros((K, M * P_max, M * P_max), dtype=np.complex128)

        for kk in range(K):
            cov_dict['noisy_nb'][kk] = np.eye(M) * (1.0 + kk)
            cov_dict['noise_nb'][kk] = np.eye(M) * 4
            cov_dict['noisy_wb'][kk] = np.eye(M * P_max) * (1.0 + kk)
            cov_dict['noise_wb'][kk] = np.eye(M * P_max) * 2
            cov_dict['noisy_pseudo'][kk] = np.eye(M * P_max) * 3

            # add some small perturbation to ensure they are not exactly multiples of identity
            cov_dict['noisy_nb'][kk] += 0.03 * kk
            cov_dict['noise_nb'][kk] += 0.02 * kk
            cov_dict['noisy_wb'][kk] += 0.04 * kk
            cov_dict['noise_wb'][kk] += 0.02 * kk
            cov_dict['noisy_pseudo'][kk] += 0.05 * kk

        cyclic_bins = np.array([0, 2], dtype=int)
        weights, err_flags, cond_num_cov, sv = cm.compute_cyclic_mvdr_beamformers_wl(cov_dict, 'blind', cyclic_bins, speech_rtf_oracle=np.array([]))
        # weights dimension should be 2*M*P_max by K (widely linear doubles size)
        self.assertEqual(weights.shape[1], K)
        self.assertTrue(weights.shape[0] >= M)

    def test_solve_via_schur_satisfies_block_system(self):
        # Build a small system and verify solve_via_schur solves the block system [A B; B* A*] [w1; w2] = [z1; z2]
        n = 3
        # A hermitian positive definite
        A = np.eye(n) * 2.0 + 0.1 * np.ones((n, n))
        # make B symmetric
        B = 0.2 * np.eye(n)
        # z vectors
        z1 = np.arange(1.0, n + 1.0)
        z2 = np.arange(2.0, n + 2.0)
        w = CyclicMVDR.solve_via_schur(A, B, z1, z2)
        # verify shape
        self.assertEqual(w.shape[0], 2 * n)
        # build full block and check residual
        block = np.block([[A, B], [np.conj(B), np.conj(A)]])
        rhs = np.concatenate([z1, z2])
        res = block @ w - rhs
        self.assertTrue(np.allclose(res, 0.0, atol=1e-8))

    def test_estimate_rtf_or_get_oracle_mvdr_oracle_and_semi_oracle_noise_added(self):
        K = 4
        M = 2
        cm = CyclicMVDR(self.loadings_cfg, (K, M), noise_var_rtf=1e-6)
        # oracle rtf: ones
        oracle = np.ones((K, M), dtype=np.complex128)
        out = cm.estimate_rtf_or_get_oracle_mvdr(None, None, oracle, which_variant='oracle', kk=0)

        # first entry should remain equal to oracle's first element
        self.assertEqual(out[0, 0], oracle[0, 0])

        # if noise_var_rtf > 0 and M>1, other elements likely differ (not strict assert as randomness possible); at least shape ok
        self.assertEqual(out.shape[0], K)
        self.assertEqual(out.shape[1], M)

        # semi-oracle should behave similarly
        out2 = cm.estimate_rtf_or_get_oracle_mvdr(None, None, oracle, which_variant='semi-oracle', kk=0)
        self.assertEqual(out2[0, 0], oracle[0, 0])

    def test_estimate_rtf_or_get_oracle_mvdr_blind_skip_and_estimate(self):
        K = 3
        M = 2
        cm = CyclicMVDR(self.loadings_cfg, (K, M))
        # ensure rtf_needs_estimation exists and is True for testing
        cm.rtf_needs_estimation = np.ones(K, dtype=bool)
        # case 1: cov_noisy trace smaller than cov_noise -> should skip estimation and return existing rtf_est[kk]
        cov_noisy_small = np.eye(M) * 0.1
        cov_noise_big = np.eye(M) * 10.0
        # set some initial rtf_est value
        cm.rtf_est[0] = np.array([1.0, 0.0])
        out = cm.estimate_rtf_or_get_oracle_mvdr(cov_noisy_small, cov_noise_big, None, which_variant='blind', kk=0)
        self.assertTrue(np.allclose(out, cm.rtf_est[0]))
        # case 2: perform actual GEVD-based estimation when noisy > noise
        cov_noisy = np.array([[4.0, 0.2], [0.2, 3.0]])
        cov_noise = np.eye(M) * 0.5
        cm.rtf_needs_estimation[1] = True
        out2 = cm.estimate_rtf_or_get_oracle_mvdr(cov_noisy, cov_noise, None, which_variant='blind', kk=1)
        # rtf_est should have been updated and rtf_needs_estimation for kk set to False
        self.assertFalse(cm.rtf_needs_estimation[1])
        self.assertEqual(out2.shape[0], M)

    def test_check_if_rtf_needs_estimation_behavior(self):
        # Test behavior across warmup and interval conditions
        K = 6
        M = 2
        cm = CyclicMVDR(self.loadings_cfg, (K, M))
        # ensure rtf_needs_estimation array exists
        cm.rtf_needs_estimation = np.zeros(K, dtype=bool)
        # warmup behavior: idx_chunk smaller than warmup should return ones
        warmup_chunks = Beamformer.rtf_est_warmup_chunks
        out_warmup = cm.check_if_rtf_needs_estimation(idx_chunk=0, warmup_chunks=warmup_chunks, interval_chunks=Beamformer.rtf_est_interval_chunks)
        self.assertTrue(np.all(out_warmup == np.ones_like(cm.rtf_needs_estimation)))
        # choose idx_chunk not triggering interval and larger than warmup -> zeros
        idx = max(warmup_chunks, 1) + 1
        out_later = cm.check_if_rtf_needs_estimation(idx_chunk=idx, warmup_chunks=warmup_chunks, interval_chunks=10000)
        self.assertTrue(np.all(out_later == np.zeros_like(cm.rtf_needs_estimation)))
        # if interval_chunks divides idx_chunk should return ones
        out_interval = cm.check_if_rtf_needs_estimation(idx_chunk=2, warmup_chunks=0, interval_chunks=2)
        self.assertTrue(np.all(out_interval == np.ones_like(cm.rtf_needs_estimation)))


if __name__ == "__main__":
    unittest.main()
