import io
import sys
import unittest
import numpy as np
from cmvdr import globs as gs
gs.rng, _ = gs.compute_rng(seed_is_random=False, rnd_seed_=123)
from cmvdr.beamformer_manager import BeamformerManager, F0ChangeAmount


class SimpleHarmonicInfo:
    """
    lightweight helper used only as a harmonic_info object for beamforming tests.
    It provides the minimal API used by BeamformerManager.beamform_signals.
    """
    def __init__(self, harmonic_bins, harmonic_sets_before_coh=False):
        self.harmonic_bins = harmonic_bins
        self.harmonic_sets_before_coh = harmonic_sets_before_coh

    def get_harmonic_set_and_num_shifts(self, kk, before_coherence=False):
        # For the tests we always return the non-modulated single set: index 0, P=1
        return 0, 1


class BeamformerManagerIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:
        # Use small shapes so initialization is fast
        self.sig_shape_k_m = (2, 3)

    def test_init_creates_components_when_requested(self):
        names = ['mvdr_blind', 'mwf_oracle', 'mf_semi-oracle']
        mgr = BeamformerManager(names, sig_shape_k_m=self.sig_shape_k_m)
        # components should be created according to names
        self.assertIsNotNone(mgr.mvdr, "mvdr instance should be created")
        self.assertIsNotNone(mgr.mwf, "mwf instance should be created")
        self.assertIsNotNone(mgr.other_bf, "other beamformer instance should be created")

    def test_harmonic_info_setter_updates_children(self):
        mgr = BeamformerManager(['mvdr_blind', 'mwf_blind'], sig_shape_k_m=self.sig_shape_k_m)
        hi = SimpleHarmonicInfo([0, 1, 2])
        mgr.harmonic_info = hi
        # ensure the manager stores it and child beamformers receive the reference
        self.assertIs(mgr.harmonic_info, hi)
        if mgr.mwf is not None:
            self.assertIs(mgr.mwf.harmonic_info, hi)
        if mgr.mvdr is not None:
            self.assertIs(mgr.mvdr.harmonic_info, hi)

    def test_make_weights_dict_symmetric_around_central_frequency(self):
        K_nfft = 8  # odd -> central index 3
        M = 2
        w = np.zeros((M, K_nfft), dtype=np.complex128)

        # set non-zero values on the left side
        w[:, 1:K_nfft // 2] = 1 + 1j
        weights = {'mvdr_test': w.copy()}
        out = BeamformerManager.make_weights_dict_symmetric_around_central_frequency(K_nfft, weights)

        # central bin must be purely real
        self.assertTrue(np.allclose(out['mvdr_test'][:, K_nfft // 2].imag, 0.0))

        # symmetry: right half equals conjugate reversed left half
        left = out['mvdr_test'][:, 1:K_nfft // 2]
        right = out['mvdr_test'][:, K_nfft // 2 + 1:]
        self.assertTrue(np.allclose(right, left[:, ::-1].conj()))

    def test_make_signals_dict_symmetric_around_central_frequency_basic(self):
        K_nfft = 8
        frames = 3
        sig = np.ones((K_nfft, frames), dtype=np.complex128)
        sig[1:K_nfft // 2] = (1 + 2j)  # some complex values in the lower freq bins
        signals = {'s': sig.copy()}
        out = BeamformerManager.make_signals_dict_symmetric_around_central_frequency(K_nfft, signals)

        # central bin must be real-valued
        self.assertTrue(np.allclose(out['s'][K_nfft // 2], out['s'][K_nfft // 2].real))

        # symmetry check for slices
        left = out['s'][1:K_nfft // 2]
        right = out['s'][K_nfft // 2 + 1:]
        self.assertTrue(np.allclose(right, left[::-1].conj()))

    def test_select_correlated_columns_min_correlation_zero_returns_same(self):
        M = 2
        PM = 4
        cov_noisy = np.eye(PM, dtype=np.complex128)
        cov_noise = 0.5 * np.eye(PM, dtype=np.complex128)
        covn_high, covnoise_high, cols = BeamformerManager.select_correlated_columns(cov_noisy, cov_noise, 0.0, M)
        self.assertTrue(np.allclose(covn_high, cov_noisy))
        self.assertTrue(np.allclose(covnoise_high, cov_noise))
        self.assertTrue(np.all(cols == np.arange(PM)))

    def test_select_correlated_columns_positive_threshold_smaller_matrix(self):
        # Build PM=M*P with distinguishable block means so some columns are selected
        M = 2
        P = 3
        PM = M * P
        cov_noisy = np.zeros((PM, PM), dtype=np.complex128)
        cov_noise = np.zeros_like(cov_noisy)
        for p in range(P):
            for q in range(P):
                val = p + q + 1.0
                cov_noisy[p*M:(p+1)*M, q*M:(q+1)*M] = np.eye(M) * val
                cov_noise[p*M:(p+1)*M, q*M:(q+1)*M] = np.eye(M) * (val * 0.1)
        covn_high, covnoise_high, cols = BeamformerManager.select_correlated_columns(cov_noisy, cov_noise, 0.2, M)
        # returned matrix should be no larger than original and indices inside bounds
        self.assertTrue(covn_high.shape[0] <= PM)
        self.assertTrue(np.all(cols >= 0) and np.all(cols < PM))

    def test_get_beamformers_names_and_infer_beamforming_methods(self):
        methods_dict = {'mvdr': True, 'mwf': True, 'cmwf': False}
        variants = ['blind', 'semi-oracle', 'oracle']
        bf_names = BeamformerManager.get_beamformers_names(methods_dict, variants)
        keys = list(bf_names.keys())
        # expected keys present
        self.assertIn('mvdr_blind', keys)
        self.assertIn('mwf_oracle', keys)
        # infer should return only selected True methods for the requested variants
        config = {'methods_dict': methods_dict, 'variants': ['blind', 'oracle']}
        inferred = BeamformerManager.infer_beamforming_methods(config)
        self.assertNotIn('cmwf_blind', inferred)
        self.assertTrue(any('mvdr' in s for s in inferred))

    def test_is_narrowband_beamformer(self):
        self.assertTrue(BeamformerManager.is_narrowband_beamformer('mvdr_blind'))
        self.assertFalse(BeamformerManager.is_narrowband_beamformer('cmvdr_oracle'))
        self.assertFalse(BeamformerManager.is_narrowband_beamformer('cmvdr-wl_oracle'))

    def test_beamform_signals_narrowband_path(self):
        mgr = BeamformerManager(['mvdr_blind'], sig_shape_k_m=self.sig_shape_k_m)
        mgr.harmonic_info = SimpleHarmonicInfo([0, 1])
        M = 2
        K = 3
        total_frames = 5
        noisy_stft = np.zeros((M, K, total_frames), dtype=np.complex128)
        # populate so that dot product is easy to verify
        for m in range(M):
            for kk in range(K):
                noisy_stft[m, kk, :] = (m + 1) * (kk + 1)
        noisy_mod = np.zeros((1, M, K, total_frames), dtype=np.complex128)  # unused by narrowband
        weights = {'mvdr_blind': np.ones((M, K), dtype=np.complex128)}
        slice_frames = slice(0, 3)
        bfd = mgr.beamform_signals(noisy_stft, noisy_mod, slice_frames, weights, mod_amount=F0ChangeAmount.no_change)
        self.assertIn('mvdr_blind', bfd)
        out = bfd['mvdr_blind']
        expected_shape = (K, min(slice_frames.stop - slice_frames.start, noisy_stft.shape[-1] - slice_frames.start))
        self.assertEqual(out.shape, expected_shape)
        # verify numeric for one bin/frame
        kk = 1
        frame_idx = 0
        expected = np.conj(weights['mvdr_blind'][:, kk]).T @ noisy_stft[:, kk, slice_frames][..., frame_idx]
        self.assertAlmostEqual(out[kk, frame_idx], expected)

    def test_beamform_signals_cyclic_path(self):
        mgr = BeamformerManager(['cmvdr_blind'], sig_shape_k_m=self.sig_shape_k_m)
        mgr.harmonic_info = SimpleHarmonicInfo([0, 1, 2])
        M = 2
        K = 3
        total_frames = 4
        noisy_stft = np.zeros((M, K, total_frames), dtype=np.complex128)
        # create modulated signals with simple pattern
        noisy_mod = np.zeros((1, M, K, total_frames), dtype=np.complex128)
        for m in range(M):
            for kk in range(K):
                noisy_mod[0, m, kk, :] = (m + 1) * (kk + 1)
        # weights shaped (M*P, K) but P=1 from SimpleHarmonicInfo
        weights = {'cmvdr_blind': np.ones((M, K), dtype=np.complex128) * 2.0}
        slice_frames = slice(0, 2)
        bfd = mgr.beamform_signals(noisy_stft, noisy_mod, slice_frames, weights, mod_amount=F0ChangeAmount.no_change)
        self.assertIn('cmvdr_blind', bfd)
        out = bfd['cmvdr_blind']
        expected_shape = (K, min(slice_frames.stop - slice_frames.start, noisy_stft.shape[-1] - slice_frames.start))
        self.assertEqual(out.shape, expected_shape)
        # check values for first bin
        kk = 0
        processed_sig = noisy_mod[0, :M, kk, slice_frames]
        expected_vec = np.conj(weights['cmvdr_blind'][:, kk]).T @ processed_sig
        self.assertTrue(np.allclose(out[kk], expected_vec))

    def test_check_beamformed_signals_non_zero_prints_message_when_zero(self):
        bfd = {'bf1': np.zeros((3, 2))}
        signals_unproc = {'wet_rank1': {'stft': np.array([[1.0], [0.0]])}}
        buf = io.StringIO()
        old_stdout = sys.stdout
        try:
            sys.stdout = buf
            BeamformerManager.check_beamformed_signals_non_zero(bfd, signals_unproc)
        finally:
            sys.stdout = old_stdout
        self.assertIn('Beamformed signal bf1 is all zeros.', buf.getvalue())

    def test_use_old_weights_if_error_raises(self):
        mgr = BeamformerManager(['mwf_blind'], sig_shape_k_m=self.sig_shape_k_m)
        with self.assertRaises(NotImplementedError):
            mgr.use_old_weights_if_error({}, {}, {})


if __name__ == '__main__':
    unittest.main()

